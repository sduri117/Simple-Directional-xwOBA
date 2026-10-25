"""Append Stage 3 correlations and create simple-directional diagnostic plots."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import matplotlib
import numpy as np
import polars as pl
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from sd_experiment_config import (  # noqa: E402
    CALCULATED_XWOBA_DIRECTORY,
    CORRELATION_RESULTS_PATH,
    PREDICTION_YEARS,
    SD_CONTACT_YEAR_AHEAD_PLOT_PATH,
    SD_VS_LOCAL_PLAYER_SEASON_PLOT_PATH,
    SD_VS_MLB_PLAYER_SEASON_PLOT_PATH,
    SD_XWOBA_COLUMN,
    SD_XWOBA_CONTACT_COLUMN,
    SD_YEAR_AHEAD_PLOT_PATH,
    SEASON_LEVEL_XWOBA_PATH,
    WOBA_CONSTANTS_PATH,
    XWOBA_LOCAL_COLUMN,
)

FloatArray = NDArray[np.float64]
RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "analysis_level",
    "comparison",
    "cohort",
    "x_variable",
    "y_variable",
    "n",
    "pearson_r",
    "r_squared",
)
SEASON_COLUMNS: Final[tuple[str, ...]] = (
    "year",
    "batter_name",
    "batter_id",
    "wOBA",
    "xwOBA-MLB",
    XWOBA_LOCAL_COLUMN,
    "wOBAcon",
    "xwOBAcon-MLB",
    "xwOBAcon-local",
    SD_XWOBA_COLUMN,
    SD_XWOBA_CONTACT_COLUMN,
)
PLOT_DPI: Final[int] = 180


class AnalysisError(RuntimeError):
    """Raised when Stage 3 analysis input or output is invalid."""


class AnalysisOutputExistsError(AnalysisError):
    """Raised when replacing an analysis artifact was not authorized."""


@dataclass(frozen=True, slots=True)
class CorrelationStatistics:
    """Pearson correlation statistics for one paired sample."""

    n: int
    pearson_r: float
    r_squared: float


@dataclass(frozen=True, slots=True)
class BattedBallWeights:
    """Batted-ball outcome weights for one season."""

    year: int
    single: float
    double: float
    triple: float
    home_run: float


@dataclass(frozen=True, slots=True)
class AnalysisSummary:
    """Locations and row counts for completed Stage 3 analysis artifacts."""

    existing_result_rows: int
    stage_three_result_rows: int
    matched_players: int
    results_path: Path
    plot_paths: tuple[Path, ...]


def pearson_statistics(
    x_values: Sequence[float] | FloatArray,
    y_values: Sequence[float] | FloatArray,
) -> CorrelationStatistics:
    """Return signed Pearson r and its square for a valid paired sample."""
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise AnalysisError("Correlation inputs must be equal-length 1D arrays")
    if x.size < 2:
        raise AnalysisError("Correlation requires at least two paired observations")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise AnalysisError("Correlation inputs must contain only finite values")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        raise AnalysisError("Pearson correlation is undefined for constant inputs")
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(pearson_r):
        raise AnalysisError("Pearson correlation calculation was non-finite")
    pearson_r = min(1.0, max(-1.0, pearson_r))
    return CorrelationStatistics(x.size, pearson_r, pearson_r * pearson_r)


def load_existing_results(path: Path) -> pl.DataFrame:
    """Load Stage 2 results and remove prior Stage 3 rows for safe reruns."""
    try:
        loaded = pl.read_csv(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise AnalysisError(
            f"Could not read existing correlation results {path}: {error}"
        ) from error
    if loaded.columns != list(RESULT_COLUMNS):
        raise AnalysisError(
            f"Existing result columns are {loaded.columns}, expected "
            f"{list(RESULT_COLUMNS)}"
        )
    if loaded.is_empty() or any(loaded.null_count().row(0)):
        raise AnalysisError("Existing correlation results are empty or have nulls")
    for column in ("pearson_r", "r_squared"):
        if loaded.filter(~pl.col(column).is_finite()).height:
            raise AnalysisError(f"Existing results contain non-finite {column}")
    if loaded.filter(
        (pl.col("n") < 2)
        | (pl.col("pearson_r").abs() > 1.0)
        | (pl.col("r_squared") < 0.0)
        | (pl.col("r_squared") > 1.0)
    ).height:
        raise AnalysisError("Existing results contain invalid statistics")
    if not np.allclose(
        loaded["r_squared"].to_numpy(),
        loaded["pearson_r"].to_numpy() ** 2,
        rtol=0.0,
        atol=1e-15,
    ):
        raise AnalysisError("Existing result R-squared values do not equal r squared")
    return loaded.filter(
        ~pl.col("x_variable").str.contains("sd-xwOBA", literal=True)
        & ~pl.col("y_variable").str.contains("sd-xwOBA", literal=True)
    )


def load_batted_ball_weights(path: Path) -> dict[int, BattedBallWeights]:
    """Load the outcome coefficients needed to calculate realized contact value."""
    columns = ["Season", "w1B", "w2B", "w3B", "wHR"]
    try:
        frame = pl.read_csv(path, columns=columns).filter(
            pl.col("Season").is_in(PREDICTION_YEARS)
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise AnalysisError(f"Could not read wOBA constants {path}: {error}") from error
    if frame.height != len(PREDICTION_YEARS) or frame["Season"].n_unique() != len(
        PREDICTION_YEARS
    ):
        raise AnalysisError("Expected one batted-ball coefficient row per year")
    result: dict[int, BattedBallWeights] = {}
    for row in frame.iter_rows(named=True):
        year = int(cast(int, row["Season"]))
        values = [float(cast(float, row[column])) for column in columns[1:]]
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise AnalysisError(f"Invalid batted-ball coefficients for {year}")
        result[year] = BattedBallWeights(year, *values)
    return result


def load_batted_ball_analysis(
    path: Path, year: int, weights: BattedBallWeights
) -> pl.DataFrame:
    """Load and validate batted-ball values needed for Stage 3 analysis."""
    columns = [
        "game_date",
        "total_bases",
        "xwOBA-MLB",
        XWOBA_LOCAL_COLUMN,
        SD_XWOBA_COLUMN,
    ]
    try:
        frame = pl.read_csv(
            path,
            columns=columns,
            schema_overrides={
                "game_date": pl.String,
                "total_bases": pl.Int64,
                "xwOBA-MLB": pl.Float64,
                XWOBA_LOCAL_COLUMN: pl.Float64,
                SD_XWOBA_COLUMN: pl.Float64,
            },
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise AnalysisError(
            f"Could not read batted-ball input {path}: {error}"
        ) from error
    if frame.is_empty() or any(frame.null_count().row(0)):
        raise AnalysisError(f"Batted-ball input for {year} is empty or has nulls")
    for column in ("xwOBA-MLB", XWOBA_LOCAL_COLUMN, SD_XWOBA_COLUMN):
        if frame.filter(~pl.col(column).is_finite()).height:
            raise AnalysisError(f"Batted-ball input has non-finite {column}")
    invalid_outcomes = set(
        cast(list[int], frame["total_bases"].unique().to_list())
    ) - set(range(5))
    if invalid_outcomes:
        raise AnalysisError(
            f"Batted-ball input has invalid outcomes {invalid_outcomes}"
        )
    dates = frame["game_date"].str.to_date(strict=False)
    if dates.null_count() or set(
        cast(list[int], dates.dt.year().unique().to_list())
    ) != {year}:
        raise AnalysisError(f"Batted-ball input does not exclusively contain {year}")
    outcome_weights = {
        0: 0.0,
        1: weights.single,
        2: weights.double,
        3: weights.triple,
        4: weights.home_run,
    }
    return frame.with_columns(
        pl.lit(year).cast(pl.Int64).alias("year"),
        pl.col("total_bases")
        .replace_strict(outcome_weights, return_dtype=pl.Float64)
        .alias("realized_woba"),
    )


def load_season_analysis(path: Path) -> pl.DataFrame:
    """Load and validate the qualified season-level analysis table."""
    try:
        frame = pl.read_csv(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise AnalysisError(
            f"Could not read season analysis {path}: {error}"
        ) from error
    if frame.columns != list(SEASON_COLUMNS):
        raise AnalysisError(
            f"Season columns are {frame.columns}, expected {list(SEASON_COLUMNS)}"
        )
    if frame.is_empty() or any(frame.null_count().row(0)):
        raise AnalysisError("Season analysis is empty or has nulls")
    if set(cast(list[int], frame["year"].unique().to_list())) != set(PREDICTION_YEARS):
        raise AnalysisError("Season analysis does not contain both prediction years")
    if frame.group_by("year", "batter_id").len().filter(pl.col("len") != 1).height:
        raise AnalysisError("Season analysis has duplicate player-season keys")
    for column in SEASON_COLUMNS[3:]:
        if frame.filter(~pl.col(column).is_finite()).height:
            raise AnalysisError(f"Season analysis contains non-finite {column}")
    return frame


def matched_player_seasons(seasons: pl.DataFrame) -> pl.DataFrame:
    """Match players qualified in both seasons by batter ID."""
    first_year, second_year = PREDICTION_YEARS
    first = seasons.filter(pl.col("year") == first_year).select(
        "batter_id",
        pl.col("batter_name").alias("batter_name_2025"),
        pl.col(SD_XWOBA_COLUMN).alias("sd-xwOBA_2025"),
        pl.col(SD_XWOBA_CONTACT_COLUMN).alias("sd-xwOBAcon_2025"),
    )
    second = seasons.filter(pl.col("year") == second_year).select(
        "batter_id",
        pl.col("batter_name").alias("batter_name_2026"),
        pl.col("wOBA").alias("wOBA_2026"),
        pl.col("wOBAcon").alias("wOBAcon_2026"),
    )
    matched = first.join(second, on="batter_id", how="inner", validate="1:1").sort(
        "batter_id"
    )
    if matched.height < 2:
        raise AnalysisError("Fewer than two players qualified in both seasons")
    return matched


def _result_record(
    analysis_level: str,
    comparison: str,
    cohort: str,
    x_variable: str,
    y_variable: str,
    x: FloatArray,
    y: FloatArray,
) -> dict[str, str | int | float]:
    statistics = pearson_statistics(x, y)
    return {
        "analysis_level": analysis_level,
        "comparison": comparison,
        "cohort": cohort,
        "x_variable": x_variable,
        "y_variable": y_variable,
        "n": statistics.n,
        "pearson_r": statistics.pearson_r,
        "r_squared": statistics.r_squared,
    }


def build_stage_three_results(
    batted_balls: Mapping[int, pl.DataFrame], seasons: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the 17 specified Stage 3 correlation rows and matched cohort."""
    if set(batted_balls) != set(PREDICTION_YEARS):
        raise AnalysisError("Batted-ball analysis requires both prediction years")
    records: list[dict[str, str | int | float]] = []
    season_frames = {
        year: seasons.filter(pl.col("year") == year) for year in PREDICTION_YEARS
    }
    for level, frames in (
        ("batted_ball", batted_balls),
        ("player_season", season_frames),
    ):
        for comparison_column in ("xwOBA-MLB", XWOBA_LOCAL_COLUMN):
            for year in PREDICTION_YEARS:
                frame = frames[year]
                records.append(
                    _result_record(
                        level,
                        f"{SD_XWOBA_COLUMN} vs {comparison_column}",
                        str(year),
                        SD_XWOBA_COLUMN,
                        comparison_column,
                        frame[SD_XWOBA_COLUMN].to_numpy(),
                        frame[comparison_column].to_numpy(),
                    )
                )
            combined = pl.concat([frames[year] for year in PREDICTION_YEARS])
            records.append(
                _result_record(
                    level,
                    f"{SD_XWOBA_COLUMN} vs {comparison_column}",
                    "2025-2026 combined",
                    SD_XWOBA_COLUMN,
                    comparison_column,
                    combined[SD_XWOBA_COLUMN].to_numpy(),
                    combined[comparison_column].to_numpy(),
                )
            )

    for year in PREDICTION_YEARS:
        frame = batted_balls[year]
        records.append(
            _result_record(
                "batted_ball_contact_value",
                f"realized wOBA value vs {SD_XWOBA_CONTACT_COLUMN}",
                str(year),
                SD_XWOBA_CONTACT_COLUMN,
                "realized wOBA value",
                frame[SD_XWOBA_COLUMN].to_numpy(),
                frame["realized_woba"].to_numpy(),
            )
        )
    combined_batted = pl.concat([batted_balls[year] for year in PREDICTION_YEARS])
    records.append(
        _result_record(
            "batted_ball_contact_value",
            f"realized wOBA value vs {SD_XWOBA_CONTACT_COLUMN}",
            "2025-2026 combined",
            SD_XWOBA_CONTACT_COLUMN,
            "realized wOBA value",
            combined_batted[SD_XWOBA_COLUMN].to_numpy(),
            combined_batted["realized_woba"].to_numpy(),
        )
    )

    matched = matched_player_seasons(seasons)
    for x_column, y_column, x_label, y_label in (
        ("sd-xwOBA_2025", "wOBA_2026", "2025 sd-xwOBA", "2026 wOBA"),
        (
            "sd-xwOBAcon_2025",
            "wOBAcon_2026",
            "2025 sd-xwOBAcon",
            "2026 wOBAcon",
        ),
    ):
        records.append(
            _result_record(
                "year_to_year" if y_column == "wOBA_2026" else "year_to_year_contact",
                f"{x_label} vs {y_label}",
                "qualified in 2025 and 2026",
                x_label,
                y_label,
                matched[x_column].to_numpy(),
                matched[y_column].to_numpy(),
            )
        )
    result = pl.DataFrame(records).select(RESULT_COLUMNS)
    if result.height != 17:
        raise AnalysisError(f"Expected 17 Stage 3 result rows, found {result.height}")
    return result, matched


def _add_fit_and_annotation(
    axes: Axes,
    x: FloatArray,
    y: FloatArray,
    statistics: CorrelationStatistics,
) -> None:
    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.linspace(float(x.min()), float(x.max()), 200)
    axes.plot(line_x, slope * line_x + intercept, color="#232323", linewidth=1.6)
    annotation = (
        f"n = {statistics.n}\n"
        f"r = {statistics.pearson_r:.3f}\n"
        f"R² = {statistics.r_squared:.3f}"
    )
    axes.text(
        0.03,
        0.97,
        annotation,
        transform=axes.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )


def _save_figure(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.part{path.suffix}")
    try:
        figure.savefig(temporary, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(figure)
        image = mpimg.imread(temporary)
        if image.ndim < 2 or min(image.shape[:2]) < 100:
            raise AnalysisError(f"Generated plot is invalid: {temporary}")
        temporary.replace(path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def _player_season_plot(
    seasons: pl.DataFrame, comparison_column: str, label: str, path: Path
) -> None:
    figure, axes = plt.subplots(figsize=(7.2, 5.6))
    colors = {2025: "#1f77b4", 2026: "#ff7f0e"}
    for year in PREDICTION_YEARS:
        frame = seasons.filter(pl.col("year") == year)
        axes.scatter(
            frame[SD_XWOBA_COLUMN].to_numpy(),
            frame[comparison_column].to_numpy(),
            s=24,
            alpha=0.7,
            color=colors[year],
            label=str(year),
        )
    x = seasons[SD_XWOBA_COLUMN].to_numpy()
    y = seasons[comparison_column].to_numpy()
    _add_fit_and_annotation(axes, x, y, pearson_statistics(x, y))
    axes.set_xlabel("Player-season sd-xwOBA")
    axes.set_ylabel(label)
    axes.set_title(f"Player-season sd-xwOBA vs {label}")
    axes.legend(title="Season")
    axes.grid(alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, path)


def _year_ahead_plot(
    matched: pl.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    path: Path,
) -> None:
    figure, axes = plt.subplots(figsize=(7.2, 5.6))
    x = matched[x_column].to_numpy()
    y = matched[y_column].to_numpy()
    axes.scatter(x, y, s=30, alpha=0.75, color="#2a788e")
    _add_fit_and_annotation(axes, x, y, pearson_statistics(x, y))
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(f"{x_label}\nvs {y_label}")
    axes.grid(alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, path)


def create_plots(
    seasons: pl.DataFrame,
    matched: pl.DataFrame,
    sd_vs_mlb_path: Path,
    sd_vs_local_path: Path,
    sd_year_ahead_path: Path,
    sd_contact_year_ahead_path: Path,
) -> tuple[Path, ...]:
    """Create and validate all four requested Stage 3 plots."""
    _player_season_plot(seasons, "xwOBA-MLB", "MLB xwOBA", sd_vs_mlb_path)
    _player_season_plot(seasons, XWOBA_LOCAL_COLUMN, "local xwOBA", sd_vs_local_path)
    _year_ahead_plot(
        matched,
        "sd-xwOBA_2025",
        "wOBA_2026",
        "2025 sd-xwOBA",
        "2026 wOBA",
        sd_year_ahead_path,
    )
    _year_ahead_plot(
        matched,
        "sd-xwOBAcon_2025",
        "wOBAcon_2026",
        "2025 sd-xwOBAcon",
        "2026 wOBAcon",
        sd_contact_year_ahead_path,
    )
    return (
        sd_vs_mlb_path,
        sd_vs_local_path,
        sd_year_ahead_path,
        sd_contact_year_ahead_path,
    )


def validate_results_output(
    path: Path, existing: pl.DataFrame, stage_three: pl.DataFrame
) -> None:
    """Validate the combined saved result table and preservation contract."""
    try:
        output = pl.read_csv(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise AnalysisError(f"Could not validate results {path}: {error}") from error
    expected = pl.concat([existing, stage_three], how="vertical")
    if not output.equals(expected, null_equal=True):
        raise AnalysisError("Written correlation results changed from expectation")
    if not output.head(existing.height).equals(existing, null_equal=True):
        raise AnalysisError("Existing correlation rows changed or reordered")
    if output.height != existing.height + 17:
        raise AnalysisError("Combined correlation result row count is incorrect")
    if not np.allclose(
        output["r_squared"].to_numpy(),
        output["pearson_r"].to_numpy() ** 2,
        rtol=0.0,
        atol=1e-15,
    ):
        raise AnalysisError("Saved R-squared values do not equal r squared")


def analyze(
    *,
    calculated_directory: Path = CALCULATED_XWOBA_DIRECTORY,
    season_path: Path = SEASON_LEVEL_XWOBA_PATH,
    constants_path: Path = WOBA_CONSTANTS_PATH,
    results_path: Path = CORRELATION_RESULTS_PATH,
    sd_vs_mlb_path: Path = SD_VS_MLB_PLAYER_SEASON_PLOT_PATH,
    sd_vs_local_path: Path = SD_VS_LOCAL_PLAYER_SEASON_PLOT_PATH,
    sd_year_ahead_path: Path = SD_YEAR_AHEAD_PLOT_PATH,
    sd_contact_year_ahead_path: Path = SD_CONTACT_YEAR_AHEAD_PLOT_PATH,
    overwrite: bool = False,
) -> AnalysisSummary:
    """Calculate, write, plot, and validate the complete Stage 3 analysis."""
    plot_paths = (
        sd_vs_mlb_path,
        sd_vs_local_path,
        sd_year_ahead_path,
        sd_contact_year_ahead_path,
    )
    existing_paths = [path for path in (results_path, *plot_paths) if path.exists()]
    if existing_paths and not overwrite:
        raise AnalysisOutputExistsError(
            "Artifact replacement was not requested for: "
            + ", ".join(str(path) for path in existing_paths)
        )

    existing = load_existing_results(results_path)
    weights = load_batted_ball_weights(constants_path)
    batted_balls = {
        year: load_batted_ball_analysis(
            calculated_directory / f"{year}_all_BBE_with_calculated_xwOBAs.csv",
            year,
            weights[year],
        )
        for year in PREDICTION_YEARS
    }
    seasons = load_season_analysis(season_path)
    stage_three, matched = build_stage_three_results(batted_balls, seasons)

    combined = pl.concat([existing, stage_three], how="vertical")
    temporary_results = results_path.with_name(f"{results_path.name}.part")
    try:
        combined.write_csv(temporary_results)
        validate_results_output(temporary_results, existing, stage_three)
        temporary_results.replace(results_path)
    finally:
        temporary_results.unlink(missing_ok=True)
    created_plots = create_plots(
        seasons,
        matched,
        sd_vs_mlb_path,
        sd_vs_local_path,
        sd_year_ahead_path,
        sd_contact_year_ahead_path,
    )
    return AnalysisSummary(
        existing.height,
        stage_three.height,
        matched.height,
        results_path,
        created_plots,
    )


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse Stage 3 analysis command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calculated-directory", type=Path, default=CALCULATED_XWOBA_DIRECTORY
    )
    parser.add_argument("--season-data", type=Path, default=SEASON_LEVEL_XWOBA_PATH)
    parser.add_argument("--constants", type=Path, default=WOBA_CONSTANTS_PATH)
    parser.add_argument("--results", type=Path, default=CORRELATION_RESULTS_PATH)
    parser.add_argument(
        "--sd-vs-mlb-plot", type=Path, default=SD_VS_MLB_PLAYER_SEASON_PLOT_PATH
    )
    parser.add_argument(
        "--sd-vs-local-plot", type=Path, default=SD_VS_LOCAL_PLAYER_SEASON_PLOT_PATH
    )
    parser.add_argument(
        "--sd-year-ahead-plot", type=Path, default=SD_YEAR_AHEAD_PLOT_PATH
    )
    parser.add_argument(
        "--sd-contact-year-ahead-plot",
        type=Path,
        default=SD_CONTACT_YEAR_AHEAD_PLOT_PATH,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run Stage 3 analysis and print a compact artifact summary."""
    args = parse_args(arguments)
    summary = analyze(
        calculated_directory=args.calculated_directory,
        season_path=args.season_data,
        constants_path=args.constants,
        results_path=args.results,
        sd_vs_mlb_path=args.sd_vs_mlb_plot,
        sd_vs_local_path=args.sd_vs_local_plot,
        sd_year_ahead_path=args.sd_year_ahead_plot,
        sd_contact_year_ahead_path=args.sd_contact_year_ahead_plot,
        overwrite=args.overwrite,
    )
    print(
        f"Preserved {summary.existing_result_rows} existing rows and appended "
        f"{summary.stage_three_result_rows} Stage 3 rows for "
        f"{summary.matched_players} matched players -> {summary.results_path}"
    )
    for path in summary.plot_paths:
        print(f"Wrote plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
