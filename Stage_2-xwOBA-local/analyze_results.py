"""Analyze Stage 2 correlations and create the requested diagnostic plots."""

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

from calculate_season_xwoba import WobaWeights, load_woba_weights  # noqa: E402
from experiment_config import (  # noqa: E402
    CALCULATED_XWOBA_DIRECTORY,
    CORRELATION_RESULTS_PATH,
    LOCAL_YEAR_AHEAD_PLOT_PATH,
    MLB_YEAR_AHEAD_PLOT_PATH,
    PLAYER_SEASON_COMPARISON_PLOT_PATH,
    PREDICTION_YEARS,
    SEASON_LEVEL_XWOBA_PATH,
    WOBA_CONSTANTS_PATH,
    XWOBA_LOCAL_COLUMN,
)

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
)
PLOT_DPI: Final[int] = 180


class AnalysisError(RuntimeError):
    """Raised when a Stage 2 analysis input or output is invalid."""


class AnalysisOutputExistsError(AnalysisError):
    """Raised when replacing an analysis artifact was not authorized."""


@dataclass(frozen=True, slots=True)
class CorrelationStatistics:
    """Pearson correlation statistics for one paired sample."""

    n: int
    pearson_r: float
    r_squared: float


@dataclass(frozen=True, slots=True)
class AnalysisSummary:
    """Locations and row counts for a completed Stage 2 analysis."""

    result_rows: int
    matched_players: int
    results_path: Path
    plot_paths: tuple[Path, ...]


def pearson_statistics(
    x_values: Sequence[float] | NDArray[np.float64],
    y_values: Sequence[float] | NDArray[np.float64],
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


def load_batted_ball_analysis(
    path: Path, year: int, weights: WobaWeights
) -> pl.DataFrame:
    """Load and validate the two batted-ball measures needed for analysis."""
    if year not in PREDICTION_YEARS:
        raise ValueError(f"Unsupported prediction year: {year}")
    try:
        frame = pl.read_csv(
            path,
            columns=[
                "game_date",
                "total_bases",
                "xwOBA-MLB",
                XWOBA_LOCAL_COLUMN,
            ],
            schema_overrides={
                "game_date": pl.String,
                "total_bases": pl.Int64,
                "xwOBA-MLB": pl.Float64,
                XWOBA_LOCAL_COLUMN: pl.Float64,
            },
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise AnalysisError(
            f"Could not read batted-ball analysis input {path}: {error}"
        ) from error
    if frame.is_empty() or any(frame.null_count().row(0)):
        raise AnalysisError(
            f"Batted-ball analysis input for {year} is empty or has nulls"
        )
    if frame.filter(
        ~pl.col("xwOBA-MLB").is_finite() | ~pl.col(XWOBA_LOCAL_COLUMN).is_finite()
    ).height:
        raise AnalysisError(
            f"Batted-ball analysis input for {year} has non-finite values"
        )
    invalid_outcomes = set(
        cast(list[int], frame["total_bases"].unique().to_list())
    ) - set(range(5))
    if invalid_outcomes:
        raise AnalysisError(
            f"Batted-ball analysis input for {year} has invalid outcomes: "
            f"{sorted(invalid_outcomes)}"
        )
    dates = frame["game_date"].str.to_date(strict=False)
    if dates.null_count():
        raise AnalysisError(
            f"Batted-ball analysis input for {year} has malformed dates"
        )
    actual_years = set(cast(list[int], dates.dt.year().unique().to_list()))
    if actual_years != {year}:
        raise AnalysisError(
            "Batted-ball analysis input contains years "
            f"{sorted(actual_years)}, expected {year}"
        )
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
    """Load and validate the qualified player-season analysis table."""
    try:
        frame = pl.read_csv(
            path,
            schema_overrides={
                "year": pl.Int64,
                "batter_name": pl.String,
                "batter_id": pl.Int64,
                "wOBA": pl.Float64,
                "xwOBA-MLB": pl.Float64,
                XWOBA_LOCAL_COLUMN: pl.Float64,
            },
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise AnalysisError(
            f"Could not read season analysis input {path}: {error}"
        ) from error
    if frame.columns != list(SEASON_COLUMNS):
        raise AnalysisError(
            f"Season analysis columns are {frame.columns}, expected "
            f"{list(SEASON_COLUMNS)}"
        )
    if frame.is_empty() or any(frame.null_count().row(0)):
        raise AnalysisError("Season analysis input is empty or contains null values")
    actual_years = set(cast(list[int], frame["year"].unique().to_list()))
    if actual_years != set(PREDICTION_YEARS):
        raise AnalysisError(
            f"Season analysis contains years {sorted(actual_years)}, expected "
            f"{list(PREDICTION_YEARS)}"
        )
    if frame.group_by("year", "batter_id").len().filter(pl.col("len") != 1).height:
        raise AnalysisError("Season analysis contains duplicate player-season keys")
    for column in (
        "wOBA",
        "xwOBA-MLB",
        XWOBA_LOCAL_COLUMN,
        "wOBAcon",
        "xwOBAcon-MLB",
        "xwOBAcon-local",
    ):
        if frame.filter(~pl.col(column).is_finite()).height:
            raise AnalysisError(f"Season analysis contains non-finite {column} values")
    return frame


def matched_player_seasons(seasons: pl.DataFrame) -> pl.DataFrame:
    """Return players qualified in both seasons, matched by MLBAMID/batter_id."""
    year_one, year_two = PREDICTION_YEARS
    first = seasons.filter(pl.col("year") == year_one).select(
        "batter_id",
        pl.col("batter_name").alias("batter_name_2025"),
        pl.col("wOBA").alias("wOBA_2025"),
        pl.col("xwOBA-MLB").alias("xwOBA-MLB_2025"),
        pl.col(XWOBA_LOCAL_COLUMN).alias("xwOBA-local_2025"),
        pl.col("wOBAcon").alias("wOBAcon_2025"),
        pl.col("xwOBAcon-MLB").alias("xwOBAcon-MLB_2025"),
        pl.col("xwOBAcon-local").alias("xwOBAcon-local_2025"),
    )
    second = seasons.filter(pl.col("year") == year_two).select(
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
    x: NDArray[np.float64],
    y: NDArray[np.float64],
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


def build_correlation_results(
    batted_balls: Mapping[int, pl.DataFrame], seasons: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build all specified correlation results and the year-ahead cohort."""
    if set(batted_balls) != set(PREDICTION_YEARS):
        raise AnalysisError("Batted-ball analysis requires both prediction years")
    records: list[dict[str, str | int | float]] = []
    for level, frames in (
        ("batted_ball", batted_balls),
        (
            "player_season",
            {year: seasons.filter(pl.col("year") == year) for year in PREDICTION_YEARS},
        ),
    ):
        for year in PREDICTION_YEARS:
            frame = frames[year]
            records.append(
                _result_record(
                    level,
                    "xwOBA-local vs xwOBA-MLB",
                    str(year),
                    XWOBA_LOCAL_COLUMN,
                    "xwOBA-MLB",
                    frame[XWOBA_LOCAL_COLUMN].to_numpy(),
                    frame["xwOBA-MLB"].to_numpy(),
                )
            )
        combined = pl.concat([frames[year] for year in PREDICTION_YEARS])
        records.append(
            _result_record(
                level,
                "xwOBA-local vs xwOBA-MLB",
                "2025-2026 combined",
                XWOBA_LOCAL_COLUMN,
                "xwOBA-MLB",
                combined[XWOBA_LOCAL_COLUMN].to_numpy(),
                combined["xwOBA-MLB"].to_numpy(),
            )
        )

    for year in PREDICTION_YEARS:
        frame = batted_balls[year]
        for expected_column, expected_label in (
            ("xwOBA-MLB", "xwOBAcon-MLB"),
            (XWOBA_LOCAL_COLUMN, "xwOBAcon-local"),
        ):
            records.append(
                _result_record(
                    "batted_ball_contact_value",
                    f"realized wOBA value vs {expected_label}",
                    str(year),
                    expected_label,
                    "realized wOBA value",
                    frame[expected_column].to_numpy(),
                    frame["realized_woba"].to_numpy(),
                )
            )
    combined_batted_balls = pl.concat([batted_balls[year] for year in PREDICTION_YEARS])
    for expected_column, expected_label in (
        ("xwOBA-MLB", "xwOBAcon-MLB"),
        (XWOBA_LOCAL_COLUMN, "xwOBAcon-local"),
    ):
        records.append(
            _result_record(
                "batted_ball_contact_value",
                f"realized wOBA value vs {expected_label}",
                "2025-2026 combined",
                expected_label,
                "realized wOBA value",
                combined_batted_balls[expected_column].to_numpy(),
                combined_batted_balls["realized_woba"].to_numpy(),
            )
        )

    matched = matched_player_seasons(seasons)
    for x_column, x_label in (
        ("xwOBA-MLB_2025", "2025 xwOBA-MLB"),
        ("xwOBA-local_2025", "2025 xwOBA-local"),
    ):
        records.append(
            _result_record(
                "year_to_year",
                f"{x_label} vs 2026 wOBA",
                "qualified in 2025 and 2026",
                x_label,
                "2026 wOBA",
                matched[x_column].to_numpy(),
                matched["wOBA_2026"].to_numpy(),
            )
        )

    for x_column, x_label in (
        ("xwOBAcon-MLB_2025", "2025 xwOBAcon-MLB"),
        ("xwOBAcon-local_2025", "2025 xwOBAcon-local"),
    ):
        records.append(
            _result_record(
                "year_to_year_contact",
                f"{x_label} vs 2026 wOBAcon",
                "qualified in 2025 and 2026",
                x_label,
                "2026 wOBAcon",
                matched[x_column].to_numpy(),
                matched["wOBAcon_2026"].to_numpy(),
            )
        )

    for x_column, y_column, x_label, y_label in (
        ("wOBA_2025", "wOBA_2026", "2025 wOBA", "2026 wOBA"),
        ("wOBAcon_2025", "wOBAcon_2026", "2025 wOBAcon", "2026 wOBAcon"),
    ):
        records.append(
            _result_record(
                "year_to_year_realized",
                f"{x_label} vs {y_label}",
                "qualified in 2025 and 2026",
                x_label,
                y_label,
                matched[x_column].to_numpy(),
                matched[y_column].to_numpy(),
            )
        )

    result = pl.DataFrame(
        records,
        schema={
            "analysis_level": pl.String,
            "comparison": pl.String,
            "cohort": pl.String,
            "x_variable": pl.String,
            "y_variable": pl.String,
            "n": pl.Int64,
            "pearson_r": pl.Float64,
            "r_squared": pl.Float64,
        },
    ).select(RESULT_COLUMNS)
    return result, matched


def _add_fit_line(ax: Axes, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.linspace(float(x.min()), float(x.max()), 200)
    ax.plot(line_x, slope * line_x + intercept, color="#252525", linewidth=2)


def _annotate_statistics(ax: Axes, statistics: CorrelationStatistics) -> None:
    ax.text(
        0.03,
        0.97,
        f"n = {statistics.n:,}\nPearson r = {statistics.pearson_r:.3f}\n"
        f"Pearson r² = {statistics.r_squared:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )


def _finish_plot(fig: Figure, ax: Axes, x_label: str, y_label: str, title: str) -> None:
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()


def _save_plot(fig: Figure, path: Path) -> None:
    temporary = path.with_name(f"{path.stem}.part{path.suffix}")
    try:
        fig.savefig(temporary, dpi=PLOT_DPI, format="png", bbox_inches="tight")
        image = mpimg.imread(temporary)
        if image.ndim not in (2, 3) or min(image.shape[:2]) < 100:
            raise AnalysisError(f"Generated plot is invalid: {temporary}")
        temporary.replace(path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def create_plots(
    seasons: pl.DataFrame,
    matched: pl.DataFrame,
    player_season_path: Path,
    mlb_year_ahead_path: Path,
    local_year_ahead_path: Path,
) -> None:
    """Create and validate the three requested correlation plots."""
    plot_paths = (player_season_path, mlb_year_ahead_path, local_year_ahead_path)
    for path in plot_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    x = seasons[XWOBA_LOCAL_COLUMN].to_numpy()
    y = seasons["xwOBA-MLB"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {2025: "#1f77b4", 2026: "#e45756"}
    for year in PREDICTION_YEARS:
        annual = seasons.filter(pl.col("year") == year)
        ax.scatter(
            annual[XWOBA_LOCAL_COLUMN].to_numpy(),
            annual["xwOBA-MLB"].to_numpy(),
            s=32,
            alpha=0.72,
            color=colors[year],
            label=str(year),
        )
    _add_fit_line(ax, x, y)
    _annotate_statistics(ax, pearson_statistics(x, y))
    _finish_plot(
        fig,
        ax,
        "Player-season xwOBA-local",
        "Player-season xwOBA-MLB",
        "Qualified player-season expected wOBA, 2025–2026",
    )
    ax.legend(title="Season")
    _save_plot(fig, player_season_path)

    for x_column, x_label, title, output_path, color in (
        (
            "xwOBA-MLB_2025",
            "2025 xwOBA-MLB",
            "2025 MLB expected wOBA vs. 2026 wOBA",
            mlb_year_ahead_path,
            "#4c78a8",
        ),
        (
            "xwOBA-local_2025",
            "2025 xwOBA-local",
            "2025 local expected wOBA vs. 2026 wOBA",
            local_year_ahead_path,
            "#f58518",
        ),
    ):
        year_x = matched[x_column].to_numpy()
        year_y = matched["wOBA_2026"].to_numpy()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(year_x, year_y, s=38, alpha=0.75, color=color)
        _add_fit_line(ax, year_x, year_y)
        _annotate_statistics(ax, pearson_statistics(year_x, year_y))
        _finish_plot(fig, ax, x_label, "2026 wOBA", title)
        _save_plot(fig, output_path)


def validate_results_output(path: Path, expected: pl.DataFrame) -> None:
    """Reload and validate the saved correlation result table."""
    try:
        actual = pl.read_csv(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise AnalysisError(
            f"Could not validate correlation results {path}: {error}"
        ) from error
    if actual.columns != list(RESULT_COLUMNS) or actual.height != 18:
        raise AnalysisError("Correlation result output has an invalid shape or schema")
    if (
        actual.select(RESULT_COLUMNS[:-2]).to_dicts()
        != expected.select(RESULT_COLUMNS[:-2]).to_dicts()
    ):
        raise AnalysisError("Correlation result identifiers changed during output")
    np.testing.assert_allclose(
        actual.select("pearson_r", "r_squared").to_numpy(),
        expected.select("pearson_r", "r_squared").to_numpy(),
        rtol=1e-14,
        atol=1e-14,
    )
    r_values = actual["pearson_r"].to_numpy()
    r_squared = actual["r_squared"].to_numpy()
    if not np.allclose(r_squared, r_values * r_values, rtol=1e-14, atol=1e-14):
        raise AnalysisError(
            "Saved R-squared values are not squared Pearson correlations"
        )


def analyze(
    calculated_directory: Path = CALCULATED_XWOBA_DIRECTORY,
    season_path: Path = SEASON_LEVEL_XWOBA_PATH,
    constants_path: Path = WOBA_CONSTANTS_PATH,
    results_path: Path = CORRELATION_RESULTS_PATH,
    player_season_plot_path: Path = PLAYER_SEASON_COMPARISON_PLOT_PATH,
    mlb_year_ahead_plot_path: Path = MLB_YEAR_AHEAD_PLOT_PATH,
    local_year_ahead_plot_path: Path = LOCAL_YEAR_AHEAD_PLOT_PATH,
    *,
    overwrite: bool = False,
) -> AnalysisSummary:
    """Run the full Stage 2 correlation analysis and write validated artifacts."""
    plot_paths = (
        player_season_plot_path,
        mlb_year_ahead_plot_path,
        local_year_ahead_plot_path,
    )
    output_paths = (results_path, *plot_paths)
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise AnalysisOutputExistsError(
            f"Analysis output replacement was not requested: {existing}"
        )

    weights = load_woba_weights(constants_path)
    batted_balls = {
        year: load_batted_ball_analysis(
            calculated_directory / f"{year}_all_BBE_with_calculated_xwOBAs.csv",
            year,
            weights[year],
        )
        for year in PREDICTION_YEARS
    }
    seasons = load_season_analysis(season_path)
    results, matched = build_correlation_results(batted_balls, seasons)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_results = results_path.with_name(f"{results_path.name}.part")
    try:
        results.write_csv(temporary_results)
        temporary_results.replace(results_path)
        validate_results_output(results_path, results)
    finally:
        temporary_results.unlink(missing_ok=True)

    create_plots(
        seasons,
        matched,
        player_season_plot_path,
        mlb_year_ahead_plot_path,
        local_year_ahead_plot_path,
    )
    return AnalysisSummary(results.height, matched.height, results_path, plot_paths)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Stage 2 analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calculated-directory", type=Path, default=CALCULATED_XWOBA_DIRECTORY
    )
    parser.add_argument("--season-data", type=Path, default=SEASON_LEVEL_XWOBA_PATH)
    parser.add_argument("--constants", type=Path, default=WOBA_CONSTANTS_PATH)
    parser.add_argument("--results", type=Path, default=CORRELATION_RESULTS_PATH)
    parser.add_argument(
        "--player-season-plot", type=Path, default=PLAYER_SEASON_COMPARISON_PLOT_PATH
    )
    parser.add_argument(
        "--mlb-year-ahead-plot", type=Path, default=MLB_YEAR_AHEAD_PLOT_PATH
    )
    parser.add_argument(
        "--local-year-ahead-plot", type=Path, default=LOCAL_YEAR_AHEAD_PLOT_PATH
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the analysis command and print a compact artifact summary."""
    args = parse_args(arguments)
    summary = analyze(
        calculated_directory=args.calculated_directory,
        season_path=args.season_data,
        constants_path=args.constants,
        results_path=args.results,
        player_season_plot_path=args.player_season_plot,
        mlb_year_ahead_plot_path=args.mlb_year_ahead_plot,
        local_year_ahead_plot_path=args.local_year_ahead_plot,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote {summary.result_rows} correlations for "
        f"{summary.matched_players} matched players to {summary.results_path}"
    )
    for path in summary.plot_paths:
        print(f"Wrote plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
