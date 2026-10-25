"""Append qualified player-season sd-xwOBA and sd-xwOBAcon values."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import polars as pl

from sd_experiment_config import (
    CALCULATED_XWOBA_DIRECTORY,
    PREDICTION_YEARS,
    QUALIFIED_PLAYER_SEASONS_PATH,
    SD_XWOBA_COLUMN,
    SD_XWOBA_CONTACT_COLUMN,
    SEASON_LEVEL_XWOBA_PATH,
    WOBA_CONSTANTS_PATH,
)

EXISTING_SEASON_COLUMNS: Final[tuple[str, ...]] = (
    "year",
    "batter_name",
    "batter_id",
    "wOBA",
    "xwOBA-MLB",
    "xwOBA-local",
    "wOBAcon",
    "xwOBAcon-MLB",
    "xwOBAcon-local",
)
QUALIFIED_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "Season",
    "Name",
    "MLBAMID",
    "AB",
    "BB",
    "IBB",
    "HBP",
    "SF",
    "wOBA",
    "xwOBA",
)
QUALIFIED_SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    "Season": pl.Int64,
    "Name": pl.String,
    "MLBAMID": pl.Int64,
    "AB": pl.Int64,
    "BB": pl.Int64,
    "IBB": pl.Int64,
    "HBP": pl.Int64,
    "SF": pl.Int64,
    "wOBA": pl.Float64,
    "xwOBA": pl.Float64,
}


class SeasonCalculationError(RuntimeError):
    """Raised when a Stage 3 season calculation or validation fails."""


class SeasonOutputExistsError(SeasonCalculationError):
    """Raised when season-output replacement was not explicitly requested."""


@dataclass(frozen=True, slots=True)
class PlateAppearanceWeights:
    """Plate-appearance wOBA coefficients for one season."""

    year: int
    unintentional_walk: float
    hit_by_pitch: float


@dataclass(frozen=True, slots=True)
class SeasonCalculationSummary:
    """Validated output details for all prediction-year qualified seasons."""

    rows: int
    rows_by_year: Mapping[int, int]
    output_path: Path


def load_existing_seasons(path: Path) -> pl.DataFrame:
    """Load the existing Stage 2 season artifact without its Stage 3 columns."""
    try:
        loaded = pl.read_csv(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise SeasonCalculationError(
            f"Could not read existing season data {path}: {error}"
        ) from error
    frame = loaded.drop(
        column
        for column in (SD_XWOBA_COLUMN, SD_XWOBA_CONTACT_COLUMN)
        if column in loaded.columns
    )
    if frame.columns != list(EXISTING_SEASON_COLUMNS):
        raise SeasonCalculationError(
            f"Existing season columns are {frame.columns}, expected "
            f"{list(EXISTING_SEASON_COLUMNS)}"
        )
    if frame.is_empty() or any(frame.null_count().row(0)):
        raise SeasonCalculationError("Existing season data is empty or has nulls")
    actual_years = set(cast(list[int], frame["year"].unique().to_list()))
    if actual_years != set(PREDICTION_YEARS):
        raise SeasonCalculationError(
            f"Existing season data contains years {sorted(actual_years)}"
        )
    if frame.group_by("year", "batter_id").len().filter(pl.col("len") != 1).height:
        raise SeasonCalculationError("Existing season data has duplicate keys")
    for column in (
        "wOBA",
        "xwOBA-MLB",
        "xwOBA-local",
        "wOBAcon",
        "xwOBAcon-MLB",
        "xwOBAcon-local",
    ):
        if frame.filter(~pl.col(column).is_finite()).height:
            raise SeasonCalculationError(
                f"Existing season data contains non-finite {column}"
            )
    return frame


def load_qualified_seasons(path: Path) -> pl.DataFrame:
    """Load and validate the qualified 2025 and 2026 player-season data."""
    try:
        frame = pl.read_csv(
            path,
            columns=list(QUALIFIED_REQUIRED_COLUMNS),
            schema_overrides=QUALIFIED_SCHEMA,
        ).filter(pl.col("Season").is_in(PREDICTION_YEARS))
    except (OSError, pl.exceptions.PolarsError) as error:
        raise SeasonCalculationError(
            f"Could not read qualified player seasons {path}: {error}"
        ) from error
    if frame.is_empty() or any(frame.null_count().row(0)):
        raise SeasonCalculationError("Qualified season data is empty or has nulls")
    actual_years = set(cast(list[int], frame["Season"].unique().to_list()))
    if actual_years != set(PREDICTION_YEARS):
        raise SeasonCalculationError(
            f"Qualified data contains years {sorted(actual_years)}"
        )
    if frame.group_by("Season", "MLBAMID").len().filter(pl.col("len") != 1).height:
        raise SeasonCalculationError("Qualified data has duplicate player-season keys")
    for column in ("AB", "BB", "IBB", "HBP", "SF"):
        if frame.filter(pl.col(column) < 0).height:
            raise SeasonCalculationError(f"Qualified data contains negative {column}")
    if frame.filter(pl.col("IBB") > pl.col("BB")).height:
        raise SeasonCalculationError("Qualified data contains IBB greater than BB")
    for column in ("wOBA", "xwOBA"):
        if frame.filter(~pl.col(column).is_finite()).height:
            raise SeasonCalculationError(f"Qualified data contains non-finite {column}")
    denominator = (
        pl.col("AB") + pl.col("BB") - pl.col("IBB") + pl.col("SF") + pl.col("HBP")
    )
    if frame.filter(denominator <= 0).height:
        raise SeasonCalculationError("Qualified data has nonpositive denominators")
    return frame


def load_plate_appearance_weights(path: Path) -> dict[int, PlateAppearanceWeights]:
    """Load and validate required walk and hit-by-pitch coefficients."""
    try:
        frame = pl.read_csv(
            path,
            columns=["Season", "wBB", "wHBP"],
            schema_overrides={
                "Season": pl.Int64,
                "wBB": pl.Float64,
                "wHBP": pl.Float64,
            },
        ).filter(pl.col("Season").is_in(PREDICTION_YEARS))
    except (OSError, pl.exceptions.PolarsError) as error:
        raise SeasonCalculationError(
            f"Could not read wOBA constants {path}: {error}"
        ) from error
    if frame.height != len(PREDICTION_YEARS) or frame["Season"].n_unique() != len(
        PREDICTION_YEARS
    ):
        raise SeasonCalculationError("Expected one coefficient row per year")
    result: dict[int, PlateAppearanceWeights] = {}
    for row in frame.iter_rows(named=True):
        year = int(cast(int, row["Season"]))
        walk = float(cast(float, row["wBB"]))
        hbp = float(cast(float, row["wHBP"]))
        if not all(math.isfinite(value) and value > 0.0 for value in (walk, hbp)):
            raise SeasonCalculationError(
                f"Plate-appearance coefficients for {year} must be positive"
            )
        result[year] = PlateAppearanceWeights(year, walk, hbp)
    return result


def aggregate_batted_ball_sd_xwoba(path: Path, year: int) -> pl.DataFrame:
    """Aggregate batted-ball sd-xwOBA sums and counts by batter."""
    if year not in PREDICTION_YEARS:
        raise ValueError(f"Unsupported prediction year: {year}")
    try:
        frame = pl.read_csv(
            path,
            columns=["game_date", "batter", SD_XWOBA_COLUMN],
            schema_overrides={
                "game_date": pl.String,
                "batter": pl.Int64,
                SD_XWOBA_COLUMN: pl.Float64,
            },
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise SeasonCalculationError(
            f"Could not read calculated batted balls {path}: {error}"
        ) from error
    if frame.is_empty() or any(frame.null_count().row(0)):
        raise SeasonCalculationError(
            f"Calculated batted-ball data for {year} is empty or has nulls"
        )
    if frame.filter(~pl.col(SD_XWOBA_COLUMN).is_finite()).height:
        raise SeasonCalculationError(
            f"Calculated batted-ball data for {year} has non-finite sd-xwOBA"
        )
    dates = frame["game_date"].str.to_date(strict=False)
    if dates.null_count() or set(
        cast(list[int], dates.dt.year().unique().to_list())
    ) != {year}:
        raise SeasonCalculationError(
            f"Calculated batted-ball data does not exclusively contain {year}"
        )
    return (
        frame.group_by("batter")
        .agg(
            pl.len().cast(pl.Int64).alias("batted_ball_count"),
            pl.col(SD_XWOBA_COLUMN).sum().alias("batted_ball_sd_xwoba_sum"),
            pl.col(SD_XWOBA_COLUMN).mean().alias(SD_XWOBA_CONTACT_COLUMN),
        )
        .with_columns(pl.lit(year).cast(pl.Int64).alias("year"))
        .rename({"batter": "batter_id"})
        .select(
            "year",
            "batter_id",
            "batted_ball_count",
            "batted_ball_sd_xwoba_sum",
            SD_XWOBA_CONTACT_COLUMN,
        )
    )


def calculate_season_frame(
    existing: pl.DataFrame,
    qualified: pl.DataFrame,
    batted_ball_aggregates: pl.DataFrame,
    weights: Mapping[int, PlateAppearanceWeights],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Calculate Stage 3 metrics and append them to existing season rows."""
    coefficient_frame = pl.DataFrame(
        {
            "year": list(PREDICTION_YEARS),
            "wBB": [weights[year].unintentional_walk for year in PREDICTION_YEARS],
            "wHBP": [weights[year].hit_by_pitch for year in PREDICTION_YEARS],
        },
        schema={"year": pl.Int64, "wBB": pl.Float64, "wHBP": pl.Float64},
    )
    calculation = (
        qualified.rename(
            {
                "Season": "year",
                "Name": "qualified_name",
                "MLBAMID": "batter_id",
                "wOBA": "qualified_wOBA",
                "xwOBA": "qualified_xwOBA_MLB",
            }
        )
        .join(
            batted_ball_aggregates, on=["year", "batter_id"], how="left", validate="1:1"
        )
        .join(coefficient_frame, on="year", how="left", validate="m:1")
        .with_columns(
            (pl.col("BB") - pl.col("IBB")).alias("uBB"),
            (
                pl.col("AB")
                + pl.col("BB")
                - pl.col("IBB")
                + pl.col("SF")
                + pl.col("HBP")
            ).alias("denominator"),
        )
        .with_columns(
            (
                pl.col("wBB") * pl.col("uBB")
                + pl.col("wHBP") * pl.col("HBP")
                + pl.col("batted_ball_sd_xwoba_sum")
            ).alias("numerator")
        )
        .with_columns(
            (pl.col("numerator") / pl.col("denominator")).alias(SD_XWOBA_COLUMN)
        )
    )
    unmatched = calculation.filter(
        pl.col("batted_ball_count").is_null()
        | pl.col("batted_ball_sd_xwoba_sum").is_null()
        | pl.col("wBB").is_null()
        | pl.col("wHBP").is_null()
    )
    if unmatched.height:
        raise SeasonCalculationError(
            "Qualified player seasons lack batted-ball or coefficient matches"
        )
    if calculation.filter(
        ~pl.col(SD_XWOBA_COLUMN).is_finite()
        | ~pl.col(SD_XWOBA_CONTACT_COLUMN).is_finite()
        | (pl.col("denominator") <= 0)
        | (pl.col("uBB") < 0)
    ).height:
        raise SeasonCalculationError("Season calculation produced invalid values")

    output = (
        existing.with_row_index("source_order")
        .join(
            calculation.select(
                "year",
                "batter_id",
                "qualified_name",
                "qualified_wOBA",
                "qualified_xwOBA_MLB",
                "AB",
                "BB",
                "IBB",
                "HBP",
                "SF",
                "wBB",
                "wHBP",
                "uBB",
                "denominator",
                "batted_ball_count",
                "batted_ball_sd_xwoba_sum",
                SD_XWOBA_COLUMN,
                SD_XWOBA_CONTACT_COLUMN,
            ),
            on=["year", "batter_id"],
            how="left",
            validate="1:1",
        )
        .sort("source_order")
    )
    if output.filter(
        (pl.col("batter_name") != pl.col("qualified_name"))
        | (pl.col("wOBA") != pl.col("qualified_wOBA"))
        | (pl.col("xwOBA-MLB") != pl.col("qualified_xwOBA_MLB"))
    ).height:
        raise SeasonCalculationError(
            "Existing season identity or source metrics differ from qualified data"
        )
    if output.filter(pl.col(SD_XWOBA_COLUMN).is_null()).height:
        raise SeasonCalculationError("Existing season rows lack calculation matches")
    final = output.select(
        *EXISTING_SEASON_COLUMNS, SD_XWOBA_COLUMN, SD_XWOBA_CONTACT_COLUMN
    )
    return final, output


def validate_season_output(
    path: Path,
    existing: pl.DataFrame,
    expected: pl.DataFrame,
    reconciliation: pl.DataFrame,
) -> None:
    """Reopen and independently validate the written season-level output."""
    try:
        output = pl.read_csv(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise SeasonCalculationError(
            f"Could not validate season output {path}: {error}"
        ) from error
    expected_columns = [
        *EXISTING_SEASON_COLUMNS,
        SD_XWOBA_COLUMN,
        SD_XWOBA_CONTACT_COLUMN,
    ]
    if output.columns != expected_columns:
        raise SeasonCalculationError(
            f"Season output columns are {output.columns}, expected {expected_columns}"
        )
    if not output.equals(expected, null_equal=True):
        raise SeasonCalculationError("Written season output changed from expectation")
    if not output.select(EXISTING_SEASON_COLUMNS).equals(existing, null_equal=True):
        raise SeasonCalculationError("Existing season columns changed or reordered")
    if any(output.null_count().row(0)):
        raise SeasonCalculationError("Season output contains null values")

    independent_sd_xwoba = (
        (
            reconciliation["wBB"] * reconciliation["uBB"]
            + reconciliation["wHBP"] * reconciliation["HBP"]
            + reconciliation["batted_ball_sd_xwoba_sum"]
        )
        / reconciliation["denominator"]
    ).to_numpy()
    independent_contact = (
        reconciliation["batted_ball_sd_xwoba_sum"] / reconciliation["batted_ball_count"]
    ).to_numpy()
    for column, independent in (
        (SD_XWOBA_COLUMN, independent_sd_xwoba),
        (SD_XWOBA_CONTACT_COLUMN, independent_contact),
    ):
        actual = output[column].to_numpy()
        if not np.isfinite(actual).all() or not np.allclose(
            actual, independent, rtol=0.0, atol=1e-15
        ):
            raise SeasonCalculationError(
                f"Season output fails independent {column} check"
            )


def calculate_all_seasons(
    *,
    calculated_directory: Path = CALCULATED_XWOBA_DIRECTORY,
    qualified_path: Path = QUALIFIED_PLAYER_SEASONS_PATH,
    constants_path: Path = WOBA_CONSTANTS_PATH,
    output_path: Path = SEASON_LEVEL_XWOBA_PATH,
    overwrite: bool = False,
) -> SeasonCalculationSummary:
    """Calculate, append, write, and validate all qualified season values."""
    if not overwrite:
        raise SeasonOutputExistsError(
            "Appending Stage 3 metrics replaces the existing season file; "
            "pass --overwrite to authorize this in-place update"
        )
    existing = load_existing_seasons(output_path)
    qualified = load_qualified_seasons(qualified_path)
    weights = load_plate_appearance_weights(constants_path)
    aggregates = pl.concat(
        [
            aggregate_batted_ball_sd_xwoba(
                calculated_directory / f"{year}_all_BBE_with_calculated_xwOBAs.csv",
                year,
            )
            for year in PREDICTION_YEARS
        ],
        how="vertical",
    )
    output, reconciliation = calculate_season_frame(
        existing, qualified, aggregates, weights
    )

    temporary_path = output_path.with_suffix(f"{output_path.suffix}.part")
    try:
        output.write_csv(temporary_path)
        validate_season_output(temporary_path, existing, output, reconciliation)
        temporary_path.replace(output_path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise SeasonCalculationError(
            f"Could not write season output {output_path}: {error}"
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)

    counts = {
        year: output.filter(pl.col("year") == year).height for year in PREDICTION_YEARS
    }
    return SeasonCalculationSummary(output.height, counts, output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse qualified season sd-xwOBA calculation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calculated-directory", type=Path, default=CALCULATED_XWOBA_DIRECTORY
    )
    parser.add_argument("--qualified", type=Path, default=QUALIFIED_PLAYER_SEASONS_PATH)
    parser.add_argument("--constants", type=Path, default=WOBA_CONSTANTS_PATH)
    parser.add_argument("--output", type=Path, default=SEASON_LEVEL_XWOBA_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Append Stage 3 metrics to the qualified season-level artifact."""
    args = parse_args(argv)
    summary = calculate_all_seasons(
        calculated_directory=args.calculated_directory,
        qualified_path=args.qualified,
        constants_path=args.constants,
        output_path=args.output,
        overwrite=args.overwrite,
    )
    formatted_counts = ", ".join(
        f"{year}: {count}" for year, count in sorted(summary.rows_by_year.items())
    )
    print(
        f"Validated {summary.rows} qualified player seasons "
        f"({formatted_counts}) -> {summary.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
