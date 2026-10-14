"""Calculate xwOBA-local for qualified 2025 and 2026 player seasons."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import polars as pl

from experiment_config import (
    CALCULATED_XWOBA_DIRECTORY,
    PREDICTION_YEARS,
    QUALIFIED_PLAYER_SEASONS_PATH,
    SEASON_LEVEL_XWOBA_PATH,
    WOBA_CONSTANTS_PATH,
    XWOBA_LOCAL_COLUMN,
    calculated_batted_ball_path,
)

QUALIFIED_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "Season",
    "Name",
    "AB",
    "BB",
    "IBB",
    "HBP",
    "SF",
    "wOBA",
    "xwOBA",
    "MLBAMID",
)
QUALIFIED_SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    "Season": pl.Int64,
    "Name": pl.String,
    "AB": pl.Int64,
    "BB": pl.Int64,
    "IBB": pl.Int64,
    "HBP": pl.Int64,
    "SF": pl.Int64,
    "wOBA": pl.Float64,
    "xwOBA": pl.Float64,
    "MLBAMID": pl.Int64,
}
SEASON_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "year",
    "batter_name",
    "batter_id",
    "wOBA",
    "xwOBA-MLB",
    XWOBA_LOCAL_COLUMN,
)
SEASON_OUTPUT_SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    "year": pl.Int64,
    "batter_name": pl.String,
    "batter_id": pl.Int64,
    "wOBA": pl.Float64,
    "xwOBA-MLB": pl.Float64,
    XWOBA_LOCAL_COLUMN: pl.Float64,
}


class SeasonCalculationError(RuntimeError):
    """Raised when season-level calculation or validation fails."""


class SeasonOutputExistsError(SeasonCalculationError):
    """Raised when season-output replacement was not explicitly requested."""


@dataclass(frozen=True, slots=True)
class PlateAppearanceWeights:
    """Walk and hit-by-pitch coefficients for one season."""

    year: int
    unintentional_walk: float
    hit_by_pitch: float


@dataclass(frozen=True, slots=True)
class SeasonCalculationSummary:
    """Validated output details for all requested qualified seasons."""

    rows: int
    rows_by_year: Mapping[int, int]
    output_path: Path


def load_qualified_seasons(path: Path) -> pl.DataFrame:
    """Load and validate qualified 2025 and 2026 player-season inputs."""
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
    if frame.is_empty():
        raise SeasonCalculationError("Qualified player-season input is empty")
    actual_years = set(cast(list[int], frame["Season"].unique().to_list()))
    if actual_years != set(PREDICTION_YEARS):
        raise SeasonCalculationError(
            f"Qualified data contains years {sorted(actual_years)}, expected "
            f"{list(PREDICTION_YEARS)}"
        )
    null_counts = {
        column: count
        for column, count in zip(frame.columns, frame.null_count().row(0), strict=True)
        if count
    }
    if null_counts:
        raise SeasonCalculationError(
            f"Qualified data contains null values: {null_counts}"
        )
    duplicates = frame.group_by(["Season", "MLBAMID"]).len().filter(pl.col("len") != 1)
    if duplicates.height:
        raise SeasonCalculationError(
            f"Qualified data contains {duplicates.height} duplicate "
            "player-season key(s)"
        )
    for column in ("AB", "BB", "IBB", "HBP", "SF"):
        if frame.filter(pl.col(column) < 0).height:
            raise SeasonCalculationError(f"Qualified data contains negative {column}")
    if frame.filter(pl.col("IBB") > pl.col("BB")).height:
        raise SeasonCalculationError("Qualified data contains IBB greater than BB")
    for column in ("wOBA", "xwOBA"):
        if frame.filter(~pl.col(column).is_finite()).height:
            raise SeasonCalculationError(
                f"Qualified data contains non-finite {column} values"
            )
    denominator = (
        pl.col("AB") + pl.col("BB") - pl.col("IBB") + pl.col("SF") + pl.col("HBP")
    )
    if frame.filter(denominator <= 0).height:
        raise SeasonCalculationError("Qualified data contains nonpositive denominators")
    return frame


def load_plate_appearance_weights(
    path: Path,
) -> dict[int, PlateAppearanceWeights]:
    """Load and validate wBB and wHBP for both prediction seasons."""
    columns = ["Season", "wBB", "wHBP"]
    try:
        frame = pl.read_csv(
            path,
            columns=columns,
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
        raise SeasonCalculationError(
            "Expected one plate-appearance coefficient row per prediction year"
        )
    result: dict[int, PlateAppearanceWeights] = {}
    for row in frame.iter_rows(named=True):
        year = int(cast(int, row["Season"]))
        walk = float(cast(float, row["wBB"]))
        hbp = float(cast(float, row["wHBP"]))
        if not all(math.isfinite(value) and value > 0.0 for value in (walk, hbp)):
            raise SeasonCalculationError(
                f"Plate-appearance coefficients for {year} must be finite and positive"
            )
        result[year] = PlateAppearanceWeights(year, walk, hbp)
    if set(result) != set(PREDICTION_YEARS):
        raise SeasonCalculationError(
            "Plate-appearance coefficient years are incomplete"
        )
    return result


def aggregate_batted_ball_xwoba(path: Path, year: int) -> pl.DataFrame:
    """Sum calculated batted-ball xwOBA by batter for one season."""
    if year not in PREDICTION_YEARS:
        raise ValueError(f"Unsupported prediction year: {year}")
    try:
        frame = pl.read_csv(
            path,
            columns=["game_date", "batter", XWOBA_LOCAL_COLUMN],
            schema_overrides={
                "game_date": pl.String,
                "batter": pl.Int64,
                XWOBA_LOCAL_COLUMN: pl.Float64,
            },
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise SeasonCalculationError(
            f"Could not read calculated batted balls {path}: {error}"
        ) from error
    if frame.is_empty():
        raise SeasonCalculationError(
            f"Calculated batted-ball input for {year} is empty"
        )
    if frame.null_count().row(0) != (0, 0, 0):
        raise SeasonCalculationError(
            f"Calculated batted-ball input for {year} contains null values"
        )
    if frame.filter(~pl.col(XWOBA_LOCAL_COLUMN).is_finite()).height:
        raise SeasonCalculationError(
            f"Calculated batted-ball input for {year} contains non-finite values"
        )
    parsed_dates = frame["game_date"].str.to_date(strict=False)
    if parsed_dates.null_count():
        raise SeasonCalculationError(
            f"Calculated batted-ball input for {year} contains malformed dates"
        )
    actual_years = set(cast(list[int], parsed_dates.dt.year().unique().to_list()))
    if actual_years != {year}:
        raise SeasonCalculationError(
            f"Calculated batted-ball input contains years {sorted(actual_years)}, "
            f"expected {year}"
        )
    return (
        frame.group_by("batter")
        .agg(
            pl.len().cast(pl.Int64).alias("batted_ball_count"),
            pl.col(XWOBA_LOCAL_COLUMN).sum().alias("batted_ball_xwoba_sum"),
        )
        .with_columns(pl.lit(year).cast(pl.Int64).alias("year"))
        .rename({"batter": "batter_id"})
        .select("year", "batter_id", "batted_ball_count", "batted_ball_xwoba_sum")
    )


def calculate_season_frame(
    qualified: pl.DataFrame,
    batted_ball_aggregates: pl.DataFrame,
    weights: Mapping[int, PlateAppearanceWeights],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Join inputs and calculate season xwOBA-local plus reconciliation fields."""
    coefficient_frame = pl.DataFrame(
        {
            "year": list(PREDICTION_YEARS),
            "wBB": [weights[year].unintentional_walk for year in PREDICTION_YEARS],
            "wHBP": [weights[year].hit_by_pitch for year in PREDICTION_YEARS],
        },
        schema={"year": pl.Int64, "wBB": pl.Float64, "wHBP": pl.Float64},
    )
    joined = (
        qualified.rename(
            {
                "Season": "year",
                "Name": "batter_name",
                "MLBAMID": "batter_id",
                "xwOBA": "xwOBA-MLB",
            }
        )
        .join(
            batted_ball_aggregates,
            on=["year", "batter_id"],
            how="left",
            validate="1:1",
        )
        .join(coefficient_frame, on="year", how="left", validate="m:1")
    )
    unmatched = joined.filter(
        pl.col("batted_ball_count").is_null()
        | pl.col("batted_ball_xwoba_sum").is_null()
    )
    if unmatched.height:
        identifiers = unmatched.select("year", "batter_id").to_dicts()
        raise SeasonCalculationError(
            f"Qualified player seasons lack batted-ball matches: {identifiers}"
        )

    reconciliation = (
        joined.with_columns(
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
                + pl.col("batted_ball_xwoba_sum")
            ).alias("numerator")
        )
        .with_columns(
            (pl.col("numerator") / pl.col("denominator")).alias(XWOBA_LOCAL_COLUMN)
        )
        .sort("year", "batter_name", "batter_id")
    )
    if reconciliation.filter(
        ~pl.col(XWOBA_LOCAL_COLUMN).is_finite()
        | (pl.col("denominator") <= 0)
        | (pl.col("uBB") < 0)
    ).height:
        raise SeasonCalculationError("Season calculation produced invalid values")

    output = reconciliation.select(SEASON_OUTPUT_COLUMNS)
    return output, reconciliation


def validate_season_output(
    path: Path,
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
    if output.columns != list(SEASON_OUTPUT_COLUMNS):
        raise SeasonCalculationError(
            f"Season output columns {output.columns} do not match "
            f"{list(SEASON_OUTPUT_COLUMNS)}"
        )
    if output.schema != SEASON_OUTPUT_SCHEMA:
        raise SeasonCalculationError(
            f"Season output schema {output.schema} does not match "
            f"{SEASON_OUTPUT_SCHEMA}"
        )
    if not output.equals(expected, null_equal=True):
        raise SeasonCalculationError("Written season output changed from expectation")
    if output.group_by(["year", "batter_id"]).len().filter(pl.col("len") != 1).height:
        raise SeasonCalculationError("Season output has duplicate player-season keys")
    if output.null_count().row(0) != (0, 0, 0, 0, 0, 0):
        raise SeasonCalculationError("Season output contains null values")
    for column in ("wOBA", "xwOBA-MLB", XWOBA_LOCAL_COLUMN):
        if output.filter(~pl.col(column).is_finite()).height:
            raise SeasonCalculationError(
                f"Season output contains non-finite {column} values"
            )

    independently_calculated = (
        (
            reconciliation["wBB"] * reconciliation["uBB"]
            + reconciliation["wHBP"] * reconciliation["HBP"]
            + reconciliation["batted_ball_xwoba_sum"]
        )
        / reconciliation["denominator"]
    ).to_numpy()
    actual = output[XWOBA_LOCAL_COLUMN].to_numpy()
    if not np.allclose(actual, independently_calculated, rtol=0.0, atol=1e-15):
        raise SeasonCalculationError("Season output fails independent formula check")


def calculate_all_seasons(
    *,
    calculated_directory: Path = CALCULATED_XWOBA_DIRECTORY,
    qualified_path: Path = QUALIFIED_PLAYER_SEASONS_PATH,
    constants_path: Path = WOBA_CONSTANTS_PATH,
    output_path: Path = SEASON_LEVEL_XWOBA_PATH,
    overwrite: bool = False,
) -> SeasonCalculationSummary:
    """Calculate, write, and validate all qualified prediction-year seasons."""
    if output_path.exists() and not overwrite:
        raise SeasonOutputExistsError(
            f"Output exists and replacement was not requested: {output_path}"
        )
    qualified = load_qualified_seasons(qualified_path)
    weights = load_plate_appearance_weights(constants_path)
    aggregates = pl.concat(
        [
            aggregate_batted_ball_xwoba(
                calculated_directory / calculated_batted_ball_path(year).name,
                year,
            )
            for year in PREDICTION_YEARS
        ],
        how="vertical",
    )
    output, reconciliation = calculate_season_frame(qualified, aggregates, weights)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.part")
    try:
        output.write_csv(temporary_path)
        validate_season_output(temporary_path, output, reconciliation)
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
    """Parse qualified player-season calculation command-line arguments."""
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
    """Generate the qualified 2025 and 2026 season-level xwOBA output."""
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
