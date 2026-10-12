"""Condense monthly Baseball Savant data into analysis-ready annual files."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl

YEARS: Final[tuple[int, ...]] = (2022, 2023, 2024, 2025, 2026)
MONTH_SLUGS: Final[tuple[str, ...]] = (
    "mar-apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep-oct",
)
DIRECTIONS: Final[tuple[str, ...]] = ("pull", "straightaway", "opposite")
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "game_date",
    "player_name",
    "batter",
    "events",
    "launch_speed",
    "launch_angle",
)
DERIVED_COLUMNS: Final[tuple[str, ...]] = (
    "hit_direction",
    "simplified_events",
    "total_bases",
)
OUTPUT_COLUMNS: Final[tuple[str, ...]] = REQUIRED_COLUMNS + DERIVED_COLUMNS
HIT_EVENTS: Final[tuple[str, ...]] = (
    "single",
    "double",
    "triple",
    "home_run",
)
EVENT_TO_TOTAL_BASES: Final[dict[str, int]] = {
    "out": 0,
    "single": 1,
    "double": 2,
    "triple": 3,
    "home_run": 4,
}
RAW_DATA_DIRECTORY: Final[Path] = Path("RawData2")
PROCESSED_DATA_DIRECTORY: Final[Path] = Path("ProcessedData")

CSV_SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    "game_date": pl.String,
    "player_name": pl.String,
    "batter": pl.Int64,
    "events": pl.String,
    "launch_speed": pl.Float64,
    "launch_angle": pl.Float64,
}


class PreprocessingError(Exception):
    """Base exception for preprocessing failures."""


class InputInventoryError(PreprocessingError):
    """Raised when a year does not have its complete monthly file inventory."""


class OutputExistsError(PreprocessingError):
    """Raised when an output exists and replacement was not requested."""


@dataclass(frozen=True)
class MonthlyDataFile:
    """One monthly, directional Baseball Savant input file."""

    year: int
    month_slug: str
    direction: str
    path: Path


@dataclass(frozen=True)
class YearPreprocessingSummary:
    """Row-count reconciliation for one annual output."""

    year: int
    input_files: int
    input_rows: int
    output_rows: int
    removed_rows: int
    output_path: Path


def expected_input_files(input_directory: Path, year: int) -> list[MonthlyDataFile]:
    """Return the expected files for a year in deterministic concatenation order."""
    _validate_year(year)
    return [
        MonthlyDataFile(
            year=year,
            month_slug=month_slug,
            direction=direction,
            path=input_directory / f"{year}_{month_slug}_{direction}_BBE.csv",
        )
        for month_slug in MONTH_SLUGS
        for direction in DIRECTIONS
    ]


def discover_year_files(input_directory: Path, year: int) -> list[MonthlyDataFile]:
    """Require and return all 18 expected input files for a year."""
    expected_files = expected_input_files(input_directory, year)
    missing_paths = [
        data_file.path for data_file in expected_files if not data_file.path.is_file()
    ]
    if missing_paths:
        formatted_paths = "\n".join(f"- {path}" for path in missing_paths)
        raise InputInventoryError(
            f"Missing {len(missing_paths)} input file(s) for {year}:\n{formatted_paths}"
        )
    return expected_files


def transform_monthly_file(data_file: MonthlyDataFile) -> tuple[pl.DataFrame, int]:
    """Select, filter, and derive columns for one monthly input file."""
    try:
        frame = pl.read_csv(
            data_file.path,
            columns=list(REQUIRED_COLUMNS),
            schema_overrides=CSV_SCHEMA,
            null_values="",
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise PreprocessingError(f"Could not read {data_file.path}: {error}") from error

    input_rows = frame.height
    frame = frame.with_columns(
        pl.col("launch_speed").fill_nan(None),
        pl.col("launch_angle").fill_nan(None),
    ).drop_nulls(list(REQUIRED_COLUMNS))

    simplified_events = (
        pl.when(pl.col("events").is_in(HIT_EVENTS))
        .then(pl.col("events"))
        .otherwise(pl.lit("out"))
        .alias("simplified_events")
    )
    frame = frame.with_columns(
        pl.lit(data_file.direction).alias("hit_direction"),
        simplified_events,
    ).with_columns(
        pl.col("simplified_events")
        .replace_strict(EVENT_TO_TOTAL_BASES, return_dtype=pl.Int8)
        .alias("total_bases")
    )

    return frame.select(OUTPUT_COLUMNS), input_rows


def preprocess_year(
    input_directory: Path,
    output_directory: Path,
    year: int,
    *,
    overwrite: bool = False,
) -> YearPreprocessingSummary:
    """Create one annual dataset from all monthly and directional inputs."""
    input_files = discover_year_files(input_directory, year)
    output_path = output_directory / f"{year}_all_BBE.csv"
    if output_path.exists() and not overwrite:
        raise OutputExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    transformed_frames: list[pl.DataFrame] = []
    input_rows = 0
    for data_file in input_files:
        transformed_frame, file_input_rows = transform_monthly_file(data_file)
        transformed_frames.append(transformed_frame)
        input_rows += file_input_rows

    annual_frame = pl.concat(transformed_frames, how="vertical")
    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.part")
    try:
        annual_frame.write_csv(temporary_path)
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    output_rows = annual_frame.height
    return YearPreprocessingSummary(
        year=year,
        input_files=len(input_files),
        input_rows=input_rows,
        output_rows=output_rows,
        removed_rows=input_rows - output_rows,
        output_path=output_path,
    )


def preprocess_years(
    input_directory: Path,
    output_directory: Path,
    years: Sequence[int],
    *,
    overwrite: bool = False,
) -> list[YearPreprocessingSummary]:
    """Create annual datasets serially for the requested years."""
    return [
        preprocess_year(
            input_directory,
            output_directory,
            year,
            overwrite=overwrite,
        )
        for year in years
    ]


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Condense monthly Baseball Savant BBE data into annual CSVs."
    )
    parser.add_argument(
        "--year",
        action="append",
        choices=YEARS,
        type=int,
        dest="years",
        help="Year to preprocess; repeat for multiple years (default: all years).",
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=RAW_DATA_DIRECTORY,
        help=f"Monthly input directory (default: {RAW_DATA_DIRECTORY}).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROCESSED_DATA_DIRECTORY,
        help=f"Annual output directory (default: {PROCESSED_DATA_DIRECTORY}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace annual output files that already exist.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run preprocessing and report row-count summaries."""
    options = parse_arguments(arguments)
    years = options.years if options.years is not None else YEARS
    try:
        summaries = preprocess_years(
            options.input_directory,
            options.output_directory,
            years,
            overwrite=options.overwrite,
        )
    except PreprocessingError as error:
        print(f"Error: {error}")
        return 1

    for summary in summaries:
        print(
            f"{summary.year}: wrote {summary.output_rows:,} rows to "
            f"{summary.output_path} ({summary.removed_rows:,} of "
            f"{summary.input_rows:,} input rows removed)"
        )
    return 0


def _validate_year(year: int) -> None:
    if year not in YEARS:
        raise ValueError(f"Unsupported year: {year}")


if __name__ == "__main__":
    raise SystemExit(main())
