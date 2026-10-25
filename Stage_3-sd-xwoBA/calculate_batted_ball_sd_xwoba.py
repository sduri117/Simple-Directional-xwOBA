"""Append sd-xwOBA to the calculated 2025 and 2026 batted-ball files."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from sd_experiment_config import (
    BATTED_BALL_COLUMNS,
    BATTED_BALL_SCHEMA,
    CALCULATED_XWOBA_DIRECTORY,
    CLASS_LABELS,
    DIRECTION_CATEGORIES,
    DIRECTION_COLUMN,
    FINAL_MODEL_METADATA_PATH,
    FINAL_MODEL_PATH,
    FINAL_TRAINING_ROW_COUNT,
    NUMERIC_FEATURE_COLUMNS,
    PREDICTION_YEARS,
    RAW_FEATURE_COLUMNS,
    SD_XWOBA_COLUMN,
    SELECTED_MODEL_CONFIG,
    WOBA_CONSTANTS_PATH,
    XWOBA_LOCAL_COLUMN,
)
from sd_model_evaluation import align_probability_columns
from train_sd_model import (
    FinalModelError,
    read_metadata,
    sha256_file,
    validate_fitted_pipeline,
    validate_metadata,
)

FloatArray = NDArray[np.float64]


class BattedBallCalculationError(RuntimeError):
    """Raised when a batted-ball sd-xwOBA calculation or validation fails."""


class CalculatedOutputExistsError(BattedBallCalculationError):
    """Raised when calculated-file replacement was not explicitly requested."""


@dataclass(frozen=True, slots=True)
class WOBAWeights:
    """Season-specific hit-event wOBA coefficients in model class order."""

    year: int
    single: float
    double: float
    triple: float
    home_run: float

    def as_array(self) -> FloatArray:
        """Return weights ordered as out, single, double, triple, home run."""
        return np.asarray(
            [0.0, self.single, self.double, self.triple, self.home_run],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class CalculationSummary:
    """Validated output details for one prediction season."""

    year: int
    rows: int
    minimum_sd_xwoba: float
    maximum_sd_xwoba: float
    mean_sd_xwoba: float
    output_path: Path


def load_final_pipeline(
    model_path: Path = FINAL_MODEL_PATH,
    metadata_path: Path = FINAL_MODEL_METADATA_PATH,
) -> Pipeline:
    """Load and validate the selected final model and its metadata."""
    try:
        metadata = read_metadata(metadata_path)
        model_sha256 = sha256_file(model_path)
        selected_configuration = metadata.get("selected_configuration")
        if not isinstance(selected_configuration, dict):
            raise BattedBallCalculationError(
                "Model metadata is missing selected_configuration"
            )
        n_jobs = selected_configuration.get("n_jobs")
        if not isinstance(n_jobs, int) or n_jobs == 0:
            raise BattedBallCalculationError("Model metadata contains invalid n_jobs")
        validate_metadata(
            metadata,
            config=SELECTED_MODEL_CONFIG,
            algorithm="ball_tree",
            n_jobs=n_jobs,
            training_rows=FINAL_TRAINING_ROW_COUNT,
            model_sha256=model_sha256,
        )
        loaded = joblib.load(model_path)
    except (OSError, TypeError, ValueError, FinalModelError) as error:
        raise BattedBallCalculationError(
            f"Could not load final model {model_path}: {error}"
        ) from error
    if not isinstance(loaded, Pipeline):
        raise BattedBallCalculationError("Final model artifact is not a Pipeline")
    try:
        validate_fitted_pipeline(
            loaded,
            SELECTED_MODEL_CONFIG,
            "ball_tree",
            FINAL_TRAINING_ROW_COUNT,
        )
    except FinalModelError as error:
        raise BattedBallCalculationError(f"Final model is invalid: {error}") from error
    return loaded


def load_woba_weights(path: Path, year: int) -> WOBAWeights:
    """Load the four batted-ball coefficients for one prediction year."""
    if year not in PREDICTION_YEARS:
        raise ValueError(f"Unsupported prediction year: {year}")
    columns = ["Season", "w1B", "w2B", "w3B", "wHR"]
    try:
        frame = pl.read_csv(
            path,
            columns=columns,
            schema_overrides={
                "Season": pl.Int64,
                "w1B": pl.Float64,
                "w2B": pl.Float64,
                "w3B": pl.Float64,
                "wHR": pl.Float64,
            },
        ).filter(pl.col("Season") == year)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise BattedBallCalculationError(
            f"Could not read wOBA constants {path}: {error}"
        ) from error
    if frame.height != 1:
        raise BattedBallCalculationError(
            f"Expected one wOBA-constant row for {year}, found {frame.height}"
        )
    values = [float(cast(float, frame[column].item())) for column in columns[1:]]
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise BattedBallCalculationError(
            f"wOBA constants for {year} must be finite and positive"
        )
    return WOBAWeights(year, *values)


def validate_prediction_input(frame: pl.DataFrame, year: int) -> None:
    """Validate an existing calculated batted-ball file before prediction."""
    if year not in PREDICTION_YEARS:
        raise ValueError(f"Unsupported prediction year: {year}")
    if frame.is_empty():
        raise BattedBallCalculationError(f"Batted-ball input for {year} is empty")
    required_columns = (*BATTED_BALL_COLUMNS, XWOBA_LOCAL_COLUMN)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise BattedBallCalculationError(
            f"Batted-ball input is missing required columns: {missing}"
        )
    if frame.columns[: len(required_columns)] != list(required_columns):
        raise BattedBallCalculationError(
            "Existing batted-ball columns are not in their required order"
        )
    expected_schema = {**BATTED_BALL_SCHEMA, XWOBA_LOCAL_COLUMN: pl.Float64}
    unexpected_types = {
        column: (frame.schema[column], expected)
        for column, expected in expected_schema.items()
        if frame.schema[column] != expected
    }
    if unexpected_types:
        raise BattedBallCalculationError(
            f"Batted-ball input has unexpected types: {unexpected_types}"
        )
    null_counts = {
        column: count
        for column, count in zip(frame.columns, frame.null_count().row(0), strict=True)
        if count
    }
    if null_counts:
        raise BattedBallCalculationError(
            f"Batted-ball input contains null values: {null_counts}"
        )
    for column in (*NUMERIC_FEATURE_COLUMNS, "xwOBA-MLB", XWOBA_LOCAL_COLUMN):
        invalid_count = frame.filter(~pl.col(column).is_finite()).height
        if invalid_count:
            raise BattedBallCalculationError(
                f"Batted-ball input has {invalid_count:,} non-finite {column} value(s)"
            )
    actual_directions = set(cast(list[str], frame[DIRECTION_COLUMN].unique().to_list()))
    unexpected_directions = actual_directions - set(DIRECTION_CATEGORIES)
    if unexpected_directions:
        raise BattedBallCalculationError(
            f"Batted-ball input contains invalid hit directions: "
            f"{sorted(unexpected_directions)}"
        )
    parsed_dates = frame["game_date"].str.to_date(strict=False)
    if parsed_dates.null_count():
        raise BattedBallCalculationError("Batted-ball input contains malformed dates")
    actual_years = set(cast(list[int], parsed_dates.dt.year().unique().to_list()))
    if actual_years != {year}:
        raise BattedBallCalculationError(
            f"Batted-ball input contains years {sorted(actual_years)}, expected {year}"
        )


def predict_probabilities_in_batches(
    pipeline: Pipeline,
    features: pl.DataFrame,
    batch_size: int,
) -> FloatArray:
    """Predict aligned five-class probabilities with bounded peak memory."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    probabilities = np.empty((features.height, len(CLASS_LABELS)), dtype=np.float64)
    classes = cast(NDArray[np.int64], pipeline.classes_)
    for start in range(0, features.height, batch_size):
        stop = min(start + batch_size, features.height)
        batch_probabilities = cast(
            FloatArray, pipeline.predict_proba(features.slice(start, stop - start))
        )
        probabilities[start:stop] = align_probability_columns(
            batch_probabilities, classes
        )
    return probabilities


def calculate_sd_xwoba(
    probabilities: FloatArray,
    weights: WOBAWeights,
) -> FloatArray:
    """Calculate probability-weighted sd-xwOBA for every observation."""
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_LABELS):
        raise ValueError("Expected probabilities with five columns")
    values = probabilities @ weights.as_array()
    if not np.isfinite(values).all():
        raise BattedBallCalculationError("Calculated sd-xwOBA is non-finite")
    return values


def validate_calculated_output(
    path: Path,
    source: pl.DataFrame,
    expected_sd_xwoba: FloatArray,
    weights: WOBAWeights,
) -> None:
    """Reopen and reconcile a temporary output before installation."""
    try:
        output = pl.read_csv(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise BattedBallCalculationError(
            f"Could not validate calculated output {path}: {error}"
        ) from error
    expected_columns = [*source.columns, SD_XWOBA_COLUMN]
    if output.columns != expected_columns:
        raise BattedBallCalculationError(
            f"Output columns {output.columns} do not match {expected_columns}"
        )
    if output.height != source.height:
        raise BattedBallCalculationError(
            f"Output has {output.height:,} rows, expected {source.height:,}"
        )
    if not output.select(source.columns).equals(source, null_equal=True):
        raise BattedBallCalculationError("Existing columns changed or reordered")
    actual = output[SD_XWOBA_COLUMN].to_numpy()
    if not np.allclose(actual, expected_sd_xwoba, rtol=0.0, atol=1e-15):
        raise BattedBallCalculationError("Written sd-xwOBA values changed")
    if not np.isfinite(actual).all():
        raise BattedBallCalculationError("Output contains non-finite sd-xwOBA")
    if np.any(actual < 0.0) or np.any(actual > weights.as_array().max()):
        raise BattedBallCalculationError(
            "Output sd-xwOBA values fall outside coefficient bounds"
        )


def calculate_year(
    pipeline: Pipeline,
    *,
    year: int,
    input_path: Path,
    constants_path: Path,
    output_path: Path,
    batch_size: int = 10_000,
    overwrite: bool = False,
) -> CalculationSummary:
    """Calculate, append, and validate one season's batted-ball sd-xwOBA."""
    if output_path.exists() and output_path != input_path and not overwrite:
        raise CalculatedOutputExistsError(
            f"Output exists and replacement was not requested: {output_path}"
        )
    if input_path == output_path and not overwrite:
        raise CalculatedOutputExistsError(
            f"In-place replacement was not requested: {output_path}"
        )
    try:
        loaded = pl.read_csv(
            input_path,
            schema_overrides={**BATTED_BALL_SCHEMA, XWOBA_LOCAL_COLUMN: pl.Float64},
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise BattedBallCalculationError(
            f"Could not read {input_path}: {error}"
        ) from error
    source = (
        loaded.drop(SD_XWOBA_COLUMN) if SD_XWOBA_COLUMN in loaded.columns else loaded
    )
    validate_prediction_input(source, year)
    weights = load_woba_weights(constants_path, year)
    features = source.select(RAW_FEATURE_COLUMNS)
    probabilities = predict_probabilities_in_batches(pipeline, features, batch_size)
    calculated = calculate_sd_xwoba(probabilities, weights)
    output = source.with_columns(
        pl.Series(SD_XWOBA_COLUMN, calculated, dtype=pl.Float64)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.part")
    try:
        output.write_csv(temporary_path)
        validate_calculated_output(temporary_path, source, calculated, weights)
        temporary_path.replace(output_path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise BattedBallCalculationError(
            f"Could not write calculated output {output_path}: {error}"
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)

    return CalculationSummary(
        year=year,
        rows=source.height,
        minimum_sd_xwoba=float(calculated.min()),
        maximum_sd_xwoba=float(calculated.max()),
        mean_sd_xwoba=float(calculated.mean()),
        output_path=output_path,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse batted-ball sd-xwOBA calculation command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, action="append", choices=PREDICTION_YEARS)
    parser.add_argument(
        "--calculated-directory", type=Path, default=CALCULATED_XWOBA_DIRECTORY
    )
    parser.add_argument("--constants", type=Path, default=WOBA_CONSTANTS_PATH)
    parser.add_argument("--model", type=Path, default=FINAL_MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=FINAL_MODEL_METADATA_PATH)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Append sd-xwOBA to calculated files for requested prediction years."""
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    years = tuple(args.year) if args.year else PREDICTION_YEARS
    if len(years) != len(set(years)):
        raise ValueError("Each --year may be specified only once")
    paths = {
        year: args.calculated_directory / f"{year}_all_BBE_with_calculated_xwOBAs.csv"
        for year in years
    }
    if not args.overwrite:
        raise CalculatedOutputExistsError(
            "Appending sd-xwOBA replaces the existing calculated files; "
            "pass --overwrite to authorize this in-place update"
        )

    pipeline = load_final_pipeline(args.model, args.metadata)
    for year in years:
        summary = calculate_year(
            pipeline,
            year=year,
            input_path=paths[year],
            constants_path=args.constants,
            output_path=paths[year],
            batch_size=args.batch_size,
            overwrite=True,
        )
        print(
            f"{summary.year}: {summary.rows:,} rows, sd-xwOBA "
            f"[{summary.minimum_sd_xwoba:.6f}, "
            f"{summary.maximum_sd_xwoba:.6f}], "
            f"mean={summary.mean_sd_xwoba:.6f} -> {summary.output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
