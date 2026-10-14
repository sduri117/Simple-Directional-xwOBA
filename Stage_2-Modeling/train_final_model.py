"""Train, serialize, and validate the user-selected Stage 2 KNN pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Final, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from experiment_config import (
    CLASS_LABELS,
    FEATURE_COLUMNS,
    FINAL_MODEL_METADATA_PATH,
    FINAL_MODEL_PATH,
    MODEL_SELECTION_RESULTS_PATH,
    PROCESSED_DATA_DIRECTORY,
    RANDOM_SEED,
    SELECTED_MODEL_CONFIG,
    SELECTED_NEIGHBOR_ALGORITHM,
    TARGET_COLUMN,
    TRAINING_YEARS,
    KNNConfig,
)
from model_evaluation import (
    METRIC_COLUMNS,
    ModelingDataset,
    NeighborAlgorithm,
    build_knn_pipeline,
    load_modeling_data,
    read_result_checkpoint,
)

FloatArray = NDArray[np.float64]
METADATA_SCHEMA_VERSION: Final[int] = 1


class FinalModelError(RuntimeError):
    """Raised when final-model training or artifact validation fails."""


class ModelArtifactExistsError(FinalModelError):
    """Raised when artifact replacement was not explicitly requested."""


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    """Validated final-model training and artifact details."""

    training_rows: int
    elapsed_seconds: float
    model_path: Path
    metadata_path: Path
    model_sha256: str


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_validation_metrics(
    results_path: Path = MODEL_SELECTION_RESULTS_PATH,
    config: KNNConfig = SELECTED_MODEL_CONFIG,
) -> dict[str, float]:
    """Read the unique model-selection row for the chosen configuration."""
    frame = read_result_checkpoint(results_path)
    selected = frame.filter(
        (frame["model_type"] == "knn")
        & (frame["feature_scaling"] == config.scaling)
        & (frame["n_neighbors"] == config.n_neighbors)
        & (frame["weights"] == config.weights)
        & (frame["p"] == config.p)
        & (frame["neighbor_algorithm"] == SELECTED_NEIGHBOR_ALGORITHM)
    )
    if selected.height != 1:
        raise FinalModelError(
            f"Expected one model-selection row for {config}, found {selected.height}"
        )
    row = selected.row(0, named=True)
    return {column: float(cast(float, row[column])) for column in METRIC_COLUMNS}


def validate_fitted_pipeline(
    pipeline: Pipeline,
    config: KNNConfig,
    algorithm: NeighborAlgorithm,
    expected_training_rows: int,
) -> None:
    """Validate fitted steps, hyperparameters, class order, and row coverage."""
    if list(pipeline.named_steps) != ["scaler", "classifier"]:
        raise FinalModelError(
            f"Unexpected pipeline steps: {list(pipeline.named_steps)}"
        )
    scaler = pipeline.named_steps["scaler"]
    if config.scaling == "none" and scaler != "passthrough":
        raise FinalModelError("Selected pipeline unexpectedly applies feature scaling")

    classifier = pipeline.named_steps["classifier"]
    if not isinstance(classifier, KNeighborsClassifier):
        raise FinalModelError("Pipeline classifier is not KNeighborsClassifier")
    parameters = classifier.get_params(deep=False)
    if parameters.get("n_neighbors") != config.n_neighbors:
        raise FinalModelError("Fitted model has the wrong n_neighbors")
    if parameters.get("weights") != config.weights:
        raise FinalModelError("Fitted model has the wrong weights")
    if parameters.get("p") != config.p:
        raise FinalModelError("Fitted model has the wrong distance power")
    if parameters.get("algorithm") != algorithm:
        raise FinalModelError("Fitted model has the wrong neighbor algorithm")
    if classifier.n_samples_fit_ != expected_training_rows:
        raise FinalModelError(
            f"Fitted model contains {classifier.n_samples_fit_:,} rows, expected "
            f"{expected_training_rows:,}"
        )
    classes = cast(NDArray[np.int64], pipeline.classes_)
    if classes.tolist() != list(CLASS_LABELS):
        raise FinalModelError(
            f"Fitted classes {classes.tolist()} do not match {list(CLASS_LABELS)}"
        )


def fixed_validation_sample(features: FloatArray, sample_size: int = 257) -> FloatArray:
    """Select deterministic, evenly spaced rows for serialization validation."""
    if features.shape[0] == 0:
        raise ValueError("Cannot sample an empty feature matrix")
    count = min(sample_size, features.shape[0])
    indices = np.linspace(0, features.shape[0] - 1, count, dtype=np.int64)
    return features[indices]


def build_metadata(
    *,
    config: KNNConfig,
    algorithm: NeighborAlgorithm,
    n_jobs: int,
    training_rows: int,
    validation_metrics: Mapping[str, float],
    input_hashes: Mapping[str, str],
    model_sha256: str,
) -> dict[str, object]:
    """Build portable metadata for the fitted final model."""
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "model_family": "KNeighborsClassifier",
        "model_sha256": model_sha256,
        "selected_configuration": {
            "feature_scaling": config.scaling,
            "n_neighbors": config.n_neighbors,
            "weights": config.weights,
            "p": config.p,
            "metric": "minkowski",
            "neighbor_algorithm": algorithm,
            "n_jobs": n_jobs,
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "class_order": list(CLASS_LABELS),
        "training_years": list(TRAINING_YEARS),
        "training_row_count": training_rows,
        "random_seed": RANDOM_SEED,
        "validation_metrics": dict(validation_metrics),
        "input_sha256": dict(input_hashes),
        "software_versions": {
            "python": platform.python_version(),
            "joblib": version("joblib"),
            "numpy": version("numpy"),
            "polars": version("polars"),
            "scikit-learn": version("scikit-learn"),
        },
    }


def read_metadata(path: Path) -> dict[str, object]:
    """Read a JSON metadata artifact as a string-keyed mapping."""
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise FinalModelError(f"Could not read metadata {path}: {error}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FinalModelError(f"Metadata {path} is not a JSON object")
    return cast(dict[str, object], value)


def validate_metadata(
    metadata: Mapping[str, object],
    *,
    config: KNNConfig,
    algorithm: NeighborAlgorithm,
    n_jobs: int,
    training_rows: int,
    model_sha256: str,
) -> None:
    """Validate the identity and training contract recorded in metadata."""
    expected_values: dict[str, object] = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "model_family": "KNeighborsClassifier",
        "model_sha256": model_sha256,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "class_order": list(CLASS_LABELS),
        "training_years": list(TRAINING_YEARS),
        "training_row_count": training_rows,
        "random_seed": RANDOM_SEED,
    }
    for key, expected in expected_values.items():
        if metadata.get(key) != expected:
            raise FinalModelError(
                f"Metadata field {key!r} is {metadata.get(key)!r}, expected "
                f"{expected!r}"
            )

    expected_config = {
        "feature_scaling": config.scaling,
        "n_neighbors": config.n_neighbors,
        "weights": config.weights,
        "p": config.p,
        "metric": "minkowski",
        "neighbor_algorithm": algorithm,
        "n_jobs": n_jobs,
    }
    if metadata.get("selected_configuration") != expected_config:
        raise FinalModelError("Metadata selected configuration is incorrect")

    validation_metrics = metadata.get("validation_metrics")
    if not isinstance(validation_metrics, dict) or set(validation_metrics) != set(
        METRIC_COLUMNS
    ):
        raise FinalModelError("Metadata validation metrics are incomplete")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in validation_metrics.values()
    ):
        raise FinalModelError("Metadata validation metrics contain invalid values")

    input_hashes = metadata.get("input_sha256")
    if not isinstance(input_hashes, dict) or len(input_hashes) != len(TRAINING_YEARS):
        raise FinalModelError("Metadata input hashes are incomplete")
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in input_hashes.values()
    ):
        raise FinalModelError("Metadata contains invalid input hashes")

    software_versions = metadata.get("software_versions")
    expected_software = {"python", "joblib", "numpy", "polars", "scikit-learn"}
    if not isinstance(software_versions, dict) or set(software_versions) != (
        expected_software
    ):
        raise FinalModelError("Metadata software versions are incomplete")


def train_and_save_model(
    dataset: ModelingDataset,
    *,
    validation_metrics: Mapping[str, float],
    input_paths: Sequence[Path],
    model_path: Path = FINAL_MODEL_PATH,
    metadata_path: Path = FINAL_MODEL_METADATA_PATH,
    config: KNNConfig = SELECTED_MODEL_CONFIG,
    algorithm: NeighborAlgorithm = "ball_tree",
    n_jobs: int = -1,
    overwrite: bool = False,
) -> TrainingSummary:
    """Fit the final pipeline and atomically write validated model artifacts."""
    if model_path == metadata_path:
        raise ValueError("Model and metadata paths must differ")
    existing_paths = [path for path in (model_path, metadata_path) if path.exists()]
    if existing_paths and not overwrite:
        raise ModelArtifactExistsError(
            "Artifact replacement was not requested for: "
            + ", ".join(str(path) for path in existing_paths)
        )
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be zero")
    if len(input_paths) != len(TRAINING_YEARS):
        raise ValueError(f"Expected {len(TRAINING_YEARS)} training input paths")

    for path in input_paths:
        if not path.is_file():
            raise FinalModelError(f"Training input does not exist: {path}")
    input_hashes = {str(path): sha256_file(path) for path in input_paths}

    start = time.perf_counter()
    pipeline = build_knn_pipeline(config, algorithm, n_jobs)
    pipeline.fit(dataset.features, dataset.target)
    elapsed_seconds = time.perf_counter() - start
    training_rows = dataset.target.size
    validate_fitted_pipeline(pipeline, config, algorithm, training_rows)

    validation_sample = fixed_validation_sample(dataset.features)
    expected_probabilities = cast(FloatArray, pipeline.predict_proba(validation_sample))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model_path = model_path.with_suffix(f"{model_path.suffix}.part")
    temporary_metadata_path = metadata_path.with_suffix(f"{metadata_path.suffix}.part")
    try:
        joblib.dump(pipeline, temporary_model_path, compress=3)
        model_sha256 = sha256_file(temporary_model_path)
        metadata = build_metadata(
            config=config,
            algorithm=algorithm,
            n_jobs=n_jobs,
            training_rows=training_rows,
            validation_metrics=validation_metrics,
            input_hashes=input_hashes,
            model_sha256=model_sha256,
        )
        with temporary_metadata_path.open("w", encoding="utf-8") as output:
            json.dump(metadata, output, indent=2, sort_keys=True)
            output.write("\n")

        reloaded = joblib.load(temporary_model_path)
        if not isinstance(reloaded, Pipeline):
            raise FinalModelError("Reloaded artifact is not a scikit-learn Pipeline")
        validate_fitted_pipeline(reloaded, config, algorithm, training_rows)
        actual_probabilities = cast(
            FloatArray, reloaded.predict_proba(validation_sample)
        )
        if not np.array_equal(expected_probabilities, actual_probabilities):
            raise FinalModelError(
                "Reloaded model did not reproduce the validation probabilities"
            )
        reloaded_metadata = read_metadata(temporary_metadata_path)
        validate_metadata(
            reloaded_metadata,
            config=config,
            algorithm=algorithm,
            n_jobs=n_jobs,
            training_rows=training_rows,
            model_sha256=model_sha256,
        )

        temporary_model_path.replace(model_path)
        temporary_metadata_path.replace(metadata_path)
    except (OSError, TypeError, ValueError) as error:
        raise FinalModelError(
            f"Could not write final model artifacts: {error}"
        ) from error
    finally:
        temporary_model_path.unlink(missing_ok=True)
        temporary_metadata_path.unlink(missing_ok=True)

    if sha256_file(model_path) != model_sha256:
        raise FinalModelError("Final model checksum changed after installation")
    final_metadata = read_metadata(metadata_path)
    validate_metadata(
        final_metadata,
        config=config,
        algorithm=algorithm,
        n_jobs=n_jobs,
        training_rows=training_rows,
        model_sha256=model_sha256,
    )
    return TrainingSummary(
        training_rows=training_rows,
        elapsed_seconds=elapsed_seconds,
        model_path=model_path,
        metadata_path=metadata_path,
        model_sha256=model_sha256,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse final-model training command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-directory", type=Path, default=PROCESSED_DATA_DIRECTORY
    )
    parser.add_argument(
        "--selection-results", type=Path, default=MODEL_SELECTION_RESULTS_PATH
    )
    parser.add_argument("--model-output", type=Path, default=FINAL_MODEL_PATH)
    parser.add_argument(
        "--metadata-output", type=Path, default=FINAL_MODEL_METADATA_PATH
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Train and validate the selected final model on all training years."""
    args = parse_args(argv)
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be zero")
    dataset = load_modeling_data(args.input_directory)
    metrics = selected_validation_metrics(args.selection_results)
    input_paths = [
        args.input_directory / f"{year}_all_BBE.csv" for year in TRAINING_YEARS
    ]
    summary = train_and_save_model(
        dataset,
        validation_metrics=metrics,
        input_paths=input_paths,
        model_path=args.model_output,
        metadata_path=args.metadata_output,
        config=SELECTED_MODEL_CONFIG,
        algorithm=cast(NeighborAlgorithm, SELECTED_NEIGHBOR_ALGORITHM),
        n_jobs=args.n_jobs,
        overwrite=args.overwrite,
    )
    print(
        f"Trained {summary.training_rows:,} rows in " f"{summary.elapsed_seconds:.3f}s"
    )
    print(f"Model: {summary.model_path}")
    print(f"Metadata: {summary.metadata_path}")
    print(f"SHA-256: {summary.model_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
