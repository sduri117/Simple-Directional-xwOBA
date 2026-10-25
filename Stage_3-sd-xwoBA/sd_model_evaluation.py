"""Evaluate the Stage 3 baseline and directional KNN probability classifiers."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from sd_experiment_config import (
    CHRONOLOGICAL_TRAINING_YEARS,
    CHRONOLOGICAL_VALIDATION_YEAR,
    CLASS_LABELS,
    CV_FOLDS,
    KNN_CONFIGS,
    MODEL_SELECTION_RESULTS_PATH,
    MODELING_REQUIRED_COLUMNS,
    MODELING_SCHEMA,
    PROCESSED_DATA_DIRECTORY,
    RANDOM_SEED,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRAINING_YEARS,
    KNNConfig,
    ScalingName,
    WeightName,
    validate_modeling_frame,
)
from sd_feature_pipeline import build_feature_preprocessor

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
NeighborAlgorithm = Literal["brute", "kd_tree", "ball_tree"]
ModelType = Literal["baseline", "knn"]

SUPPORTED_ALGORITHMS: Final[tuple[NeighborAlgorithm, ...]] = (
    "brute",
    "kd_tree",
    "ball_tree",
)
RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "model_type",
    "feature_scaling",
    "n_neighbors",
    "weights",
    "p",
    "neighbor_algorithm",
    "cv_log_loss",
    "cv_brier_score",
    "cv_accuracy",
    "chronological_log_loss",
    "chronological_brier_score",
    "chronological_accuracy",
)
RESULT_SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    "model_type": pl.String,
    "feature_scaling": pl.String,
    "n_neighbors": pl.Int64,
    "weights": pl.String,
    "p": pl.Int64,
    "neighbor_algorithm": pl.String,
    "cv_log_loss": pl.Float64,
    "cv_brier_score": pl.Float64,
    "cv_accuracy": pl.Float64,
    "chronological_log_loss": pl.Float64,
    "chronological_brier_score": pl.Float64,
    "chronological_accuracy": pl.Float64,
}
METRIC_COLUMNS: Final[tuple[str, ...]] = RESULT_COLUMNS[6:]


class ModelEvaluationError(RuntimeError):
    """Raised when Stage 3 model evaluation or result validation fails."""


@dataclass(frozen=True, slots=True)
class ModelingDataset:
    """Validated raw model features, targets, and observation years."""

    features: pl.DataFrame
    target: IntArray
    years: IntArray

    def __post_init__(self) -> None:
        row_count = self.target.shape[0]
        if self.features.shape != (row_count, len(RAW_FEATURE_COLUMNS)):
            raise ValueError("Feature shape does not match the target")
        if tuple(self.features.columns) != RAW_FEATURE_COLUMNS:
            raise ValueError("Feature columns are not in the required order")
        if self.years.shape != (row_count,):
            raise ValueError("Year shape does not match the target")


@dataclass(frozen=True, slots=True)
class MetricResults:
    """Probability and classification metrics for one validation design."""

    log_loss: float
    brier_score: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class EvaluationResults:
    """Cross-validation and chronological results for one candidate."""

    cross_validation: MetricResults
    chronological: MetricResults


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Timing and equivalence result for one neighbor-search implementation."""

    algorithm: NeighborAlgorithm
    elapsed_seconds: float
    equivalent_to_brute: bool


@dataclass(frozen=True, slots=True)
class ResultRow:
    """One checkpointable row in the model-selection output."""

    model_type: ModelType
    feature_scaling: ScalingName | None
    n_neighbors: int | None
    weights: WeightName | None
    p: int | None
    neighbor_algorithm: NeighborAlgorithm | None
    cv_log_loss: float
    cv_brier_score: float
    cv_accuracy: float
    chronological_log_loss: float
    chronological_brier_score: float
    chronological_accuracy: float

    def as_record(self) -> dict[str, str | int | float | None]:
        """Return a dictionary in the declared result-column order."""
        record = asdict(self)
        return {column: record[column] for column in RESULT_COLUMNS}


class TrainingFrequencyClassifier:
    """Return the class distribution learned from the current training set."""

    classes_: IntArray
    probabilities_: FloatArray

    def fit(self, target: IntArray) -> TrainingFrequencyClassifier:
        """Learn explicitly ordered class frequencies."""
        if target.ndim != 1 or target.size == 0:
            raise ModelEvaluationError("Baseline target must be a nonempty vector")
        invalid_classes = set(target.tolist()) - set(CLASS_LABELS)
        if invalid_classes:
            raise ModelEvaluationError(
                f"Baseline target contains invalid classes: {sorted(invalid_classes)}"
            )
        counts = np.bincount(target, minlength=len(CLASS_LABELS)).astype(np.float64)
        if np.any(counts == 0):
            raise ModelEvaluationError(
                "Baseline training data must contain all classes"
            )
        self.classes_ = np.asarray(CLASS_LABELS, dtype=np.int64)
        self.probabilities_ = counts / target.size
        return self

    def predict_proba(self, row_count: int) -> FloatArray:
        """Repeat the learned class probabilities for every requested row."""
        if row_count < 0 or not hasattr(self, "probabilities_"):
            raise ModelEvaluationError("Baseline must be fitted before prediction")
        return np.tile(self.probabilities_, (row_count, 1))


def load_modeling_data(
    data_directory: Path = PROCESSED_DATA_DIRECTORY,
) -> ModelingDataset:
    """Load and validate the annual Stage 3 training inputs."""
    frames: list[pl.DataFrame] = []
    for year in TRAINING_YEARS:
        path = data_directory / f"{year}_all_BBE.csv"
        try:
            frame = pl.read_csv(
                path,
                columns=list(MODELING_REQUIRED_COLUMNS),
                schema_overrides=MODELING_SCHEMA,
            )
        except (OSError, pl.exceptions.PolarsError) as error:
            raise ModelEvaluationError(
                f"Could not read modeling data {path}: {error}"
            ) from error
        validate_modeling_frame(frame, year)
        frames.append(frame.with_columns(pl.lit(year).cast(pl.Int64).alias("year")))

    combined = pl.concat(frames, how="vertical")
    features = combined.select(RAW_FEATURE_COLUMNS)
    target = combined[TARGET_COLUMN].to_numpy().astype(np.int64, copy=False)
    years = combined["year"].to_numpy().astype(np.int64, copy=False)
    return ModelingDataset(features, target, years)


def build_knn_pipeline(
    config: KNNConfig,
    algorithm: NeighborAlgorithm,
    n_jobs: int,
) -> Pipeline:
    """Construct a leakage-safe preprocessing and KNN classifier pipeline."""
    classifier = KNeighborsClassifier(
        n_neighbors=config.n_neighbors,
        weights=config.weights,
        algorithm=algorithm,
        p=config.p,
        metric="minkowski",
        n_jobs=n_jobs,
    )
    return Pipeline(
        [
            ("preprocessor", build_feature_preprocessor(config.scaling)),
            ("classifier", classifier),
        ]
    )


def align_probability_columns(
    probabilities: FloatArray,
    estimator_classes: IntArray,
) -> FloatArray:
    """Align estimator probability columns to required class order 0 through 4."""
    if probabilities.ndim != 2:
        raise ModelEvaluationError("Probability output must be two-dimensional")
    if probabilities.shape[1] != estimator_classes.size:
        raise ModelEvaluationError("Probability columns do not match estimator classes")
    if set(estimator_classes.tolist()) != set(CLASS_LABELS):
        raise ModelEvaluationError(
            f"Estimator classes {estimator_classes.tolist()} do not match "
            f"{list(CLASS_LABELS)}"
        )
    class_positions = {
        int(class_label): index
        for index, class_label in enumerate(estimator_classes.tolist())
    }
    aligned = probabilities[
        :, [class_positions[class_label] for class_label in CLASS_LABELS]
    ]
    validate_probabilities(aligned)
    return aligned


def validate_probabilities(probabilities: FloatArray) -> None:
    """Require finite five-class probability distributions summing to one."""
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_LABELS):
        raise ModelEvaluationError("Expected a probability matrix with five columns")
    if not np.isfinite(probabilities).all():
        raise ModelEvaluationError("Probability matrix contains non-finite values")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ModelEvaluationError("Probability matrix contains values outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ModelEvaluationError("Probability rows do not sum to one")


def calculate_metrics(target: IntArray, probabilities: FloatArray) -> MetricResults:
    """Calculate log loss, summed multiclass Brier score, and accuracy."""
    if target.shape != (probabilities.shape[0],):
        raise ModelEvaluationError("Targets and probabilities have different lengths")
    validate_probabilities(probabilities)
    one_hot = np.eye(len(CLASS_LABELS), dtype=np.float64)[target]
    brier_score = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    predicted_classes = np.asarray(CLASS_LABELS, dtype=np.int64)[
        np.argmax(probabilities, axis=1)
    ]
    return MetricResults(
        log_loss=float(log_loss(target, probabilities, labels=CLASS_LABELS)),
        brier_score=brier_score,
        accuracy=float(accuracy_score(target, predicted_classes)),
    )


def _fit_predict_probabilities(
    train_features: pl.DataFrame,
    train_target: IntArray,
    validation_features: pl.DataFrame,
    config: KNNConfig | None,
    algorithm: NeighborAlgorithm,
    n_jobs: int,
) -> FloatArray:
    if config is None:
        baseline = TrainingFrequencyClassifier().fit(train_target)
        return align_probability_columns(
            baseline.predict_proba(validation_features.height), baseline.classes_
        )
    if config.n_neighbors > train_target.size:
        raise ModelEvaluationError(
            f"n_neighbors={config.n_neighbors} exceeds the training row count "
            f"{train_target.size}"
        )
    pipeline = build_knn_pipeline(config, algorithm, n_jobs)
    pipeline.fit(train_features, train_target)
    probabilities = cast(FloatArray, pipeline.predict_proba(validation_features))
    classes = cast(IntArray, pipeline.classes_)
    return align_probability_columns(probabilities, classes)


def cross_validated_probabilities(
    dataset: ModelingDataset,
    config: KNNConfig | None,
    algorithm: NeighborAlgorithm = "brute",
    n_jobs: int = 1,
    folds: int = CV_FOLDS,
) -> FloatArray:
    """Generate one seeded, stratified out-of-fold prediction per observation."""
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    probabilities = np.empty((dataset.target.size, len(CLASS_LABELS)), dtype=np.float64)
    predicted = np.zeros(dataset.target.size, dtype=np.bool_)
    for train_indices, validation_indices in splitter.split(
        np.zeros(dataset.target.size), dataset.target
    ):
        probabilities[validation_indices] = _fit_predict_probabilities(
            dataset.features[train_indices],
            dataset.target[train_indices],
            dataset.features[validation_indices],
            config,
            algorithm,
            n_jobs,
        )
        predicted[validation_indices] = True
    if not predicted.all():
        raise ModelEvaluationError("Cross-validation did not predict every observation")
    validate_probabilities(probabilities)
    return probabilities


def chronological_probabilities(
    dataset: ModelingDataset,
    config: KNNConfig | None,
    algorithm: NeighborAlgorithm = "brute",
    n_jobs: int = 1,
) -> tuple[IntArray, FloatArray]:
    """Train on 2022–2023 and predict every 2024 observation."""
    training_mask = np.isin(dataset.years, CHRONOLOGICAL_TRAINING_YEARS)
    validation_mask = dataset.years == CHRONOLOGICAL_VALIDATION_YEAR
    if not training_mask.any() or not validation_mask.any():
        raise ModelEvaluationError("Chronological split lacks training or test rows")
    unexpected_years = set(dataset.years.tolist()) - set(TRAINING_YEARS)
    if unexpected_years:
        raise ModelEvaluationError(
            f"Chronological data contains unexpected years: {sorted(unexpected_years)}"
        )
    validation_target = dataset.target[validation_mask]
    probabilities = _fit_predict_probabilities(
        dataset.features.filter(pl.Series(training_mask)),
        dataset.target[training_mask],
        dataset.features.filter(pl.Series(validation_mask)),
        config,
        algorithm,
        n_jobs,
    )
    return validation_target, probabilities


def evaluate_candidate(
    dataset: ModelingDataset,
    config: KNNConfig | None,
    algorithm: NeighborAlgorithm = "brute",
    n_jobs: int = 1,
    folds: int = CV_FOLDS,
) -> EvaluationResults:
    """Evaluate one baseline or KNN candidate under both validation designs."""
    cv_probabilities = cross_validated_probabilities(
        dataset, config, algorithm, n_jobs, folds
    )
    chronological_target, chronological_prediction = chronological_probabilities(
        dataset, config, algorithm, n_jobs
    )
    return EvaluationResults(
        cross_validation=calculate_metrics(dataset.target, cv_probabilities),
        chronological=calculate_metrics(chronological_target, chronological_prediction),
    )


def make_result_row(
    config: KNNConfig | None,
    results: EvaluationResults,
    algorithm: NeighborAlgorithm,
) -> ResultRow:
    """Combine a candidate definition and its metrics into one result row."""
    if config is None:
        model_type: ModelType = "baseline"
        scaling = None
        n_neighbors = None
        weights = None
        distance_power = None
        recorded_algorithm = None
    else:
        model_type = "knn"
        scaling = config.scaling
        n_neighbors = config.n_neighbors
        weights = config.weights
        distance_power = config.p
        recorded_algorithm = algorithm
    return ResultRow(
        model_type=model_type,
        feature_scaling=scaling,
        n_neighbors=n_neighbors,
        weights=weights,
        p=distance_power,
        neighbor_algorithm=recorded_algorithm,
        cv_log_loss=results.cross_validation.log_loss,
        cv_brier_score=results.cross_validation.brier_score,
        cv_accuracy=results.cross_validation.accuracy,
        chronological_log_loss=results.chronological.log_loss,
        chronological_brier_score=results.chronological.brier_score,
        chronological_accuracy=results.chronological.accuracy,
    )


def benchmark_neighbor_algorithms(
    features: pl.DataFrame,
    target: IntArray,
    *,
    train_size: int = 5_000,
    query_size: int = 500,
    n_neighbors: int = 100,
    n_jobs: int = 1,
) -> tuple[NeighborAlgorithm, tuple[BenchmarkResult, ...]]:
    """Choose the fastest algorithm equivalent to brute force on encoded data."""
    required_rows = train_size + query_size
    if features.height < required_rows or target.shape[0] < required_rows:
        raise ValueError(f"Benchmark requires at least {required_rows:,} rows")
    generator = np.random.default_rng(RANDOM_SEED)
    indices = generator.choice(features.height, size=required_rows, replace=False)
    encoded = cast(
        FloatArray,
        build_feature_preprocessor("none").fit_transform(features[indices]),
    )
    encoded += generator.normal(0.0, 1e-10, encoded.shape)
    sampled_target = target[indices]
    train_features = encoded[:train_size]
    train_target = sampled_target[:train_size]
    query_features = encoded[train_size:]
    neighbor_count = min(n_neighbors, train_size)

    reference: FloatArray | None = None
    results: list[BenchmarkResult] = []
    for algorithm in SUPPORTED_ALGORITHMS:
        classifier = KNeighborsClassifier(
            n_neighbors=neighbor_count,
            algorithm=algorithm,
            metric="minkowski",
            p=2,
            weights="uniform",
            n_jobs=n_jobs,
        )
        start = time.perf_counter()
        classifier.fit(train_features, train_target)
        probabilities = cast(FloatArray, classifier.predict_proba(query_features))
        elapsed = time.perf_counter() - start
        equivalent = reference is None or bool(
            np.allclose(probabilities, reference, rtol=0.0, atol=1e-12)
        )
        if reference is None:
            reference = probabilities
        results.append(BenchmarkResult(algorithm, elapsed, equivalent))
    equivalent_results = [result for result in results if result.equivalent_to_brute]
    if not equivalent_results:
        raise ModelEvaluationError("No neighbor-search algorithm matched brute force")
    selected = min(equivalent_results, key=lambda result: result.elapsed_seconds)
    return selected.algorithm, tuple(results)


def _empty_result_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=RESULT_SCHEMA)


def _candidate_key_from_values(
    model_type: str,
    scaling: str | None,
    n_neighbors: int | None,
    weights: str | None,
    distance_power: int | None,
) -> tuple[str, str | None, int | None, str | None, int | None]:
    return model_type, scaling, n_neighbors, weights, distance_power


def _candidate_key(
    config: KNNConfig | None,
) -> tuple[str, str | None, int | None, str | None, int | None]:
    if config is None:
        return _candidate_key_from_values("baseline", None, None, None, None)
    return _candidate_key_from_values(
        "knn", config.scaling, config.n_neighbors, config.weights, config.p
    )


def validate_result_frame(frame: pl.DataFrame) -> None:
    """Validate a complete or partial checkpoint result table."""
    if frame.columns != list(RESULT_COLUMNS):
        raise ModelEvaluationError(
            f"Result columns {frame.columns} do not match {list(RESULT_COLUMNS)}"
        )
    unexpected_types = {
        column: (frame.schema[column], expected_type)
        for column, expected_type in RESULT_SCHEMA.items()
        if frame.schema[column] != expected_type
    }
    if unexpected_types:
        raise ModelEvaluationError(
            f"Result table has unexpected types: {unexpected_types}"
        )

    allowed_knn_keys = {_candidate_key(config) for config in KNN_CONFIGS}
    observed_keys: set[tuple[str, str | None, int | None, str | None, int | None]] = (
        set()
    )
    for row in frame.iter_rows(named=True):
        model_type = cast(str, row["model_type"])
        scaling = cast(str | None, row["feature_scaling"])
        n_neighbors = cast(int | None, row["n_neighbors"])
        weights = cast(str | None, row["weights"])
        distance_power = cast(int | None, row["p"])
        algorithm = cast(str | None, row["neighbor_algorithm"])
        key = _candidate_key_from_values(
            model_type, scaling, n_neighbors, weights, distance_power
        )
        if key in observed_keys:
            raise ModelEvaluationError(
                f"Result table contains duplicate candidate {key}"
            )
        observed_keys.add(key)
        if model_type == "baseline":
            if key != _candidate_key(None) or algorithm is not None:
                raise ModelEvaluationError("Baseline row contains KNN settings")
        elif model_type == "knn":
            if key not in allowed_knn_keys:
                raise ModelEvaluationError(f"Result table contains unknown KNN {key}")
            if algorithm not in SUPPORTED_ALGORITHMS:
                raise ModelEvaluationError(
                    f"Result row contains unsupported algorithm {algorithm}"
                )
        else:
            raise ModelEvaluationError(f"Unknown result model type: {model_type}")

        for column in METRIC_COLUMNS:
            value = cast(float, row[column])
            if not math.isfinite(value):
                raise ModelEvaluationError(f"Result {column} is non-finite")
            if column.endswith("accuracy") and not 0.0 <= value <= 1.0:
                raise ModelEvaluationError(f"Result {column} is outside [0, 1]")
            if not column.endswith("accuracy") and value < 0.0:
                raise ModelEvaluationError(f"Result {column} is negative")
            if "brier_score" in column and value > 2.0:
                raise ModelEvaluationError(f"Result {column} exceeds 2")

    algorithms = set(
        cast(list[str], frame["neighbor_algorithm"].drop_nulls().to_list())
    )
    if len(algorithms) > 1:
        raise ModelEvaluationError("Result table mixes neighbor-search algorithms")


def read_result_checkpoint(path: Path) -> pl.DataFrame:
    """Read and validate an existing partial or complete result table."""
    try:
        frame = pl.read_csv(path, schema_overrides=RESULT_SCHEMA)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ModelEvaluationError(
            f"Could not read result checkpoint {path}: {error}"
        ) from error
    validate_result_frame(frame)
    return frame


def write_result_checkpoint(frame: pl.DataFrame, path: Path) -> None:
    """Atomically validate and replace a result checkpoint."""
    validate_result_frame(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.part")
    try:
        frame.write_csv(temporary_path)
        written = read_result_checkpoint(temporary_path)
        if not frame.equals(written, null_equal=True):
            raise ModelEvaluationError(
                f"Written result checkpoint {temporary_path} did not round-trip"
            )
        temporary_path.replace(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ModelEvaluationError(
            f"Could not write result checkpoint {path}: {error}"
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _completed_keys(
    frame: pl.DataFrame,
) -> set[tuple[str, str | None, int | None, str | None, int | None]]:
    return {
        _candidate_key_from_values(
            cast(str, row["model_type"]),
            cast(str | None, row["feature_scaling"]),
            cast(int | None, row["n_neighbors"]),
            cast(str | None, row["weights"]),
            cast(int | None, row["p"]),
        )
        for row in frame.iter_rows(named=True)
    }


def _recorded_algorithm(frame: pl.DataFrame) -> NeighborAlgorithm | None:
    algorithms = cast(
        list[str], frame["neighbor_algorithm"].drop_nulls().unique().to_list()
    )
    if not algorithms:
        return None
    if len(algorithms) != 1 or algorithms[0] not in SUPPORTED_ALGORITHMS:
        raise ModelEvaluationError("Checkpoint has inconsistent neighbor algorithms")
    return algorithms[0]


def run_experiment(
    dataset: ModelingDataset,
    *,
    configs: Sequence[KNNConfig] = KNN_CONFIGS,
    output_path: Path = MODEL_SELECTION_RESULTS_PATH,
    overwrite: bool = False,
    algorithm: NeighborAlgorithm | None = None,
    n_jobs: int = 1,
    folds: int = CV_FOLDS,
    benchmark_train_size: int = 5_000,
    benchmark_query_size: int = 500,
) -> pl.DataFrame:
    """Run missing candidates and atomically checkpoint every completed row."""
    results = (
        read_result_checkpoint(output_path)
        if output_path.exists() and not overwrite
        else _empty_result_frame()
    )
    completed = _completed_keys(results)
    candidates: list[KNNConfig | None] = [None, *configs]
    missing = [
        candidate
        for candidate in candidates
        if _candidate_key(candidate) not in completed
    ]

    checkpoint_algorithm = _recorded_algorithm(results)
    if checkpoint_algorithm is not None:
        if algorithm is not None and algorithm != checkpoint_algorithm:
            raise ModelEvaluationError(
                f"Checkpoint uses {checkpoint_algorithm}, not requested {algorithm}"
            )
        algorithm = checkpoint_algorithm
    if any(candidate is not None for candidate in missing) and algorithm is None:
        algorithm, benchmark_results = benchmark_neighbor_algorithms(
            dataset.features,
            dataset.target,
            train_size=benchmark_train_size,
            query_size=benchmark_query_size,
            n_jobs=n_jobs,
        )
        for benchmark in benchmark_results:
            print(
                f"Benchmark {benchmark.algorithm}: "
                f"{benchmark.elapsed_seconds:.3f}s, "
                f"equivalent={benchmark.equivalent_to_brute}"
            )
        print(f"Selected neighbor algorithm: {algorithm}")
    selected_algorithm: NeighborAlgorithm = algorithm or "brute"

    for index, candidate in enumerate(missing, start=1):
        label = "baseline" if candidate is None else str(candidate)
        print(f"Evaluating {index}/{len(missing)}: {label}")
        evaluation = evaluate_candidate(
            dataset, candidate, selected_algorithm, n_jobs, folds
        )
        row = make_result_row(candidate, evaluation, selected_algorithm)
        results = pl.concat(
            [results, pl.DataFrame([row.as_record()], schema=RESULT_SCHEMA)],
            how="vertical",
        )
        write_result_checkpoint(results, output_path)
    validate_result_frame(results)
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse Stage 3 model-selection command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=PROCESSED_DATA_DIRECTORY)
    parser.add_argument("--output", type=Path, default=MODEL_SELECTION_RESULTS_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--algorithm", choices=SUPPORTED_ALGORITHMS)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--folds", type=int, default=CV_FOLDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Stage 3 model-selection experiment."""
    args = parse_args(argv)
    dataset = load_modeling_data(args.data_directory)
    results = run_experiment(
        dataset,
        output_path=args.output,
        overwrite=args.overwrite,
        algorithm=args.algorithm,
        n_jobs=args.n_jobs,
        folds=args.folds,
    )
    print(f"Validated {results.height} model-selection result row(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
