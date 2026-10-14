from __future__ import annotations

import json
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from experiment_config import (
    CLASS_LABELS,
    RANDOM_SEED,
    SELECTED_MODEL_CONFIG,
    SELECTED_NEIGHBOR_ALGORITHM,
    KNNConfig,
)
from model_evaluation import (
    RESULT_SCHEMA,
    EvaluationResults,
    MetricResults,
    ModelingDataset,
    make_result_row,
    write_result_checkpoint,
)
from train_final_model import (
    FinalModelError,
    ModelArtifactExistsError,
    build_metadata,
    fixed_validation_sample,
    parse_args,
    read_metadata,
    selected_validation_metrics,
    sha256_file,
    train_and_save_model,
    validate_fitted_pipeline,
    validate_metadata,
)

VALIDATION_METRICS = {
    "cv_log_loss": 0.57,
    "cv_brier_score": 0.31,
    "cv_accuracy": 0.77,
    "chronological_log_loss": 0.56,
    "chronological_brier_score": 0.30,
    "chronological_accuracy": 0.78,
}


def synthetic_dataset(rows_per_class: int = 3) -> ModelingDataset:
    target = np.repeat(np.asarray(CLASS_LABELS, dtype=np.int64), rows_per_class)
    features = np.column_stack(
        (
            75.0 + target * 5.0 + np.arange(target.size) * 0.01,
            -10.0 + target * 8.0 + np.arange(target.size) * 0.02,
        )
    ).astype(np.float64)
    years = np.resize(np.asarray([2022, 2023, 2024], dtype=np.int64), target.size)
    return ModelingDataset(features, target, years)


def make_input_paths(directory: Path) -> list[Path]:
    paths: list[Path] = []
    for year in (2022, 2023, 2024):
        path = directory / f"{year}.csv"
        path.write_text(f"training fixture {year}\n", encoding="utf-8")
        paths.append(path)
    return paths


def test_selected_model_matches_user_choice() -> None:
    assert SELECTED_MODEL_CONFIG == KNNConfig("none", 1000, "uniform", 1)
    assert SELECTED_NEIGHBOR_ALGORITHM == "ball_tree"
    assert RANDOM_SEED == 99


def test_selected_validation_metrics_reads_unique_result(tmp_path: Path) -> None:
    results = EvaluationResults(
        MetricResults(0.57, 0.31, 0.77),
        MetricResults(0.56, 0.30, 0.78),
    )
    row = make_result_row(SELECTED_MODEL_CONFIG, results, "ball_tree")
    frame = pl.DataFrame([row.as_record()], schema=RESULT_SCHEMA)
    path = tmp_path / "results.csv"
    write_result_checkpoint(frame, path)

    metrics = selected_validation_metrics(path)

    assert metrics == VALIDATION_METRICS


def test_selected_validation_metrics_requires_selected_row(tmp_path: Path) -> None:
    results = EvaluationResults(
        MetricResults(0.6, 0.4, 0.7),
        MetricResults(0.6, 0.4, 0.7),
    )
    row = make_result_row(KNNConfig("none", 500, "uniform", 1), results, "ball_tree")
    frame = pl.DataFrame([row.as_record()], schema=RESULT_SCHEMA)
    path = tmp_path / "results.csv"
    write_result_checkpoint(frame, path)

    with pytest.raises(FinalModelError, match="Expected one model-selection row"):
        selected_validation_metrics(path)


def test_fixed_validation_sample_is_deterministic_and_bounded() -> None:
    features = np.arange(40, dtype=np.float64).reshape(20, 2)

    first = fixed_validation_sample(features, sample_size=5)
    second = fixed_validation_sample(features, sample_size=5)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[[0, -1]], features[[0, -1]])


def test_build_and_validate_metadata(tmp_path: Path) -> None:
    paths = make_input_paths(tmp_path)
    hashes = {str(path): sha256_file(path) for path in paths}
    config = KNNConfig("none", 5, "uniform", 1)
    metadata = build_metadata(
        config=config,
        algorithm="brute",
        n_jobs=1,
        training_rows=15,
        validation_metrics=VALIDATION_METRICS,
        input_hashes=hashes,
        model_sha256="a" * 64,
    )

    validate_metadata(
        metadata,
        config=config,
        algorithm="brute",
        n_jobs=1,
        training_rows=15,
        model_sha256="a" * 64,
    )
    assert metadata["feature_columns"] == ["launch_speed", "launch_angle"]
    assert metadata["class_order"] == list(CLASS_LABELS)


def test_validate_metadata_rejects_wrong_class_order(tmp_path: Path) -> None:
    paths = make_input_paths(tmp_path)
    config = KNNConfig("none", 5, "uniform", 1)
    metadata = build_metadata(
        config=config,
        algorithm="brute",
        n_jobs=1,
        training_rows=15,
        validation_metrics=VALIDATION_METRICS,
        input_hashes={str(path): sha256_file(path) for path in paths},
        model_sha256="a" * 64,
    )
    metadata["class_order"] = [4, 3, 2, 1, 0]

    with pytest.raises(FinalModelError, match="class_order"):
        validate_metadata(
            metadata,
            config=config,
            algorithm="brute",
            n_jobs=1,
            training_rows=15,
            model_sha256="a" * 64,
        )


def test_train_save_reload_and_validate_predictions(tmp_path: Path) -> None:
    dataset = synthetic_dataset()
    paths = make_input_paths(tmp_path)
    model_path = tmp_path / "artifacts" / "model.joblib"
    metadata_path = tmp_path / "artifacts" / "metadata.json"
    config = KNNConfig("none", 5, "uniform", 1)

    summary = train_and_save_model(
        dataset,
        validation_metrics=VALIDATION_METRICS,
        input_paths=paths,
        model_path=model_path,
        metadata_path=metadata_path,
        config=config,
        algorithm="brute",
        n_jobs=1,
    )

    assert summary.training_rows == dataset.target.size
    assert summary.elapsed_seconds >= 0.0
    assert summary.model_sha256 == sha256_file(model_path)
    assert model_path.is_file()
    assert metadata_path.is_file()
    assert not model_path.with_suffix(".joblib.part").exists()
    assert not metadata_path.with_suffix(".json.part").exists()

    reloaded = joblib.load(model_path)
    assert isinstance(reloaded, Pipeline)
    validate_fitted_pipeline(reloaded, config, "brute", dataset.target.size)
    metadata = read_metadata(metadata_path)
    validate_metadata(
        metadata,
        config=config,
        algorithm="brute",
        n_jobs=1,
        training_rows=dataset.target.size,
        model_sha256=summary.model_sha256,
    )


def test_train_requires_overwrite_for_existing_artifact(tmp_path: Path) -> None:
    dataset = synthetic_dataset()
    paths = make_input_paths(tmp_path)
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    model_path.write_bytes(b"existing")

    with pytest.raises(ModelArtifactExistsError, match="replacement was not requested"):
        train_and_save_model(
            dataset,
            validation_metrics=VALIDATION_METRICS,
            input_paths=paths,
            model_path=model_path,
            metadata_path=metadata_path,
            config=KNNConfig("none", 5, "uniform", 1),
            algorithm="brute",
            n_jobs=1,
        )


def test_read_metadata_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(FinalModelError, match="not a JSON object"):
        read_metadata(path)


def test_parse_args_supports_artifact_controls() -> None:
    args = parse_args(
        [
            "--input-directory",
            "inputs",
            "--selection-results",
            "selection.csv",
            "--model-output",
            "model.joblib",
            "--metadata-output",
            "metadata.json",
            "--n-jobs",
            "2",
            "--overwrite",
        ]
    )

    assert args.input_directory == Path("inputs")
    assert args.selection_results == Path("selection.csv")
    assert args.model_output == Path("model.joblib")
    assert args.metadata_output == Path("metadata.json")
    assert args.n_jobs == 2
    assert args.overwrite is True
