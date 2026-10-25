from __future__ import annotations

import json
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from sd_experiment_config import (
    CLASS_LABELS,
    DIRECTION_CATEGORIES,
    RANDOM_SEED,
    SELECTED_MODEL_CONFIG,
    SELECTED_NEIGHBOR_ALGORITHM,
    TRANSFORMED_FEATURE_COLUMNS,
    KNNConfig,
)
from sd_model_evaluation import (
    RESULT_SCHEMA,
    EvaluationResults,
    MetricResults,
    ModelingDataset,
    make_result_row,
    write_result_checkpoint,
)
from train_sd_model import (
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
    "cv_log_loss": 0.5378433518688378,
    "cv_brier_score": 0.29869844698388,
    "cv_accuracy": 0.7886757383949147,
    "chronological_log_loss": 0.5391981853442623,
    "chronological_brier_score": 0.2985086345061233,
    "chronological_accuracy": 0.7889788541185434,
}


def synthetic_dataset(rows_per_class: int = 3) -> ModelingDataset:
    target = np.repeat(np.asarray(CLASS_LABELS, dtype=np.int64), rows_per_class)
    row_indices = np.arange(target.size)
    directions = np.asarray(DIRECTION_CATEGORIES, dtype=object)
    features = pl.DataFrame(
        {
            "launch_speed": 75.0 + target * 5.0 + row_indices * 0.01,
            "launch_angle": -10.0 + target * 8.0 + row_indices * 0.02,
            "hit_direction": directions[row_indices % len(directions)].tolist(),
        },
        schema={
            "launch_speed": pl.Float64,
            "launch_angle": pl.Float64,
            "hit_direction": pl.String,
        },
    )
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
    assert SELECTED_MODEL_CONFIG == KNNConfig("robust", 1000, "uniform", 2)
    assert SELECTED_NEIGHBOR_ALGORITHM == "ball_tree"
    assert RANDOM_SEED == 99


def test_selected_validation_metrics_reads_unique_result(tmp_path: Path) -> None:
    results = EvaluationResults(
        MetricResults(
            VALIDATION_METRICS["cv_log_loss"],
            VALIDATION_METRICS["cv_brier_score"],
            VALIDATION_METRICS["cv_accuracy"],
        ),
        MetricResults(
            VALIDATION_METRICS["chronological_log_loss"],
            VALIDATION_METRICS["chronological_brier_score"],
            VALIDATION_METRICS["chronological_accuracy"],
        ),
    )
    row = make_result_row(SELECTED_MODEL_CONFIG, results, "ball_tree")
    frame = pl.DataFrame([row.as_record()], schema=RESULT_SCHEMA)
    path = tmp_path / "results.csv"
    write_result_checkpoint(frame, path)

    assert selected_validation_metrics(path) == VALIDATION_METRICS


def test_selected_validation_metrics_requires_selected_row(tmp_path: Path) -> None:
    results = EvaluationResults(
        MetricResults(0.6, 0.4, 0.7), MetricResults(0.6, 0.4, 0.7)
    )
    row = make_result_row(KNNConfig("robust", 500, "uniform", 2), results, "ball_tree")
    path = tmp_path / "results.csv"
    write_result_checkpoint(pl.DataFrame([row.as_record()], schema=RESULT_SCHEMA), path)

    with pytest.raises(FinalModelError, match="Expected one model-selection row"):
        selected_validation_metrics(path)


def test_fixed_validation_sample_is_deterministic_and_bounded() -> None:
    features = synthetic_dataset().features

    first = fixed_validation_sample(features, sample_size=5)
    second = fixed_validation_sample(features, sample_size=5)

    assert first.equals(second)
    assert first.row(0) == features.row(0)
    assert first.row(-1) == features.row(-1)


def test_build_and_validate_metadata(tmp_path: Path) -> None:
    paths = make_input_paths(tmp_path)
    config = KNNConfig("robust", 5, "uniform", 2)
    metadata = build_metadata(
        config=config,
        algorithm="brute",
        n_jobs=1,
        training_rows=15,
        validation_metrics=VALIDATION_METRICS,
        input_hashes={str(path): sha256_file(path) for path in paths},
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
    assert metadata["direction_categories"] == list(DIRECTION_CATEGORIES)
    assert metadata["transformed_feature_columns"] == list(TRANSFORMED_FEATURE_COLUMNS)
    assert metadata["class_order"] == list(CLASS_LABELS)


def test_validate_metadata_rejects_wrong_direction_order(tmp_path: Path) -> None:
    paths = make_input_paths(tmp_path)
    config = KNNConfig("robust", 5, "uniform", 2)
    metadata = build_metadata(
        config=config,
        algorithm="brute",
        n_jobs=1,
        training_rows=15,
        validation_metrics=VALIDATION_METRICS,
        input_hashes={str(path): sha256_file(path) for path in paths},
        model_sha256="a" * 64,
    )
    metadata["direction_categories"] = ["opposite", "straightaway", "pull"]

    with pytest.raises(FinalModelError, match="direction_categories"):
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
    config = KNNConfig("robust", 5, "uniform", 2)

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
    validate_metadata(
        read_metadata(metadata_path),
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
            config=KNNConfig("robust", 5, "uniform", 2),
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
