from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.preprocessing import RobustScaler, StandardScaler

import model_evaluation
from experiment_config import (
    CLASS_LABELS,
    RANDOM_SEED,
    KNNConfig,
    ModelingDataError,
)
from model_evaluation import (
    RESULT_COLUMNS,
    BenchmarkResult,
    EvaluationResults,
    MetricResults,
    ModelEvaluationError,
    ModelingDataset,
    ResultRow,
    TrainingFrequencyClassifier,
    align_probability_columns,
    benchmark_neighbor_algorithms,
    build_knn_pipeline,
    calculate_metrics,
    chronological_probabilities,
    cross_validated_probabilities,
    evaluate_candidate,
    load_modeling_data,
    make_result_row,
    parse_args,
    read_result_checkpoint,
    run_experiment,
    validate_probabilities,
    validate_result_frame,
    write_result_checkpoint,
)


def synthetic_dataset(rows_per_class_year: int = 5) -> ModelingDataset:
    features: list[list[float]] = []
    target: list[int] = []
    years: list[int] = []
    for year in (2022, 2023, 2024):
        for class_label in CLASS_LABELS:
            for repeat in range(rows_per_class_year):
                features.append(
                    [
                        70.0 + class_label * 7.0 + repeat * 0.1,
                        -15.0 + class_label * 10.0 + repeat * 0.2,
                    ]
                )
                target.append(class_label)
                years.append(year)
    return ModelingDataset(
        np.asarray(features, dtype=np.float64),
        np.asarray(target, dtype=np.int64),
        np.asarray(years, dtype=np.int64),
    )


def sample_evaluation() -> EvaluationResults:
    return EvaluationResults(
        cross_validation=MetricResults(0.9, 0.5, 0.7),
        chronological=MetricResults(1.0, 0.6, 0.65),
    )


def sample_result_row(config: KNNConfig | None = None) -> ResultRow:
    return make_result_row(config, sample_evaluation(), "brute")


def write_modeling_inputs(directory: Path) -> None:
    for year in (2022, 2023, 2024):
        pl.DataFrame(
            {
                "game_date": [f"{year}-04-{day:02d}" for day in range(1, 6)],
                "launch_speed": [80.0, 85.0, 90.0, 95.0, 100.0],
                "launch_angle": [-10.0, 0.0, 10.0, 20.0, 30.0],
                "total_bases": list(CLASS_LABELS),
                "unused": ["ignored"] * 5,
            },
            schema={
                "game_date": pl.String,
                "launch_speed": pl.Float64,
                "launch_angle": pl.Float64,
                "total_bases": pl.Int64,
                "unused": pl.String,
            },
        ).write_csv(directory / f"{year}_all_BBE.csv")


def test_load_modeling_data_validates_and_concatenates_years(tmp_path: Path) -> None:
    write_modeling_inputs(tmp_path)

    dataset = load_modeling_data(tmp_path)

    assert dataset.features.shape == (15, 2)
    assert dataset.target.tolist() == list(CLASS_LABELS) * 3
    assert dataset.years.tolist() == [2022] * 5 + [2023] * 5 + [2024] * 5
    assert dataset.features.dtype == np.float64
    assert dataset.target.dtype == np.int64


def test_load_modeling_data_wraps_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ModelEvaluationError, match="Could not read"):
        load_modeling_data(tmp_path)


def test_load_modeling_data_applies_input_contract(tmp_path: Path) -> None:
    write_modeling_inputs(tmp_path)
    path = tmp_path / "2023_all_BBE.csv"
    frame = pl.read_csv(path).with_columns(pl.lit(9).alias("total_bases"))
    frame.write_csv(path)

    with pytest.raises(ModelingDataError, match="invalid target classes"):
        load_modeling_data(tmp_path)


def test_modeling_dataset_rejects_misaligned_shapes() -> None:
    with pytest.raises(ValueError, match="Feature shape"):
        ModelingDataset(
            np.ones((2, 3), dtype=np.float64),
            np.ones(2, dtype=np.int64),
            np.ones(2, dtype=np.int64),
        )


def test_training_frequency_classifier_returns_training_distribution() -> None:
    target = np.asarray([0, 0, 0, 1, 1, 2, 3, 4], dtype=np.int64)

    classifier = TrainingFrequencyClassifier().fit(target)
    probabilities = classifier.predict_proba(2)

    assert classifier.classes_.tolist() == list(CLASS_LABELS)
    np.testing.assert_allclose(
        probabilities,
        np.asarray([[3 / 8, 2 / 8, 1 / 8, 1 / 8, 1 / 8]] * 2),
    )


@pytest.mark.parametrize(
    ("scaling", "expected_scaler_name"),
    [
        ("none", "passthrough"),
        ("standard", "standard"),
        ("robust", "robust"),
    ],
)
def test_build_knn_pipeline_uses_requested_scaler(
    scaling: str,
    expected_scaler_name: str,
) -> None:
    config = KNNConfig(scaling, 5, "uniform", 2)  # type: ignore[arg-type]

    pipeline = build_knn_pipeline(config, "brute", 1)

    scaler = pipeline.named_steps["scaler"]
    if expected_scaler_name == "passthrough":
        assert scaler == "passthrough"
    elif expected_scaler_name == "standard":
        assert isinstance(scaler, StandardScaler)
    else:
        assert isinstance(scaler, RobustScaler)
    classifier = pipeline.named_steps["classifier"]
    assert classifier.n_neighbors == 5
    assert classifier.weights == "uniform"
    assert classifier.p == 2
    assert classifier.n_jobs == 1


def test_scaler_is_fitted_only_on_training_rows() -> None:
    config = KNNConfig("standard", 1, "uniform", 2)
    pipeline = build_knn_pipeline(config, "brute", 1)
    training_features = np.asarray([[0.0, 0.0], [2.0, 4.0]], dtype=np.float64)
    training_target = np.asarray([0, 1], dtype=np.int64)
    pipeline.fit(training_features, training_target)

    scaler = pipeline.named_steps["scaler"]
    assert isinstance(scaler, StandardScaler)
    assert scaler.mean_ is not None
    np.testing.assert_allclose(scaler.mean_, np.asarray([1.0, 2.0], dtype=np.float64))


def test_align_probability_columns_reorders_explicit_classes() -> None:
    probabilities = np.asarray([[0.4, 0.3, 0.2, 0.09, 0.01]], dtype=np.float64)
    classes = np.asarray([4, 3, 2, 1, 0], dtype=np.int64)

    aligned = align_probability_columns(probabilities, classes)

    np.testing.assert_allclose(aligned, [[0.01, 0.09, 0.2, 0.3, 0.4]])


@pytest.mark.parametrize(
    "probabilities",
    [
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        np.asarray([[np.nan, 0.25, 0.25, 0.25, 0.25]], dtype=np.float64),
        np.asarray([[1.1, 0.0, 0.0, 0.0, -0.1]], dtype=np.float64),
        np.asarray([[0.1, 0.1, 0.1, 0.1, 0.1]], dtype=np.float64),
    ],
)
def test_validate_probabilities_rejects_invalid_matrices(
    probabilities: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> None:
    with pytest.raises(ModelEvaluationError):
        validate_probabilities(probabilities)


def test_calculate_metrics_is_zero_for_perfect_probabilities() -> None:
    target = np.asarray(CLASS_LABELS, dtype=np.int64)
    probabilities = np.eye(len(CLASS_LABELS), dtype=np.float64)

    metrics = calculate_metrics(target, probabilities)

    assert metrics.log_loss == pytest.approx(0.0, abs=1e-12)
    assert metrics.brier_score == 0.0
    assert metrics.accuracy == 1.0


def test_seeded_cross_validation_is_reproducible() -> None:
    dataset = synthetic_dataset()
    config = KNNConfig("standard", 3, "distance", 2)

    first = cross_validated_probabilities(dataset, config, "brute", 1)
    second = cross_validated_probabilities(dataset, config, "brute", 1)

    np.testing.assert_array_equal(first, second)
    assert RANDOM_SEED == 99


def test_cross_validation_baseline_learns_each_training_fold() -> None:
    dataset = synthetic_dataset()

    probabilities = cross_validated_probabilities(dataset, None, folds=5)

    np.testing.assert_allclose(probabilities, np.full((75, 5), 0.2))


def test_chronological_validation_uses_2024_as_test_data() -> None:
    dataset = synthetic_dataset()

    target, probabilities = chronological_probabilities(dataset, None)

    assert target.tolist() == [
        class_label for class_label in CLASS_LABELS for _ in range(5)
    ]
    np.testing.assert_allclose(probabilities, np.full((25, 5), 0.2))


def test_evaluate_candidate_returns_both_validation_designs() -> None:
    results = evaluate_candidate(synthetic_dataset(), None)

    assert results.cross_validation.log_loss == pytest.approx(np.log(5.0))
    assert results.chronological.log_loss == pytest.approx(np.log(5.0))
    assert results.cross_validation.brier_score == pytest.approx(0.8)
    assert results.chronological.accuracy == pytest.approx(0.2)


def test_benchmark_checks_all_algorithms_for_equivalence() -> None:
    generator = np.random.default_rng(RANDOM_SEED)
    features = generator.normal(size=(120, 2))
    target = np.tile(np.asarray(CLASS_LABELS, dtype=np.int64), 24)

    selected, results = benchmark_neighbor_algorithms(
        features,
        target,
        train_size=100,
        query_size=20,
        n_neighbors=10,
    )

    assert selected in {"brute", "kd_tree", "ball_tree"}
    assert [result.algorithm for result in results] == [
        "brute",
        "kd_tree",
        "ball_tree",
    ]
    assert all(result.equivalent_to_brute for result in results)
    assert all(result.elapsed_seconds >= 0.0 for result in results)


def test_make_result_row_distinguishes_baseline_and_knn() -> None:
    baseline = sample_result_row()
    config = KNNConfig("robust", 250, "distance", 1)
    knn = sample_result_row(config)

    assert baseline.model_type == "baseline"
    assert baseline.feature_scaling is None
    assert baseline.neighbor_algorithm is None
    assert knn.model_type == "knn"
    assert knn.feature_scaling == "robust"
    assert knn.n_neighbors == 250
    assert knn.neighbor_algorithm == "brute"


def test_result_checkpoint_round_trips_atomically(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    frame = pl.DataFrame(
        [sample_result_row().as_record()], schema=model_evaluation.RESULT_SCHEMA
    ).select(RESULT_COLUMNS)

    write_result_checkpoint(frame, path)
    result = read_result_checkpoint(path)

    assert result.equals(frame, null_equal=True)
    assert not path.with_suffix(".csv.part").exists()


def test_result_validation_rejects_duplicate_candidate() -> None:
    record = sample_result_row().as_record()
    frame = pl.DataFrame([record, record], schema=model_evaluation.RESULT_SCHEMA)

    with pytest.raises(ModelEvaluationError, match="duplicate candidate"):
        validate_result_frame(frame)


def test_result_validation_rejects_impossible_brier_score() -> None:
    record = sample_result_row().as_record()
    record["cv_brier_score"] = 2.1
    frame = pl.DataFrame([record], schema=model_evaluation.RESULT_SCHEMA)

    with pytest.raises(ModelEvaluationError, match="exceeds 2"):
        validate_result_frame(frame)


def test_run_experiment_checkpoints_and_resumes_without_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "results.csv"
    dataset = synthetic_dataset()
    config = KNNConfig("none", 5, "uniform", 2)

    first = run_experiment(
        dataset,
        configs=[config],
        output_path=path,
        algorithm="brute",
    )
    assert first.height == 2

    def unexpected_evaluation(*args: object, **kwargs: object) -> EvaluationResults:
        raise AssertionError("A completed candidate was recomputed")

    monkeypatch.setattr(model_evaluation, "evaluate_candidate", unexpected_evaluation)
    resumed = run_experiment(
        dataset,
        configs=[config],
        output_path=path,
        algorithm="brute",
    )

    assert resumed.equals(first, null_equal=True)


def test_run_experiment_overwrite_replaces_old_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    old_config = KNNConfig("none", 5, "uniform", 2)
    new_config = KNNConfig("standard", 5, "uniform", 2)
    dataset = synthetic_dataset()
    run_experiment(
        dataset,
        configs=[old_config],
        output_path=path,
        algorithm="brute",
    )

    result = run_experiment(
        dataset,
        configs=[new_config],
        output_path=path,
        overwrite=True,
        algorithm="brute",
    )

    assert result.height == 2
    assert result.get_column("feature_scaling").to_list() == [None, "standard"]


def test_resume_requires_checkpoint_neighbor_algorithm(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    config = KNNConfig("none", 5, "uniform", 2)
    dataset = synthetic_dataset()
    run_experiment(
        dataset,
        configs=[config],
        output_path=path,
        algorithm="brute",
    )

    with pytest.raises(ModelEvaluationError, match="Checkpoint uses brute"):
        run_experiment(
            dataset,
            configs=[config, KNNConfig("standard", 5, "uniform", 2)],
            output_path=path,
            algorithm="kd_tree",
        )


def test_parse_args_supports_safe_operational_controls() -> None:
    args = parse_args(
        [
            "--input-directory",
            "inputs",
            "--output",
            "results.csv",
            "--algorithm",
            "kd_tree",
            "--n-jobs",
            "2",
            "--max-knn-candidates",
            "4",
            "--overwrite",
        ]
    )

    assert args.input_directory == Path("inputs")
    assert args.output == Path("results.csv")
    assert args.algorithm == "kd_tree"
    assert args.n_jobs == 2
    assert args.max_knn_candidates == 4
    assert args.overwrite is True


def test_benchmark_result_is_typed_record() -> None:
    result = BenchmarkResult("kd_tree", 0.5, True)

    assert result.algorithm == "kd_tree"
