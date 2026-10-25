from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier

import sd_model_evaluation
from sd_experiment_config import CLASS_LABELS, RANDOM_SEED, KNNConfig
from sd_model_evaluation import (
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
    launch_speed: list[float] = []
    launch_angle: list[float] = []
    hit_direction: list[str] = []
    target: list[int] = []
    years: list[int] = []
    directions = ("pull", "straightaway", "opposite")
    for year in (2022, 2023, 2024):
        for class_label in CLASS_LABELS:
            for repeat in range(rows_per_class_year):
                launch_speed.append(70.0 + class_label * 7.0 + repeat * 0.1)
                launch_angle.append(-15.0 + class_label * 10.0 + repeat * 0.2)
                hit_direction.append(directions[(class_label + repeat) % 3])
                target.append(class_label)
                years.append(year)
    return ModelingDataset(
        pl.DataFrame(
            {
                "launch_speed": launch_speed,
                "launch_angle": launch_angle,
                "hit_direction": hit_direction,
            }
        ),
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
    directions = ["pull", "straightaway", "opposite", "pull", "straightaway"]
    for year in (2022, 2023, 2024):
        pl.DataFrame(
            {
                "game_date": [f"{year}-04-{day:02d}" for day in range(1, 6)],
                "launch_speed": [80.0, 85.0, 90.0, 95.0, 100.0],
                "launch_angle": [-10.0, 0.0, 10.0, 20.0, 30.0],
                "hit_direction": directions,
                "total_bases": list(CLASS_LABELS),
                "unused": ["ignored"] * 5,
            },
            schema={
                "game_date": pl.String,
                "launch_speed": pl.Float64,
                "launch_angle": pl.Float64,
                "hit_direction": pl.String,
                "total_bases": pl.Int64,
                "unused": pl.String,
            },
        ).write_csv(directory / f"{year}_all_BBE.csv")


def test_load_modeling_data_validates_and_concatenates_years(tmp_path: Path) -> None:
    write_modeling_inputs(tmp_path)

    dataset = load_modeling_data(tmp_path)

    assert dataset.features.shape == (15, 3)
    assert dataset.features.columns == [
        "launch_speed",
        "launch_angle",
        "hit_direction",
    ]
    assert dataset.target.tolist() == list(CLASS_LABELS) * 3
    assert dataset.years.tolist() == [2022] * 5 + [2023] * 5 + [2024] * 5
    assert dataset.target.dtype == np.int64


def test_load_modeling_data_wraps_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ModelEvaluationError, match="Could not read"):
        load_modeling_data(tmp_path)


def test_modeling_dataset_rejects_bad_shape_and_order() -> None:
    dataset = synthetic_dataset()
    with pytest.raises(ValueError, match="Feature shape"):
        ModelingDataset(
            dataset.features.drop("hit_direction"), dataset.target, dataset.years
        )
    with pytest.raises(ValueError, match="required order"):
        ModelingDataset(
            dataset.features.select("hit_direction", "launch_speed", "launch_angle"),
            dataset.target,
            dataset.years,
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


def test_build_knn_pipeline_contains_preprocessor_and_classifier() -> None:
    config = KNNConfig("standard", 5, "distance", 1)

    pipeline = build_knn_pipeline(config, "ball_tree", 2)

    assert isinstance(pipeline.named_steps["preprocessor"], ColumnTransformer)
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, KNeighborsClassifier)
    parameters = classifier.get_params()
    assert parameters["n_neighbors"] == 5
    assert parameters["weights"] == "distance"
    assert parameters["p"] == 1
    assert parameters["algorithm"] == "ball_tree"
    assert parameters["n_jobs"] == 2


def test_fitted_pipeline_retains_five_transformed_features() -> None:
    dataset = synthetic_dataset(rows_per_class_year=2)
    pipeline = build_knn_pipeline(KNNConfig("robust", 3, "uniform", 2), "brute", 1)

    pipeline.fit(dataset.features, dataset.target)
    preprocessor = pipeline.named_steps["preprocessor"]

    assert isinstance(preprocessor, ColumnTransformer)
    assert preprocessor.get_feature_names_out().tolist() == [
        "launch_speed",
        "launch_angle",
        "hit_direction_pull",
        "hit_direction_straightaway",
        "hit_direction_opposite",
    ]
    probabilities = pipeline.predict_proba(dataset.features.head(4))
    assert probabilities.shape == (4, 5)


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
    probabilities = cross_validated_probabilities(synthetic_dataset(), None, folds=5)

    np.testing.assert_allclose(probabilities, np.full((75, 5), 0.2))


def test_chronological_validation_uses_2024_as_test_data() -> None:
    target, probabilities = chronological_probabilities(synthetic_dataset(), None)

    assert target.tolist() == [
        class_label for class_label in CLASS_LABELS for _ in range(5)
    ]
    np.testing.assert_allclose(probabilities, np.full((25, 5), 0.2))


def test_evaluate_candidate_returns_both_validation_designs() -> None:
    results = evaluate_candidate(synthetic_dataset(), None)

    assert results.cross_validation.log_loss == pytest.approx(np.log(5.0))
    assert results.chronological.log_loss == pytest.approx(np.log(5.0))
    assert results.cross_validation.brier_score == pytest.approx(0.8)


def test_neighbor_algorithm_benchmark_checks_equivalence() -> None:
    dataset = synthetic_dataset(rows_per_class_year=10)

    selected, results = benchmark_neighbor_algorithms(
        dataset.features,
        dataset.target,
        train_size=100,
        query_size=25,
        n_neighbors=5,
    )

    assert selected in {"brute", "kd_tree", "ball_tree"}
    assert [result.algorithm for result in results] == [
        "brute",
        "kd_tree",
        "ball_tree",
    ]
    assert all(result.equivalent_to_brute for result in results)
    assert all(result.elapsed_seconds >= 0.0 for result in results)


def test_result_rows_distinguish_baseline_and_knn() -> None:
    baseline = sample_result_row()
    config = KNNConfig("none", 5, "uniform", 1)
    knn = sample_result_row(config)

    assert baseline.model_type == "baseline"
    assert baseline.neighbor_algorithm is None
    assert knn.model_type == "knn"
    assert knn.feature_scaling == "none"
    assert knn.neighbor_algorithm == "brute"


def test_checkpoint_round_trip_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    frame = pl.DataFrame(
        [sample_result_row().as_record()], schema=sd_model_evaluation.RESULT_SCHEMA
    )

    write_result_checkpoint(frame, path)
    restored = read_result_checkpoint(path)

    assert restored.equals(frame, null_equal=True)
    assert restored.columns == list(RESULT_COLUMNS)
    assert not path.with_suffix(".csv.part").exists()


def test_result_validation_rejects_duplicate_candidate() -> None:
    record = sample_result_row().as_record()
    frame = pl.DataFrame([record, record], schema=sd_model_evaluation.RESULT_SCHEMA)

    with pytest.raises(ModelEvaluationError, match="duplicate candidate"):
        validate_result_frame(frame)


def test_run_experiment_checkpoints_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = synthetic_dataset()
    configs = [KNNConfig("none", 5, "uniform", 1)]
    output = tmp_path / "results.csv"
    evaluations: list[KNNConfig | None] = []

    def fake_evaluate(
        unused_dataset: ModelingDataset,
        config: KNNConfig | None,
        unused_algorithm: str,
        unused_n_jobs: int,
        unused_folds: int,
    ) -> EvaluationResults:
        evaluations.append(config)
        return sample_evaluation()

    monkeypatch.setattr(sd_model_evaluation, "evaluate_candidate", fake_evaluate)
    monkeypatch.setattr(
        sd_model_evaluation,
        "benchmark_neighbor_algorithms",
        lambda *args, **kwargs: (
            "brute",
            (BenchmarkResult("brute", 0.01, True),),
        ),
    )

    first = run_experiment(dataset, configs=configs, output_path=output)
    second = run_experiment(dataset, configs=configs, output_path=output)

    assert first.height == 2
    assert second.equals(first, null_equal=True)
    assert evaluations == [None, configs[0]]


def test_run_experiment_rejects_algorithm_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = synthetic_dataset()
    config = KNNConfig("none", 5, "uniform", 1)
    output = tmp_path / "results.csv"
    frame = pl.DataFrame(
        [sample_result_row(config).as_record()],
        schema=sd_model_evaluation.RESULT_SCHEMA,
    )
    write_result_checkpoint(frame, output)

    monkeypatch.setattr(
        sd_model_evaluation, "evaluate_candidate", lambda *args: sample_evaluation()
    )
    with pytest.raises(ModelEvaluationError, match="not requested"):
        run_experiment(
            dataset,
            configs=[config],
            output_path=output,
            algorithm="kd_tree",
        )


def test_parse_args_supports_experiment_controls() -> None:
    args = parse_args(
        [
            "--data-directory",
            "inputs",
            "--output",
            "results.csv",
            "--algorithm",
            "ball_tree",
            "--n-jobs",
            "3",
            "--folds",
            "4",
            "--overwrite",
        ]
    )

    assert args.data_directory == Path("inputs")
    assert args.output == Path("results.csv")
    assert args.algorithm == "ball_tree"
    assert args.n_jobs == 3
    assert args.folds == 4
    assert args.overwrite is True
