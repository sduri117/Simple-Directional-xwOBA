from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from calculate_batted_ball_xwoba import (
    BattedBallCalculationError,
    CalculatedOutputExistsError,
    WOBAWeights,
    calculate_local_xwoba,
    calculate_year,
    load_woba_weights,
    parse_args,
    predict_probabilities_in_batches,
    validate_calculated_output,
    validate_prediction_input,
)
from experiment_config import (
    BATTED_BALL_COLUMNS,
    BATTED_BALL_SCHEMA,
    FEATURE_COLUMNS,
    XWOBA_LOCAL_COLUMN,
    KNNConfig,
)
from model_evaluation import build_knn_pipeline


def valid_batted_ball_frame(year: int = 2025) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_date": [f"{year}-04-{day:02d}" for day in range(1, 6)],
            "player_name": [f"Player {index}" for index in range(5)],
            "batter": [100, 101, 102, 103, 104],
            "events": ["field_out", "single", "double", "triple", "home_run"],
            "launch_speed": [75.0, 85.0, 95.0, 105.0, 110.0],
            "launch_angle": [-10.0, 0.0, 10.0, 20.0, 30.0],
            "xwOBA-MLB": [0.1, 0.3, 0.5, 0.7, 0.9],
            "hit_direction": ["pull"] * 5,
            "simplified_events": [
                "out",
                "single",
                "double",
                "triple",
                "home_run",
            ],
            "total_bases": [0, 1, 2, 3, 4],
        },
        schema=BATTED_BALL_SCHEMA,
    )


def fitted_pipeline() -> Pipeline:
    frame = valid_batted_ball_frame()
    pipeline = build_knn_pipeline(KNNConfig("none", 3, "uniform", 1), "brute", 1)
    pipeline.fit(
        frame.select(FEATURE_COLUMNS).to_numpy(),
        frame["total_bases"].to_numpy(),
    )
    return pipeline


def write_constants(path: Path) -> None:
    pl.DataFrame(
        {
            "Season": [2025, 2026],
            "w1B": [0.88, 0.89],
            "w2B": [1.25, 1.26],
            "w3B": [1.58, 1.59],
            "wHR": [2.04, 2.05],
        }
    ).write_csv(path)


def test_woba_weights_are_in_model_class_order() -> None:
    weights = WOBAWeights(2025, 0.88, 1.25, 1.58, 2.04)

    np.testing.assert_array_equal(
        weights.as_array(), np.asarray([0.0, 0.88, 1.25, 1.58, 2.04])
    )


def test_load_woba_weights_selects_requested_season(tmp_path: Path) -> None:
    path = tmp_path / "constants.csv"
    write_constants(path)

    weights = load_woba_weights(path, 2026)

    assert weights == WOBAWeights(2026, 0.89, 1.26, 1.59, 2.05)


def test_load_woba_weights_requires_unique_season(tmp_path: Path) -> None:
    path = tmp_path / "constants.csv"
    pl.DataFrame(
        {
            "Season": [2025, 2025],
            "w1B": [0.88, 0.88],
            "w2B": [1.25, 1.25],
            "w3B": [1.58, 1.58],
            "wHR": [2.04, 2.04],
        }
    ).write_csv(path)

    with pytest.raises(BattedBallCalculationError, match="Expected one"):
        load_woba_weights(path, 2025)


def test_validate_prediction_input_accepts_stage_one_contract() -> None:
    validate_prediction_input(valid_batted_ball_frame(), 2025)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda frame: frame.drop("launch_angle"), "Input columns"),
        (
            lambda frame: frame.with_columns(pl.col("total_bases").cast(pl.Int8)),
            "unexpected types",
        ),
        (
            lambda frame: frame.with_columns(
                pl.when(pl.int_range(pl.len()) == 0)
                .then(None)
                .otherwise(pl.col("player_name"))
                .alias("player_name")
            ),
            "null values",
        ),
        (
            lambda frame: frame.with_columns(
                pl.when(pl.int_range(pl.len()) == 0)
                .then(float("nan"))
                .otherwise(pl.col("launch_speed"))
                .alias("launch_speed")
            ),
            "non-finite launch_speed",
        ),
        (
            lambda frame: frame.with_columns(pl.lit("2026-04-01").alias("game_date")),
            "contains years",
        ),
    ],
)
def test_validate_prediction_input_rejects_invalid_data(
    change: Callable[[pl.DataFrame], pl.DataFrame],
    message: str,
) -> None:
    frame = change(valid_batted_ball_frame())

    with pytest.raises(BattedBallCalculationError, match=message):
        validate_prediction_input(frame, 2025)


def test_predict_probabilities_batches_and_aligns_classes() -> None:
    pipeline = fitted_pipeline()
    assert hasattr(pipeline, "predict_proba")
    frame = valid_batted_ball_frame()
    features = frame.select(FEATURE_COLUMNS).to_numpy()

    probabilities = predict_probabilities_in_batches(pipeline, features, 2)

    assert probabilities.shape == (5, 5)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_calculate_local_xwoba_is_probability_weighted_sum() -> None:
    probabilities = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.25, 0.25, 0.25],
        ],
        dtype=np.float64,
    )
    weights = WOBAWeights(2025, 0.8, 1.2, 1.6, 2.0)

    result = calculate_local_xwoba(probabilities, weights)

    np.testing.assert_allclose(result, [0.0, 1.4])


def test_calculate_year_preserves_source_and_appends_xwoba(tmp_path: Path) -> None:
    input_path = tmp_path / "2025_all_BBE.csv"
    constants_path = tmp_path / "constants.csv"
    output_path = tmp_path / "calculated" / "2025_output.csv"
    source = valid_batted_ball_frame()
    source.write_csv(input_path)
    write_constants(constants_path)
    pipeline = fitted_pipeline()

    summary = calculate_year(
        pipeline,
        year=2025,
        input_path=input_path,
        constants_path=constants_path,
        output_path=output_path,
        batch_size=2,
    )

    assert summary.rows == 5
    assert summary.output_path == output_path
    assert 0.0 <= summary.minimum_xwoba <= summary.maximum_xwoba <= 2.04
    output = pl.read_csv(output_path)
    assert output.columns == [*BATTED_BALL_COLUMNS, XWOBA_LOCAL_COLUMN]
    assert output.select(BATTED_BALL_COLUMNS).equals(source, null_equal=True)
    assert output[XWOBA_LOCAL_COLUMN].is_finite().all()


def test_calculate_year_requires_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / "output.csv"
    output_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(CalculatedOutputExistsError, match="replacement"):
        calculate_year(
            fitted_pipeline(),
            year=2025,
            input_path=tmp_path / "unused.csv",
            constants_path=tmp_path / "unused_constants.csv",
            output_path=output_path,
        )


def test_validate_calculated_output_detects_changed_source(tmp_path: Path) -> None:
    source = valid_batted_ball_frame()
    weights = WOBAWeights(2025, 0.88, 1.25, 1.58, 2.04)
    values = np.full(source.height, 0.5, dtype=np.float64)
    output = source.with_columns(
        pl.Series(XWOBA_LOCAL_COLUMN, values, dtype=pl.Float64)
    ).with_columns(pl.lit(999).alias("batter"))
    path = tmp_path / "output.csv"
    output.write_csv(path)

    with pytest.raises(BattedBallCalculationError, match="changed or reordered"):
        validate_calculated_output(path, source, values, weights)


def test_parse_args_supports_calculation_controls() -> None:
    args = parse_args(
        [
            "--year",
            "2025",
            "--input-directory",
            "inputs",
            "--output-directory",
            "outputs",
            "--constants",
            "constants.csv",
            "--model",
            "model.joblib",
            "--metadata",
            "metadata.json",
            "--batch-size",
            "250",
            "--overwrite",
        ]
    )

    assert args.year == [2025]
    assert args.input_directory == Path("inputs")
    assert args.output_directory == Path("outputs")
    assert args.constants == Path("constants.csv")
    assert args.model == Path("model.joblib")
    assert args.metadata == Path("metadata.json")
    assert args.batch_size == 250
    assert args.overwrite is True
