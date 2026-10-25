from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from calculate_batted_ball_sd_xwoba import (
    BattedBallCalculationError,
    CalculatedOutputExistsError,
    WOBAWeights,
    calculate_sd_xwoba,
    calculate_year,
    load_woba_weights,
    parse_args,
    predict_probabilities_in_batches,
    validate_calculated_output,
    validate_prediction_input,
)
from sd_experiment_config import (
    BATTED_BALL_COLUMNS,
    BATTED_BALL_SCHEMA,
    RAW_FEATURE_COLUMNS,
    SD_XWOBA_COLUMN,
    XWOBA_LOCAL_COLUMN,
    KNNConfig,
)
from sd_model_evaluation import build_knn_pipeline


def valid_calculated_frame(year: int = 2025) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_date": [f"{year}-04-{day:02d}" for day in range(1, 7)],
            "player_name": [f"Player {index}" for index in range(6)],
            "batter": [100, 101, 102, 103, 104, 105],
            "events": [
                "field_out",
                "single",
                "double",
                "triple",
                "home_run",
                "field_out",
            ],
            "launch_speed": [75.0, 85.0, 95.0, 105.0, 110.0, 80.0],
            "launch_angle": [-10.0, 0.0, 10.0, 20.0, 30.0, -5.0],
            "xwOBA-MLB": [0.1, 0.3, 0.5, 0.7, 0.9, 0.2],
            "hit_direction": [
                "pull",
                "straightaway",
                "opposite",
                "pull",
                "straightaway",
                "opposite",
            ],
            "simplified_events": [
                "out",
                "single",
                "double",
                "triple",
                "home_run",
                "out",
            ],
            "total_bases": [0, 1, 2, 3, 4, 0],
            XWOBA_LOCAL_COLUMN: [0.11, 0.31, 0.51, 0.71, 0.91, 0.21],
        },
        schema={**BATTED_BALL_SCHEMA, XWOBA_LOCAL_COLUMN: pl.Float64},
    )


def fitted_pipeline() -> Pipeline:
    frame = valid_calculated_frame()
    pipeline = build_knn_pipeline(KNNConfig("robust", 3, "uniform", 2), "brute", 1)
    pipeline.fit(frame.select(RAW_FEATURE_COLUMNS), frame["total_bases"].to_numpy())
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

    assert load_woba_weights(path, 2026) == WOBAWeights(2026, 0.89, 1.26, 1.59, 2.05)


def test_validate_prediction_input_accepts_existing_calculated_file() -> None:
    validate_prediction_input(valid_calculated_frame(), 2025)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda frame: frame.drop("hit_direction"), "missing required columns"),
        (
            lambda frame: frame.select(
                "game_date", "player_name", "batter", *reversed(frame.columns[3:])
            ),
            "required order",
        ),
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
            lambda frame: frame.with_columns(pl.lit("center").alias("hit_direction")),
            "invalid hit directions",
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
    with pytest.raises(BattedBallCalculationError, match=message):
        validate_prediction_input(change(valid_calculated_frame()), 2025)


def test_predict_probabilities_batches_and_aligns_classes() -> None:
    pipeline = fitted_pipeline()
    features = valid_calculated_frame().select(RAW_FEATURE_COLUMNS)

    probabilities = predict_probabilities_in_batches(pipeline, features, 2)

    assert probabilities.shape == (6, 5)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_calculate_sd_xwoba_is_probability_weighted_sum() -> None:
    probabilities = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.25, 0.25, 0.25],
        ],
        dtype=np.float64,
    )

    result = calculate_sd_xwoba(probabilities, WOBAWeights(2025, 0.8, 1.2, 1.6, 2.0))

    np.testing.assert_allclose(result, [0.0, 1.4])


def test_calculate_year_preserves_existing_columns_and_appends_sd_xwoba(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.csv"
    constants_path = tmp_path / "constants.csv"
    output_path = tmp_path / "calculated" / "output.csv"
    source = valid_calculated_frame()
    source.write_csv(input_path)
    write_constants(constants_path)

    summary = calculate_year(
        fitted_pipeline(),
        year=2025,
        input_path=input_path,
        constants_path=constants_path,
        output_path=output_path,
        batch_size=2,
    )

    assert summary.rows == 6
    assert 0.0 <= summary.minimum_sd_xwoba <= summary.maximum_sd_xwoba <= 2.04
    output = pl.read_csv(output_path)
    assert output.columns == [*BATTED_BALL_COLUMNS, XWOBA_LOCAL_COLUMN, SD_XWOBA_COLUMN]
    assert output.select(source.columns).equals(source, null_equal=True)
    assert output[SD_XWOBA_COLUMN].is_finite().all()


def test_calculate_year_replaces_prior_sd_column_idempotently(tmp_path: Path) -> None:
    path = tmp_path / "calculated.csv"
    constants_path = tmp_path / "constants.csv"
    valid_calculated_frame().with_columns(
        pl.lit(-1.0).alias(SD_XWOBA_COLUMN)
    ).write_csv(path)
    write_constants(constants_path)

    calculate_year(
        fitted_pipeline(),
        year=2025,
        input_path=path,
        constants_path=constants_path,
        output_path=path,
        batch_size=2,
        overwrite=True,
    )

    output = pl.read_csv(path)
    assert output.columns.count(SD_XWOBA_COLUMN) == 1
    assert float(output[SD_XWOBA_COLUMN].to_numpy().min()) >= 0.0


def test_calculate_year_requires_overwrite_for_in_place_update(tmp_path: Path) -> None:
    path = tmp_path / "output.csv"
    valid_calculated_frame().write_csv(path)

    with pytest.raises(CalculatedOutputExistsError, match="In-place replacement"):
        calculate_year(
            fitted_pipeline(),
            year=2025,
            input_path=path,
            constants_path=tmp_path / "unused.csv",
            output_path=path,
        )


def test_validate_calculated_output_detects_changed_source(tmp_path: Path) -> None:
    source = valid_calculated_frame()
    values = np.full(source.height, 0.5, dtype=np.float64)
    output = source.with_columns(
        pl.Series(SD_XWOBA_COLUMN, values, dtype=pl.Float64)
    ).with_columns(pl.lit(999).alias("batter"))
    path = tmp_path / "output.csv"
    output.write_csv(path)

    with pytest.raises(BattedBallCalculationError, match="changed or reordered"):
        validate_calculated_output(
            path, source, values, WOBAWeights(2025, 0.88, 1.25, 1.58, 2.04)
        )


def test_parse_args_supports_calculation_controls() -> None:
    args = parse_args(
        [
            "--year",
            "2025",
            "--calculated-directory",
            "calculated",
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
    assert args.calculated_directory == Path("calculated")
    assert args.constants == Path("constants.csv")
    assert args.model == Path("model.joblib")
    assert args.metadata == Path("metadata.json")
    assert args.batch_size == 250
    assert args.overwrite is True
