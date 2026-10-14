from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from experiment_config import (
    CLASS_LABELS,
    CV_FOLDS,
    FEATURE_COLUMNS,
    KNN_CANDIDATE_COUNT,
    KNN_CONFIGS,
    MODEL_SELECTION_RESULTS_PATH,
    MODELING_REQUIRED_COLUMNS,
    NEIGHBOR_OPTIONS,
    P_OPTIONS,
    RANDOM_SEED,
    SCALING_OPTIONS,
    TARGET_COLUMN,
    TOTAL_CANDIDATE_COUNT,
    TRAINING_YEARS,
    WEIGHT_OPTIONS,
    ModelingDataError,
    annual_batted_ball_path,
    generate_knn_configs,
    validate_modeling_frame,
)


def valid_modeling_frame(year: int = 2022) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_date": [f"{year}-04-{day:02d}" for day in range(1, 6)],
            "launch_speed": [80.0, 85.0, 90.0, 95.0, 100.0],
            "launch_angle": [-10.0, 0.0, 10.0, 20.0, 30.0],
            "total_bases": list(CLASS_LABELS),
        },
        schema={
            "game_date": pl.String,
            "launch_speed": pl.Float64,
            "launch_angle": pl.Float64,
            "total_bases": pl.Int64,
        },
    )


def test_reproducibility_and_modeling_contract() -> None:
    assert RANDOM_SEED == 99
    assert CV_FOLDS == 5
    assert TRAINING_YEARS == (2022, 2023, 2024)
    assert FEATURE_COLUMNS == ("launch_speed", "launch_angle")
    assert TARGET_COLUMN == "total_bases"
    assert CLASS_LABELS == (0, 1, 2, 3, 4)
    assert MODELING_REQUIRED_COLUMNS == (
        "game_date",
        "launch_speed",
        "launch_angle",
        "total_bases",
    )


def test_candidate_grid_is_complete_unique_and_deterministic() -> None:
    expected_count = (
        len(SCALING_OPTIONS)
        * len(NEIGHBOR_OPTIONS)
        * len(WEIGHT_OPTIONS)
        * len(P_OPTIONS)
    )

    assert expected_count == KNN_CANDIDATE_COUNT == 96
    assert TOTAL_CANDIDATE_COUNT == 97
    assert len(KNN_CONFIGS) == expected_count
    assert len(set(KNN_CONFIGS)) == expected_count
    assert KNN_CONFIGS == generate_knn_configs()
    assert KNN_CONFIGS[0] == replace(
        KNN_CONFIGS[0],
        scaling="none",
        n_neighbors=5,
        weights="uniform",
        p=1,
    )
    assert KNN_CONFIGS[-1] == replace(
        KNN_CONFIGS[-1],
        scaling="robust",
        n_neighbors=1000,
        weights="distance",
        p=2,
    )
    assert {
        (config.scaling, config.n_neighbors, config.weights, config.p)
        for config in KNN_CONFIGS
    } == {
        (scaling, neighbors, weights, distance_power)
        for scaling in SCALING_OPTIONS
        for neighbors in NEIGHBOR_OPTIONS
        for weights in WEIGHT_OPTIONS
        for distance_power in P_OPTIONS
    }


def test_canonical_paths_preserve_agreed_names() -> None:
    assert annual_batted_ball_path(2022).as_posix() == (
        "ProcessedData/2022_all_BBE.csv"
    )
    assert MODEL_SELECTION_RESULTS_PATH.name == "ModelSelectionResuts.CSV"


def test_annual_batted_ball_path_rejects_unsupported_year() -> None:
    with pytest.raises(ValueError, match="Unsupported Stage 2 year: 2021"):
        annual_batted_ball_path(2021)


def test_validate_modeling_frame_accepts_valid_contract() -> None:
    validate_modeling_frame(valid_modeling_frame(), 2022)


def test_validate_modeling_frame_rejects_missing_column() -> None:
    frame = valid_modeling_frame().drop("launch_angle")

    with pytest.raises(ModelingDataError, match="missing required columns"):
        validate_modeling_frame(frame, 2022)


def test_validate_modeling_frame_rejects_unexpected_type() -> None:
    frame = valid_modeling_frame().with_columns(pl.col("total_bases").cast(pl.Int8))

    with pytest.raises(ModelingDataError, match="unexpected column types"):
        validate_modeling_frame(frame, 2022)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("launch_speed", None, "contains null values"),
        ("launch_angle", float("nan"), "non-finite launch_angle"),
        ("launch_speed", float("inf"), "non-finite launch_speed"),
        ("game_date", "not-a-date", "malformed dates"),
        ("game_date", "2023-04-01", "contains years"),
        ("total_bases", 5, "invalid target classes"),
    ],
)
def test_validate_modeling_frame_rejects_invalid_values(
    column: str,
    value: object,
    message: str,
) -> None:
    frame = valid_modeling_frame()
    values = frame.get_column(column).to_list()
    values[0] = value
    frame = frame.with_columns(pl.Series(column, values).cast(frame.schema[column]))

    with pytest.raises(ModelingDataError, match=message):
        validate_modeling_frame(frame, 2022)


def test_validate_modeling_frame_requires_every_target_class() -> None:
    frame = valid_modeling_frame().filter(pl.col("total_bases") != 4)

    with pytest.raises(ModelingDataError, match=r"missing target classes: \[4\]"):
        validate_modeling_frame(frame, 2022)


def test_validate_modeling_frame_rejects_empty_data() -> None:
    with pytest.raises(ModelingDataError, match="is empty"):
        validate_modeling_frame(valid_modeling_frame().clear(), 2022)


def test_validate_modeling_frame_rejects_unsupported_training_year() -> None:
    with pytest.raises(ValueError, match="Unsupported training year: 2025"):
        validate_modeling_frame(valid_modeling_frame(2025), 2025)
