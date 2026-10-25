"""Shared, typed configuration and input contracts for Stage 3."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Final, Literal, cast

import polars as pl

RANDOM_SEED: Final[int] = 99

TRAINING_YEARS: Final[tuple[int, ...]] = (2022, 2023, 2024)
CHRONOLOGICAL_TRAINING_YEARS: Final[tuple[int, ...]] = (2022, 2023)
CHRONOLOGICAL_VALIDATION_YEAR: Final[int] = 2024
PREDICTION_YEARS: Final[tuple[int, ...]] = (2025, 2026)
ALL_STAGE_3_YEARS: Final[tuple[int, ...]] = (*TRAINING_YEARS, *PREDICTION_YEARS)

NUMERIC_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "launch_speed",
    "launch_angle",
)
DIRECTION_COLUMN: Final[str] = "hit_direction"
DIRECTION_CATEGORIES: Final[tuple[str, ...]] = (
    "pull",
    "straightaway",
    "opposite",
)
RAW_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *NUMERIC_FEATURE_COLUMNS,
    DIRECTION_COLUMN,
)
TRANSFORMED_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *NUMERIC_FEATURE_COLUMNS,
    *(f"{DIRECTION_COLUMN}_{category}" for category in DIRECTION_CATEGORIES),
)
TARGET_COLUMN: Final[str] = "total_bases"
DATE_COLUMN: Final[str] = "game_date"
CLASS_LABELS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
XWOBA_LOCAL_COLUMN: Final[str] = "xwOBA-local"
SD_XWOBA_COLUMN: Final[str] = "sd-xwOBA"
SD_XWOBA_CONTACT_COLUMN: Final[str] = "sd-xwOBAcon"
BATTED_BALL_COLUMNS: Final[tuple[str, ...]] = (
    "game_date",
    "player_name",
    "batter",
    "events",
    "launch_speed",
    "launch_angle",
    "xwOBA-MLB",
    "hit_direction",
    "simplified_events",
    "total_bases",
)
BATTED_BALL_SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    "game_date": pl.String,
    "player_name": pl.String,
    "batter": pl.Int64,
    "events": pl.String,
    "launch_speed": pl.Float64,
    "launch_angle": pl.Float64,
    "xwOBA-MLB": pl.Float64,
    "hit_direction": pl.String,
    "simplified_events": pl.String,
    "total_bases": pl.Int64,
}

MODELING_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    DATE_COLUMN,
    *RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
MODELING_SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    DATE_COLUMN: pl.String,
    "launch_speed": pl.Float64,
    "launch_angle": pl.Float64,
    DIRECTION_COLUMN: pl.String,
    TARGET_COLUMN: pl.Int64,
}

CV_FOLDS: Final[int] = 5
SCALING_OPTIONS: Final[tuple[ScalingName, ...]] = (
    "none",
    "standard",
    "robust",
)
NEIGHBOR_OPTIONS: Final[tuple[int, ...]] = (
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
)
WEIGHT_OPTIONS: Final[tuple[WeightName, ...]] = ("uniform", "distance")
P_OPTIONS: Final[tuple[DistancePower, ...]] = (1, 2)

PROCESSED_DATA_DIRECTORY: Final[Path] = Path("ProcessedData")
STAGE_3_DIRECTORY: Final[Path] = Path("Stage_3-sd-xwoBA")
CALCULATED_XWOBA_DIRECTORY: Final[Path] = Path("Calculated-xwOBAs")
MODEL_SELECTION_RESULTS_PATH: Final[Path] = (
    STAGE_3_DIRECTORY / "ModelSelectionResults.CSV"
)
MODEL_ARTIFACT_DIRECTORY: Final[Path] = Path("ModelArtifacts")
FINAL_MODEL_PATH: Final[Path] = (
    MODEL_ARTIFACT_DIRECTORY / "sd_xwoba_knn_2022_2024.joblib"
)
FINAL_MODEL_METADATA_PATH: Final[Path] = (
    MODEL_ARTIFACT_DIRECTORY / "sd_xwoba_knn_2022_2024.metadata.json"
)
SEASON_LEVEL_XWOBA_PATH: Final[Path] = (
    CALCULATED_XWOBA_DIRECTORY / "SeasonLevelCalculatedxwOBA.csv"
)
CORRELATION_RESULTS_PATH: Final[Path] = (
    CALCULATED_XWOBA_DIRECTORY / "CorrelationResults.csv"
)
PLOTS_DIRECTORY: Final[Path] = CALCULATED_XWOBA_DIRECTORY / "plots"
SD_VS_MLB_PLAYER_SEASON_PLOT_PATH: Final[Path] = (
    PLOTS_DIRECTORY / "player_season_sd_xwoba_vs_mlb.png"
)
SD_VS_LOCAL_PLAYER_SEASON_PLOT_PATH: Final[Path] = (
    PLOTS_DIRECTORY / "player_season_sd_xwoba_vs_local.png"
)
SD_YEAR_AHEAD_PLOT_PATH: Final[Path] = (
    PLOTS_DIRECTORY / "year1_sd_xwoba_vs_year2_woba.png"
)
SD_CONTACT_YEAR_AHEAD_PLOT_PATH: Final[Path] = (
    PLOTS_DIRECTORY / "year1_sd_xwobacon_vs_year2_wobacon.png"
)
WOBA_CONSTANTS_PATH: Final[Path] = PROCESSED_DATA_DIRECTORY / "wOBAConstants.csv"
QUALIFIED_PLAYER_SEASONS_PATH: Final[Path] = (
    PROCESSED_DATA_DIRECTORY / "QualifiedPlayerSeasonsData.csv"
)

ScalingName = Literal["none", "standard", "robust"]
WeightName = Literal["uniform", "distance"]
DistancePower = Literal[1, 2]


class ModelingDataError(ValueError):
    """Raised when a modeling input violates the Stage 3 data contract."""


@dataclass(frozen=True, slots=True)
class KNNConfig:
    """One member of the complete Stage 3 KNN model-selection grid."""

    scaling: ScalingName
    n_neighbors: int
    weights: WeightName
    p: DistancePower


def generate_knn_configs() -> tuple[KNNConfig, ...]:
    """Return all KNN candidates in deterministic configuration order."""
    return tuple(
        KNNConfig(scaling, n_neighbors, weights, distance_power)
        for scaling, n_neighbors, weights, distance_power in product(
            SCALING_OPTIONS,
            NEIGHBOR_OPTIONS,
            WEIGHT_OPTIONS,
            P_OPTIONS,
        )
    )


KNN_CONFIGS: Final[tuple[KNNConfig, ...]] = generate_knn_configs()
KNN_CANDIDATE_COUNT: Final[int] = 96
TOTAL_CANDIDATE_COUNT: Final[int] = KNN_CANDIDATE_COUNT + 1
SELECTED_MODEL_CONFIG: Final[KNNConfig] = KNNConfig(
    scaling="robust",
    n_neighbors=1000,
    weights="uniform",
    p=2,
)
SELECTED_NEIGHBOR_ALGORITHM: Final[str] = "ball_tree"
FINAL_TRAINING_ROW_COUNT: Final[int] = 370_161


def annual_batted_ball_path(year: int) -> Path:
    """Return the canonical Stage 1 output path for a supported year."""
    if year not in ALL_STAGE_3_YEARS:
        raise ValueError(f"Unsupported Stage 3 year: {year}")
    return PROCESSED_DATA_DIRECTORY / f"{year}_all_BBE.csv"


def calculated_batted_ball_path(year: int) -> Path:
    """Return the shared calculated batted-ball path for a prediction year."""
    if year not in PREDICTION_YEARS:
        raise ValueError(f"Unsupported prediction year: {year}")
    return CALCULATED_XWOBA_DIRECTORY / (f"{year}_all_BBE_with_calculated_xwOBAs.csv")


def validate_modeling_frame(frame: pl.DataFrame, expected_year: int) -> None:
    """Validate one annual training frame against the Stage 3 model contract."""
    if expected_year not in TRAINING_YEARS:
        raise ValueError(f"Unsupported training year: {expected_year}")
    if frame.is_empty():
        raise ModelingDataError(f"Modeling data for {expected_year} is empty")

    missing_columns = [
        column for column in MODELING_REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ModelingDataError(
            f"Modeling data for {expected_year} is missing required columns: "
            f"{missing_columns}"
        )

    unexpected_types = {
        column: (frame.schema[column], expected_type)
        for column, expected_type in MODELING_SCHEMA.items()
        if frame.schema[column] != expected_type
    }
    if unexpected_types:
        raise ModelingDataError(
            f"Modeling data for {expected_year} has unexpected column types: "
            f"{unexpected_types}"
        )

    null_counts = {
        column: count
        for column, count in zip(
            MODELING_REQUIRED_COLUMNS,
            frame.select(MODELING_REQUIRED_COLUMNS).null_count().row(0),
            strict=True,
        )
        if count
    }
    if null_counts:
        raise ModelingDataError(
            f"Modeling data for {expected_year} contains null values: {null_counts}"
        )

    for feature in NUMERIC_FEATURE_COLUMNS:
        nonfinite_count = frame.filter(~pl.col(feature).is_finite()).height
        if nonfinite_count:
            raise ModelingDataError(
                f"Modeling data for {expected_year} has {nonfinite_count:,} "
                f"non-finite {feature} value(s)"
            )

    actual_directions = set(
        cast(list[str], frame.get_column(DIRECTION_COLUMN).unique().to_list())
    )
    invalid_directions = actual_directions - set(DIRECTION_CATEGORIES)
    if invalid_directions:
        raise ModelingDataError(
            f"Modeling data for {expected_year} contains invalid hit directions: "
            f"{sorted(invalid_directions)}"
        )
    missing_directions = set(DIRECTION_CATEGORIES) - actual_directions
    if missing_directions:
        raise ModelingDataError(
            f"Modeling data for {expected_year} is missing hit directions: "
            f"{sorted(missing_directions)}"
        )

    parsed_dates = frame.get_column(DATE_COLUMN).str.to_date(strict=False)
    if parsed_dates.null_count():
        raise ModelingDataError(
            f"Modeling data for {expected_year} contains malformed dates"
        )
    actual_years = set(cast(list[int], parsed_dates.dt.year().unique().to_list()))
    if actual_years != {expected_year}:
        raise ModelingDataError(
            f"Modeling data for {expected_year} contains years "
            f"{sorted(actual_years)}"
        )

    actual_classes = set(
        cast(list[int], frame.get_column(TARGET_COLUMN).unique().to_list())
    )
    invalid_classes = actual_classes - set(CLASS_LABELS)
    if invalid_classes:
        raise ModelingDataError(
            f"Modeling data for {expected_year} contains invalid target classes: "
            f"{sorted(invalid_classes)}"
        )
    missing_classes = set(CLASS_LABELS) - actual_classes
    if missing_classes:
        raise ModelingDataError(
            f"Modeling data for {expected_year} is missing target classes: "
            f"{sorted(missing_classes)}"
        )
