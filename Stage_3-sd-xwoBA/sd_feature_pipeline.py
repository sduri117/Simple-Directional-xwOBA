"""Construct deterministic mixed-feature preprocessing for Stage 3."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from sd_experiment_config import (
    DIRECTION_CATEGORIES,
    DIRECTION_COLUMN,
    NUMERIC_FEATURE_COLUMNS,
    ScalingName,
)


def build_feature_preprocessor(scaling: ScalingName) -> ColumnTransformer:
    """Build numeric scaling plus fixed, unscaled direction one-hot encoding."""
    numeric_transformer: StandardScaler | RobustScaler | Literal["passthrough"]
    if scaling == "standard":
        numeric_transformer = StandardScaler()
    elif scaling == "robust":
        numeric_transformer = RobustScaler()
    else:
        numeric_transformer = "passthrough"

    direction_encoder = OneHotEncoder(
        categories=[list(DIRECTION_CATEGORIES)],
        drop=None,
        handle_unknown="error",
        sparse_output=False,
        dtype=np.float64,
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, list(NUMERIC_FEATURE_COLUMNS)),
            ("direction", direction_encoder, [DIRECTION_COLUMN]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
