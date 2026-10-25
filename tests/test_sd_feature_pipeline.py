from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from sd_experiment_config import TRANSFORMED_FEATURE_COLUMNS
from sd_feature_pipeline import build_feature_preprocessor


def feature_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "launch_speed": [70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
            "launch_angle": [-20.0, -10.0, 0.0, 10.0, 20.0, 30.0],
            "hit_direction": [
                "pull",
                "straightaway",
                "opposite",
                "pull",
                "straightaway",
                "opposite",
            ],
        }
    )


@pytest.mark.parametrize(
    ("scaling", "scaler_type"),
    [
        ("none", None),
        ("standard", StandardScaler),
        ("robust", RobustScaler),
    ],
)
def test_preprocessor_uses_requested_numeric_scaling(
    scaling: str,
    scaler_type: type[StandardScaler] | type[RobustScaler] | None,
) -> None:
    preprocessor = build_feature_preprocessor(scaling)  # type: ignore[arg-type]
    configured_numeric_transformer = preprocessor.get_params()["numeric"]
    transformed = preprocessor.fit_transform(feature_frame())

    numeric_transformer = preprocessor.named_transformers_["numeric"]
    if scaler_type is None:
        assert configured_numeric_transformer == "passthrough"
        np.testing.assert_array_equal(
            transformed[:, :2], feature_frame().select("launch_speed", "launch_angle")
        )
    else:
        assert isinstance(numeric_transformer, scaler_type)
    assert transformed.shape == (6, 5)
    assert transformed.dtype == np.float64


def test_preprocessor_retains_all_three_unscaled_direction_columns() -> None:
    preprocessor = build_feature_preprocessor("standard")
    transformed = preprocessor.fit_transform(feature_frame())

    np.testing.assert_array_equal(
        transformed[:, 2:],
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    np.testing.assert_array_equal(transformed[:, 2:].sum(axis=1), np.ones(6))
    assert tuple(preprocessor.get_feature_names_out()) == TRANSFORMED_FEATURE_COLUMNS
    encoder = preprocessor.named_transformers_["direction"]
    assert isinstance(encoder, OneHotEncoder)
    encoder_parameters = encoder.get_params()
    assert encoder_parameters["drop"] is None
    assert encoder_parameters["handle_unknown"] == "error"


def test_numeric_scaler_is_fitted_only_on_training_rows() -> None:
    training = feature_frame().head(3)
    validation = pl.DataFrame(
        {
            "launch_speed": [1000.0],
            "launch_angle": [500.0],
            "hit_direction": ["pull"],
        }
    )
    preprocessor = build_feature_preprocessor("standard")
    preprocessor.fit(training)

    scaler = preprocessor.named_transformers_["numeric"]
    assert isinstance(scaler, StandardScaler)
    assert scaler.mean_ is not None
    np.testing.assert_allclose(scaler.mean_, np.asarray([80.0, -10.0]))
    transformed = preprocessor.transform(validation)
    assert transformed[0, 0] > 100.0


def test_preprocessor_rejects_unknown_direction() -> None:
    preprocessor = build_feature_preprocessor("none").fit(feature_frame())
    unknown = (
        feature_frame().head(1).with_columns(pl.lit("center").alias("hit_direction"))
    )

    with pytest.raises(ValueError, match="unknown categories"):
        preprocessor.transform(unknown)
