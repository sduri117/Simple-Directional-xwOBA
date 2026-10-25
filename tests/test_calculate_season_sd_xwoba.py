from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from calculate_season_sd_xwoba import (
    EXISTING_SEASON_COLUMNS,
    PlateAppearanceWeights,
    SeasonCalculationError,
    SeasonOutputExistsError,
    aggregate_batted_ball_sd_xwoba,
    calculate_all_seasons,
    calculate_season_frame,
    load_existing_seasons,
    load_plate_appearance_weights,
    load_qualified_seasons,
    parse_args,
    validate_season_output,
)
from sd_experiment_config import SD_XWOBA_COLUMN, SD_XWOBA_CONTACT_COLUMN


def existing_seasons() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "year": [2025, 2025, 2025, 2026, 2026, 2026],
            "batter_name": ["Alpha", "Beta", "Gamma", "Alpha", "Beta", "Delta"],
            "batter_id": [10, 20, 30, 10, 20, 40],
            "wOBA": [0.31, 0.35, 0.29, 0.34, 0.32, 0.30],
            "xwOBA-MLB": [0.30, 0.34, 0.28, 0.33, 0.31, 0.29],
            "xwOBA-local": [0.29, 0.36, 0.30, 0.32, 0.30, 0.27],
            "wOBAcon": [0.42, 0.48, 0.39, 0.45, 0.43, 0.40],
            "xwOBAcon-MLB": [0.41, 0.47, 0.38, 0.44, 0.42, 0.39],
            "xwOBAcon-local": [0.40, 0.49, 0.40, 0.43, 0.41, 0.37],
        },
        schema={
            "year": pl.Int64,
            "batter_name": pl.String,
            "batter_id": pl.Int64,
            "wOBA": pl.Float64,
            "xwOBA-MLB": pl.Float64,
            "xwOBA-local": pl.Float64,
            "wOBAcon": pl.Float64,
            "xwOBAcon-MLB": pl.Float64,
            "xwOBAcon-local": pl.Float64,
        },
    )


def qualified_seasons() -> pl.DataFrame:
    existing = existing_seasons()
    return pl.DataFrame(
        {
            "Season": existing["year"],
            "Name": existing["batter_name"],
            "MLBAMID": existing["batter_id"],
            "AB": [500, 450, 400, 510, 460, 420],
            "BB": [50, 40, 30, 55, 42, 35],
            "IBB": [5, 2, 1, 6, 2, 1],
            "HBP": [4, 5, 3, 4, 6, 2],
            "SF": [3, 4, 2, 3, 5, 2],
            "wOBA": existing["wOBA"],
            "xwOBA": existing["xwOBA-MLB"],
        }
    )


def batted_ball_aggregates() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "year": [2025, 2025, 2025, 2026, 2026, 2026],
            "batter_id": [10, 20, 30, 10, 20, 40],
            "batted_ball_count": [300, 280, 250, 310, 285, 260],
            "batted_ball_sd_xwoba_sum": [120.0, 105.0, 90.0, 125.0, 108.0, 95.0],
            SD_XWOBA_CONTACT_COLUMN: [0.4, 0.375, 0.36, 125 / 310, 108 / 285, 95 / 260],
        },
        schema={
            "year": pl.Int64,
            "batter_id": pl.Int64,
            "batted_ball_count": pl.Int64,
            "batted_ball_sd_xwoba_sum": pl.Float64,
            SD_XWOBA_CONTACT_COLUMN: pl.Float64,
        },
    )


def weights() -> dict[int, PlateAppearanceWeights]:
    return {
        2025: PlateAppearanceWeights(2025, 0.70, 0.72),
        2026: PlateAppearanceWeights(2026, 0.71, 0.73),
    }


def write_inputs(directory: Path) -> tuple[Path, Path, Path, Path]:
    calculated = directory / "calculated"
    calculated.mkdir()
    batted_rows = {
        2025: [(10, 0.4), (10, 0.6), (20, 0.3), (20, 0.5), (30, 0.2)],
        2026: [(10, 0.5), (10, 0.7), (20, 0.4), (20, 0.6), (40, 0.3)],
    }
    for year, rows in batted_rows.items():
        pl.DataFrame(
            {
                "game_date": [
                    f"{year}-04-{index + 1:02d}" for index in range(len(rows))
                ],
                "batter": [row[0] for row in rows],
                SD_XWOBA_COLUMN: [row[1] for row in rows],
            }
        ).write_csv(calculated / f"{year}_all_BBE_with_calculated_xwOBAs.csv")

    qualified_path = directory / "qualified.csv"
    qualified_seasons().write_csv(qualified_path)
    constants_path = directory / "constants.csv"
    pl.DataFrame(
        {"Season": [2025, 2026], "wBB": [0.70, 0.71], "wHBP": [0.72, 0.73]}
    ).write_csv(constants_path)
    output_path = directory / "seasons.csv"
    existing_seasons().write_csv(output_path)
    return calculated, qualified_path, constants_path, output_path


def test_load_existing_seasons_removes_prior_stage_three_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "seasons.csv"
    existing_seasons().with_columns(
        pl.lit(0.3).alias(SD_XWOBA_COLUMN),
        pl.lit(0.4).alias(SD_XWOBA_CONTACT_COLUMN),
    ).write_csv(path)

    loaded = load_existing_seasons(path)

    assert loaded.columns == list(EXISTING_SEASON_COLUMNS)
    assert loaded.equals(existing_seasons())


def test_qualified_and_coefficient_loaders_validate_inputs(tmp_path: Path) -> None:
    qualified_path = tmp_path / "qualified.csv"
    constants_path = tmp_path / "constants.csv"
    qualified_seasons().write_csv(qualified_path)
    pl.DataFrame(
        {"Season": [2025, 2026], "wBB": [0.70, 0.71], "wHBP": [0.72, 0.73]}
    ).write_csv(constants_path)

    qualified = load_qualified_seasons(qualified_path)
    loaded_weights = load_plate_appearance_weights(constants_path)

    assert qualified.height == 6
    assert loaded_weights == weights()


def test_aggregate_batted_ball_values_by_batter(tmp_path: Path) -> None:
    path = tmp_path / "batted.csv"
    pl.DataFrame(
        {
            "game_date": ["2025-04-01", "2025-04-02", "2025-04-03"],
            "batter": [10, 10, 20],
            SD_XWOBA_COLUMN: [0.2, 0.6, 0.5],
        }
    ).write_csv(path)

    aggregate = aggregate_batted_ball_sd_xwoba(path, 2025).sort("batter_id")

    assert aggregate["batted_ball_count"].to_list() == [2, 1]
    np.testing.assert_allclose(
        aggregate["batted_ball_sd_xwoba_sum"].to_numpy(), [0.8, 0.5]
    )
    np.testing.assert_allclose(
        aggregate[SD_XWOBA_CONTACT_COLUMN].to_numpy(), [0.4, 0.5]
    )


def test_calculate_season_frame_preserves_existing_and_applies_formulas() -> None:
    existing = existing_seasons()

    output, reconciliation = calculate_season_frame(
        existing, qualified_seasons(), batted_ball_aggregates(), weights()
    )

    assert output.columns == [
        *EXISTING_SEASON_COLUMNS,
        SD_XWOBA_COLUMN,
        SD_XWOBA_CONTACT_COLUMN,
    ]
    assert output.select(EXISTING_SEASON_COLUMNS).equals(existing)
    first = reconciliation.row(0, named=True)
    expected = (0.70 * (50 - 5) + 0.72 * 4 + 120.0) / (500 + 50 - 5 + 3 + 4)
    assert first[SD_XWOBA_COLUMN] == pytest.approx(expected)
    assert first[SD_XWOBA_CONTACT_COLUMN] == pytest.approx(0.4)


def test_calculate_season_frame_rejects_missing_batted_ball_match() -> None:
    aggregates = batted_ball_aggregates().filter(pl.col("batter_id") != 40)

    with pytest.raises(SeasonCalculationError, match="lack batted-ball"):
        calculate_season_frame(
            existing_seasons(), qualified_seasons(), aggregates, weights()
        )


def test_validate_season_output_detects_changed_existing_value(tmp_path: Path) -> None:
    existing = existing_seasons()
    output, reconciliation = calculate_season_frame(
        existing, qualified_seasons(), batted_ball_aggregates(), weights()
    )
    changed = output.with_columns((pl.col("wOBA") + 0.01).alias("wOBA"))
    path = tmp_path / "output.csv"
    changed.write_csv(path)

    with pytest.raises(SeasonCalculationError, match="changed from expectation"):
        validate_season_output(path, existing, output, reconciliation)


def test_calculate_all_seasons_writes_validated_output(tmp_path: Path) -> None:
    calculated, qualified_path, constants_path, output_path = write_inputs(tmp_path)
    prior = pl.read_csv(output_path)

    summary = calculate_all_seasons(
        calculated_directory=calculated,
        qualified_path=qualified_path,
        constants_path=constants_path,
        output_path=output_path,
        overwrite=True,
    )

    assert summary.rows == 6
    assert summary.rows_by_year == {2025: 3, 2026: 3}
    output = pl.read_csv(output_path)
    assert output.select(EXISTING_SEASON_COLUMNS).equals(prior)
    assert output[SD_XWOBA_COLUMN].is_finite().all()
    assert output[SD_XWOBA_CONTACT_COLUMN].is_finite().all()
    assert not output_path.with_suffix(".csv.part").exists()


def test_calculate_all_seasons_requires_overwrite(tmp_path: Path) -> None:
    with pytest.raises(SeasonOutputExistsError, match="pass --overwrite"):
        calculate_all_seasons(output_path=tmp_path / "seasons.csv")


def test_parse_args_supports_season_controls() -> None:
    args = parse_args(
        [
            "--calculated-directory",
            "calculated",
            "--qualified",
            "qualified.csv",
            "--constants",
            "constants.csv",
            "--output",
            "seasons.csv",
            "--overwrite",
        ]
    )

    assert args.calculated_directory == Path("calculated")
    assert args.qualified == Path("qualified.csv")
    assert args.constants == Path("constants.csv")
    assert args.output == Path("seasons.csv")
    assert args.overwrite is True
