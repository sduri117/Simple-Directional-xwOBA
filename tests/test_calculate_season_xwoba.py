from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from calculate_season_xwoba import (
    SEASON_OUTPUT_COLUMNS,
    SeasonCalculationError,
    SeasonOutputExistsError,
    WobaWeights,
    aggregate_batted_ball_xwoba,
    calculate_all_seasons,
    calculate_season_frame,
    load_qualified_seasons,
    load_woba_weights,
    parse_args,
    validate_season_output,
)
from experiment_config import XWOBA_LOCAL_COLUMN


def qualified_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Season": [2025, 2025, 2026, 2026],
            "Name": ["Alpha", "Beta", "Alpha", "Gamma"],
            "AB": [4, 5, 6, 7],
            "BB": [2, 1, 2, 1],
            "IBB": [1, 0, 0, 0],
            "HBP": [1, 0, 1, 0],
            "SF": [1, 1, 0, 1],
            "wOBA": [0.31, 0.32, 0.33, 0.34],
            "xwOBA": [0.30, 0.31, 0.32, 0.33],
            "MLBAMID": [10, 20, 10, 30],
        }
    )


def write_qualified(path: Path, frame: pl.DataFrame | None = None) -> None:
    (frame if frame is not None else qualified_frame()).write_csv(path)


def write_constants(path: Path) -> None:
    pl.DataFrame(
        {
            "Season": [2025, 2026],
            "wBB": [0.70, 0.71],
            "wHBP": [0.72, 0.73],
            "w1B": [0.90, 0.91],
            "w2B": [1.30, 1.31],
            "w3B": [1.60, 1.61],
            "wHR": [2.00, 2.01],
        }
    ).write_csv(path)


def write_calculated_batted_balls(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "game_date": ["2025-04-01", "2025-04-02", "2025-04-03"],
            "batter": [10, 10, 20],
            "total_bases": [1, 0, 4],
            "xwOBA-MLB": [0.3, 0.2, 0.9],
            XWOBA_LOCAL_COLUMN: [0.2, 0.4, 0.8],
        }
    ).write_csv(directory / "2025_all_BBE_with_calculated_xwOBAs.csv")
    pl.DataFrame(
        {
            "game_date": ["2026-04-01", "2026-04-02", "2026-04-03"],
            "batter": [10, 30, 30],
            "total_bases": [2, 0, 1],
            "xwOBA-MLB": [0.6, 0.5, 0.8],
            XWOBA_LOCAL_COLUMN: [0.5, 0.6, 0.7],
        }
    ).write_csv(directory / "2026_all_BBE_with_calculated_xwOBAs.csv")


def test_load_qualified_seasons_validates_contract(tmp_path: Path) -> None:
    path = tmp_path / "qualified.csv"
    write_qualified(path)

    result = load_qualified_seasons(path)

    assert result.height == 4
    assert result["Season"].to_list() == [2025, 2025, 2026, 2026]


def test_load_qualified_seasons_rejects_duplicate_key(tmp_path: Path) -> None:
    path = tmp_path / "qualified.csv"
    frame = pl.concat([qualified_frame(), qualified_frame().head(1)])
    write_qualified(path, frame)

    with pytest.raises(SeasonCalculationError, match="duplicate"):
        load_qualified_seasons(path)


def test_load_qualified_seasons_rejects_ibb_greater_than_bb(tmp_path: Path) -> None:
    path = tmp_path / "qualified.csv"
    frame = qualified_frame().with_columns(
        pl.when(pl.col("MLBAMID") == 20).then(2).otherwise(pl.col("IBB")).alias("IBB")
    )
    write_qualified(path, frame)

    with pytest.raises(SeasonCalculationError, match="IBB greater than BB"):
        load_qualified_seasons(path)


def test_load_woba_weights(tmp_path: Path) -> None:
    path = tmp_path / "constants.csv"
    write_constants(path)

    result = load_woba_weights(path)

    assert result == {
        2025: WobaWeights(2025, 0.70, 0.72, 0.90, 1.30, 1.60, 2.00),
        2026: WobaWeights(2026, 0.71, 0.73, 0.91, 1.31, 1.61, 2.01),
    }


def test_aggregate_batted_ball_xwoba_sums_by_batter(tmp_path: Path) -> None:
    write_calculated_batted_balls(tmp_path)

    result = aggregate_batted_ball_xwoba(
        tmp_path / "2025_all_BBE_with_calculated_xwOBAs.csv",
        2025,
        WobaWeights(2025, 0.70, 0.72, 0.90, 1.30, 1.60, 2.00),
    ).sort("batter_id")

    assert result["batter_id"].to_list() == [10, 20]
    assert result["batted_ball_count"].to_list() == [2, 1]
    np.testing.assert_allclose(result["batted_ball_xwoba_sum"].to_numpy(), [0.6, 0.8])
    np.testing.assert_allclose(result["wOBAcon"].to_numpy(), [0.45, 2.0])
    np.testing.assert_allclose(result["xwOBAcon-MLB"].to_numpy(), [0.25, 0.9])
    np.testing.assert_allclose(result["xwOBAcon-local"].to_numpy(), [0.3, 0.8])


def test_aggregate_batted_ball_xwoba_validates_year(tmp_path: Path) -> None:
    write_calculated_batted_balls(tmp_path)

    with pytest.raises(SeasonCalculationError, match="contains years"):
        aggregate_batted_ball_xwoba(
            tmp_path / "2025_all_BBE_with_calculated_xwOBAs.csv",
            2026,
            WobaWeights(2026, 0.71, 0.73, 0.91, 1.31, 1.61, 2.01),
        )


def test_calculate_season_frame_applies_formula() -> None:
    qualified = qualified_frame()
    aggregates = pl.DataFrame(
        {
            "year": [2025, 2025, 2026, 2026],
            "batter_id": [10, 20, 10, 30],
            "batted_ball_count": [2, 1, 1, 2],
            "batted_ball_xwoba_sum": [0.6, 0.8, 0.5, 1.3],
            "batted_ball_mlb_xwoba_sum": [0.5, 0.9, 0.6, 1.3],
            "batted_ball_woba_sum": [0.9, 2.0, 1.31, 0.91],
            "wOBAcon": [0.45, 2.0, 1.31, 0.455],
            "xwOBAcon-MLB": [0.25, 0.9, 0.6, 0.65],
            "xwOBAcon-local": [0.3, 0.8, 0.5, 0.65],
        },
        schema={
            "year": pl.Int64,
            "batter_id": pl.Int64,
            "batted_ball_count": pl.Int64,
            "batted_ball_xwoba_sum": pl.Float64,
            "batted_ball_mlb_xwoba_sum": pl.Float64,
            "batted_ball_woba_sum": pl.Float64,
            "wOBAcon": pl.Float64,
            "xwOBAcon-MLB": pl.Float64,
            "xwOBAcon-local": pl.Float64,
        },
    )
    weights = {
        2025: WobaWeights(2025, 0.70, 0.72, 0.90, 1.30, 1.60, 2.00),
        2026: WobaWeights(2026, 0.71, 0.73, 0.91, 1.31, 1.61, 2.01),
    }

    output, reconciliation = calculate_season_frame(qualified, aggregates, weights)

    assert output.columns == list(SEASON_OUTPUT_COLUMNS)
    alpha_2025 = output.filter((pl.col("year") == 2025) & (pl.col("batter_id") == 10))[
        XWOBA_LOCAL_COLUMN
    ].item()
    assert alpha_2025 == pytest.approx((0.70 * 1 + 0.72 * 1 + 0.6) / 7)
    assert (
        reconciliation.filter((pl.col("year") == 2025) & (pl.col("batter_id") == 10))[
            "denominator"
        ].item()
        == 7
    )


def test_calculate_season_frame_requires_batted_ball_match() -> None:
    aggregates = pl.DataFrame(
        {
            "year": [2025, 2025, 2026],
            "batter_id": [10, 20, 10],
            "batted_ball_count": [2, 1, 1],
            "batted_ball_xwoba_sum": [0.6, 0.8, 0.5],
            "batted_ball_mlb_xwoba_sum": [0.5, 0.9, 0.6],
            "batted_ball_woba_sum": [0.9, 2.0, 1.31],
            "wOBAcon": [0.45, 2.0, 1.31],
            "xwOBAcon-MLB": [0.25, 0.9, 0.6],
            "xwOBAcon-local": [0.3, 0.8, 0.5],
        }
    )
    weights = {
        2025: WobaWeights(2025, 0.70, 0.72, 0.90, 1.30, 1.60, 2.00),
        2026: WobaWeights(2026, 0.71, 0.73, 0.91, 1.31, 1.61, 2.01),
    }

    with pytest.raises(SeasonCalculationError, match="lack batted-ball matches"):
        calculate_season_frame(qualified_frame(), aggregates, weights)


def test_calculate_all_seasons_writes_expected_output(tmp_path: Path) -> None:
    calculated_directory = tmp_path / "calculated"
    qualified_path = tmp_path / "qualified.csv"
    constants_path = tmp_path / "constants.csv"
    output_path = tmp_path / "output" / "season.csv"
    write_calculated_batted_balls(calculated_directory)
    write_qualified(qualified_path)
    write_constants(constants_path)

    summary = calculate_all_seasons(
        calculated_directory=calculated_directory,
        qualified_path=qualified_path,
        constants_path=constants_path,
        output_path=output_path,
    )

    assert summary.rows == 4
    assert summary.rows_by_year == {2025: 2, 2026: 2}
    assert output_path.is_file()
    assert not output_path.with_suffix(".csv.part").exists()
    output = pl.read_csv(output_path)
    assert output.columns == list(SEASON_OUTPUT_COLUMNS)
    assert output["batter_name"].to_list() == ["Alpha", "Beta", "Alpha", "Gamma"]


def test_calculate_all_seasons_requires_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / "season.csv"
    output_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(SeasonOutputExistsError, match="replacement"):
        calculate_all_seasons(output_path=output_path)


def test_validate_season_output_detects_changed_value(tmp_path: Path) -> None:
    expected = pl.DataFrame(
        {
            "year": [2025],
            "batter_name": ["Alpha"],
            "batter_id": [10],
            "wOBA": [0.31],
            "xwOBA-MLB": [0.30],
            XWOBA_LOCAL_COLUMN: [0.29],
            "wOBAcon": [0.45],
            "xwOBAcon-MLB": [0.25],
            "xwOBAcon-local": [0.30],
        }
    )
    reconciliation = pl.DataFrame(
        {
            "wBB": [0.7],
            "uBB": [1],
            "wHBP": [0.72],
            "HBP": [1],
            "batted_ball_xwoba_sum": [0.61],
            "batted_ball_mlb_xwoba_sum": [0.5],
            "batted_ball_woba_sum": [0.9],
            "batted_ball_count": [2],
            "denominator": [7],
        }
    )
    path = tmp_path / "season.csv"
    expected.with_columns(pl.lit(0.5).alias(XWOBA_LOCAL_COLUMN)).write_csv(path)

    with pytest.raises(SeasonCalculationError, match="changed from expectation"):
        validate_season_output(path, expected, reconciliation)


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
            "season.csv",
            "--overwrite",
        ]
    )

    assert args.calculated_directory == Path("calculated")
    assert args.qualified == Path("qualified.csv")
    assert args.constants == Path("constants.csv")
    assert args.output == Path("season.csv")
    assert args.overwrite is True
