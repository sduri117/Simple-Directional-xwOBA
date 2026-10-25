from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import polars as pl
import pytest

from analyze_sd_results import (
    RESULT_COLUMNS,
    AnalysisError,
    AnalysisOutputExistsError,
    BattedBallWeights,
    analyze,
    build_stage_three_results,
    load_batted_ball_analysis,
    load_existing_results,
    load_season_analysis,
    matched_player_seasons,
    parse_args,
    pearson_statistics,
)


def season_frame() -> pl.DataFrame:
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
            "sd-xwOBA": [0.28, 0.37, 0.31, 0.34, 0.29, 0.26],
            "sd-xwOBAcon": [0.39, 0.50, 0.41, 0.45, 0.40, 0.36],
        }
    )


def batted_ball_frames() -> dict[int, pl.DataFrame]:
    return {
        2025: pl.DataFrame(
            {
                "game_date": ["2025-04-01", "2025-04-02", "2025-04-03"],
                "total_bases": [0, 1, 4],
                "xwOBA-MLB": [0.1, 0.6, 0.9],
                "xwOBA-local": [0.2, 0.5, 0.8],
                "sd-xwOBA": [0.15, 0.55, 0.85],
                "realized_woba": [0.0, 0.9, 2.0],
                "year": [2025, 2025, 2025],
            }
        ),
        2026: pl.DataFrame(
            {
                "game_date": ["2026-04-01", "2026-04-02", "2026-04-03"],
                "total_bases": [0, 2, 1],
                "xwOBA-MLB": [0.2, 0.7, 0.8],
                "xwOBA-local": [0.1, 0.6, 0.9],
                "sd-xwOBA": [0.12, 0.65, 0.88],
                "realized_woba": [0.0, 1.31, 0.91],
                "year": [2026, 2026, 2026],
            }
        ),
    }


def existing_results() -> pl.DataFrame:
    records = [
        {
            "analysis_level": "batted_ball",
            "comparison": "xwOBA-local vs xwOBA-MLB",
            "cohort": "2025",
            "x_variable": "xwOBA-local",
            "y_variable": "xwOBA-MLB",
            "n": 3,
            "pearson_r": 0.9,
            "r_squared": 0.81,
        },
        {
            "analysis_level": "year_to_year",
            "comparison": "2025 local xwOBA vs 2026 wOBA",
            "cohort": "qualified in 2025 and 2026",
            "x_variable": "2025 xwOBA-local",
            "y_variable": "2026 wOBA",
            "n": 2,
            "pearson_r": 0.5,
            "r_squared": 0.25,
        },
    ]
    return pl.DataFrame(records).select(RESULT_COLUMNS)


def write_analysis_inputs(directory: Path) -> tuple[Path, Path, Path, Path]:
    calculated = directory / "calculated"
    calculated.mkdir()
    for year, frame in batted_ball_frames().items():
        frame.drop("year", "realized_woba").write_csv(
            calculated / f"{year}_all_BBE_with_calculated_xwOBAs.csv"
        )
    season_path = directory / "seasons.csv"
    season_frame().write_csv(season_path)
    constants_path = directory / "constants.csv"
    pl.DataFrame(
        {
            "Season": [2025, 2026],
            "w1B": [0.90, 0.91],
            "w2B": [1.30, 1.31],
            "w3B": [1.60, 1.61],
            "wHR": [2.00, 2.01],
        }
    ).write_csv(constants_path)
    results_path = directory / "correlations.csv"
    existing_results().write_csv(results_path)
    return calculated, season_path, constants_path, results_path


def test_pearson_statistics_reports_signed_r_and_square() -> None:
    result = pearson_statistics([1.0, 2.0, 3.0], [6.0, 4.0, 2.0])

    assert result.n == 3
    assert result.pearson_r == pytest.approx(-1.0)
    assert result.r_squared == pytest.approx(1.0)


def test_pearson_statistics_rejects_constant_input() -> None:
    with pytest.raises(AnalysisError, match="constant"):
        pearson_statistics([1.0, 1.0], [2.0, 3.0])


def test_load_existing_results_removes_prior_stage_three_rows(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    prior = existing_results()
    stage_three = prior.head(1).with_columns(pl.lit("sd-xwOBA").alias("x_variable"))
    pl.concat([prior, stage_three]).write_csv(path)

    loaded = load_existing_results(path)

    assert loaded.equals(prior)


def test_batted_ball_and_season_loaders_validate_inputs(tmp_path: Path) -> None:
    batted_path = tmp_path / "batted.csv"
    batted_ball_frames()[2025].drop("year", "realized_woba").write_csv(batted_path)
    seasons_path = tmp_path / "seasons.csv"
    season_frame().write_csv(seasons_path)

    batted = load_batted_ball_analysis(
        batted_path, 2025, BattedBallWeights(2025, 0.9, 1.3, 1.6, 2.0)
    )
    seasons = load_season_analysis(seasons_path)

    assert batted["realized_woba"].to_list() == [0.0, 0.9, 2.0]
    assert seasons.height == 6


def test_matched_player_seasons_uses_batter_id() -> None:
    matched = matched_player_seasons(season_frame())

    assert matched["batter_id"].to_list() == [10, 20]
    assert matched.height == 2


def test_build_stage_three_results_creates_all_17_rows() -> None:
    results, matched = build_stage_three_results(batted_ball_frames(), season_frame())

    assert results.columns == list(RESULT_COLUMNS)
    assert results.height == 17
    assert matched.height == 2
    assert results.group_by("analysis_level").len().sort(
        "analysis_level"
    ).to_dicts() == [
        {"analysis_level": "batted_ball", "len": 6},
        {"analysis_level": "batted_ball_contact_value", "len": 3},
        {"analysis_level": "player_season", "len": 6},
        {"analysis_level": "year_to_year", "len": 1},
        {"analysis_level": "year_to_year_contact", "len": 1},
    ]
    np.testing.assert_allclose(
        results["r_squared"].to_numpy(), results["pearson_r"].to_numpy() ** 2
    )


def test_analyze_preserves_results_and_writes_four_plots(tmp_path: Path) -> None:
    calculated, season_path, constants_path, results_path = write_analysis_inputs(
        tmp_path
    )
    prior = pl.read_csv(results_path)
    plots = tuple(tmp_path / f"plot_{index}.png" for index in range(4))

    summary = analyze(
        calculated_directory=calculated,
        season_path=season_path,
        constants_path=constants_path,
        results_path=results_path,
        sd_vs_mlb_path=plots[0],
        sd_vs_local_path=plots[1],
        sd_year_ahead_path=plots[2],
        sd_contact_year_ahead_path=plots[3],
        overwrite=True,
    )

    assert summary.existing_result_rows == 2
    assert summary.stage_three_result_rows == 17
    assert summary.matched_players == 2
    output = pl.read_csv(results_path)
    assert output.height == 19
    assert output.head(2).equals(prior)
    for plot in plots:
        assert plot.is_file()
        assert min(mpimg.imread(plot).shape[:2]) >= 100
        assert not plot.with_name(f"{plot.stem}.part{plot.suffix}").exists()


def test_analyze_requires_overwrite_for_existing_artifact(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    existing_results().write_csv(results_path)

    with pytest.raises(AnalysisOutputExistsError, match="replacement"):
        analyze(results_path=results_path)


def test_parse_args_supports_analysis_controls() -> None:
    args = parse_args(
        [
            "--calculated-directory",
            "calculated",
            "--season-data",
            "seasons.csv",
            "--constants",
            "constants.csv",
            "--results",
            "results.csv",
            "--sd-vs-mlb-plot",
            "one.png",
            "--sd-vs-local-plot",
            "two.png",
            "--sd-year-ahead-plot",
            "three.png",
            "--sd-contact-year-ahead-plot",
            "four.png",
            "--overwrite",
        ]
    )

    assert args.calculated_directory == Path("calculated")
    assert args.season_data == Path("seasons.csv")
    assert args.constants == Path("constants.csv")
    assert args.results == Path("results.csv")
    assert args.sd_vs_mlb_plot == Path("one.png")
    assert args.sd_vs_local_plot == Path("two.png")
    assert args.sd_year_ahead_plot == Path("three.png")
    assert args.sd_contact_year_ahead_plot == Path("four.png")
    assert args.overwrite is True
