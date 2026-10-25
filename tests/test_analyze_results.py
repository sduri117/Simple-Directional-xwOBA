from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import polars as pl
import pytest

from analyze_results import (
    RESULT_COLUMNS,
    AnalysisError,
    AnalysisOutputExistsError,
    analyze,
    build_correlation_results,
    load_batted_ball_analysis,
    load_season_analysis,
    parse_args,
    pearson_statistics,
)
from calculate_season_xwoba import WobaWeights


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


def batted_ball_frames() -> dict[int, pl.DataFrame]:
    return {
        2025: pl.DataFrame(
            {
                "game_date": ["2025-04-01", "2025-04-02", "2025-04-03"],
                "total_bases": [0, 1, 4],
                "xwOBA-MLB": [0.1, 0.6, 0.9],
                "xwOBA-local": [0.2, 0.5, 0.8],
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
                "realized_woba": [0.0, 1.31, 0.91],
                "year": [2026, 2026, 2026],
            }
        ),
    }


def write_analysis_inputs(directory: Path) -> tuple[Path, Path, Path]:
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
            "wBB": [0.70, 0.71],
            "wHBP": [0.72, 0.73],
            "w1B": [0.90, 0.91],
            "w2B": [1.30, 1.31],
            "w3B": [1.60, 1.61],
            "wHR": [2.00, 2.01],
        }
    ).write_csv(constants_path)
    return calculated, season_path, constants_path


def test_pearson_statistics_squares_signed_correlation() -> None:
    result = pearson_statistics([1.0, 2.0, 3.0], [6.0, 4.0, 2.0])

    assert result.n == 3
    assert result.pearson_r == pytest.approx(-1.0)
    assert result.r_squared == pytest.approx(1.0)


def test_pearson_statistics_matches_numpy() -> None:
    x = np.array([0.1, 0.4, 0.7, 1.1])
    y = np.array([0.3, 0.2, 0.9, 0.8])

    result = pearson_statistics(x, y)
    expected_r = np.corrcoef(x, y)[0, 1]

    assert result.pearson_r == pytest.approx(expected_r)
    assert result.r_squared == pytest.approx(expected_r**2)


def test_pearson_statistics_rejects_constant_input() -> None:
    with pytest.raises(AnalysisError, match="constant"):
        pearson_statistics([1.0, 1.0], [2.0, 3.0])


def test_load_batted_ball_analysis_validates_year(tmp_path: Path) -> None:
    path = tmp_path / "batted.csv"
    pl.DataFrame(
        {
            "game_date": ["2024-04-01", "2024-04-02"],
            "total_bases": [0, 1],
            "xwOBA-MLB": [0.1, 0.2],
            "xwOBA-local": [0.2, 0.3],
        }
    ).write_csv(path)

    with pytest.raises(AnalysisError, match="contains years"):
        load_batted_ball_analysis(
            path, 2025, WobaWeights(2025, 0.70, 0.72, 0.90, 1.30, 1.60, 2.00)
        )


def test_load_season_analysis_rejects_duplicate_key(tmp_path: Path) -> None:
    path = tmp_path / "seasons.csv"
    pl.concat([season_frame(), season_frame().head(1)]).write_csv(path)

    with pytest.raises(AnalysisError, match="duplicate"):
        load_season_analysis(path)


def test_build_correlation_results_creates_all_results() -> None:
    results, matched = build_correlation_results(batted_ball_frames(), season_frame())

    assert results.columns == list(RESULT_COLUMNS)
    assert results.height == 18
    assert matched["batter_id"].to_list() == [10, 20]
    assert results.group_by("analysis_level").len().sort(
        "analysis_level"
    ).to_dicts() == [
        {"analysis_level": "batted_ball", "len": 3},
        {"analysis_level": "batted_ball_contact_value", "len": 6},
        {"analysis_level": "player_season", "len": 3},
        {"analysis_level": "year_to_year", "len": 2},
        {"analysis_level": "year_to_year_contact", "len": 2},
        {"analysis_level": "year_to_year_realized", "len": 2},
    ]
    np.testing.assert_allclose(
        results["r_squared"].to_numpy(),
        results["pearson_r"].to_numpy() ** 2,
    )


def test_analyze_writes_validated_results_and_plots(tmp_path: Path) -> None:
    calculated, season_path, constants_path = write_analysis_inputs(tmp_path)
    output_directory = tmp_path / "outputs"
    results_path = output_directory / "correlations.csv"
    plots = tuple(output_directory / f"plot_{index}.png" for index in range(3))

    summary = analyze(
        calculated_directory=calculated,
        season_path=season_path,
        constants_path=constants_path,
        results_path=results_path,
        player_season_plot_path=plots[0],
        mlb_year_ahead_plot_path=plots[1],
        local_year_ahead_plot_path=plots[2],
    )

    assert summary.result_rows == 18
    assert summary.matched_players == 2
    results = pl.read_csv(results_path)
    assert results.columns == list(RESULT_COLUMNS)
    np.testing.assert_allclose(
        results["r_squared"].to_numpy(), results["pearson_r"].to_numpy() ** 2
    )
    for plot in plots:
        assert plot.is_file()
        assert min(mpimg.imread(plot).shape[:2]) >= 100
        assert not plot.with_name(f"{plot.stem}.part{plot.suffix}").exists()
    assert not results_path.with_name(f"{results_path.name}.part").exists()


def test_analyze_requires_overwrite_for_existing_output(tmp_path: Path) -> None:
    results_path = tmp_path / "correlations.csv"
    results_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(AnalysisOutputExistsError, match="replacement"):
        analyze(results_path=results_path)


def test_parse_args_supports_analysis_controls() -> None:
    args = parse_args(
        [
            "--calculated-directory",
            "calculated",
            "--season-data",
            "seasons.csv",
            "--results",
            "results.csv",
            "--constants",
            "constants.csv",
            "--player-season-plot",
            "one.png",
            "--mlb-year-ahead-plot",
            "two.png",
            "--local-year-ahead-plot",
            "three.png",
            "--overwrite",
        ]
    )

    assert args.calculated_directory == Path("calculated")
    assert args.season_data == Path("seasons.csv")
    assert args.results == Path("results.csv")
    assert args.constants == Path("constants.csv")
    assert args.player_season_plot == Path("one.png")
    assert args.mlb_year_ahead_plot == Path("two.png")
    assert args.local_year_ahead_plot == Path("three.png")
    assert args.overwrite is True
