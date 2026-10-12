import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl
import pytest

from data_preprocessing import (
    DERIVED_COLUMNS,
    DIRECTIONS,
    EVENT_TO_TOTAL_BASES,
    MONTH_SLUGS,
    OUTPUT_COLUMNS,
    REQUIRED_COLUMNS,
    InputInventoryError,
    MonthlyDataFile,
    OutputExistsError,
    PreprocessingError,
    discover_year_files,
    expected_input_files,
    main,
    preprocess_year,
    transform_monthly_file,
)

CSV_COLUMNS = REQUIRED_COLUMNS + ("unused_column",)


def row(
    *,
    game_date: object = "2025-04-01",
    player_name: object = "Hitter, Test",
    batter: object = 123,
    events: object = "single",
    launch_speed: object = 100.1,
    launch_angle: object = 20.0,
    unused_column: object = "unused",
) -> dict[str, object]:
    return {
        "game_date": game_date,
        "player_name": player_name,
        "batter": batter,
        "events": events,
        "launch_speed": launch_speed,
        "launch_angle": launch_angle,
        "unused_column": unused_column,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def populate_year(input_directory: Path, year: int) -> None:
    for index, data_file in enumerate(expected_input_files(input_directory, year)):
        write_csv(
            data_file.path,
            [
                row(
                    game_date="2025-04-01",
                    player_name=f"Player {index}",
                    batter=index,
                )
            ],
        )


def test_output_column_contract() -> None:
    assert REQUIRED_COLUMNS == (
        "game_date",
        "player_name",
        "batter",
        "events",
        "launch_speed",
        "launch_angle",
    )
    assert DERIVED_COLUMNS == (
        "hit_direction",
        "simplified_events",
        "total_bases",
    )
    assert OUTPUT_COLUMNS == REQUIRED_COLUMNS + DERIVED_COLUMNS


def test_total_base_mapping() -> None:
    assert EVENT_TO_TOTAL_BASES == {
        "out": 0,
        "single": 1,
        "double": 2,
        "triple": 3,
        "home_run": 4,
    }


def test_expected_input_files_have_deterministic_month_direction_order(
    tmp_path: Path,
) -> None:
    files = expected_input_files(tmp_path, 2025)

    assert len(files) == 18
    assert [(item.month_slug, item.direction) for item in files] == [
        (month, direction) for month in MONTH_SLUGS for direction in DIRECTIONS
    ]
    assert files[0].path == tmp_path / "2025_mar-apr_pull_BBE.csv"
    assert files[-1].path == tmp_path / "2025_sep-oct_opposite_BBE.csv"


def test_expected_input_files_reject_unsupported_year(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported year: 2021"):
        expected_input_files(tmp_path, 2021)


def test_discover_year_files_requires_complete_inventory(tmp_path: Path) -> None:
    populate_year(tmp_path, 2025)
    missing_path = tmp_path / "2025_jul_straightaway_BBE.csv"
    missing_path.unlink()

    with pytest.raises(InputInventoryError, match=missing_path.name):
        discover_year_files(tmp_path, 2025)


def test_transform_monthly_file_maps_events_and_direction(tmp_path: Path) -> None:
    input_path = tmp_path / "2025_may_opposite_BBE.csv"
    events = [
        "field_out",
        "single",
        "double",
        "triple",
        "home_run",
        "grounded_into_double_play",
    ]
    write_csv(input_path, [row(events=event) for event in events])
    data_file = MonthlyDataFile(2025, "may", "opposite", input_path)

    result, input_rows = transform_monthly_file(data_file)

    assert input_rows == len(events)
    assert result.columns == list(OUTPUT_COLUMNS)
    assert result["events"].to_list() == events
    assert result["hit_direction"].to_list() == ["opposite"] * len(events)
    assert result["simplified_events"].to_list() == [
        "out",
        "single",
        "double",
        "triple",
        "home_run",
        "out",
    ]
    assert result["total_bases"].to_list() == [0, 1, 2, 3, 4, 0]
    assert result.schema["total_bases"] == pl.Int8


def test_transform_monthly_file_removes_only_missing_required_values(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "2025_jun_pull_BBE.csv"
    missing_required_rows = [
        row(game_date=""),
        row(player_name=""),
        row(batter=""),
        row(events=""),
        row(launch_speed=""),
        row(launch_angle=""),
        row(launch_speed="NaN"),
        row(launch_angle="NaN"),
    ]
    write_csv(input_path, [row(unused_column=""), *missing_required_rows])
    data_file = MonthlyDataFile(2025, "jun", "pull", input_path)

    result, input_rows = transform_monthly_file(data_file)

    assert input_rows == 9
    assert result.height == 1
    assert result["hit_direction"].item() == "pull"
    assert "unused_column" not in result.columns


def test_transform_monthly_file_rejects_missing_required_column(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "2025_jun_pull_BBE.csv"
    input_path.write_text(
        "game_date,player_name,batter,events,launch_speed\n"
        "2025-06-01,Player One,123,single,100.1\n",
        encoding="utf-8",
    )
    data_file = MonthlyDataFile(2025, "jun", "pull", input_path)

    with pytest.raises(PreprocessingError, match="Could not read"):
        transform_monthly_file(data_file)


def test_preprocess_year_combines_inputs_and_writes_expected_output(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "raw"
    output_directory = tmp_path / "processed"
    populate_year(input_directory, 2025)

    summary = preprocess_year(input_directory, output_directory, 2025)

    assert summary.year == 2025
    assert summary.input_files == 18
    assert summary.input_rows == 18
    assert summary.output_rows == 18
    assert summary.removed_rows == 0
    assert summary.output_path == output_directory / "2025_all_BBE.csv"
    assert summary.output_path.is_file()
    assert not (output_directory / "2025_all_BBE.csv.part").exists()

    result = pl.read_csv(summary.output_path)
    assert result.columns == list(OUTPUT_COLUMNS)
    assert result["player_name"].to_list() == [f"Player {index}" for index in range(18)]
    assert result["hit_direction"].to_list() == list(DIRECTIONS) * len(MONTH_SLUGS)


def test_preprocess_year_reports_removed_rows(tmp_path: Path) -> None:
    input_directory = tmp_path / "raw"
    output_directory = tmp_path / "processed"
    populate_year(input_directory, 2025)
    first_file = expected_input_files(input_directory, 2025)[0]
    write_csv(first_file.path, [row(), row(events="")])

    summary = preprocess_year(input_directory, output_directory, 2025)

    assert summary.input_rows == 19
    assert summary.output_rows == 18
    assert summary.removed_rows == 1


def test_preprocess_year_requires_overwrite_for_existing_output(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "raw"
    output_directory = tmp_path / "processed"
    populate_year(input_directory, 2025)
    preprocess_year(input_directory, output_directory, 2025)

    with pytest.raises(OutputExistsError, match="--overwrite"):
        preprocess_year(input_directory, output_directory, 2025)

    summary = preprocess_year(
        input_directory,
        output_directory,
        2025,
        overwrite=True,
    )
    assert summary.output_rows == 18


def test_main_returns_error_for_incomplete_inventory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "--year",
            "2025",
            "--input-directory",
            str(tmp_path / "missing"),
            "--output-directory",
            str(tmp_path / "processed"),
        ]
    )

    assert exit_code == 1
    assert "Missing 18 input file(s) for 2025" in capsys.readouterr().out
