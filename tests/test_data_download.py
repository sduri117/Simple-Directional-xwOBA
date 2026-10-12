from __future__ import annotations

import csv
from collections.abc import Iterable
from io import StringIO
from pathlib import Path

import httpx
import pytest

import data_download
from data_download import (
    DIRECTIONS,
    MONTH_SLUGS,
    YEARS,
    DataValidationError,
    DownloadError,
    DownloadRequest,
    build_download_requests,
    download_all,
    download_one,
    parse_args,
    validate_csv,
)
from data_preprocessing import REQUIRED_COLUMNS


def csv_bytes(rows: Iterable[Iterable[object]]) -> bytes:
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(REQUIRED_COLUMNS)
    writer.writerows(rows)
    return output.getvalue().encode()


def valid_row(game_date: str = "2025-03-29") -> tuple[object, ...]:
    return (game_date, "Judge, Aaron", 592450, "home_run", 107.3, 28, 1.82)


def test_download_dimensions_cover_the_stage_scope() -> None:
    assert YEARS == (2022, 2023, 2024, 2025, 2026)
    assert MONTH_SLUGS == {
        4: "mar-apr",
        5: "may",
        6: "jun",
        7: "jul",
        8: "aug",
        9: "sep-oct",
    }
    assert DIRECTIONS == ("pull", "straightaway", "opposite")
    assert len(build_download_requests()) == 90


def test_download_request_uses_canonical_filename_and_query() -> None:
    request = DownloadRequest(year=2025, month=4, direction="straightaway")

    assert request.filename == "2025_mar-apr_straightaway_BBE.csv"
    assert request.query_parameters() == {
        "all": "true",
        "type": "details",
        "hfGT": "R|",
        "hfPR": r"hit\.\.into\.\.play|",
        "hfPull": "Straightaway|",
        "hfSea": "2025|",
        "hfMo": "4|",
        "player_type": "batter",
        "group_by": "name",
        "sort_col": "pitches",
        "sort_order": "desc",
        "min_pitches": "0",
        "min_results": "0",
        "min_pas": "0",
    }


@pytest.mark.parametrize(
    ("year", "month", "direction"),
    [(2021, 4, "pull"), (2025, 3, "pull"), (2025, 4, "center")],
)
def test_download_request_rejects_unsupported_dimensions(
    year: int,
    month: int,
    direction: str,
) -> None:
    with pytest.raises(ValueError):
        DownloadRequest(year=year, month=month, direction=direction)


def test_validate_csv_accepts_dates_in_combined_month_buckets(tmp_path: Path) -> None:
    request = DownloadRequest(2025, 4, "pull")
    path = tmp_path / request.filename
    path.write_bytes(csv_bytes([valid_row("2025-03-29"), valid_row("2025-04-01")]))

    assert validate_csv(path, request) == 2


def test_validate_csv_rejects_missing_required_columns(tmp_path: Path) -> None:
    request = DownloadRequest(2025, 4, "pull")
    path = tmp_path / request.filename
    path.write_text("game_date,events\n2025-04-01,single\n")

    with pytest.raises(DataValidationError, match="Could not read"):
        validate_csv(path, request)


def test_validate_csv_rejects_dates_outside_request(tmp_path: Path) -> None:
    request = DownloadRequest(2025, 5, "pull")
    path = tmp_path / request.filename
    path.write_bytes(csv_bytes([valid_row("2025-06-01")]))

    with pytest.raises(DataValidationError, match="outside the may bucket"):
        validate_csv(path, request)


def test_validate_csv_rejects_an_empty_download(tmp_path: Path) -> None:
    request = DownloadRequest(2025, 4, "pull")
    path = tmp_path / request.filename
    path.write_bytes(csv_bytes([]))

    with pytest.raises(DataValidationError, match="contains no data rows"):
        validate_csv(path, request)


def test_validate_csv_rejects_the_suspected_row_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = DownloadRequest(2025, 4, "pull")
    path = tmp_path / request.filename
    path.write_bytes(csv_bytes([valid_row(), valid_row()]))
    monkeypatch.setattr(data_download, "SUSPECTED_TRUNCATION_ROW_COUNT", 2)

    with pytest.raises(DataValidationError, match="possible truncation"):
        validate_csv(path, request)


def test_validate_csv_allows_missing_analysis_values(tmp_path: Path) -> None:
    request = DownloadRequest(2025, 4, "pull")
    path = tmp_path / request.filename
    row_with_missing_launch_speed = (
        "2025-04-01",
        "Judge, Aaron",
        592450,
        "field_out",
        None,
        28,
        0.12,
    )
    path.write_bytes(csv_bytes([row_with_missing_launch_speed]))

    assert validate_csv(path, request) == 1


def test_download_one_writes_validated_file_atomically(tmp_path: Path) -> None:
    captured_parameters: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_parameters.update(dict(request.url.params))
        return httpx.Response(200, content=csv_bytes([valid_row()]), request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    request = DownloadRequest(2025, 4, "pull")

    result = download_one(client, request, tmp_path, sleep=lambda _: None)

    assert result.downloaded is True
    assert result.row_count == 1
    assert result.path == tmp_path / request.filename
    assert result.path.exists()
    assert not result.path.with_suffix(".csv.part").exists()
    assert captured_parameters == request.query_parameters()


def test_download_one_skips_valid_existing_file(tmp_path: Path) -> None:
    request = DownloadRequest(2025, 4, "pull")
    destination = tmp_path / request.filename
    destination.write_bytes(csv_bytes([valid_row()]))

    def unexpected_request(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"Unexpected HTTP request: {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(unexpected_request))

    result = download_one(client, request, tmp_path)

    assert result.downloaded is False
    assert result.row_count == 1


def test_download_one_retries_server_error(tmp_path: Path) -> None:
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(200, content=csv_bytes([valid_row()]), request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    result = download_one(
        client,
        DownloadRequest(2025, 4, "pull"),
        tmp_path,
        retry_delay_seconds=0.25,
        sleep=delays.append,
    )

    assert result.downloaded is True
    assert attempts == 2
    assert delays == [0.25]


def test_download_one_removes_partial_file_after_final_failure(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    request = DownloadRequest(2025, 4, "pull")

    with pytest.raises(DownloadError, match="after 2 attempt"):
        download_one(
            client,
            request,
            tmp_path,
            max_attempts=2,
            retry_delay_seconds=0,
            sleep=lambda _: None,
        )

    assert not (tmp_path / request.filename).exists()
    assert not (tmp_path / f"{request.filename}.part").exists()


def test_download_all_processes_requests_in_order(tmp_path: Path) -> None:
    observed_months: list[str] = []
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_months.append(request.url.params["hfMo"])
        month = int(request.url.params["hfMo"].rstrip("|"))
        game_date = "2025-04-01" if month == 4 else "2025-05-01"
        return httpx.Response(
            200,
            content=csv_bytes([valid_row(game_date)]),
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    requests = [
        DownloadRequest(2025, 4, "pull"),
        DownloadRequest(2025, 5, "pull"),
    ]

    results = download_all(
        client,
        requests,
        tmp_path,
        request_delay_seconds=0.5,
        sleep=delays.append,
    )

    assert [result.path.name for result in results] == [
        "2025_mar-apr_pull_BBE.csv",
        "2025_may_pull_BBE.csv",
    ]
    assert observed_months == ["4|", "5|"]
    assert delays == [0.5]


def test_parse_args_defaults_to_the_complete_request_scope() -> None:
    options = parse_args([])

    assert options.years == YEARS
    assert options.months == tuple(MONTH_SLUGS)
    assert options.directions == DIRECTIONS
    assert options.overwrite is False


def test_parse_args_supports_narrow_downloads() -> None:
    options = parse_args(
        [
            "--year",
            "2025",
            "--month",
            "4",
            "--direction",
            "opposite",
            "--overwrite",
        ]
    )

    assert options.years == (2025,)
    assert options.months == (4,)
    assert options.directions == ("opposite",)
    assert options.overwrite is True
