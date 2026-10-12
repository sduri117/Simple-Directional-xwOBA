"""Download complete monthly batted-ball data from Baseball Savant."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import httpx
import polars as pl

from data_preprocessing import REQUIRED_COLUMNS

BASEBALL_SAVANT_CSV_URL: Final[str] = (
    "https://baseballsavant.mlb.com/statcast_search/csv"
)
YEARS: Final[tuple[int, ...]] = tuple(range(2022, 2027))
MONTH_SLUGS: Final[dict[int, str]] = {
    4: "mar-apr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sep-oct",
}
MONTHS_IN_BUCKET: Final[dict[int, frozenset[int]]] = {
    4: frozenset((3, 4)),
    5: frozenset((5,)),
    6: frozenset((6,)),
    7: frozenset((7,)),
    8: frozenset((8,)),
    9: frozenset((9, 10)),
}
DIRECTIONS: Final[tuple[str, ...]] = ("pull", "straightaway", "opposite")
DIRECTION_QUERY_VALUES: Final[dict[str, str]] = {
    "pull": "Pull",
    "straightaway": "Straightaway",
    "opposite": "Opposite",
}
RAW_DATA_DIRECTORY: Final[Path] = Path("RawData2")
SUSPECTED_TRUNCATION_ROW_COUNT: Final[int] = 25_000
DEFAULT_TIMEOUT_SECONDS: Final[float] = 120.0
DEFAULT_MAX_ATTEMPTS: Final[int] = 3
DEFAULT_RETRY_DELAY_SECONDS: Final[float] = 2.0
DEFAULT_REQUEST_DELAY_SECONDS: Final[float] = 1.0
USER_AGENT: Final[str] = "simple-directional-xwoba/0.1"


class DownloadError(RuntimeError):
    """Report a download that could not be completed safely."""


class DataValidationError(ValueError):
    """Report downloaded data that violates the expected data contract."""


@dataclass(frozen=True, slots=True)
class DownloadRequest:
    """Identify one Baseball Savant monthly batted-ball download."""

    year: int
    month: int
    direction: str

    def __post_init__(self) -> None:
        if self.year not in YEARS:
            raise ValueError(f"Unsupported year: {self.year}")
        if self.month not in MONTH_SLUGS:
            raise ValueError(f"Unsupported month bucket: {self.month}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"Unsupported batted-ball direction: {self.direction}")

    @property
    def filename(self) -> str:
        """Return the canonical filename for this download."""
        return f"{self.year}_{MONTH_SLUGS[self.month]}_{self.direction}_BBE.csv"

    def query_parameters(self) -> dict[str, str]:
        """Return Baseball Savant parameters for individual-pitch CSV data."""
        return {
            "all": "true",
            "type": "details",
            "hfGT": "R|",
            "hfPR": r"hit\.\.into\.\.play|",
            "hfPull": f"{DIRECTION_QUERY_VALUES[self.direction]}|",
            "hfSea": f"{self.year}|",
            "hfMo": f"{self.month}|",
            "player_type": "batter",
            "group_by": "name",
            "sort_col": "pitches",
            "sort_order": "desc",
            "min_pitches": "0",
            "min_results": "0",
            "min_pas": "0",
        }


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Describe the validated local result of one requested download."""

    path: Path
    row_count: int
    downloaded: bool


@dataclass(frozen=True, slots=True)
class DownloaderOptions:
    """Hold validated downloader command-line options."""

    years: tuple[int, ...]
    months: tuple[int, ...]
    directions: tuple[str, ...]
    output_directory: Path
    overwrite: bool
    timeout_seconds: float
    max_attempts: int
    retry_delay_seconds: float
    request_delay_seconds: float


def build_download_requests(
    years: Iterable[int] = YEARS,
    months: Iterable[int] = MONTH_SLUGS,
    directions: Iterable[str] = DIRECTIONS,
) -> list[DownloadRequest]:
    """Build requests in deterministic year, month, then direction order."""
    return [
        DownloadRequest(year=year, month=month, direction=direction)
        for year in years
        for month in months
        for direction in directions
    ]


def validate_csv(path: Path, request: DownloadRequest) -> int:
    """Validate one CSV and return its number of data rows."""
    try:
        frame = pl.read_csv(
            path,
            columns=list(REQUIRED_COLUMNS),
            schema_overrides={"game_date": pl.String},
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise DataValidationError(f"Could not read {path}: {error}") from error

    if frame.is_empty():
        raise DataValidationError(f"{path} contains no data rows")
    if frame.height >= SUSPECTED_TRUNCATION_ROW_COUNT:
        raise DataValidationError(
            f"{path} has {frame.height:,} rows, indicating possible truncation"
        )

    game_dates = frame.get_column("game_date")
    parsed_dates = game_dates.str.to_date(strict=False)
    malformed_dates = game_dates.is_not_null() & parsed_dates.is_null()
    if malformed_dates.any():
        raise DataValidationError(f"{path} contains malformed game_date values")

    non_null_dates = parsed_dates.drop_nulls()
    if non_null_dates.is_empty():
        raise DataValidationError(f"{path} has no usable game_date values")

    actual_years = set(non_null_dates.dt.year().unique().to_list())
    if actual_years != {request.year}:
        raise DataValidationError(
            f"{path} contains years {sorted(actual_years)}, expected {request.year}"
        )

    actual_months = set(non_null_dates.dt.month().unique().to_list())
    allowed_months = MONTHS_IN_BUCKET[request.month]
    unexpected_months = actual_months - allowed_months
    if unexpected_months:
        raise DataValidationError(
            f"{path} contains months {sorted(unexpected_months)} outside the "
            f"{MONTH_SLUGS[request.month]} bucket"
        )

    return frame.height


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, (httpx.TransportError, DataValidationError)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code == 429 or error.response.status_code >= 500
    return False


def _download_to_path(
    client: httpx.Client,
    request: DownloadRequest,
    temporary_path: Path,
) -> None:
    with client.stream(
        "GET",
        BASEBALL_SAVANT_CSV_URL,
        params=request.query_parameters(),
    ) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as output_file:
            for chunk in response.iter_bytes():
                output_file.write(chunk)


def download_one(
    client: httpx.Client,
    request: DownloadRequest,
    output_directory: Path = RAW_DATA_DIRECTORY,
    *,
    overwrite: bool = False,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> DownloadResult:
    """Download and validate one request, retrying transient failures."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds cannot be negative")

    destination = output_directory / request.filename
    if destination.exists() and not overwrite:
        row_count = validate_csv(destination, request)
        return DownloadResult(destination, row_count, downloaded=False)

    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(f"{destination.suffix}.part")

    for attempt in range(1, max_attempts + 1):
        temporary_path.unlink(missing_ok=True)
        try:
            _download_to_path(client, request, temporary_path)
            row_count = validate_csv(temporary_path, request)
            temporary_path.replace(destination)
            return DownloadResult(destination, row_count, downloaded=True)
        except (
            DataValidationError,
            httpx.HTTPStatusError,
            httpx.TransportError,
        ) as error:
            temporary_path.unlink(missing_ok=True)
            if not _is_retryable(error) or attempt == max_attempts:
                raise DownloadError(
                    f"Failed to download {request.filename} after {attempt} "
                    f"attempt(s): {error}"
                ) from error
            sleep(retry_delay_seconds * (2 ** (attempt - 1)))

    raise AssertionError("download retry loop exited unexpectedly")


def download_all(
    client: httpx.Client,
    requests: Sequence[DownloadRequest],
    output_directory: Path = RAW_DATA_DIRECTORY,
    *,
    overwrite: bool = False,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> list[DownloadResult]:
    """Download requests serially and return their local results."""
    if request_delay_seconds < 0:
        raise ValueError("request_delay_seconds cannot be negative")

    results: list[DownloadResult] = []
    for index, request in enumerate(requests):
        result = download_one(
            client,
            request,
            output_directory,
            overwrite=overwrite,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
            sleep=sleep,
        )
        results.append(result)
        action = "downloaded" if result.downloaded else "skipped"
        print(
            f"[{index + 1}/{len(requests)}] {action} {result.path} "
            f"({result.row_count:,} rows)"
        )
        if result.downloaded and index < len(requests) - 1:
            sleep(request_delay_seconds)
    return results


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args(arguments: Sequence[str] | None = None) -> DownloaderOptions:
    """Parse downloader CLI arguments into typed options."""
    parser = argparse.ArgumentParser(
        description="Download monthly Baseball Savant batted-ball CSV files."
    )
    parser.add_argument(
        "--year",
        action="append",
        type=int,
        choices=YEARS,
        help="Year to download; repeat to select multiple years (default: all).",
    )
    parser.add_argument(
        "--month",
        action="append",
        type=int,
        choices=tuple(MONTH_SLUGS),
        help="Month bucket number; repeat to select multiple (default: all).",
    )
    parser.add_argument(
        "--direction",
        action="append",
        choices=DIRECTIONS,
        help="Direction to download; repeat to select multiple (default: all).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=RAW_DATA_DIRECTORY,
        help="Destination directory (default: RawData2).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-attempts",
        type=_positive_int,
        default=DEFAULT_MAX_ATTEMPTS,
    )
    parser.add_argument(
        "--retry-delay",
        type=_nonnegative_float,
        default=DEFAULT_RETRY_DELAY_SECONDS,
        help="Initial retry delay in seconds; retries use exponential backoff.",
    )
    parser.add_argument(
        "--request-delay",
        type=_nonnegative_float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
        help="Delay between successful network downloads in seconds.",
    )
    namespace = parser.parse_args(arguments)

    selected_years = cast(list[int] | None, namespace.year)
    selected_months = cast(list[int] | None, namespace.month)
    selected_directions = cast(list[str] | None, namespace.direction)
    return DownloaderOptions(
        years=tuple(selected_years or YEARS),
        months=tuple(selected_months or MONTH_SLUGS),
        directions=tuple(selected_directions or DIRECTIONS),
        output_directory=cast(Path, namespace.output_directory),
        overwrite=cast(bool, namespace.overwrite),
        timeout_seconds=cast(float, namespace.timeout),
        max_attempts=cast(int, namespace.max_attempts),
        retry_delay_seconds=cast(float, namespace.retry_delay),
        request_delay_seconds=cast(float, namespace.request_delay),
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the downloader command-line interface."""
    options = parse_args(arguments)
    requests = build_download_requests(
        options.years,
        options.months,
        options.directions,
    )
    try:
        with httpx.Client(
            timeout=options.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            download_all(
                client,
                requests,
                options.output_directory,
                overwrite=options.overwrite,
                max_attempts=options.max_attempts,
                retry_delay_seconds=options.retry_delay_seconds,
                request_delay_seconds=options.request_delay_seconds,
            )
    except DownloadError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
