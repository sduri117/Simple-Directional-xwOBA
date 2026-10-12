"""Download complete monthly batted-ball data from Baseball Savant."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

YEARS: Final[tuple[int, ...]] = tuple(range(2022, 2027))
MONTH_SLUGS: Final[dict[int, str]] = {
    4: "mar-apr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sep-oct",
}
DIRECTIONS: Final[tuple[str, ...]] = ("pull", "straightaway", "opposite")
RAW_DATA_DIRECTORY: Final[Path] = Path("RawData2")


@dataclass(frozen=True, slots=True)
class DownloadRequest:
    """Identify one Baseball Savant monthly batted-ball download."""

    year: int
    month: int
    direction: str

    @property
    def filename(self) -> str:
        """Return the canonical filename for this download."""
        return f"{self.year}_{MONTH_SLUGS[self.month]}_{self.direction}_BBE.csv"
