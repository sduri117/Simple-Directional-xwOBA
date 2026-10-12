"""Condense monthly Baseball Savant data into analysis-ready annual files."""

from pathlib import Path
from typing import Final

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "game_date",
    "player_name",
    "batter",
    "events",
    "launch_speed",
    "launch_angle",
)
DERIVED_COLUMNS: Final[tuple[str, ...]] = (
    "hit_direction",
    "simplified_events",
    "total_bases",
)
EVENT_TO_TOTAL_BASES: Final[dict[str, int]] = {
    "out": 0,
    "single": 1,
    "double": 2,
    "triple": 3,
    "home_run": 4,
}
PROCESSED_DATA_DIRECTORY: Final[Path] = Path("ProcessedData")
