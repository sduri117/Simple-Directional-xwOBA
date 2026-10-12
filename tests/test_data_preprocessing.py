from data_preprocessing import (
    DERIVED_COLUMNS,
    EVENT_TO_TOTAL_BASES,
    REQUIRED_COLUMNS,
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


def test_total_base_mapping() -> None:
    assert EVENT_TO_TOTAL_BASES == {
        "out": 0,
        "single": 1,
        "double": 2,
        "triple": 3,
        "home_run": 4,
    }
