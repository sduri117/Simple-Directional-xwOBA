from data_download import DIRECTIONS, MONTH_SLUGS, YEARS, DownloadRequest


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
    assert len(YEARS) * len(MONTH_SLUGS) * len(DIRECTIONS) == 90


def test_download_request_uses_the_canonical_filename() -> None:
    request = DownloadRequest(year=2025, month=4, direction="straightaway")

    assert request.filename == "2025_mar-apr_straightaway_BBE.csv"
