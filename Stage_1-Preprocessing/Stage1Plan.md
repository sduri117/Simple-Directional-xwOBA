# Stage 1: Data Download and Preprocessing

## Goal

Create a reproducible pipeline that downloads complete regular-season batted-ball
event data from Baseball Savant and condenses it into one analysis-ready CSV file
per season.

Stage 1 covers the 2022 through 2026 seasons. The existing FanGraphs files in
`ProcessedData` are already final and must not be modified.

## Data download specification

The downloader will retrieve individual-pitch CSV data from Baseball Savant's
Statcast search CSV endpoint. Every request will use the filters below:

- Pitch Result: `In play`
- Season Type: `Regular Season`
- Season: exactly one year from 2022 through 2026
- Batted Ball Direction: exactly one of `Pull`, `Straightaway`, or `Opposite`
- Player Type: `Batter`
- Month: exactly one of the six monthly options, excluding `1st Half` and
  `2nd Half`

The Baseball Savant monthly options and local filename slugs are:

| Baseball Savant option | Filename slug |
| --- | --- |
| Mar/Apr | `mar-apr` |
| May | `may` |
| Jun | `jun` |
| Jul | `jul` |
| Aug | `aug` |
| Sep/Oct | `sep-oct` |

Combining five years, six monthly options, and three directions produces 90
downloads. Requests must execute serially so only one file is downloaded at a
time.

Each file will be saved directly to `RawData2` with this naming convention:

```text
{year}_{month_slug}_{direction}_BBE.csv
```

For example:

```text
2025_mar-apr_pull_BBE.csv
```

Direction slugs are `pull`, `straightaway`, and `opposite`.

Downloads will first be written to a temporary `.part` path. A temporary file
will be renamed to its final CSV filename only after it passes validation. Valid
existing files will be skipped by default to support safe resumption, while an
explicit command-line option will permit replacement.

Each downloaded CSV must be nonempty and contain all six columns needed by the
preprocessing pipeline. The downloader will also validate that the data matches
the requested year and month bucket. Retries and bounded request timeouts will
handle transient network failures.

## Preprocessing specification

The preprocessing pipeline will read the monthly files in `RawData2`, retain
only these source columns, and preserve their order:

1. `game_date`
2. `player_name`
3. `batter`
4. `events`
5. `launch_speed`
6. `launch_angle`

After selecting these six columns, any row with a missing value in at least one
of them will be removed. Missing values in unneeded Baseball Savant columns do
not affect whether a row is retained.

The pipeline will add three columns:

- `hit_direction`: the lowercase direction represented by the input filename;
  one of `pull`, `straightaway`, or `opposite`.
- `simplified_events`: preserves source values `single`, `double`, and `triple`,
  converts the source value `home_run` to `home run`, and maps every other event
  to `out`.
- `total_bases`: maps `out` to 0, `single` to 1, `double` to 2, `triple` to 3,
  and `home run` to 4.

The final files will be written without an index to `ProcessedData` as:

```text
{year}_all_BBE.csv
```

There will be one final file for every year from 2022 through 2026. Each file
will contain the six source columns followed by the three derived columns.

## Technology

- Python 3.13, constrained to `>=3.13,<3.14`
- uv for dependency and environment management
- Polars for CSV transformation and validation
- HTTPX for serial HTTP downloads
- Type hints and strict mypy checks
- pytest for unit tests
- Black for formatting
- Ruff for linting; Black remains the sole formatter

The direct dependencies and their transitive dependencies are locked in
`uv.lock`. Libraries needed only by later stages, including scikit-learn, SciPy,
and plotting libraries, will be selected and added during those stages.

## Code organization

```text
Stage_1-Preprocessing/
├── Stage1Plan.md
├── data_download.py
└── data_preprocessing.py
tests/
├── test_data_download.py
└── test_data_preprocessing.py
```

## Milestones

### Milestone 1: Project foundation and revised plan

- Create and configure `pyproject.toml`, `uv.lock`, and `.python-version`.
- Record the agreed Stage 1 specification and milestones in this document.
- Add typed constants and data definitions shared by the upcoming downloader
  and preprocessing implementations.
- Establish the unit-test structure.
- Run Black, Ruff, mypy, and pytest, then pause for review.

### Milestone 2: Implement the downloader

- Generate the complete 90-request matrix.
- Add serial HTTP requests, retries, timeouts, atomic writes, resume behavior,
  and command-line filters.
- Validate HTTP responses, CSV structure, required columns, nonempty results,
  requested years, requested month buckets, and filenames.
- Test downloader behavior with mocked HTTP responses; unit tests must not
  contact Baseball Savant.
- Run Black, Ruff, mypy, and pytest, then pause for review.

### Milestone 3: Download and validate the complete raw data

- Run all 90 downloads serially into `RawData2`.
- Validate every file's year, direction, month range, schema, and row count.
- Confirm that no monthly file reaches the 25,000-row ceiling.
- Run Black, Ruff, mypy, and pytest, then pause for review.

### Milestone 4: Implement preprocessing

- Implement column selection, required-value filtering, derived columns,
  annual concatenation, and CSV output.
- Add unit tests for mappings, null removal, directions, schemas, input
  discovery, concatenation, and output naming.
- Run Black, Ruff, mypy, and pytest, then pause for review.

### Milestone 5: Generate and validate the final datasets

- Generate all five annual files in `ProcessedData`.
- Reconcile output row counts with monthly inputs and removed-row counts.
- Validate schemas, types, categories, dates, and event mappings.
- Update this document to match the completed implementation and record the
  validation results.
- Run Black, Ruff, mypy, and pytest, then pause for final Stage 1 review.

## Quality gate

After every implementation change, run the tools in this order:

1. `uv run black .`
2. `uv run ruff check .`
3. `uv run mypy`
4. `uv run pytest`

All four commands must pass before a milestone is presented for review.

