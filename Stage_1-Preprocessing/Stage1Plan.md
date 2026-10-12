# Stage 1: Data Download and Preprocessing

**Status:** Complete as of October 12, 2026.

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

Each downloaded CSV must be nonempty and contain all seven columns needed by the
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
7. `estimated_woba_using_speedangle`, renamed to `xwOBA-MLB` in the condensed
   output

After selecting these seven columns, any row with a missing value in at least one
of them will be removed. Missing values in unneeded Baseball Savant columns do
not affect whether a row is retained.

The pipeline will add three columns:

- `hit_direction`: the lowercase direction represented by the input filename;
  one of `pull`, `straightaway`, or `opposite`.
- `simplified_events`: preserves source values `single`, `double`, `triple`, and
  `home_run`, and maps every other event to `out`.
- `total_bases`: maps `out` to 0, `single` to 1, `double` to 2, `triple` to 3,
  and `home_run` to 4.

The final files will be written without an index to `ProcessedData` as:

```text
{year}_all_BBE.csv
```

There will be one final file for every year from 2022 through 2026. Each file
will contain the seven retained source columns, with the Statcast column renamed
to `xwOBA-MLB`, followed by the three derived columns.

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

## Completed implementation

The downloader generated the full matrix of 90 monthly and directional files
in `RawData2`. The preprocessing command requires all 18 expected inputs for a
requested year, processes years serially in deterministic month-then-direction
order, and refuses to replace an existing annual file unless `--overwrite` is
specified.

The annual CSV is first written to a `.part` file. Before that temporary file is
renamed, the pipeline reopens it and validates its column order, inferred data
types, row count, null count, dates, year, category domains, simplified-event
mapping, and total-bases mapping. This preserves an existing output if writing
or validation fails.

Generate every annual dataset from the repository root with:

```text
uv run python Stage_1-Preprocessing/data_preprocessing.py
```

Use repeated `--year` options to generate selected years. Use `--overwrite`
only when intentionally regenerating existing annual files.

## Final validation results

All five annual files were generated in `ProcessedData` and independently
reconciled against the 90 monthly inputs. Only rows missing at least one of the
seven required values were removed.

| Year | Input files | Raw rows | Removed rows | Final rows | Date range |
| --- | ---: | ---: | ---: | ---: | --- |
| 2022 | 18 | 124,266 | 935 | 123,331 | 2022-04-07 to 2022-10-05 |
| 2023 | 18 | 124,233 | 784 | 123,449 | 2023-03-30 to 2023-10-01 |
| 2024 | 18 | 124,201 | 820 | 123,381 | 2024-03-20 to 2024-09-30 |
| 2025 | 18 | 124,887 | 997 | 123,890 | 2025-03-18 to 2025-09-28 |
| 2026 | 18 | 124,611 | 1,230 | 123,381 | 2026-03-25 to 2026-09-27 |
| **Total** | **90** | **622,198** | **4,766** | **617,432** | — |

Direction counts in the final files were:

| Year | Opposite | Pull | Straightaway |
| --- | ---: | ---: | ---: |
| 2022 | 30,914 | 46,763 | 45,654 |
| 2023 | 30,534 | 47,981 | 44,934 |
| 2024 | 30,344 | 47,724 | 45,313 |
| 2025 | 30,259 | 48,532 | 45,099 |
| 2026 | 29,601 | 48,774 | 45,006 |

Simplified-event counts in the final files were:

| Year | Out | Single | Double | Triple | Home run |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | 83,792 | 25,758 | 7,926 | 642 | 5,213 |
| 2023 | 82,690 | 25,958 | 8,224 | 711 | 5,866 |
| 2024 | 83,621 | 25,843 | 7,771 | 696 | 5,450 |
| 2025 | 83,822 | 26,053 | 7,740 | 627 | 5,648 |
| 2026 | 83,637 | 26,021 | 7,487 | 671 | 5,565 |

Every output has the specified ten columns in the specified order. Inferred
CSV types are strings for dates, names, event fields, and direction fields;
64-bit integers for `batter` and `total_bases`; and 64-bit floats for
`launch_speed`, `launch_angle`, and `xwOBA-MLB`. An independent row-for-row
comparison confirmed that every `xwOBA-MLB` value exactly matches its retained
`estimated_woba_using_speedangle` source value. Validation found zero nulls,
malformed dates, unexpected directions, unexpected simplified events,
incorrect event mappings, or incorrect total-base mappings. The pre-existing
FanGraphs files were not modified.

The final quality gate completed with Black, Ruff, and strict mypy passing, and
all 31 pytest tests passing on Python 3.13.9.

