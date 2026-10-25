# Stage 2: Model Selection and Calculated xwOBA

**Status:** Milestone 9 complete and awaiting review as of October 24, 2026.

## Goal

Develop and validate a K-nearest-neighbors multiclass probability model that
uses launch speed and launch angle to estimate the probability of each possible
total-base outcome for a batted ball. Use the selected model to calculate local
expected wOBA values for 2025 and 2026, aggregate those values to qualified
player seasons, and compare them with MLB xwOBA and following-season wOBA.

Stage 2 will use the analysis-ready data produced by Stage 1. The Stage 1 source
and output files must not be modified.

## Reproducibility

- Use a fixed random seed of `99` for every process that accepts a random state
  or uses a random-number generator.
- Read annual inputs in deterministic year order and preserve deterministic row
  order in generated outputs.
- Record the complete candidate grid and software dependencies in versioned
  project files.
- Fit feature scaling inside each training fold so validation data cannot
  influence the fitted scaler.
- Explicitly align every probability matrix to the class order `0, 1, 2, 3,
  4`; do not assume that probability-column position implies a class label.
- Write generated CSV and model artifacts through temporary `.part` files and
  replace final paths only after validation succeeds.

## Input data

Model development will use these files:

- `ProcessedData/2022_all_BBE.csv`
- `ProcessedData/2023_all_BBE.csv`
- `ProcessedData/2024_all_BBE.csv`

Only `launch_speed` and `launch_angle` will be model features. `total_bases`
will be the five-class target, with allowed values 0 through 4 corresponding to
out, single, double, triple, and home run.

Calculation and analysis will additionally use:

- `ProcessedData/2025_all_BBE.csv`
- `ProcessedData/2026_all_BBE.csv`
- `ProcessedData/wOBAConstants.csv`
- `ProcessedData/QualifiedPlayerSeasonsData.csv`

The `MLBAMID` field in the qualified player-season data is the key that matches
the `batter` field in the batted-ball data.

All inputs will be validated before use for required columns, data types, nulls,
allowed target values, expected seasons, unique keys where required, and finite
feature values.

## Technology

- Python 3.13, constrained to `>=3.13,<3.14`
- uv for dependency and environment management
- Polars for CSV input, validation, joins, aggregation, and output
- scikit-learn for model pipelines, cross-validation, scaling, KNN, and metrics
- NumPy for probability-array and weighted-sum calculations
- Joblib for fitted-model serialization
- Matplotlib for reproducible, non-interactive plots
- Type hints and strict mypy checks
- pytest for unit and integration tests
- Black for formatting
- Ruff for linting; Black remains the sole formatter

## Model candidates

### Baseline

The control model will return the class-frequency distribution observed in its
training data for every validation observation. For cross-validation, the
frequency distribution will be learned separately from each fold's training
partition. For chronological validation, it will be learned from the combined
2022 and 2023 training data.

### KNN grid

Every combination of the following settings will be evaluated:

| Setting | Candidate values |
| --- | --- |
| Feature scaling | none, `StandardScaler`, `RobustScaler` |
| `n_neighbors` | 5, 10, 25, 50, 100, 250, 500, 1000 |
| `weights` | `uniform`, `distance` |
| `p` | 1 (Manhattan), 2 (Euclidean) |

This produces 96 KNN candidates plus the baseline. The deliberately broad
neighbor grid includes large neighborhoods suited to probability estimation on
approximately 370,000 training observations. The final candidate list will be
stored as part of the experiment configuration and written to the results.

The initially proposed `n_neighbors=2000` candidates were deferred during
Milestone 3 at the user's request.

A small timing and equivalence benchmark will select a supported scikit-learn
neighbor-search algorithm before the full experiment. This computational choice
will not change the statistical candidate grid. Candidate configurations will
run sequentially by default to keep memory use safe on a Mac with 24 GB of RAM.
The runner will checkpoint completed configurations so an interrupted experiment
can resume without recomputing valid results.

## Validation designs

### Stratified five-fold cross-validation

Use `StratifiedKFold` with five folds, shuffling enabled, and
`random_state=99` across the combined 2022 through 2024 dataset. Each candidate
will generate exactly one out-of-fold probability prediction for every input
row. Cross-validation metrics will be calculated over the complete aligned set
of out-of-fold predictions rather than averaging rounded fold-level metrics.

### Chronological validation

Fit each candidate on all 2022 and 2023 observations and evaluate it on all 2024
observations. Scalers and baseline class frequencies must be learned only from
the 2022 and 2023 training data.

## Metrics and model-selection results

Each validation design will report:

- Multiclass log loss, using all five explicitly ordered classes. This is the
  primary optimization metric.
- Multiclass Brier score, defined as the mean across observations of the sum of
  squared differences between the five predicted probabilities and the five
  one-hot outcome indicators. Lower values are better.
- Accuracy, using the class with the greatest predicted probability.

The baseline and every KNN candidate will be written to
`Stage_2-xwOBA-local/ModelSelectionResults.CSV`. Its columns will
include model type, feature scaling, every varied hyperparameter, and log loss,
Brier score, and accuracy for both cross-validation and chronological
validation. Numeric results will be stored at full practical precision.

The experiment will validate that every probability is finite and in `[0, 1]`,
each probability row sums to one within a strict floating-point tolerance, all
five classes are represented, and every observation receives the expected
prediction.

The completed result table will be presented for review. The user will choose
the final configuration based on performance under both validation designs; the
code will not silently select a final model.

## Final model

After the user identifies the final configuration, refit its entire
scikit-learn pipeline on all 2022 through 2024 observations. Save the fitted
pipeline, including its scaler when applicable, as a versioned Joblib artifact.
Store accompanying metadata containing the selected settings, feature order,
class order, training years, training row count, seed, dependency versions, and
validation metrics. Reload the artifact and verify that it reproduces a fixed
sample of probabilities before accepting it.

## Milestone 3 validation results

The completed `ModelSelectionResults.CSV` contains 97 unique rows: one baseline
and the complete Cartesian product of the three scaling choices, eight approved
neighbor counts, two weighting choices, and two distance powers. All 96 KNN
rows used the ball-tree search algorithm selected by the timing and equivalence
benchmark. The result has no duplicate candidates, missing configurations,
unexpected nulls, invalid metrics, or `n_neighbors=2000` rows.

The principal model-selection tradeoff is:

| Candidate | CV log loss | Chronological log loss | CV Brier | Chronological Brier | CV accuracy | Chronological accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.936974 | 0.932276 | 0.493381 | 0.490854 | 0.675660 | 0.677746 |
| No scaling, 1000 neighbors, uniform, p=1 | **0.569540** | **0.566065** | 0.315680 | 0.313385 | 0.773674 | 0.773993 |
| No scaling, 500 neighbors, uniform, p=2 | 0.579283 | 0.579190 | **0.315514** | **0.313077** | 0.773588 | 0.774422 |

The 1,000-neighbor Manhattan candidate has the best log loss in both validation
designs. The 500-neighbor Euclidean candidate has the best Brier score in both
designs. These results are reported without selecting between them; final model
selection remains the user's decision.

As a reproducibility and resource check, the full six-fit evaluation of the
included no-scaling, 1,000-neighbor, uniform, Euclidean candidate took 69.64
seconds on the local Mac using all available scikit-learn worker threads. macOS
reported a maximum resident set size of 4,886,577,152 bytes (about 4.55 GiB),
with zero swaps. This representative heaviest-neighborhood measurement is well
within the machine's 24 GB memory limit.

## Final-model artifact

The user selected the no-scaling, 1,000-neighbor, uniform-weighted Manhattan
distance model because it achieved the lowest log loss under both validation
designs while remaining close to the best Brier scores. The final pipeline was
fitted on all 370,161 batted balls from 2022 through 2024 and saved as:

- `ModelArtifacts/xwoba_knn_2022_2024.joblib`
- `ModelArtifacts/xwoba_knn_2022_2024.metadata.json`

The metadata records the feature and class order, selected configuration,
training years and row count, seed, validation metrics, dependency versions,
input checksums, and model checksum. The model SHA-256 is
`868d97a3cd191e7211bfa381c1611e61199cfc5839ae0e7a328416f472d15d3e`.
An independent reload verified the fitted configuration and training coverage,
then reproduced valid five-class probability distributions for a fixed sample
of 257 observations with every row summing to one.

## Batted-ball xwOBA-local calculation

For every 2025 and 2026 batted ball, call the final pipeline's
`predict_proba()` method and align its output to total-base classes 0 through 4.
Calculate:

```text
xwOBA-local =
    P(single)   * w1B
  + P(double)   * w2B
  + P(triple)   * w3B
  + P(home_run) * wHR
```

The out contribution is zero. Coefficients must come from the matching season
in `ProcessedData/wOBAConstants.csv`.

Preserve the existing columns and row order, append only `xwOBA-local`, and
write:

- `Calculated-xwOBAs/2025_all_BBE_with_calculated_xwOBAs.csv`
- `Calculated-xwOBAs/2026_all_BBE_with_calculated_xwOBAs.csv`

Validate the output schema, row count, identifiers, values, and weighted-sum
calculation against the source data and model predictions.

The completed batted-ball outputs contain:

| Year | Rows | Columns | Minimum xwOBA-local | Maximum xwOBA-local | Mean xwOBA-local |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025 | 123,890 | 11 | 0.000000 | 2.029717 | 0.376792 |
| 2026 | 123,381 | 11 | 0.000000 | 2.040888 | 0.365150 |

Both outputs preserve every source row and all ten source columns in their
original order, with only `xwOBA-local` appended. An independent full-row
recalculation from freshly generated class probabilities and the matching
season coefficients agreed with the saved values to a maximum absolute
difference of `4.44e-16` in each season. Both files contain finite values and
have no temporary `.part` files remaining.

## Qualified player-season xwOBA-local

For each qualified 2025 and 2026 player season, sum batted-ball
`xwOBA-local` values after joining `batter` to `MLBAMID`, then calculate:

```text
uBB = BB - IBB

season xwOBA-local =
    (wBB * uBB + wHBP * HBP + sum(batted-ball xwOBA-local))
    / (AB + BB - IBB + SF + HBP)
```

Use the matching season's `wBB` and `wHBP` coefficients. Write
`Calculated-xwOBAs/SeasonLevelCalculatedxwOBA.csv` with exactly these columns:

1. `year`
2. `batter_name`
3. `batter_id`
4. `wOBA`
5. `xwOBA-MLB`
6. `xwOBA-local`
7. `wOBAcon`
8. `xwOBAcon-MLB`
9. `xwOBAcon-local`

Source `wOBA` and `xwOBA-MLB` from `wOBA` and `xwOBA`, respectively, in the
qualified player-season input. Source `batter_name` from `Name` and `batter_id`
from `MLBAMID`. Validate join coverage, uniqueness of player-season keys,
denominators, row counts, finite results, and independent formula
recalculations.

The completed `SeasonLevelCalculatedxwOBA.csv` contains:

| Year | Qualified seasons | Minimum xwOBA-local | Maximum xwOBA-local | Mean xwOBA-local |
| --- | ---: | ---: | ---: | ---: |
| 2025 | 145 | 0.270105 | 0.464249 | 0.338498 |
| 2026 | 135 | 0.249332 | 0.443428 | 0.331656 |

All 280 qualified player-season keys are unique and matched a batted-ball
aggregate through `MLBAMID = batter`. The saved `wOBA` and `xwOBA-MLB` values
match the qualified-season source exactly. An independent full-row
reconstruction of unintentional walks, numerator, denominator, season-specific
coefficients, and batted-ball sums produced a maximum absolute difference of
zero from the saved `xwOBA-local` values.

## Correlation analysis and plots

Calculate coefficient-of-determination (`R^2`) values for:

- Batted-ball `xwOBA-local` versus `xwOBA-MLB` in 2025, in 2026, and across the
  combined seasons.
- Qualified player-season `xwOBA-local` versus `xwOBA-MLB` in 2025, in 2026,
  and across the combined seasons.
- 2025 player-season `xwOBA-MLB` versus 2026 player-season `wOBA` among players
  qualified in both seasons.
- 2025 player-season `xwOBA-local` versus 2026 player-season `wOBA` among
  players qualified in both seasons.

For every comparison, calculate the Pearson correlation coefficient (`r`) and
report `R^2` as its square: `R^2 = r * r`. This is squared Pearson correlation,
not scikit-learn's regression `r2_score`. Save both signed `r` and `R^2` so the
direction of each relationship remains explicit.

Save a machine-readable table of all reported R2 values and create three PNG
plots:

1. Player-season `xwOBA-local` versus `xwOBA-MLB` across 2025 and 2026, with
   seasons visually distinguished.
2. 2025 `xwOBA-MLB` versus 2026 `wOBA` for players qualified in both seasons.
3. 2025 `xwOBA-local` versus 2026 `wOBA` for players qualified in both seasons.

Each plot will include clear labels, sample size, the relevant Pearson `r` and
`R^2` values, and an appropriate fitted line. Plot source rows and metric
calculations will be independently reconciled with the season-level output.

The completed analysis is saved in
`Calculated-xwOBAs/CorrelationResults.csv`. It contains:

| Analysis level | Cohort/comparison | n | Pearson r | Pearson R^2 |
| --- | --- | ---: | ---: | ---: |
| Batted ball | 2025 local vs MLB | 123,890 | 0.989782 | 0.979668 |
| Batted ball | 2026 local vs MLB | 123,381 | 0.995164 | 0.990350 |
| Batted ball | 2025-2026 combined local vs MLB | 247,271 | 0.992195 | 0.984450 |
| Player season | 2025 local vs MLB | 145 | 0.989676 | 0.979459 |
| Player season | 2026 local vs MLB | 135 | 0.991255 | 0.982586 |
| Player season | 2025-2026 combined local vs MLB | 280 | 0.988489 | 0.977110 |
| Year to year | 2025 MLB xwOBA vs 2026 wOBA | 80 | 0.297376 | 0.088433 |
| Year to year | 2025 local xwOBA vs 2026 wOBA | 80 | 0.298469 | 0.089084 |

The three completed plots are saved in `Calculated-xwOBAs/plots`. Each was
visually inspected for legibility, labeling, cohort size, fitted line, and
reported statistics. An independent NumPy and Polars reconstruction from the
saved batted-ball and player-season artifacts reproduced all eight `r` and
`R^2` values with a maximum absolute difference of `5.55e-17`. It also
independently confirmed that 80 players qualified in both 2025 and 2026.

## Contact-only metrics and correlations

The reusable `calculate_season_xwoba.py` command also calculates contact-only
metrics for every qualified 2025 and 2026 player-season. For each batted ball,
map `total_bases` 0 through 4 to the season's out, `w1B`, `w2B`, `w3B`, and
`wHR` values. The out value is zero. Then calculate:

```text
wOBAcon = sum(realized batted-ball wOBA values) / batted-ball count
xwOBAcon-MLB = sum(batted-ball xwOBA-MLB) / batted-ball count
xwOBAcon-local = sum(batted-ball xwOBA-local) / batted-ball count
```

The script validates annual coefficient coverage, allowed outcomes, finite
values, player-season join coverage, counts, and independent sum/count
recalculations before atomically replacing the season-level output. Run it with:

```text
uv run python Stage_2-xwOBA-local/calculate_season_xwoba.py --overwrite
```

The completed contact-only season metrics are:

| Year | Metric | Minimum | Mean | Maximum |
| --- | --- | ---: | ---: | ---: |
| 2025 | wOBAcon | 0.281776 | 0.383259 | 0.598591 |
| 2025 | xwOBAcon-MLB | 0.289799 | 0.385913 | 0.589183 |
| 2025 | xwOBAcon-local | 0.290623 | 0.394451 | 0.601044 |
| 2026 | wOBAcon | 0.286711 | 0.383511 | 0.509305 |
| 2026 | xwOBAcon-MLB | 0.265608 | 0.379503 | 0.532911 |
| 2026 | xwOBAcon-local | 0.253134 | 0.382478 | 0.534252 |

As a diagnostic rather than a hard invariant, `wOBAcon` exceeded overall
`wOBA` in 267 of 280 qualified seasons, `xwOBAcon-MLB` exceeded overall
`xwOBA-MLB` in 262 of 280, and `xwOBAcon-local` exceeded overall
`xwOBA-local` in 266 of 280.

At the batted-ball level, the realized contact value is the matching season's
out/single/double/triple/home-run wOBA coefficient. Correlate that value with
each expected batted-ball value. At the year-to-year player level, compare
2025 `xwOBAcon-MLB` and `xwOBAcon-local` with 2026 `wOBAcon` for the 80 players
qualified in both seasons. As elsewhere in Stage 2, report signed Pearson `r`
and `R^2 = r^2`:

| Level | Cohort/comparison | n | Pearson r | Pearson R^2 |
| --- | --- | ---: | ---: | ---: |
| Batted ball | 2025 MLB expected vs realized contact value | 123,890 | 0.669786 | 0.448613 |
| Batted ball | 2025 local expected vs realized contact value | 123,890 | 0.669024 | 0.447593 |
| Batted ball | 2026 MLB expected vs realized contact value | 123,381 | 0.672093 | 0.451709 |
| Batted ball | 2026 local expected vs realized contact value | 123,381 | 0.672146 | 0.451780 |
| Batted ball | Combined MLB expected vs realized contact value | 247,271 | 0.670909 | 0.450118 |
| Batted ball | Combined local expected vs realized contact value | 247,271 | 0.670367 | 0.449392 |
| Year to year | 2025 xwOBAcon-MLB vs 2026 wOBAcon | 80 | 0.618115 | 0.382066 |
| Year to year | 2025 xwOBAcon-local vs 2026 wOBAcon | 80 | 0.609573 | 0.371579 |
| Year to year | 2025 wOBA vs 2026 wOBA | 80 | 0.266818 | 0.071192 |
| Year to year | 2025 wOBAcon vs 2026 wOBAcon | 80 | 0.611126 | 0.373475 |

These ten rows are appended to the original eight rows in
`Calculated-xwOBAs/CorrelationResults.csv`.

An independent reconstruction that did not import the project calculation
functions reproduced all 280 saved contact-only player-season values exactly.
It also reproduced the six batted-ball correlation rows exactly and the two
year-ahead rows to a maximum absolute difference of `1.67e-16`, while
independently confirming the 80-player matched cohort.

The two realized year-to-year comparisons were also independently calculated
from the season-level artifact. They reproduce the saved Pearson `r` and
`R^2 = r^2` values at full stored precision.

## Code organization

The intended organization is:

```text
Stage_2-xwOBA-local/
├── Stage2Plan.md
├── experiment_config.py
├── model_evaluation.py
├── train_final_model.py
├── calculate_batted_ball_xwoba.py
├── calculate_season_xwoba.py
└── analyze_results.py
tests/
├── test_experiment_config.py
├── test_model_evaluation.py
├── test_train_final_model.py
├── test_calculate_batted_ball_xwoba.py
├── test_calculate_season_xwoba.py
└── test_analyze_results.py
```

Small shared utilities may be introduced if implementation shows they remove
meaningful duplication. Command-line entry points will support explicit input
and output paths, safe resume behavior where relevant, and intentional
overwrite flags.

## Milestones

### Milestone 1: Project foundation and approved experiment specification

- Review and approve this Stage 2 plan, including metric definitions, the KNN
  grid, artifact names, and analysis conventions.
- Add and lock the Stage 2 dependencies.
- Extend formatting, linting, strict typing, and test configuration to Stage 2.
- Add typed experiment configuration and shared data definitions.
- Add initial tests for configuration completeness, seed propagation, class
  order, and input-schema validation.
- Run the full quality gate, then pause for review.

### Milestone 2: Model-evaluation implementation

- Implement validated loading of the 2022 through 2024 modeling data.
- Implement the training-frequency baseline and leakage-safe KNN pipelines.
- Implement seeded five-fold stratified out-of-fold evaluation.
- Implement 2022–2023 to 2024 chronological evaluation.
- Implement log loss, multiclass Brier score, and accuracy calculations.
- Implement the neighbor-search benchmark, checkpointed result writing, safe
  resume behavior, and command-line controls.
- Test split reproducibility, fold isolation, scaling isolation, probability
  alignment, metrics, candidate coverage, checkpointing, and resume behavior on
  compact synthetic fixtures.
- Run the full quality gate, then pause for review.

### Milestone 3: Full model-selection experiment

- Run the baseline and all 96 currently approved KNN candidates on the complete
  dataset.
- Produce and independently validate `ModelSelectionResults.CSV`.
- Summarize rankings and tradeoffs under both validation designs without making
  the final choice on the user's behalf.
- Record runtime and peak-memory observations for reproducibility.
- Run the full quality gate, then pause for the user's model selection.

### Milestone 4: Final-model training and validation

- Encode the user-selected configuration explicitly.
- Train it on all 2022 through 2024 data.
- Save and reload the fitted pipeline and metadata.
- Validate feature order, class order, training coverage, and reproducible
  probabilities.
- Add tests for artifact metadata and save/load prediction equivalence.
- Run the full quality gate, then pause for review.

### Milestone 5: Batted-ball xwOBA-local calculation

- Generate the 2025 and 2026 batted-ball output files in
  `Calculated-xwOBAs`.
- Validate model inputs, season-specific coefficient lookup, probability
  alignment, weighted sums, schemas, row order, and row counts.
- Add focused unit and integration tests for coefficient and xwOBA calculations.
- Run the full quality gate, then pause for review.

### Milestone 6: Qualified player-season xwOBA-local calculation

- Generate `SeasonLevelCalculatedxwOBA.csv` for qualified 2025 and 2026 player
  seasons.
- Reconcile batter-to-player joins and batted-ball aggregation.
- Independently validate every component of the season formula and final schema.
- Add focused unit and integration tests for joins, aggregation, and formula
  behavior.
- Run the full quality gate, then pause for review.

### Milestone 7: Correlation analysis, plots, and final validation

- Calculate and save Pearson `r` and `R^2 = r^2` for every specified
  correlation.
- Generate and visually inspect the three requested plots.
- Independently reproduce the analysis metrics from the saved artifacts.
- Update this document to reflect the completed implementation and record final
  validation results.
- Run the full quality gate, then pause for final Stage 2 review.

### Milestone 8: Contact-only metrics and correlations

- Extend the reusable player-season calculation script to calculate `wOBAcon`,
  `xwOBAcon-MLB`, and `xwOBAcon-local` from batted-ball inputs and annual
  coefficients.
- Append the three metrics to every qualified player-season and independently
  validate each calculation.
- Calculate batted-ball realized-versus-expected contact correlations and 2025
  expected-contact versus 2026 realized-contact correlations using Pearson
  `r` and `R^2 = r^2`.
- Regenerate and independently validate the season and correlation artifacts.
- Run the full quality gate, then pause for review.

### Milestone 9: Realized year-to-year correlations

- Compare 2025 `wOBA` with 2026 `wOBA` for players qualified in both seasons.
- Compare 2025 `wOBAcon` with 2026 `wOBAcon` for the same cohort.
- Save signed Pearson `r` and `R^2 = r^2` alongside the prior correlations.
- Independently validate the matched cohort and both calculations.
- Run the full quality gate, then pause for review.

## Quality gate

After every implementation change, run the tools in this order:

1. `uv run black .`
2. `uv run ruff check .`
3. `uv run mypy`
4. `uv run pytest`

All four commands must pass before a milestone is presented for review. Full
data experiments and artifact generation are additional milestone-specific
checks and do not replace this gate.
