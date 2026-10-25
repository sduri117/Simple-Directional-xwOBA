# Stage 3: Simple Directional xwOBA

**Status:** Milestone 1 complete and awaiting review as of October 24, 2026.

## Goal

Develop and validate a K-nearest-neighbors multiclass probability model that
uses launch speed, launch angle, and batted-ball direction to estimate the
probability of each possible total-base outcome. Use the selected model to
calculate simple-directional expected wOBA (`sd-xwOBA`) values for 2025 and
2026, aggregate those values to qualified player seasons, and compare them
with MLB xwOBA, the Stage 2 local xwOBA model, realized contact value, and
following-season results.

Stage 3 will use the analysis-ready data produced by Stage 1 and the calculated
artifacts produced by Stage 2. Existing Stage 1 and Stage 2 values must not be
modified. Stage 3 values will be appended as new columns or rows in the shared
calculated artifacts.

## Reproducibility

- Use a fixed random seed of `99` for every process that accepts a random state
  or uses a random-number generator.
- Read annual inputs in deterministic year order and preserve deterministic row
  order in generated outputs.
- Record the complete candidate grid and software dependencies in versioned
  project files.
- Fit numeric feature scaling and categorical encoding inside each training
  fold so validation data cannot influence fitted preprocessing.
- Use an explicit direction category order of `pull`, `straightaway`, and
  `opposite`; retain all three one-hot columns and reject unknown categories.
- Scale only `launch_speed` and `launch_angle`. Never scale the one-hot direction
  columns.
- Explicitly align every probability matrix to class order `0, 1, 2, 3, 4`.
- Write generated CSV and model artifacts through temporary `.part` files and
  replace final paths only after validation succeeds.

## Input data

Model development will use:

- `ProcessedData/2022_all_BBE.csv`
- `ProcessedData/2023_all_BBE.csv`
- `ProcessedData/2024_all_BBE.csv`

The raw model features are `launch_speed`, `launch_angle`, and `hit_direction`.
The five-class target is `total_bases`, where 0 through 4 represent an out,
single, double, triple, and home run.

Calculation and analysis will additionally use:

- `Calculated-xwOBAs/2025_all_BBE_with_calculated_xwOBAs.csv`
- `Calculated-xwOBAs/2026_all_BBE_with_calculated_xwOBAs.csv`
- `Calculated-xwOBAs/SeasonLevelCalculatedxwOBA.csv`
- `Calculated-xwOBAs/CorrelationResults.csv`
- `ProcessedData/wOBAConstants.csv`
- `ProcessedData/QualifiedPlayerSeasonsData.csv`

The `MLBAMID` field in qualified player-season data matches `batter` in the
batted-ball data. All inputs will be validated for required columns, data
types, nulls, finite numeric features, allowed categories and target values,
expected seasons, and unique keys where required.

## Feature preprocessing

A scikit-learn `ColumnTransformer` will produce features in this order:

1. `launch_speed`
2. `launch_angle`
3. `hit_direction_pull`
4. `hit_direction_straightaway`
5. `hit_direction_opposite`

For a no-scaling candidate, the numeric columns pass through unchanged. For a
standard- or robust-scaling candidate, the requested scaler applies only to the
two numeric columns. `OneHotEncoder` will use fixed categories, `drop=None`,
`handle_unknown="error"`, and dense floating-point output. Thus every direction
is represented symmetrically and the three indicators always remain 0 or 1.

## Technology

- Python 3.13, constrained to `>=3.13,<3.14`
- uv for dependency and environment management
- Polars for CSV input, validation, joins, aggregation, and output
- scikit-learn for column-wise preprocessing, cross-validation, scaling,
  one-hot encoding, KNN, and metrics
- NumPy for probability arrays, weighted sums, and correlations
- Joblib for fitted-model serialization
- Matplotlib for reproducible, non-interactive plots
- Type hints and strict mypy checks
- pytest for unit and integration tests
- Black for formatting
- Ruff for linting; Black remains the sole formatter

## Model candidates

The control model will return the class-frequency distribution learned from
its training data. Every combination of the following KNN settings will be
evaluated:

| Setting | Candidate values |
| --- | --- |
| Numeric feature scaling | none, `StandardScaler`, `RobustScaler` |
| `n_neighbors` | 5, 10, 25, 50, 100, 250, 500, 1000 |
| `weights` | `uniform`, `distance` |
| `p` | 1 (Manhattan), 2 (Euclidean) |

This produces 96 KNN candidates plus the baseline. A timing and equivalence
benchmark will select a supported neighbor-search algorithm before the full
experiment. Candidates will run sequentially by default and checkpoint after
each completed configuration.

## Validation and metrics

Use the same two designs as Stage 2:

- Seeded five-fold `StratifiedKFold` over combined 2022–2024 data, with metrics
  calculated from one aligned out-of-fold probability prediction per row.
- Chronological validation trained on 2022–2023 and evaluated on 2024.

Report multiclass log loss as the primary metric, summed multiclass Brier
score, and accuracy. Save all 97 results to
`Stage_3-sd-xwoBA/ModelSelectionResults.CSV`. Present the completed results for
review; the user will select the final configuration.

## Final model and calculated values

After selection, refit the complete preprocessing-and-model pipeline on all
2022–2024 observations and save distinct Joblib and metadata artifacts:

- `ModelArtifacts/sd_xwoba_knn_2022_2024.joblib`
- `ModelArtifacts/sd_xwoba_knn_2022_2024.metadata.json`

For each 2025 and 2026 batted ball, calculate:

```text
sd-xwOBA =
    P(single)   * w1B
  + P(double)   * w2B
  + P(triple)   * w3B
  + P(home_run) * wHR
```

Use coefficients from the matching season. Preserve every existing batted-ball
column and row, then append `sd-xwOBA` to the existing calculated annual files.

For each qualified player-season, calculate:

```text
uBB = BB - IBB

season sd-xwOBA =
    (wBB * uBB + wHBP * HBP + sum(batted-ball sd-xwOBA))
    / (AB + BB - IBB + SF + HBP)

sd-xwOBAcon = sum(batted-ball sd-xwOBA) / batted-ball count
```

Append `sd-xwOBA` and `sd-xwOBAcon` to
`Calculated-xwOBAs/SeasonLevelCalculatedxwOBA.csv` without changing existing
values.

## Correlations and plots

Preserve all Stage 2 correlation rows and append signed Pearson `r` and
`R^2 = r * r` for:

- Batted-ball `sd-xwOBA` versus `xwOBA-MLB`, for 2025, 2026, and combined.
- Batted-ball `sd-xwOBA` versus `xwOBA-local`, for 2025, 2026, and combined.
- Player-season `sd-xwOBA` versus `xwOBA-MLB`, for 2025, 2026, and combined.
- Player-season `sd-xwOBA` versus `xwOBA-local`, for 2025, 2026, and combined.
- Batted-ball `sd-xwOBA` versus realized contact value, for 2025, 2026, and
  combined.
- 2025 player-season `sd-xwOBA` versus 2026 `wOBA` for players qualified in
  both seasons.
- 2025 `sd-xwOBAcon` versus 2026 `wOBAcon` for the same matched cohort.

Create plots for the player-season comparisons and, specifically, both
year-to-year comparisons over the expected 80-player matched cohort. Each plot
will show the fitted line, sample size, signed Pearson `r`, and `R^2`.

## Code organization

```text
Stage_3-sd-xwoBA/
├── Stage3Plan.md
├── sd_experiment_config.py
├── sd_feature_pipeline.py
├── sd_model_evaluation.py
├── train_sd_model.py
├── calculate_batted_ball_sd_xwoba.py
├── calculate_season_sd_xwoba.py
├── analyze_sd_results.py
└── ModelSelectionResults.CSV
tests/
├── test_sd_experiment_config.py
├── test_sd_feature_pipeline.py
├── test_sd_model_evaluation.py
├── test_train_sd_model.py
├── test_calculate_batted_ball_sd_xwoba.py
├── test_calculate_season_sd_xwoba.py
└── test_analyze_sd_results.py
```

## Milestones

### Milestone 1: Foundation and approved experiment specification

- Record the approved Stage 3 specification and milestones.
- Extend formatting, linting, strict typing, and test discovery to Stage 3.
- Add typed experiment configuration and input contracts.
- Implement the deterministic mixed-feature preprocessor.
- Test grid completeness, seed propagation, category order, retained one-hot
  columns, scaling isolation, paths, and input validation.
- Run the full quality gate, then pause for review.

### Milestone 2: Model-evaluation implementation

- Implement validated modeling-data loading and the training-frequency baseline.
- Implement leakage-safe KNN pipelines using the Stage 3 preprocessor.
- Implement five-fold and chronological validation plus all three metrics.
- Implement algorithm benchmarking, checkpointing, resume behavior, and CLI
  controls.
- Add focused unit and integration tests.
- Run the full quality gate, then pause for review.

### Milestone 3: Full model-selection experiment

- Run the baseline and all 96 KNN candidates on the complete dataset.
- Independently validate and summarize the result table.
- Record runtime and peak-memory observations.
- Run the full quality gate, then pause for the user's model selection.

### Milestone 4: Final-model training and validation

- Encode the selected configuration explicitly and train on 2022–2024.
- Save, reload, and validate the fitted pipeline and metadata.
- Run the full quality gate, then pause for review.

### Milestone 5: Batted-ball sd-xwOBA calculation

- Append validated `sd-xwOBA` values to both existing annual calculated files.
- Independently reconcile model probabilities and weighted sums.
- Run the full quality gate, then pause for review.

### Milestone 6: Qualified player-season calculations

- Append validated `sd-xwOBA` and `sd-xwOBAcon` values to the existing season
  artifact.
- Independently validate joins, counts, sums, denominators, and formulas.
- Run the full quality gate, then pause for review.

### Milestone 7: Correlations, plots, and final validation

- Append all specified Stage 3 correlations while preserving Stage 2 results.
- Generate and visually inspect the requested plots, including both matched
  year-to-year comparisons.
- Independently reproduce all statistics from saved artifacts.
- Update this document with completed results.
- Run the full quality gate, then pause for final Stage 3 review.

## Quality gate

After every implementation change, run these tools in order:

1. `uv run black .`
2. `uv run ruff check .`
3. `uv run mypy`
4. `uv run pytest`

All four commands must pass before a milestone is presented for review.
