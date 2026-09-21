# Methods

## Dataset and signal

All models use every released `binnedTX` threshold-crossing channel. `spikePow` is excluded for all participants because it is exactly duplicated by `binnedTX` for T12 and a uniform feature definition was fixed before fitting. No anatomical channel selection is performed.

T12 and T16 use 20 ms bins; T15 and T17 use 10 ms bins. Requested 250 ms windows are floored to a whole number of bins, so the effective T12/T16 window is 240 ms. The 500 ms and 1,000 ms windows are exact.

T12 listening trials align to the source-defined delay epoch; other conditions align to GO. MATLAB closed intervals are converted to Python half-open indexing. Every extracted prefix stays inside one source epoch and one recording block.

## Forward block split

For each participant, blocks retain their recorded order:

- last `max(1, floor(0.2 × number_of_blocks))` blocks: test;
- preceding block: validation;
- all earlier blocks: training.

The model is not refit with validation data. Trials are never randomly distributed across folds.

| Participant | Training blocks | Validation | Test blocks |
|---|---|---|---|
| T12 | 5–11 | 12 | 13–14 |
| T15 | 3–5 | 7 | 8 |
| T16 | 3, 6–11 | 12 | 13–14 |
| T17 | 16–17 | 18 | 19 |

## Tasks

1. Seven-way word classification within imagined trials.
2. Seven-way word classification within attempted trials.
3. Seven-way word classification within listening trials.
4. Four-way instructed-condition classification: attempted, imagined, listening, and idle.

The fourth task identifies cued task condition. It is not treated as an isolated measure of volition because cues, sensory input, behavior, and task phase can all contribute.

## Models

- Shrinkage LDA: scikit-learn `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` with equal class priors.
- Diagonal LDA: class means and a pooled per-feature diagonal variance, followed by a softmax over negative diagonal Mahalanobis distances.

No neural-network training or hyperparameter search is performed.

## Permutation diagnostic

For each 500 ms model/task pair, training labels are permuted 49 times independently within training blocks. Test labels and features remain untouched. These permutations are a structured diagnostic and are not presented as multiplicity-corrected confirmatory inference.

## Streaming replay

The 500 ms condition and imagined-word classifiers run every 100 ms within each recording block. Windows never cross a block boundary. The trigger emits after two consecutive updates that:

1. classify the condition as imagined;
2. predict the same word;
3. exceed the selected imagined-state score threshold.

After emission, the trigger disarms until at least two non-imagined updates and a one-second refractory interval have occurred.

Threshold candidates are `0`, `0.5`, `0.7`, `0.9`, `0.99`, and `1.01`. The last is an explicit mute candidate because classifier scores cannot exceed one. Selection uses validation data only. A threshold is eligible only if it produces at most one event per minute across **all recorded time outside imagined GO epochs** and zero duplicate events during imagined epochs. The false-output denominator therefore includes delays, inter-trial intervals, and attempted, listening, and idle epochs. Among eligible thresholds, selection maximizes the fraction of *all* imagined trials whose first output is correct within one second; ties favor the lower threshold.

## Outcome accounting

The first event in an imagined GO epoch is labeled fast-correct, fast-wrong, late-first, or miss. Later events are counted as duplicates. `continuous_nontarget_events` includes every event outside imagined GO epochs; `continuous_nontarget_seconds` includes every recorded bin outside those epochs. Events within idle, listening, and attempted epochs are additionally broken out with their own exposure durations, and the remainder stays visible as `outside_scored_events`.

## Causality boundary

Feature extraction does not use bins after the evaluated time point, and there is no symmetric smoothing or whole-test-block normalization. This establishes causality only at the level of the released binned features. The online causality of the source study's regression reference, filtering, RMS threshold calibration, and other upstream preprocessing is not re-established here.
