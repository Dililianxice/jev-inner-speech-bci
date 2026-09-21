# Jev × Inner-Speech BCI

### Can a BCI do more with less neural input?

[中文说明](README_zh.md)

This project started with a practical question rather than a new decoder architecture:

> If a future BCI can recover only a few noisy bits of intent, can a fast semantic model help turn those bits into something useful without pretending to read more from the brain than it actually did?

That question led us to [Jev](https://docs.typesafe.ai/introduction), TypeSafe's System One model for typed judgments, choices, probabilities, and confidence. Jev looked interesting for a BCI because it can act as a structured semantic prior: application context proposes likely goals, neural evidence supplies the user's choice, and ordinary code decides whether there is enough evidence to act.

We tested the attractive part first. In a separate synthetic interaction pilot, Jev reduced the number of simulated user inputs when context was informative. Then we tested the uncomfortable part: whether public intracortical inner-speech data could support the fast, autonomous expression event that such a system would need.

The answer was mixed—and more useful than a clean demo:

- **Jev showed an input-efficiency signal:** 94.2% task success with 6.507 mean inputs, versus 96.0% and 8.043 inputs for a local embedding baseline.
- **The signal was not uniquely Jev:** a matched fast general-purpose LLM reached 94.2% with 6.128 inputs.
- **Neural content could be strong for one participant:** T16 reached 78.6% seven-word accuracy from 500 ms of held-out data.
- **Autonomous output was not solved:** every non-mute trigger violated the frozen validation constraints, so the only eligible policy stayed silent and missed all test expressions.

This repository keeps all four facts together. Jev is the interaction-layer idea that motivated the experiment; the neural benchmark shows exactly where that idea still lacks a reliable control signal.

![Seven-word decoding across prefix lengths](figures/word_prefix.png)

## Why Jev is interesting here

Most language-model integrations generate text and then ask software to parse it. Jev instead returns structured decisions that code can combine directly. For a future BCI, that suggests a useful division of labor:

```text
neural evidence  +  Jev semantic prior  +  explicit decision rules
      what?              likely goal              whether to act
```

The synthetic pilot supports this as a hypothesis, not yet as a product claim. Jev atomic features reduced inputs by 19.1% relative to the local embedding baseline while success decreased by 1.83 percentage points. A six-topic paired cluster bootstrap gave an exploratory 95% interval of 3.7%–30.5% for input reduction and −3.50–0.00 percentage points for the success difference.

The benefit also depended on context. When the suggestion was uninformative, Jev atomic success fell to 87.3%; when it opposed the simulated user's target, success fell to 83.7%. The full sanitized aggregates are in [`results/jev_synthetic_pilot_summary.csv`](results/jev_synthetic_pilot_summary.csv) and the interpretation is in [JEV_EVALUATION.md](docs/JEV_EVALUATION.md).

Jev was **not** used to extract neural features, fit neural classifiers, tune test results, or process participant data. No neural data were sent to TypeSafe or any other model provider. That boundary matters: a semantic prior may reduce interaction, but it cannot create neural information that was never decoded.

## The neural benchmark

The benchmark asks two questions that are easy to blur together:

1. **Event-aligned content:** if an instructed expression onset is already known, how much word identity is available in the next 250, 500, or 1,000 ms?
2. **Autonomous output:** can the same simple models decide *when* to emit a word while controlling false and repeated outputs across continuous recorded time?

Using forward held-out recording blocks from four participants in the Stanford `interleavedVerbalBehaviors` dataset, shrinkage LDA reached the following seven-word inner-speech accuracies at 500 ms (chance: 14.3%):

| Participant | Accuracy | Correct / test trials |
|---|---:|---:|
| T12 | 32.1% | 9 / 28 |
| T15 | 30.0% | 21 / 70 |
| T16 | **78.6%** | **22 / 28** |
| T17 | 32.9% | 23 / 70 |

T16 reached 11/14 in each of two held-out blocks. This is promising short-window information for one participant, not uniform performance or unrestricted thought decoding.

No non-mute streaming threshold passed the validation constraints. The selected policy emitted nothing and missed all 196 held-out inner-speech trials per model. Zero false outputs from a mute policy are reported as failure, not success.

## What this work contributes

The classifiers are intentionally standard: shrinkage LDA and diagonal-covariance LDA. The contribution is the **evaluation contract**:

- recording-order block splits instead of random trial mixing;
- train-only fitting and validation-only trigger selection;
- prefix features with no test-block centering or temporal smoothing;
- separate reporting of cued content decoding and autonomous triggering;
- false-output exposure covering all recorded time outside imagined GO epochs;
- every miss, wrong word, duplicate, and negative result retained;
- an explicit mute candidate that cannot be mislabeled as a useful interface;
- participant- and block-level results rather than a pooled headline number.

The appropriate description is a reproducible evaluation benchmark and methods case study. It is not a new neural decoder or a new neuroscience mechanism. See [Novelty and claim boundaries](docs/NOVELTY_AND_SCOPE.md).

## Data and reproduction

This repository does **not** redistribute neural data. Download `interleavedVerbalBehaviors.zip` from the official [Dryad dataset](https://datadryad.org/dataset/doi:10.5061/dryad.gf1vhhn1j), then extract it locally.

Expected archive:

- size: `777419058` bytes
- SHA-256: `19f90f09f2ea32f1428b7cc1c7dd8c0606dfbc988c57fcaf64aea03e77b9d409`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'

inner-speech-benchmark \
  --data-dir /path/to/interleavedVerbalBehaviors \
  --output-dir reproduced_results

inner-speech-report \
  --output-dir reproduced_results \
  --figure-dir reproduced_figures

pytest
```

For the exact release environment, use `requirements-lock.txt` and consult [`runtime_environment.json`](runtime_environment.json). The neural benchmark requires no provider API and makes no external model calls.

## Repository map

- `src/inner_speech_benchmark/`: feature extraction, LDA baselines, streaming trigger, and reports.
- `results/`: complete frozen neural results and the sanitized Jev pilot aggregate.
- `figures/`: publication-ready PNG and PDF summaries.
- `docs/METHODS.md`: exact split, feature, null, and streaming definitions.
- `docs/RESULTS.md`: participant-level results and limitations.
- `docs/JEV_EVALUATION.md`: Jev protocol, results, stress tests, and claim boundaries.
- `results/manifest.json`, `runtime_environment.json`, and `CHECKSUMS.sha256`: provenance and integrity.
- `tests/`: temporal, split, exposure, and scoring invariants.

## Citation and attribution

Benchmark author and maintainer: [Dililianxice](https://github.com/Dililianxice). Machine-readable metadata are in [CITATION.cff](CITATION.cff).

If you reuse this benchmark, cite this repository, the original study, and the dataset:

> Dililianxice (2026). *Jev × Inner-Speech BCI* (v0.1.0). GitHub repository.

> Kunz, E., Abramovich Krasa, B., Kamdar, F., et al. (2025). *Inner speech in motor cortex and implications for speech neuroprostheses*. Cell. https://doi.org/10.1016/j.cell.2025.06.015

> Kunz, E., Abramovich Krasa, B., Kamdar, F., et al. (2025). *Inner speech in motor cortex and implications for speech neuroprostheses* [Dataset]. Dryad. https://doi.org/10.5061/dryad.gf1vhhn1j

Code, documentation, and figures use the [MIT License](LICENSE). Derived CSV and JSON tables use [CC0 1.0](LICENSE-DATA.md).

## Responsible interpretation

The words and task epochs were prompted. Decodable activity may include cue memory, preparation, and inner-speech execution. GO-relative latency is not private-thought onset. Four participants, one session each, and only 4–5 idle test trials per participant do not establish cross-day stability, daily-life safety, or arbitrary thought reading.
