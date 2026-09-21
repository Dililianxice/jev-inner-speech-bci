# Jev evaluation and disclosure

## Why Jev was considered

The long-term project asks whether a future BCI can help a user reach a goal with fewer neural inputs. [TypeSafe describes Jev](https://docs.typesafe.ai/introduction) as a System One model that evaluates typed questions against application state and returns structured choices, scores, truth values, probability distributions, and confidence. That interface made Jev a plausible candidate for a **semantic prior**: application context could rank likely goals, while neural evidence would remain responsible for user-specific selection.

This is an interaction-layer hypothesis. Jev does not add information to neural recordings and should not be described as a neural decoder.

## Separate synthetic pilot

Before the public neural benchmark, Jev was evaluated in a frozen synthetic three-slot workspace-configuration task. Each simulated episode had a private target, noisy four-choice user inputs, and a more reliable yes/no confirmation channel. The semantic model saw the suggested theme and visible candidates, not the private target. Each reported condition contains 600 simulated episodes nested within six independent held-out topic clusters; repeated episodes are not treated as 600 independent topics.

The main aggregate results were:

| Method | All three slots correct | Mean user inputs |
|---|---:|---:|
| User input only | 93.8% | 9.027 |
| TF-IDF prior + fusion | 95.5% | 8.945 |
| Local embedding prior + fusion | 96.0% | 8.043 |
| Jev atomic features + fusion | 94.2% | 6.507 |
| Jev Choice prior + fusion | 93.8% | 5.947 |

Relative to the local embedding baseline, Jev atomic features reduced inputs by 19.1% while success fell by 1.83 percentage points. A paired cluster bootstrap resampled the six held-out topic clusters 4,000 times: the exploratory 95% interval was 3.7% to 30.5% for relative input reduction and −3.50 to 0.00 percentage points for the success difference. The predeclared continuation target required at least a 20% input reduction, at least 90% absolute success, and no more than a two-point success loss. The point estimate missed the input gate, and the success interval crossed below the −2 point non-inferiority margin. With only six independent clusters, this is a screening result rather than a formal non-inferiority study.

A matched fast general-purpose LLM produced 94.2% success with 6.128 mean inputs for its atomic variant. This did not show a unique Jev advantage. Provider names, endpoints, payloads, and raw responses are intentionally omitted because they are not needed to interpret or reproduce the neural benchmark.

## Stress tests

| Context condition | User-only success | Jev atomic success | Jev Choice success |
|---|---:|---:|---:|
| Informative suggestion | 93.8% | 94.2% | 93.8% |
| Suggestion uninformative | 93.8% | 87.3% | 85.2% |
| User target opposes suggestion | 93.8% | 83.7% | 81.0% |

The input reduction depended on context being informative about the hidden target. When that assumption failed, semantic priors could make the interaction worse.

The sanitized aggregate table for all five context/channel conditions is committed as [`results/jev_synthetic_pilot_summary.csv`](../results/jev_synthetic_pilot_summary.csv). It contains no prompts, provider responses, credentials, user data, or neural data. The synthetic pilot is disclosed as historical component screening; this repository does not claim to reproduce provider inference.

## Decision for this benchmark

Jev was not used in the neural pipeline because:

1. the neural dataset already displays the target word as a cue, so giving that cue to a semantic model would leak the answer rather than decode it;
2. the unresolved bottleneck was autonomous expression-state detection, not candidate semantics;
3. the synthetic pilot did not establish a unique or robust Jev benefit;
4. neural data should remain local unless an explicit, scientifically justified protocol requires otherwise.

No provider credentials, endpoints, request payloads, raw responses, or participant data are included in this repository. Reproduction of the neural benchmark requires no API account.

## Appropriate wording

> Jev was evaluated as a structured semantic prior in a separate synthetic interaction pilot. It reduced simulated input under informative context but did not show a unique robust advantage, and it was not used for neural decoding in this benchmark.

This wording records the work accurately without implying that Jev improved neural accuracy or enabled thought decoding.
