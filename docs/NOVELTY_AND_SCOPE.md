# Novelty and claim boundaries

## Objective assessment

### What is not novel

- Linear discriminant analysis for neural decoding.
- Diagonal covariance or shrinkage covariance estimation.
- Fixed post-cue spike-count/rate windows.
- Sliding-window state classification and thresholded emission.
- Offline decoding of a small inner-speech vocabulary.

Prior intracortical studies have already shown offline and online inner-speech decoding. The source Stanford study reports real-time imagined-sentence decoding, an intent-related neural dimension, and privacy-oriented lock/unlock strategies. Wandelt et al. also reported offline and online word decoding from supramarginal gyrus recordings. This repository should not claim to originate inner-speech decoding or privacy gating.

### What is useful and differentiating

The benchmark makes a systems distinction that is often blurred in informal descriptions:

> **Evidence that word identity is decodable after a known cue is not evidence that a continuous BCI knows when the user intends to communicate.**

The repository operationalizes that distinction with forward block holdout, a validation-only trigger policy, explicit exposure denominators, a mute candidate, and outcome accounting in which every miss and duplicate remains visible. The strong T16 event-aligned result and the failed autonomous trigger are evaluated on the same released features, making the gap concrete.

This is a meaningful engineering and reproducibility contribution. A suitable claim is:

> We provide a reproducible benchmark that separates cued content decodability from autonomous output utility in a public intracortical inner-speech dataset.

An unsuitable claim is:

> We developed a novel thought decoder that outputs a word as soon as a person thinks it.

## Scientific scope

Supported:

- In one held-out session block split, T16 had strong seven-word identity information in a 500 ms post-event window.
- The fixed simple trigger did not find a useful safety/recall operating point on the validation blocks.
- Participant heterogeneity is material and should not be hidden by pooled accuracy.

Not supported:

- arbitrary or unseen-word decoding;
- spontaneous thought-onset detection;
- private-thought reading;
- cross-day or cross-participant generalization;
- complete real-time causality of upstream preprocessing;
- a neural mechanism or causal role for any cortical area;
- superiority over the source paper's recurrent decoder or privacy methods.

## Relation to prior work

1. Kunz et al., *Inner speech in motor cortex and implications for speech neuroprostheses*, Cell (2025); [Dryad dataset](https://doi.org/10.5061/dryad.gf1vhhn1j).
2. Wandelt et al., *Representation of internal speech by single neurons in human supramarginal gyrus*, Nature Human Behaviour 8, 1136–1149 (2024), [DOI](https://doi.org/10.1038/s41562-024-01867-y).

