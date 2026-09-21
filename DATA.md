# Data acquisition and provenance

## Source

- Dataset: [Inner speech in motor cortex and implications for speech neuroprostheses](https://datadryad.org/dataset/doi:10.5061/dryad.gf1vhhn1j)
- Dataset DOI: `10.5061/dryad.gf1vhhn1j`
- File used: `interleavedVerbalBehaviors.zip`
- Expected size: `777419058` bytes
- Expected SHA-256: `19f90f09f2ea32f1428b7cc1c7dd8c0606dfbc988c57fcaf64aea03e77b9d409`

The source archive contains four MATLAB files, one each for participants T12, T15, T16, and T17, plus a README. It contains binned threshold-crossing counts, block identifiers, trial epochs, cue identifiers, cue text, channel-set metadata, and bin size.

Dryad publishes research datasets under the [CC0 1.0 public-domain dedication](https://datadryad.org/terms). Scholarly citation is still expected. The derived CSV and JSON tables in this repository are likewise dedicated under [CC0 1.0](LICENSE-DATA.md).

## What this repository contains

The `results/` directory contains derived predictions, confusion counts, summary metrics, and streaming-event tables. It does not contain the released neural matrices or reconstructed trial-level neural features.

## Local integrity check

```bash
stat -c '%s' interleavedVerbalBehaviors.zip
sha256sum interleavedVerbalBehaviors.zip
unzip -t interleavedVerbalBehaviors.zip
```

## Human-subject data

Dryad states that the data are anonymized and referenced by participant code, and that informed consent for sharing was obtained. Users remain responsible for following the original dataset's terms, institutional policies, and citation requirements.
