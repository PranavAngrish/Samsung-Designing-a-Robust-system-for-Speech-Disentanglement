# Fairness Report

## Status

The final Stage 7 artifact was validated on internal GSC-style Q1/Q2/Q3/Q4 trials plus
40,000 external false-accept trials from Common Voice, LibriSpeech, background noise,
and UrbanSound8K. Those sources improve acoustic diversity, but they do not provide a
complete demographic fairness evaluation.

Demographic fields such as gender, age bucket, and accent bucket are not reliably
available in the committed manifests. This report therefore documents the limitation
instead of claiming demographic parity.

## Final Stage 7 Metrics

| Metric | Value |
|---|---:|
| TA clean | 93.97% |
| Q2 imposter rejection | 95.17% |
| Q3 wrong-word rejection | 97.83% |
| Q4 background rejection | 100.00% |
| Quadrant minimum | 93.97% |
| External FA rate | 0.300% |
| External FA count | 120 / 40,000 |

## Available Subgroup Metadata

The committed smoke manifests are kept for CI and local wiring checks. They are not a
fairness dataset. In those smoke manifests, optional demographic fields are unavailable:

| Field | Available Rows | Missing Rows |
|---|---:|---:|
| gender | 0 | all smoke rows |
| age_bucket | 0 | all smoke rows |
| accent_bucket | 0 | all smoke rows |

## Dataset Skew Acknowledgement

The training and validation sources are likely skewed toward English speech, public
speech datasets, and recording conditions that differ from real Samsung-device use.
Common Voice and LibriSpeech add variety, but they do not guarantee balanced coverage
over gender, age, accent, language, microphone type, disability, or noisy household
conditions.

## Planned Mitigations

For v1.1, add a pinned multilingual Common Voice release and require demographic
availability checks before reporting subgroup metrics.

For shared-device use, add enrollment warnings when two profiles have high template
similarity. This reduces multi-user confusion risk and should be reported separately
from demographic fairness.

Before any commercial claim, run device-recorded subgroup evaluation with consented
metadata and report confidence intervals for each subgroup.
