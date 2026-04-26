# Fairness Report

## Status

This report is complete for the smoke workflow and intentionally caveated for final
evaluation. Optional demographic metadata is unavailable in the smoke manifests, so only
the mandatory keyword-syllable subgroup is populated.

## Subgroup KPI Table

Copied from `reports/subgroup_report.md` after the Phase 4 smoke evaluation:

| Subgroup | Bucket | Share / Metric |
|---|---|---:|
| keyword_syllable_count | 2 | 0.0000 |
| keyword_syllable_count | 3 | 0.0000 |
| keyword_syllable_count | 4+ | 1.0000 |
| gender | unavailable | null |
| age_bucket | unavailable | null |
| accent_bucket | unavailable | null |

## Mandatory Subgroup Result

`keyword_syllable_count` is the required subgroup. On the smoke report, the best bucket
is `4+` at `1.0000` and the lowest populated numeric buckets are `2` and `3` at `0.0000`,
so the smoke macro-TA gap is `1.0000`.

This is not a real fairness claim because smoke manifests are tiny generated fixtures.
The final report must recompute this after full data preparation and real evaluation.

## Dataset Skew Acknowledgement

The planned full training mix includes VoxCeleb-style speaker data, which is known to be
skewed toward English-speaking adult public figures and underrepresents children,
elderly speakers, many accents, and many languages. In the current smoke data,
demographic fields are absent, so the measured available counts are:

| Field | Available Rows | Missing Rows |
|---|---:|---:|
| gender | 0 | 46 |
| age_bucket | 0 | 46 |
| accent_bucket | 0 | 46 |

## Planned Mitigations

For v1.1, add a pinned multilingual Common Voice release and require demographic
availability checks before reporting subgroup metrics. The target date is post-hackathon
hardening in Q3 2026.

For shared-device use, add enrollment warnings when two profiles have high template
similarity. This reduces multi-user confusion risk and should be reported separately from
demographic fairness.
