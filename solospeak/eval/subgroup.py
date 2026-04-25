"""Stratified evaluation by demographic subgroup.

Reports KPIs stratified by: gender, age bucket, accent, keyword syllable count.
Documents gaps honestly — required for the Phase 2 report.
"""

from __future__ import annotations

from pathlib import Path

from solospeak.utils.types import KPIResult

SUBGROUP_AXES = ["gender", "age_bucket", "accent", "keyword_syllables"]
AGE_BUCKETS = ["18-30", "30-50", "50+"]
ACCENT_BUCKETS = ["US", "UK", "Other"]


def run_subgroup_eval(
    checkpoint_path: Path,
    test_manifest: Path,
) -> dict[str, dict[str, KPIResult]]:
    """Run KPI evaluation per subgroup. Returns {axis: {subgroup: KPIResult}}."""
    raise NotImplementedError("Implement in Phase 4")
