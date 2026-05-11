# SoloSpeak Subagent Role Definitions

This document defines the autonomous agent roles that can be invoked via Claude Code skills to parallelize hackathon work.

## data-curator
**Trigger:** `skills/data_curation/SKILL.md`
**Scope:** Download, verify, and prepare training manifests.
**Input:** Dataset name or "all"
**Output:** Manifests in `data/manifests/`, STATS.md, and confirmation of speaker-disjoint splits.
**Must not:** Modify any file outside `data/` or `scripts/`.

## trainer
**Trigger:** `skills/training/SKILL.md`
**Scope:** Run a single training stage or the production Stage 1-7 path.
**Input:** Stage number (1, 2, 3, 4, 4d, 5, 6, or 7), optional resume checkpoint path.
**Output:** Checkpoint in `checkpoints/stage<N>_*.pt`, W&B run link when available, GO/NO-GO result.
**Must not:** Advance to next stage if GO/NO-GO gate fails.

## evaluator
**Trigger:** `skills/evaluation/SKILL.md`
**Scope:** Run KPI checks, Stage 7 verification, ablation study, and probe verification.
**Input:** Checkpoint path, optional ablation config.
**Output:** `reports/kpi_final.json`, `reports/ablation_table.md`, `reports/probes.json`, or Stage 7 verification summaries.
**Must not:** Use test set for any hyperparameter tuning decision.

## deployer
**Trigger:** `skills/deployment/SKILL.md`
**Scope:** Export the corrected Stage 7 deployable, export to ONNX, quantize to INT8, run validation gates, package OTA artifact.
**Input:** Checkpoint path or Stage 7 deployable path.
**Output:** `exports/solospeak_stage7_deployable_corrected.pt`, `artifacts/solospeak_int8.onnx`, `artifacts/ValidationReport.json`.
**Must not:** Mark any gate as passed unless the numeric check actually passes.

## debugger
**Trigger:** `skills/debugging/SKILL.md`
**Scope:** Diagnose failing GO/NO-GO gates, loss divergence, or evaluation regressions.
**Input:** Stage number, symptom description, W&B run ID (optional).
**Output:** Root-cause hypothesis and concrete remediation steps from the Failure Mode Playbook.
**Must not:** Modify training code without first documenting the hypothesis and expected fix.
