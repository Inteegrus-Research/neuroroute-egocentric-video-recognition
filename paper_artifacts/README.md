# Paper Artifact Package

Generated: 2026-09-29T10:12:09.164390

This directory is derived from the frozen experimental artifacts and contains the figures, tables, manuscript numbers, claim audit, and reference-planning material for the IRIS paper.

## Main figures

- `fig01_study_framework` — conceptual study/method overview.
- `fig02_dataset_and_split` — dataset scale and subject-independent split.
- `fig03_visual_baselines` — frozen video-only baseline ladder.
- `fig04_eeg_teacher_diagnostics` — EEG teacher versus chance and shuffle control.
- `fig05_compute_budget` — macro-F1 versus temporal budget / estimated FLOPs.
- `fig06_temporal_robustness` — corruption robustness and NeuroRoute minus baseline.
- `fig07_controls_and_deployment` — negative controls + EEG-free deployment audit.
- `figS01_subject_level_delta` — supplementary paired subject-level differences.

## Main tables

- `table01_dataset_and_split`
- `table02_visual_baselines`
- `table03_eeg_teacher`
- `table04_compute_budget`
- `table05_temporal_robustness`
- `table06_controls_and_deployment`
- `table07_claim_evidence_matrix`

## Core recorded numbers

- synchronized rows: **227,864**
- subjects: **40**
- semantic classes: **27**
- chance macro-F1: **0.03704**
- Phase-2 visual macro-F1: **0.03757**
- EEG teacher test macro-F1: **0.03888**
- NeuroRoute 50% macro-F1: **0.03483**
- robustness mean: **0.03530 → 0.03293**
- temporal FLOPs: **35,927,552 → 17,905,152**
- FLOP reduction: **50.16%**
- final NeuroRoute gate: **INCONCLUSIVE**

## Important scientific constraint

This package is designed to make the manuscript **stronger through precision, not overclaiming**. In particular, the final record does not support a statement that NeuroRoute improved recognition accuracy or corruption robustness. The most concrete positive efficiency result is the approximately 50% temporal FLOP reduction at the predefined 50% budget, together with an EEG-free deployment audit.
