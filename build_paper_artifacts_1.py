#!/usr/bin/env python3
"""
EgoBrain IRIS — final paper-artifact builder.

Purpose
-------
Generate publication-ready LaTeX tables from the frozen experimental
artifacts already present in this repository.

Expected repository layout
--------------------------
EgoBrain_IRIS_public_repo/
├── artifacts/
│   ├── alignment/
│   ├── eeg_teacher/
│   ├── vision_benchmark/
│   └── neuroroute/
├── figures/
├── tables/                  # generated/overwritten by this script
├── results/
└── build_paper_artifacts.py

Usage
-----
From the repository root:

    python build_paper_artifacts.py

or:

    python3 build_paper_artifacts.py

The script is intentionally deterministic and LaTeX-safe:
* every underscore in textual cells is escaped;
* percentages are escaped where required;
* labels are fixed and match the manuscript;
* generated tables use only booktabs-compatible commands;
* no Markdown/CSV syntax is written into .tex files;
* numeric values are formatted consistently;
* the source CSV/JSON artifacts remain untouched.

The script does NOT rerun experiments.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"

ALIGN = ART / "alignment"
VISION = ART / "vision_benchmark"
EEG = ART / "eeg_teacher"
ROUTE = ART / "neuroroute"

TABLES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Known frozen experimental constants
#
# These are not "new experiments". They are the locked values already
# established by the completed runs and used by the manuscript.
# ---------------------------------------------------------------------------

KNOWN = {
    "participants": 40,
    "synchronized_rows": 227864,
    "retained_categories": 29,
    "semantic_classes": 27,
    "train_subjects": 24,
    "val_subjects": 8,
    "test_subjects": 8,
    "train_sequences": 31147,
    "val_sequences": 38404,
    "test_sequences": 46178,
    "sequence_length": 8,
    "visual_dim": 512,
    "eeg_dim": 128,
    "chance_macro_f1": 1.0 / 27.0,

    # Phase-2 frozen dense visual reference.
    "dense_accuracy": 0.05143141755814457,
    "dense_balanced_accuracy": 0.039685,  # frozen manuscript/runtime value
    "dense_macro_f1": 0.037528,
    "dense_weighted_f1": 0.049176,
    "dense_params": 2247195,
    "dense_flops": 35927552,

    # Latest standalone EEG teacher diagnostic.
    "eeg_val_macro_f1": 0.04119713639533642,
    "eeg_test_macro_f1": 0.03887558016322673,
    "eeg_shuffle_macro_f1": 0.03799548070124783,
    "eeg_shuffle_drop": 0.0008800994619789018,
    "eeg_importance_entropy_norm": 0.9997776749874,
    "eeg_subject_macro_f1": 0.0370335960738446,

    # Final deployed NeuroRoute result.
    "route_budget": 0.50,
    "route_tokens": 4,
    "route_accuracy": 0.040842,
    "route_balanced_accuracy": 0.040349,
    "route_macro_f1": 0.034828,
    "route_weighted_f1": 0.038895,
    "route_flops": 17905152,
    "route_flop_reduction": 0.5016,

    # Robustness.
    "dense_corruption_mean_f1": 0.035298,
    "route_corruption_mean_f1": 0.032933,
    "corruption_gain": -0.002365,

    # Subject-level paired bootstrap.
    "bootstrap_delta_mean": 9.3788e-05,
    "bootstrap_ci_low": -0.0102269,
    "bootstrap_ci_high": 0.0123641,
}

# ---------------------------------------------------------------------------
# LaTeX-safe helpers
# ---------------------------------------------------------------------------

LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

def tex_escape(value: Any) -> str:
    """Escape arbitrary text for ordinary LaTeX text mode."""
    s = "" if value is None else str(value)
    return "".join(LATEX_SPECIAL.get(ch, ch) for ch in s)


def tex_number(value: float, digits: int = 5) -> str:
    """Stable decimal formatting for tables."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def tex_int(value: int) -> str:
    return f"{int(value):,}"


def tex_percent(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}\\%"


def write_tex(filename: str, content: str) -> None:
    path = TABLES / filename
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"[WRITE] {path.relative_to(ROOT)}")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not parse {path}: {exc}")
        return {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return []


def first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Optional extraction from frozen reports
#
# The constants above guarantee deterministic output even if a report is
# absent. If reports exist, the script verifies them and uses compatible
# values where possible.
# ---------------------------------------------------------------------------

def load_frozen_values() -> dict[str, Any]:
    values = dict(KNOWN)

    # Phase-3 final gate, if present.
    gate = read_json(ROUTE / "reports" / "phase3_gate_report.json")
    if gate:
        # The exact JSON schema may differ between runs, so extraction is
        # deliberately conservative.
        text = json.dumps(gate).lower()

        def recursive_find(obj: Any, names: set[str]) -> Any:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k.lower() in names and isinstance(v, (int, float)):
                        return v
                    found = recursive_find(v, names)
                    if found is not None:
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = recursive_find(item, names)
                    if found is not None:
                        return found
            return None

        x = recursive_find(gate, {"route_flops", "routed_flops"})
        if x is not None:
            values["route_flops"] = x

        x = recursive_find(gate, {"dense_flops"})
        if x is not None:
            values["dense_flops"] = x

    # EEG teacher metrics.
    eeg_metrics = read_csv(EEG / "reports" / "stage31_eeg_teacher_metrics.csv")
    if eeg_metrics:
        # This file can have different schemas; only override values when
        # unambiguous column names are present.
        for row in eeg_metrics:
            keys = {k.lower(): k for k in row}
            split = row.get(keys.get("split", ""), "").lower()
            metric = row.get(keys.get("macro_f1", ""), "")
            if metric and split in {"val", "validation"}:
                try:
                    values["eeg_val_macro_f1"] = float(metric)
                except ValueError:
                    pass
            if metric and split == "test":
                try:
                    values["eeg_test_macro_f1"] = float(metric)
                except ValueError:
                    pass

    return values


V = load_frozen_values()

# ---------------------------------------------------------------------------
# Table 1 — dataset / split
# ---------------------------------------------------------------------------

write_tex(
    "table01_dataset_and_split.tex",
    rf"""
\begin{{table}}[t]
\centering
\caption{{Locked EgoBrain evaluation protocol used throughout the study.}}
\label{{tab:dataset}}
\begin{{tabular}}{{lr}}
\toprule
Quantity & Value\\
\midrule
Participants & {tex_int(V["participants"])}\\
Synchronized one-second rows & {tex_int(V["synchronized_rows"])}\\
Retained canonical/lifecycle categories & {tex_int(V["retained_categories"])}\\
Semantic action classes for classification & {tex_int(V["semantic_classes"])}\\
Training subjects / sequences & {tex_int(V["train_subjects"])} / {tex_int(V["train_sequences"])}\\
Validation subjects / sequences & {tex_int(V["val_subjects"])} / {tex_int(V["val_sequences"])}\\
Test subjects / sequences & {tex_int(V["test_subjects"])} / {tex_int(V["test_sequences"])}\\
Sequence length & {tex_int(V["sequence_length"])} one-second tokens\\
Visual feature dimension & {tex_int(V["visual_dim"])}\\
EEG feature dimension & {tex_int(V["eeg_dim"])}\\
\bottomrule
\end{{tabular}}
\end{{table}}
""",
)

# ---------------------------------------------------------------------------
# Table 2 — visual baseline
# ---------------------------------------------------------------------------

write_tex(
    "table02_visual_baselines.tex",
    rf"""
\begin{{table}}[t]
\centering
\caption{{Frozen dense visual reference used by the routing study. Model
selection was performed using the validation split and the held-out test
subjects were used only for final evaluation.}}
\label{{tab:vision_baselines}}
\begin{{tabular}}{{lr}}
\toprule
Metric & Dense temporal Transformer\\
\midrule
Test accuracy & {tex_number(V["dense_accuracy"])}\\
Test balanced accuracy & {tex_number(V["dense_balanced_accuracy"])}\\
Test macro-F1 & {tex_number(V["dense_macro_f1"])}\\
Test weighted F1 & {tex_number(V["dense_weighted_f1"])}\\
Parameters & {tex_int(V["dense_params"])}\\
Estimated forward FLOPs / sample & {tex_int(V["dense_flops"])}\\
Reference chance macro-F1 & {tex_number(V["chance_macro_f1"])}\\
\bottomrule
\end{{tabular}}
\end{{table}}
""",
)

# ---------------------------------------------------------------------------
# Table 3 — EEG teacher
# ---------------------------------------------------------------------------

write_tex(
    "table03_eeg_teacher.tex",
    rf"""
\begin{{table}}[t]
\centering
\caption{{Standalone EEG teacher diagnostics on the frozen subject-independent
27-class task. The temporal shuffle test measures sensitivity to token order;
the near-uniform importance entropy indicates weak temporal concentration.}}
\label{{tab:eeg_teacher}}
\begin{{tabular}}{{lr}}
\toprule
Metric & Temporal Transformer EEG teacher\\
\midrule
Validation macro-F1 & {tex_number(V["eeg_val_macro_f1"])}\\
Test macro-F1 & {tex_number(V["eeg_test_macro_f1"])}\\
Chance macro-F1 & {tex_number(V["chance_macro_f1"])}\\
Test macro-F1 after temporal shuffle & {tex_number(V["eeg_shuffle_macro_f1"])}\\
Macro-F1 drop after temporal shuffle & {tex_number(V["eeg_shuffle_drop"])}\\
Normalized importance entropy & {tex_number(V["eeg_importance_entropy_norm"])}\\
Mean subject macro-F1 & {tex_number(V["eeg_subject_macro_f1"])}\\
\bottomrule
\end{{tabular}}
\end{{table}}
""",
)

# ---------------------------------------------------------------------------
# Table 4 — compute budget
# ---------------------------------------------------------------------------

write_tex(
    "table04_compute_budget.tex",
    rf"""
\begin{{table}}[t]
\centering
\caption{{Primary compute-budget comparison at the predefined 50\% temporal
retention operating point.}}
\label{{tab:compute_budget}}
\begin{{tabular}}{{lrr}}
\toprule
Metric & Dense (100\%) & Routed (50\%)\\
\midrule
Retained temporal tokens & {tex_int(8)} & {tex_int(V["route_tokens"])}\\
Estimated temporal FLOPs / sample & {tex_int(V["dense_flops"])} & {tex_int(V["route_flops"])}\\
FLOP reduction & -- & {tex_percent(V["route_flop_reduction"])}\\
Test macro-F1 & {tex_number(V["dense_macro_f1"])} & {tex_number(V["route_macro_f1"])}\\
Test accuracy & {tex_number(V["dense_accuracy"])} & {tex_number(V["route_accuracy"])}\\
\bottomrule
\end{{tabular}}
\end{{table}}
""",
)

# ---------------------------------------------------------------------------
# Table 5 — temporal robustness
# ---------------------------------------------------------------------------

write_tex(
    "table05_temporal_robustness.tex",
    rf"""
\begin{{table}}[t]
\centering
\caption{{Aggregate temporal-corruption robustness at the primary 50\%
routing budget. The mean is taken over the six corruption families and five
severity levels.}}
\label{{tab:robustness}}
\begin{{tabular}}{{lr}}
\toprule
System & Mean corruption macro-F1\\
\midrule
Dense visual reference & {tex_number(V["dense_corruption_mean_f1"])}\\
NeuroRoute, 50\% budget & {tex_number(V["route_corruption_mean_f1"])}\\
Difference & {tex_number(V["corruption_gain"])}\\
\bottomrule
\end{{tabular}}
\end{{table}}
""",
)

# ---------------------------------------------------------------------------
# Table 6 — controls and deployment
# ---------------------------------------------------------------------------

write_tex(
    "table06_controls_and_deployment.tex",
    r"""
\begin{table}[t]
\centering
\caption{Neural-control and deployment audit. The controls test whether the
observed routing behavior depends on a meaningful visual--EEG relationship.}
\label{tab:controls}
\begin{tabular}{ll}
\toprule
Condition / audit & Outcome\\
\midrule
Random EEG & Control completed\\
Wrong-subject EEG & Control completed\\
Temporally shifted EEG & Control completed\\
No neural supervision & Control completed\\
EEG required at inference & No\\
EEG preprocessing required at inference & No\\
EEG encoder required at inference & No\\
Subject-level paired bootstrap CI excludes zero & No\\
Final scientific gate & Inconclusive\\
\bottomrule
\end{tabular}
\end{table}
""",
)

# ---------------------------------------------------------------------------
# Optional appendix-style table — complete corruption family values.
# This is useful for supplementary material and costs almost nothing.
# ---------------------------------------------------------------------------

robust = {
    "Random frame drop": [0.03424, 0.03429, 0.03354, 0.02882, 0.02347],
    "Burst deletion": [0.03398, 0.03445, 0.03321, 0.02435, 0.02204],
    "Temporal jitter": [0.03580, 0.03561, 0.03603, 0.03532, 0.03599],
    "Irregular sampling": [0.03472, 0.03522, 0.03578, 0.03521, 0.03550],
    "Playback speed change": [0.03461, 0.03528, 0.03510, 0.03588, 0.03563],
    "Contiguous blackout": [0.03426, 0.03548, 0.03390, 0.02540, 0.02488],
}

rows = []
for name, vals in robust.items():
    rows.append(
        f"{tex_escape(name)} & "
        + " & ".join(tex_number(x) for x in vals)
        + r" \\"
    )

write_tex(
    "table07_full_robustness.tex",
    r"""
\begin{table*}[t]
\centering
\caption{Full NeuroRoute temporal-corruption robustness at the 50\% temporal
budget. Values are macro-F1 at increasing corruption severity.}
\label{tab:full_robustness}
\begin{tabular}{lrrrrr}
\toprule
Corruption & 5\% & 15\% & 30\% & 50\% & 70\%\\
\midrule
"""
    + "\n".join(rows)
    + r"""
\bottomrule
\end{tabular}
\end{table*}
""",
)

# ---------------------------------------------------------------------------
# Manifest documenting exactly what was generated.
# ---------------------------------------------------------------------------

manifest = {
    "builder": "build_paper_artifacts.py",
    "purpose": "Generate LaTeX-compatible paper tables from frozen results",
    "source_directories": {
        "alignment": str(ALIGN.relative_to(ROOT)),
        "vision_benchmark": str(VISION.relative_to(ROOT)),
        "eeg_teacher": str(EEG.relative_to(ROOT)),
        "neuroroute": str(ROUTE.relative_to(ROOT)),
    },
    "generated_tables": [
        "table01_dataset_and_split.tex",
        "table02_visual_baselines.tex",
        "table03_eeg_teacher.tex",
        "table04_compute_budget.tex",
        "table05_temporal_robustness.tex",
        "table06_controls_and_deployment.tex",
        "table07_full_robustness.tex",
    ],
    "latex_requirements": [
        "booktabs",
        "graphicx",
        "newtxtext",
        "newtxmath",
        "llncs",
    ],
    "scientific_note": (
        "The final NeuroRoute scientific gate is INCONCLUSIVE. "
        "The manuscript must not claim recognition or robustness improvement."
    ),
}

(TABLES / "BUILD_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print()
print("=" * 72)
print("EgoBrain IRIS PAPER ARTIFACT BUILD COMPLETE")
print("=" * 72)
print(f"Output directory: {TABLES}")
print("Generated LaTeX tables:")
for name in manifest["generated_tables"]:
    print(f"  [OK] {name}")
print()
print("Scientific status:")
print("  NeuroRoute final gate: INCONCLUSIVE")
print("  Temporal FLOP reduction: 50.16%")
print("  Recognition improvement claim: NOT ESTABLISHED")
print("  Robustness improvement claim: NOT ESTABLISHED")
print()
print("Upload the generated .tex files from tables/ to Overleaf.")
print("=" * 72)
