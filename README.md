# EgoBrain — IRIS Research Reproduction Repository

This repository contains the final practical notebooks used to build and evaluate the EgoBrain multimodal computer-vision study.

The code is organized by **reproducible function**, not by the internal experimental "phase" numbering used during development.

## Project focus

The study uses synchronized first-person video and EEG from EgoBrain to investigate whether neural temporal information can act as training-time privileged supervision for temporal video computation.

The deployed route is **video-only**. EEG is used only during training/evaluation experiments.

The current experimental record is deliberately conservative:

- the visual action-recognition baseline is difficult and remains close to 27-class chance;
- the latest standalone EEG teacher shows only weak action-discriminative information and very limited temporal concentration;
- the completed adaptive-routing experiment achieved a measured temporal-computation reduction of **50.16%** at a 50% token budget;
- the final routing gate is **INCONCLUSIVE** because clean recognition and average temporal-robustness performance were not preserved.

The repository therefore records both positive engineering results and negative scientific findings rather than presenting the routing experiment as an accuracy-improving method.

## Final repository structure

```text
EgoBrain_IRIS_public_repo/
├── README.md
├── notebooks/
│   ├── 01_visual_feature_extraction_worker.ipynb
│   ├── 02_eeg_feature_extraction.ipynb
│   ├── 03_temporal_alignment_and_dataset_lock.ipynb
│   ├── 04_visual_baselines_and_temporal_robustness.ipynb
│   ├── 05_eeg_teacher_diagnostics.ipynb
│   └── 06_neuroroute_reproduction.ipynb
├── results/
│   ├── FROZEN_EXPERIMENTAL_RECORD.md
│   └── frozen_experimental_record.json
├── docs/
├── artifacts/
└── .gitignore
```

Raw EgoBrain data and large `.pt` / `.zip` artifacts are intentionally not committed to Git.

## Notebook order

### 1. `01_visual_feature_extraction_worker.ipynb`

GPU worker for 512-D R3D-18 video features.

Input:
- gated EgoBrain dataset
- Kaggle Secret `HF_TOKEN`

Output:
- `features_Pxxxx.pt` for the assigned worker subjects
- worker ZIP

The original production extraction used **5 workers** covering all 40 subjects.

To reproduce:
1. Attach/create a Kaggle notebook with GPU enabled.
2. Add the `HF_TOKEN` Kaggle secret.
3. Set `WORKER_ID = 1,2,3,4,5` in separate runs/notebooks.
4. Collect all 40 subject feature files under one `uniform/` directory.

The canonical feature format is `[N_seconds, 512]`, `torch.float16`.

### 2. `02_eeg_feature_extraction.ipynb`

CPU-friendly EEG extraction.

Input:
- gated EgoBrain dataset
- Kaggle Secret `HF_TOKEN`

Output:
- `eeg_Pxxxx.pt`
- EEG feature archive

The canonical feature format is `[N_seconds, 128]` with time indexing retained in the saved object where available.

### 3. `03_temporal_alignment_and_dataset_lock.ipynb`

This is the authoritative synchronization notebook.

It consumes the extracted visual and EEG features plus the EgoBrain video/marker metadata and produces:

```text
all_subjects_synchronized_index.csv
all_subjects_synchronized_labeled_index.csv
visual_feature_timeline.csv
eeg_feature_timeline.csv
marker_timeline_raw_canonicalized.csv
audit/*.csv
manifests/*.json
```

The frozen record contains:

```text
40 subjects
227,864 synchronized 1-second rows
29 retained marker/lifecycle classes
27 semantic action classes
```

Do not alter the resulting synchronized index after modeling begins.

### 4. `04_visual_baselines_and_temporal_robustness.ipynb`

Pure computer-vision benchmark.

Input:
- synchronized labeled index
- 512-D visual features
- locked subject split

Models:
- mean pooling
- max pooling
- mean+max pooling
- temporal convolution
- temporal Transformer
- masked temporal Transformer

Robustness:
- random frame/clip drop
- burst deletion
- temporal jitter
- irregular sampling
- playback-speed change
- contiguous temporal blackout

All six corruption families use severities:

```text
0.05, 0.15, 0.30, 0.50, 0.70
```

The canonical subject-disjoint sequence split is:

```text
TRAIN = 31,147 sequences
VAL   = 38,404 sequences
TEST  = 46,178 sequences

SUBJECT SPLIT = 24 / 8 / 8
```

### 5. `05_eeg_teacher_diagnostics.ipynb`

Standalone EEG-only diagnostic.

It reuses the frozen sequence split and tests:

- pooled baselines
- temporal convolution
- temporal Transformer V2
- token-level discrimination
- temporal representation diversity
- temporal-order sensitivity
- temporal importance concentration
- held-out subject performance
- randomized-label sanity check

Latest recorded result:

```text
chance Macro-F1          0.037037
validation Macro-F1      0.041197
test Macro-F1            0.038876
shuffle Macro-F1 drop    0.000880
normalized entropy       0.999778
```

Interpretation: **weak neural action information with limited evidence of strong temporal concentration**.

### 6. `06_neuroroute_reproduction.ipynb`

Full adaptive-routing reproduction.

The route uses:

```text
EEG teacher
   ↓
neural temporal relation / importance targets
   ↓
visual temporal model
   ↓
adaptive top-k temporal routing
   ↓
video-only deployment
```

Budgets:

```text
100%, 75%, 50%, 25%, 12.5%
```

The recorded primary route uses 50% = 4 of 8 tokens.

Completed-run record:

```text
Dense temporal FLOPs       35,927,552
50% route temporal FLOPs   17,905,152
Reduction                  50.16%

Phase-2 Macro-F1            0.037528
NeuroRoute Macro-F1         0.034828

Baseline mean corruption    0.035298
NeuroRoute mean corruption  0.032933
Robustness gain            -0.002365

Final gate                 INCONCLUSIVE
```

The final deployment audit recorded:

```text
EEG required                 False
EEG preprocessing required  False
EEG encoder required         False
```

These results must be reported as an efficiency/deployment result with an inconclusive recognition claim, not as an improvement claim.

## Reproducing on Kaggle

Create the feature datasets/inputs first:

```text
visual_features/
└── uniform/
    ├── features_P0001.pt
    ├── ...
    └── features_P0040.pt

eeg_features/
└── eeg_P0001.pt
    ...
    └── eeg_P0040.pt

alignment_output/
└── all_subjects_synchronized_labeled_index.csv

vision_baseline_output/
├── sequence_index_with_split.csv
├── models/
│   └── temporal_transformer/
│       └── best_checkpoint.pt
└── reports/
```

Then run notebooks in this exact order:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_visual_feature_extraction_worker.ipynb   --output executed_01_visual_feature_extraction_worker.ipynb

jupyter nbconvert --to notebook --execute notebooks/02_eeg_feature_extraction.ipynb   --output executed_02_eeg_feature_extraction.ipynb

jupyter nbconvert --to notebook --execute notebooks/03_temporal_alignment_and_dataset_lock.ipynb   --output executed_03_temporal_alignment_and_dataset_lock.ipynb

jupyter nbconvert --to notebook --execute notebooks/04_visual_baselines_and_temporal_robustness.ipynb   --output executed_04_visual_baselines_and_temporal_robustness.ipynb

jupyter nbconvert --to notebook --execute notebooks/05_eeg_teacher_diagnostics.ipynb   --output executed_05_eeg_teacher_diagnostics.ipynb

jupyter nbconvert --to notebook --execute notebooks/06_neuroroute_reproduction.ipynb   --output executed_06_neuroroute_reproduction.ipynb
```

For the extraction notebooks, the Kaggle-specific authentication and storage setup in the notebooks is authoritative; local `jupyter nbconvert` execution requires equivalent credentials, data mounts, and GPU resources.

## Kaggle downstream input convention

The downstream modeling notebooks were built around these production-style mounts:

```text
/kaggle/input/datasets/keerthikumarkj/iris-egobrain-visual-features/uniform
/kaggle/input/datasets/keerthikumarkj/iris-eeg-features/eeg_features
/kaggle/input/datasets/keerthikumarkj/iris-egobrain-actual-phase-1-output
/kaggle/input/datasets/keerthikumarkj/iris-phase-2-actual-outputs
```

The notebooks contain root-candidate discovery where appropriate, so equivalent dataset names can be used after changing only the input mounts.

## Frozen results snapshot

| Item | Frozen record |
|---|---:|
| Subjects | 40 |
| Synchronized 1-second rows | 227,864 |
| Semantic classes | 27 |
| Train / Val / Test sequences | 31,147 / 38,404 / 46,178 |
| 27-class chance Macro-F1 | 0.037037 |
| Latest EEG teacher test Macro-F1 | 0.038876 |
| EEG temporal-shuffle Macro-F1 drop | 0.000880 |
| EEG importance normalized entropy | 0.999778 |
| NeuroRoute primary budget | 50% (4/8 tokens) |
| Temporal FLOP reduction | 50.16% |
| NeuroRoute deployment Macro-F1 | 0.034828 |
| Final routing gate | INCONCLUSIVE |

## Scientific recordkeeping

The repository intentionally keeps two related but distinct records:

1. the **later standalone EEG-teacher diagnostic**, which repaired the teacher formulation and selected a Temporal Transformer V2; and
2. the **completed full routing run**, which is the exact end-to-end experiment whose final gate was INCONCLUSIVE.

They are not silently merged into one metric stream.

For manuscript writing, use the `results/` record as the frozen experimental ledger and generate publication figures/tables only after the notebook structure is locked.

## Dataset and model licensing note

EgoBrain is gated and remains governed by its own dataset terms. Do not redistribute raw EEG/video files through this repository.

The pretrained R3D-18 backbone is obtained through torchvision's published pretrained weights.

## Development cleanup

The old development tree contained duplicate notebooks and names such as `Phase1`, `Phase2`, `Phase3`, `complete`, and `final`. Those internal development names are intentionally absent from the public-facing notebook filenames above. Historical copies can be removed from the public repository once this structure is adopted.
