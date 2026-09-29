# Frozen Experimental Record

This directory is an archival record of the experiments completed for the IRIS project. It is intentionally separate from publication figures/tables.

## Dataset and alignment

- Subjects: 40
- Synchronized 1-second rows: 227,864
- Retained marker/lifecycle classes: 29
- Semantic action classes used for classification: 27
- Frozen sequence split: 31,147 train / 38,404 validation / 46,178 test
- Subject split: 24 / 8 / 8

## Visual baseline

- 27-class uniform chance Macro-F1: 0.037037
- Frozen visual baseline test Macro-F1: approximately 0.0375
- Six temporal corruption families, each evaluated at severities 0.05, 0.15, 0.30, 0.50, 0.70

## Latest standalone EEG-teacher diagnostic

Selected teacher: temporal Transformer

- Validation Macro-F1: 0.041197
- Test Macro-F1: 0.038876
- Temporal-shuffle Macro-F1 drop: 0.000880
- Normalized temporal-importance entropy: 0.999778
- Mean subject-level test Macro-F1: 0.037034

## Completed adaptive-routing experiment

Primary budget: 50% = 4 of 8 temporal tokens.

- Frozen visual baseline Macro-F1 in the final route gate: 0.037528
- NeuroRoute visual-only deployment Macro-F1: 0.034828
- Mean corruption Macro-F1 — baseline: 0.035298
- Mean corruption Macro-F1 — NeuroRoute: 0.032933
- Robustness gain: -0.002365
- Dense temporal FLOPs: 35,927,552
- 50% route temporal FLOPs: 17,905,152
- Estimated temporal FLOP reduction: 50.16%
- EEG required at inference: no
- EEG preprocessing required at inference: no
- EEG encoder required at inference: no
- Subject paired Macro-F1 delta: +0.000094
- 95% bootstrap CI: [-0.010227, 0.012364]
- Final scientific gate: **INCONCLUSIVE**

## Interpretation rule

The routing experiment establishes an efficiency result and an EEG-free deployment path, but it does **not** establish recognition or robustness improvement. The paper must preserve that distinction.

## Important experiment distinction

The standalone EEG-teacher notebook is the later diagnostic run and uses the repaired Temporal Transformer V2 teacher. The completed full routing run is an earlier end-to-end reproduction with its own teacher implementation. These are preserved as separate experimental records rather than silently merged.
