# IRIS Manuscript Plan

## Working title

**NeuroRoute: Neural-Temporal Supervision for Adaptive Computation in Egocentric Video Recognition**

The title deliberately states the mechanism and problem without claiming a recognition or robustness improvement that was not established by the frozen experiments.

## Manuscript thesis

The manuscript asks whether synchronized EEG can act as training-time privileged information for learning how a visual model allocates temporal computation in egocentric action recognition. The paper reports both the positive engineering outcome (substantial temporal FLOP reduction and EEG-free deployment) and the negative findings (recognition and corruption-robustness preservation were not established under the frozen subject-independent protocol).

## Main structure

### 1. Introduction
- Egocentric video recognition is temporally dense and computationally redundant.
- Temporal information can be missing, corrupted, or expensive to process.
- Neural signals provide a temporally resolved training-time signal that is unavailable at deployment.
- Research question: can privileged neural temporal information teach a visual model what temporal evidence to retain?
- Contributions are stated as an investigation and controlled evaluation, not as an unsupported accuracy claim.

### 2. Related Work
2.1 Egocentric action recognition and temporal video modeling
2.2 EEG / neural signals for visual and action understanding
2.3 Privileged information and cross-modal knowledge distillation
2.4 Adaptive temporal computation, token/frame pruning, and efficient video understanding
2.5 Temporal robustness and corrupted-video evaluation

### 3. Dataset, Synchronization, and Problem Formulation
3.1 EgoBrain and synchronized modalities
3.2 Temporal feature representation (512-D visual / 128-D EEG)
3.3 1-second synchronized windows
3.4 27-class semantic action-recognition task
3.5 Subject-independent 24/8/8 split
3.6 Leakage and temporal-integrity controls

### 4. Neural-Temporal Supervision and Adaptive Routing
4.1 Frozen visual temporal backbone
4.2 EEG temporal teacher
4.3 Neural temporal representation
4.4 Relation supervision
4.5 Temporal-importance supervision
4.6 Adaptive temporal router
4.7 Explicit token/computation budgets
4.8 Training-time EEG and video-only deployment
4.9 Objective function and optimization

### 5. Experimental Protocol
5.1 Clean recognition metrics
5.2 Temporal corruption benchmark (six corruption families; five severities)
5.3 Temporal budget sweep
5.4 FLOPs and latency accounting
5.5 Negative controls: random EEG, wrong-subject EEG, shifted EEG, no neural supervision
5.6 Subject-level paired analysis and bootstrap interval
5.7 Reproducibility and fixed-model-selection rules

### 6. Results
6.1 Dataset and split integrity
6.2 Video-only baseline ladder
6.3 EEG teacher diagnostics
6.4 Neural supervision and routing outcomes
6.5 Accuracy–compute trade-off
6.6 Temporal corruption robustness
6.7 Negative controls and EEG-free deployment
6.8 Subject-level variability

### 7. Discussion
- What was established.
- What was not established.
- Why weak EEG temporal structure limits the downstream transfer claim.
- Why the compute reduction remains useful as an efficiency finding.
- Implications for future neural-prior supervision and multimodal training.

### 8. Limitations
- Low absolute recognition performance on the frozen 27-class subject-independent task.
- Weak EEG temporal discriminability in the current representation.
- No explicit cross-scene benchmark available in the frozen metadata for the primary split.
- Limited sequence length (8 seconds) and feature-level rather than raw-video modeling.

### 9. Conclusion
One paragraph summarizing the controlled investigation, the 50% temporal-compute reduction at the predefined budget, EEG-free deployment, and the negative recognition/robustness findings.

## Main figure plan

1. Study and method framework.
2. Dataset synchronization and subject-independent split.
3. Video-only baseline ladder.
4. EEG teacher diagnostics.
5. Compute budget versus recognition.
6. Temporal robustness across corruption severity.
7. Negative controls + deployment audit.

Supplementary: paired unseen-subject deltas and additional class-level/robustness plots if page budget permits.

## Main table plan

1. Dataset scale and subject split.
2. Clean video-only baseline comparison.
3. EEG-teacher performance and diagnostics.
4. Temporal budget / compute sweep.
5. Temporal corruption comparison.
6. Negative controls and deployment audit.
7. Claim-to-evidence audit (primarily internal; only include in the paper if useful for limitations/ablation clarity).

## Abstract strategy

Write the abstract only after the figures/tables are frozen. The abstract must not claim that EEG improved recognition or robustness. It should state the measured compute reduction and the fact that recognition/robustness preservation was not established.
