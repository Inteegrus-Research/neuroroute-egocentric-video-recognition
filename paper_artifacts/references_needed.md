# Reference Plan for the IRIS Manuscript

Target approximately 30–40 references, distributed so that the paper is positioned in the **computer-vision** literature first and the EEG/neural literature second.

## A. Dataset and benchmark anchors
1. EgoBrain primary dataset / benchmark paper.
2. Public EgoBrain task definition, experimental protocol, and cross-subject/cross-scene evaluation.
3. Any official preprocessing or annotation reference required to reproduce the synchronized labels.

## B. Egocentric action recognition
4. Foundational egocentric video understanding / first-person activity recognition.
5. Modern temporal convolution / sequence modeling for action recognition.
6. Video Transformer / temporal attention backbone papers.
7. Recent egocentric action recognition papers using subject-independent evaluation.

## C. Efficient video understanding
8. Adaptive frame selection.
9. Temporal token pruning / token merging.
10. Learnable frame or clip sparsification.
11. Compute-aware video recognition / dynamic inference.
12. Recent 2024–2026 efficient video/token-routing papers.

## D. Temporal robustness
13. Temporal corruption benchmarks.
14. Frame-drop / missing-frame robustness.
15. Temporal consistency under irregular sampling / speed changes.
16. Recent video robustness work from CVPR/ICCV/ECCV/AAAI/ACM MM.

## E. EEG / neural signals for visual understanding
17. EEG + visual recognition / action understanding.
18. EEG-guided or brain-supervised visual representation learning.
19. EEG as privileged information / training-only modality.
20. Neural decoding of temporal information relevant to perception/action.
21. Subject-independent EEG representation learning.

## F. Knowledge distillation / privileged information
22. Teacher–student distillation foundations.
23. Learning Using Privileged Information (LUPI) foundations.
24. Cross-modal / privileged multimodal distillation.
25. Training-only auxiliary modality methods.

## G. Statistical / methodological support
26. Subject-level bootstrap / paired evaluation.
27. Macro-F1 / balanced accuracy justification for class imbalance.
28. Reproducibility / leakage prevention guidance for multimodal temporal datasets.

## Reference-selection rules
- Prefer original papers over surveys when establishing novelty.
- Prefer recent venue papers for active CV areas (2024–2026) while retaining seminal foundations.
- Every novelty sentence must cite the closest competing mechanism.
- Do not cite a paper merely because it uses EEG; cite only when the mechanism is genuinely relevant.
- Explicitly cite prior EEG+video fusion and training-only EEG supervision so the novelty claim is about the **specific neural-temporal-to-adaptive-visual-computation mechanism**, not about generic multimodal fusion.
