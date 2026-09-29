# Manuscript Claim Matrix

This matrix is the controlling record for manuscript wording. Claims marked *Not established* must not be presented as contributions or improvements.

| Claim                                               | Evidence                                            | Status          |
|:----------------------------------------------------|:----------------------------------------------------|:----------------|
| Frozen synchronized multimodal dataset              | 40 subjects; 227,864 synchronized 1-s rows          | Established     |
| Subject-independent visual benchmark                | 24/8/8 train/val/test subjects; 27 semantic classes | Established     |
| EEG contains weak action-discriminative information | Test macro-F1 0.03888 vs chance 0.03704             | Supported, weak |
| Strong temporal EEG prior                           | Shuffle Δ=0.00088; normalized entropy=0.99978       | Not established |
| 50% temporal compute reduction                      | 35,927,552 → 17,905,152 FLOPs/sample                | Established     |
| Recognition preserved by NeuroRoute                 | 0.03757 → 0.03483                                   | Not established |
| Temporal robustness improved                        | Mean corruption F1 0.03530 → 0.03293                | Not established |
| Inference requires no EEG                           | Deployment audit flags all EEG requirements false   | Established     |
