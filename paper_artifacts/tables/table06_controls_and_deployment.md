| Condition              | Clean macro-F1       | Meaningful neural supervision   |
|:-----------------------|:---------------------|:--------------------------------|
| Phase-2 dense visual   | 0.0375711273849615   | No                              |
| NeuroRoute 50%         | 0.034828474673674364 | Yes (training)                  |
| random eeg             | 0.0369938056724456   | Control                         |
| wrong subject eeg      | 0.0352696391835094   | Control                         |
| shifted eeg            | 0.0344448763141499   | Control                         |
| no neural supervision  | 0.0401410977305335   | Control                         |
| Deployment requirement | EEG-free             | No EEG preprocessing / encoder  |