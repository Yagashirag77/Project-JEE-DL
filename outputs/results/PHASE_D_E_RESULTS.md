# Phases D–E — optimisation et Test final

| model | accuracy | precision | recall | f1 | roc_auc | pr_auc | tn | fp | fn | tp | threshold | training_tuning_time_s | inference_time_s | inference_ms_per_tree |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| siamese_cnn | 0.9662 | 0.3015 | 0.346 | 0.3222 | 0.8739 | 0.2205 | 9778 | 190 | 155 | 82 | 0.8815 | 70.7332 | 0.3963 | 0.0388 |
| tcn | 0.719 | 0.0554 | 0.692 | 0.1026 | 0.7635 | 0.0617 | 7173 | 2795 | 73 | 164 | 0.6234 | 21.3124 | 0.0562 | 0.0055 |
| cusum | 0.9131 | 0.0474 | 0.1435 | 0.0712 | 0.6522 | 0.039 | 9284 | 684 | 203 | 34 | 104.12 | 0.0564 | 0.0003 | 0.0 |
| threshold | 0.9315 | 0.0324 | 0.0675 | 0.0438 | 0.6254 | 0.0334 | 9490 | 478 | 221 | 16 | 114.12 | 0.0375 | 0.0001 | 0.0 |
| isolation_forest | 0.4073 | 0.0267 | 0.692 | 0.0514 | 0.5558 | 0.0259 | 3993 | 5975 | 73 | 164 | -0.0951 | 1.6827 | 0.1777 | 0.0174 |

Configurations retenues sur Validation (PR-AUC):

```json
{
  "threshold": {
    "model": "threshold",
    "config": "max_annual_decline",
    "params": {
      "mode": "max_annual_decline"
    },
    "accuracy": 0.9274646523339144,
    "precision": 0.08960573476702509,
    "recall": 0.1718213058419244,
    "f1": 0.11778563015312132,
    "roc_auc": 0.6090138124810586,
    "pr_auc": 0.05426714076553852,
    "tn": 9527,
    "fp": 508,
    "fn": 241,
    "tp": 50,
    "threshold": 114.11999999999999
  },
  "cusum": {
    "model": "cusum",
    "config": "relative=False_k=10.0",
    "params": {
      "relative": false,
      "k": 10.0
    },
    "accuracy": 0.9137129575828007,
    "precision": 0.08100558659217877,
    "recall": 0.19931271477663232,
    "f1": 0.11519364448857994,
    "roc_auc": 0.6150963038300655,
    "pr_auc": 0.052521727909662114,
    "tn": 9377,
    "fp": 658,
    "fn": 233,
    "tp": 58,
    "threshold": 104.11999999999999
  },
  "isolation_forest": {
    "model": "isolation_forest",
    "config": "iso_2",
    "params": {
      "n_estimators": 350,
      "max_samples": 1024,
      "max_features": 0.8
    },
    "fit_time_s": 0.6481839999905787,
    "accuracy": 0.41313190005810574,
    "precision": 0.032723149198120846,
    "recall": 0.6941580756013745,
    "f1": 0.0625,
    "roc_auc": 0.5382194963675246,
    "pr_auc": 0.02950313901090211,
    "tn": 4064,
    "fp": 5971,
    "fn": 89,
    "tp": 202,
    "threshold": -0.09512983429259009
  },
  "siamese_cnn": {
    "model": "siamese_cnn",
    "config": "siamese_2",
    "params": {
      "lr": 0.0007,
      "width": 16,
      "dropout": 0.35
    },
    "fit_time_s": 20.011721100017894,
    "accuracy": 0.9533217121828395,
    "precision": 0.18688524590163935,
    "recall": 0.1958762886597938,
    "f1": 0.1912751677852349,
    "roc_auc": 0.8110198497697919,
    "pr_auc": 0.1664528409789664,
    "tn": 9787,
    "fp": 248,
    "fn": 234,
    "tp": 57,
    "threshold": 0.8814916014671326
  },
  "tcn": {
    "model": "tcn",
    "config": "tcn_0",
    "params": {
      "lr": 0.001,
      "channels": 12,
      "dropout": 0.1,
      "kernel": 2
    },
    "fit_time_s": 5.213678800093476,
    "accuracy": 0.7120859965136549,
    "precision": 0.050904219691895514,
    "recall": 0.5223367697594502,
    "f1": 0.09276777540433323,
    "roc_auc": 0.6633502671919758,
    "pr_auc": 0.04901115091746683,
    "tn": 7201,
    "fp": 2834,
    "fn": 139,
    "tp": 152,
    "threshold": 0.6233952045440674
  }
}
```
