# Offline analysis of `/Users/mark/Documents/Capstone2/okplatform-main/artifacts/full`

_generated 2026-11-08 18:06_

## Thresholds: global vs per-language + Clopper-Pearson 95%

| system                         | group      |   n_pos |   n_neg |   recall |   fpr |   roc_auc |
|:-------------------------------|:-----------|--------:|--------:|---------:|------:|----------:|
| lgb global (old rule)          | ALL        |   48797 |   52432 |    0.638 | 0.010 |     0.962 |
| lgb global (old rule)          | lang=en    |   13244 |   21525 |    0.693 | 0.007 |     0.981 |
| lgb global (old rule)          | lang=ko    |   32839 |   30313 |    0.598 | 0.013 |     0.946 |
| lgb global (old rule)          | lang=mixed |    2714 |     594 |    0.856 | 0.022 |     0.985 |
| lgb per-lang + CP95            | ALL        |   48797 |   52432 |    0.609 | 0.010 |     0.962 |
| lgb per-lang + CP95            | lang=en    |   13244 |   21525 |    0.751 | 0.012 |     0.981 |
| lgb per-lang + CP95            | lang=ko    |   32839 |   30313 |    0.550 | 0.008 |     0.946 |
| lgb per-lang + CP95            | lang=mixed |    2714 |     594 |    0.638 | 0.003 |     0.985 |
| guard_strict global (old rule) | ALL        |   48797 |   52432 |    0.600 | 0.009 |     0.958 |
| guard_strict global (old rule) | lang=en    |   13244 |   21525 |    0.652 | 0.006 |     0.980 |
| guard_strict global (old rule) | lang=ko    |   32839 |   30313 |    0.562 | 0.011 |     0.942 |
| guard_strict global (old rule) | lang=mixed |    2714 |     594 |    0.810 | 0.019 |     0.982 |
| guard_strict per-lang + CP95   | ALL        |   48797 |   52432 |    0.588 | 0.011 |     0.958 |
| guard_strict per-lang + CP95   | lang=en    |   13244 |   21525 |    0.756 | 0.015 |     0.980 |
| guard_strict per-lang + CP95   | lang=ko    |   32839 |   30313 |    0.518 | 0.008 |     0.942 |
| guard_strict per-lang + CP95   | lang=mixed |    2714 |     594 |    0.618 | 0.002 |     0.982 |


## Graded actions (conversation level, per-language thresholds)

**okchat_lgb — ALL**

|     y |         n |   allow |   careful_answer |   safe_answer |   block |
|------:|----------:|--------:|-----------------:|--------------:|--------:|
| 0.000 | 52432.000 |   0.934 |            0.011 |         0.045 |   0.010 |
| 1.000 | 48797.000 |   0.160 |            0.013 |         0.217 |   0.609 |


**okchat_lgb — external**

|     y |         n |   allow |   careful_answer |   safe_answer |   block |
|------:|----------:|--------:|-----------------:|--------------:|--------:|
| 0.000 | 12506.000 |   0.927 |            0.001 |         0.057 |   0.014 |
| 1.000 | 10918.000 |   0.237 |            0.003 |         0.213 |   0.547 |


**okchat_lgb — human_multiturn**

|     y |         n |   allow |   careful_answer |   safe_answer |   block |
|------:|----------:|--------:|-----------------:|--------------:|--------:|
| 0.000 |  7059.000 |   0.924 |            0.019 |         0.046 |   0.011 |
| 1.000 | 12487.000 |   0.258 |            0.034 |         0.308 |   0.400 |


**okchat_lgb — in_distribution**

|     y |         n |   allow |   careful_answer |   safe_answer |   block |
|------:|----------:|--------:|-----------------:|--------------:|--------:|
| 0.000 | 32867.000 |   0.938 |            0.014 |         0.041 |   0.007 |
| 1.000 | 25392.000 |   0.079 |            0.007 |         0.175 |   0.739 |


**guard_strict — ALL**

|     y |         n |   allow |   careful_answer |   safe_answer |   block |
|------:|----------:|--------:|-----------------:|--------------:|--------:|
| 0.000 | 52432.000 |   0.938 |            0.009 |         0.042 |   0.011 |
| 1.000 | 48797.000 |   0.189 |            0.010 |         0.214 |   0.588 |


**guard_strict — external**

|     y |         n |   allow |   careful_answer |   safe_answer |   block |
|------:|----------:|--------:|-----------------:|--------------:|--------:|
| 0.000 | 12506.000 |   0.927 |            0.001 |         0.054 |   0.019 |
| 1.000 | 10918.000 |   0.265 |            0.002 |         0.196 |   0.537 |


**guard_strict — human_multiturn**

|     y |         n |   allow |   careful_answer |   safe_answer |   block |
|------:|----------:|--------:|-----------------:|--------------:|--------:|
| 0.000 |  7059.000 |   0.929 |            0.016 |         0.045 |   0.011 |
| 1.000 | 12487.000 |   0.283 |            0.023 |         0.310 |   0.384 |


**guard_strict — in_distribution**

|     y |         n |   allow |   careful_answer |   safe_answer |   block |
|------:|----------:|--------:|-----------------:|--------------:|--------:|
| 0.000 | 32867.000 |   0.945 |            0.011 |         0.036 |   0.008 |
| 1.000 | 25392.000 |   0.109 |            0.006 |         0.174 |   0.710 |


## Stack vs guard-only (P(Unsafe)+P(Controversial), per-language thresholds)

Recommended: **guard**

| index           |   n_pos |   n_neg |   recall_stack |   recall_guard |   fpr_stack |   fpr_guard |   stack_only_caught |   guard_only_caught |   mcnemar_p | stack_wins   |
|:----------------|--------:|--------:|---------------:|---------------:|------------:|------------:|--------------------:|--------------------:|------------:|:-------------|
| external        |   10918 |   12506 |          0.547 |          0.537 |       0.014 |       0.019 |                 381 |                 265 |       0.000 | False        |
| human_multiturn |   12487 |    7059 |          0.400 |          0.384 |       0.011 |       0.011 |                 702 |                 508 |       0.000 | False        |


Stack without memory vs guard:

| index           |   n_pos |   n_neg |   recall_stack |   recall_guard |   fpr_stack |   fpr_guard |   stack_only_caught |   guard_only_caught |   mcnemar_p | stack_wins   |
|:----------------|--------:|--------:|---------------:|---------------:|------------:|------------:|--------------------:|--------------------:|------------:|:-------------|
| external        |   10918 |   12506 |          0.567 |          0.537 |       0.016 |       0.019 |                 457 |                 128 |       0.000 | True         |
| human_multiturn |   12487 |    7059 |          0.408 |          0.384 |       0.011 |       0.011 |                 749 |                 447 |       0.000 | True         |


## English flips (zero-shot caught, fine-tuned missed; strict score at 0.5)

```json
{
 "orbench_toxic": {
  "positives_en": 655,
  "zs_caught": 652,
  "ft_caught": 533,
  "flipped_to_missed": 119,
  "gained": 0,
  "neighbour_label_share": {
   "Safe": 0.866,
   "Unsafe": 0.134
  },
  "neighbour_source_share": {
   "orbench_80k": 0.748,
   "wgm_ko_train": 0.064,
   "aegis2": 0.059,
   "wjb_ko_train": 0.056,
   "polyguardmix": 0.042,
   "do_not_answer": 0.02
  },
  "nearest_is_safe_and_close": 0.1092436974789916
 },
 "xstest_ko": {
  "positives_en": 200,
  "zs_caught": 195,
  "ft_caught": 181,
  "flipped_to_missed": 19,
  "gained": 5,
  "neighbour_label_share": {
   "Safe": 0.614,
   "Unsafe": 0.386
  },
  "neighbour_source_share": {
   "aegis2": 0.333,
   "wjb_ko_train": 0.211,
   "wgm_ko_train": 0.158,
   "polyguardmix": 0.158,
   "orbench_80k": 0.105,
   "toxic_chat_train": 0.018
  },
  "nearest_is_safe_and_close": 0.15789473684210525
 }
}
```

Examples with nearest guard_train neighbours: `flip_examples.csv`.

## Base-feature ablation (same protocol, global threshold at the FPR budget)

{"with_base": {"oof_auc": 0.9861756091750222, "n_features": 32}, "without_base": {"oof_auc": 0.9861756091750222, "n_features": 32}}


| system       | group                 |   n_pos |   n_neg |   recall |   fpr |   roc_auc |
|:-------------|:----------------------|--------:|--------:|---------:|------:|----------:|
| with_base    | ALL                   |   48797 |   52432 |    0.627 | 0.009 |     0.962 |
| with_base    | suite=external        |   10918 |   12506 |    0.549 | 0.011 |     0.930 |
| with_base    | suite=human_multiturn |   12487 |    7059 |    0.433 | 0.013 |     0.926 |
| with_base    | lang=en               |   13244 |   21525 |    0.686 | 0.007 |     0.981 |
| with_base    | lang=ko               |   32839 |   30313 |    0.584 | 0.011 |     0.946 |
| without_base | ALL                   |   48797 |   52432 |    0.627 | 0.009 |     0.962 |
| without_base | suite=external        |   10918 |   12506 |    0.549 | 0.011 |     0.930 |
| without_base | suite=human_multiturn |   12487 |    7059 |    0.433 | 0.013 |     0.926 |
| without_base | lang=en               |   13244 |   21525 |    0.686 | 0.007 |     0.981 |
| without_base | lang=ko               |   32839 |   30313 |    0.584 | 0.011 |     0.946 |

