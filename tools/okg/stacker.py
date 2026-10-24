"""LightGBM trajectory model on top of the guard scores.

Protocol
* Trained ONLY on the `stack` role, which is disjoint (by seed group) from the data the
  guard was fine-tuned on, so its inputs are honest out-of-sample guard scores [R11][D21].
* Outer StratifiedGroupKFold -> out-of-fold (OOF) predictions; inner group split for early
  stopping [R11][R13].
* Class imbalance via scale_pos_weight; LightGBM warns this distorts probabilities, so the
  OOF scores are calibrated afterwards (isotonic when >= 1000 samples, otherwise Platt/
  sigmoid, per scikit-learn guidance) [R12][R14].
* Threshold = the smallest score cut whose CONVERSATION-level false-positive rate on benign
  stack conversations is <= fpr_budget (max recall under an FPR budget) [R15][D22].
* SHAP TreeExplainer for global importances [R16].
"""
from __future__ import annotations

import math

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


class Calibrator:
    def __init__(self, method: str):
        self.method = method
        self.model = None

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw, y):
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)   # [R14]
        else:
            self.model = LogisticRegression(C=1e6).fit(self._logit(raw), y)                        # Platt [R14]
        return self

    def transform(self, raw):
        if self.method == "isotonic":
            return self.model.predict(np.asarray(raw, float))
        return self.model.predict_proba(self._logit(raw))[:, 1]


def conv_level_threshold(conv_scores_benign: np.ndarray, fpr_budget: float) -> float:
    """Flag a conversation when its max score > tau. Pick the smallest tau with FPR <= budget. [D22]"""
    s = np.sort(np.asarray(conv_scores_benign, float))[::-1]
    if len(s) == 0:
        return 0.5
    k = int(math.floor(fpr_budget * len(s)))       # number of benign convs we may flag
    return float(s[k]) if k < len(s) else float(s[-1]) - 1e-12


def _params(cfg, y):
    pos, neg = max(int(y.sum()), 1), max(int((1 - y).sum()), 1)
    p = dict(cfg.lgb_params)
    p.update({"scale_pos_weight": neg / pos, "metric": ["binary_logloss", "auc"], "seed": cfg.seed,
              "deterministic": True, "force_col_wise": True})          # [R12]
    return p


class FoldEnsemble:
    """Per-fold booster + per-fold calibrator; prediction = mean of calibrated fold outputs.
    Same scheme as scikit-learn CalibratedClassifierCV(ensemble=True): each copy is fitted on
    the training part and calibrated on its held-out part, and predictions are averaged [R14].
    Test-time scores therefore come from the same models that produced the OOF scores used to
    pick the threshold (a single refit model would shift the score distribution). [D21]"""

    def __init__(self, boosters, calibrators):
        self.boosters, self.calibrators = boosters, calibrators

    def predict_raw(self, X):
        return np.mean([b.predict(X, num_iteration=b.best_iteration) for b in self.boosters], axis=0)

    def predict(self, X):
        return np.mean([c.transform(b.predict(X, num_iteration=b.best_iteration))
                        for b, c in zip(self.boosters, self.calibrators)], axis=0)


def fit(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg) -> dict:
    y = np.asarray(y, int)
    n_splits = max(2, min(cfg.lgb_cv_folds, len(set(groups))))
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)     # [R11]
    oof_raw = np.full(len(y), np.nan)
    oof_cal = np.full(len(y), np.nan)
    boosters, cals, best_iters, methods = [], [], [], []
    for k, (tr, te) in enumerate(outer.split(X, y, groups)):
        inner = StratifiedGroupKFold(n_splits=max(2, min(5, len(set(groups[tr])))), shuffle=True,
                                     random_state=cfg.seed + k)
        itr, iva = next(inner.split(X.iloc[tr], y[tr], groups[tr]))
        dtr = lgb.Dataset(X.iloc[tr].iloc[itr], y[tr][itr])
        dva = lgb.Dataset(X.iloc[tr].iloc[iva], y[tr][iva], reference=dtr)
        bst = lgb.train(_params(cfg, y[tr][itr]), dtr, num_boost_round=cfg.lgb_num_boost_round, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(cfg.lgb_early_stopping_rounds, verbose=False),    # [R13]
                                   lgb.log_evaluation(0)])
        best_iters.append(bst.best_iteration or cfg.lgb_num_boost_round)
        raw = bst.predict(X.iloc[te], num_iteration=bst.best_iteration)
        method = "isotonic" if len(te) >= 1000 else "sigmoid"                                    # [R14]
        cal = Calibrator(method).fit(raw, y[te]) if len(set(y[te])) > 1 else Calibrator("sigmoid").fit(
            np.r_[raw, 0.0, 1.0], np.r_[y[te], 0, 1])
        oof_raw[te], oof_cal[te] = raw, cal.transform(raw)
        boosters.append(bst)
        cals.append(cal)
        methods.append(method)
    return {"model": FoldEnsemble(boosters, cals), "oof_raw": oof_raw, "oof_cal": oof_cal,
            "best_iters": best_iters, "cal_methods": methods}


def shap_importance(boosters, X: pd.DataFrame, max_rows: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Mean |SHAP| (log-odds units) averaged over the fold boosters [R16]."""
    import warnings

    import shap                                                                                  # [R16]
    boosters = boosters if isinstance(boosters, (list, tuple)) else [boosters]
    Xs = X.sample(min(len(X), max_rows), random_state=seed)
    acc = np.zeros(X.shape[1])
    for b in boosters:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = shap.TreeExplainer(b).shap_values(Xs)
        if isinstance(sv, list):          # some shap versions return [neg, pos] for binary models
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., -1]
        acc += np.abs(sv).mean(0)
    return (pd.DataFrame({"feature": X.columns, "mean_abs_shap": acc / len(boosters)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
