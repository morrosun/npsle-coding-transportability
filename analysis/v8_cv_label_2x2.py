import os
# -*- coding: utf-8 -*-
"""2x2 comparison: cross-validation scheme x label set.

The manuscript previously attributed the rise in eICU internal AUC from 0.639
to 0.699 to label cleaning alone, but that comparison had also changed the
cross-validation scheme. This script runs all four cells with features,
preprocessing, hyper-parameters and split seeds held fixed, so the two changes
can be separated.

Labels
  legacy     whole-path matching (the algorithm being replaced)
  restricted audited leaf rules (primary)

CV
  stay-level StratifiedKFold          the original scheme
  patient-grouped GroupKFold          the corrected scheme

Grouping uses uniquepid for eICU-CRD and subject_id for MIMIC-IV; the unit of
analysis stays the ICU stay, only the CV split changes.
"""
import json
import math
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import npsle_io  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
SEED = 20240607
N_SPLIT = 5
N_REPEAT = 6

FEATURES = ["age", "female", "aps", "gcs_min", "hr", "map", "rr", "temp",
            "spo2", "wbc", "hgb", "plt", "creat", "na", "bili", "lactate",
            "vent24", "steroid_any"]

lines = []


def A(s=""):
    lines.append(str(s))


def auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    ok = ~(np.isnan(p) | (p < 0) | (p > 1))
    y, p = y[ok], p[ok]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(p).rank()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def prep(d, cols):
    """Training-fold median imputation and standardisation."""
    tr = d[d._first] if "_first" in d.columns else d
    med = tr[cols].apply(pd.to_numeric, errors="coerce").median()
    X = d[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    mu, sd = X.mean(), X.std(ddof=0).replace(0, 1.0)
    return (X - mu) / sd


def fit_predict(X, y, groups, grouped):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (GroupKFold, StratifiedKFold)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if grouped:
        # each patient contributes every stay to the same fold
        splitter = GroupKFold(n_splits=N_SPLIT)
        folds = list(splitter.split(X, y, groups=groups))
    else:
        folds = []
        for r in range(N_REPEAT):
            skf = StratifiedKFold(n_splits=N_SPLIT, shuffle=True,
                                  random_state=SEED + r)
            folds.extend(list(skf.split(X, y)))
    prob = np.full(len(y), np.nan)
    for tr, te in folds:
        if len(np.unique(y.iloc[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0))
        m.fit(X.iloc[tr], y.iloc[tr])
        prob[te] = m.predict_proba(X.iloc[te])[:, 1]
    return prob


def boot_ci(y, p, groups, grouped, n=400):
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups) if grouped else np.arange(len(y))
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        a = auc(y.iloc[idx], p[idx])
        if a == a:
            vals.append(a)
    if len(vals) < 50:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    A("2x2 COMPARISON: label set x cross-validation scheme")
    A("=" * 78)
    A("features fixed: %d variables; preprocessing fitted on the training fold;"
      % len(FEATURES))
    A("hyper-parameters fixed (L2, C=1); splits seeded; unit of analysis = ICU stay")
    A("")

    res = {}
    for db in ("mimiciv", "eicu"):
        A("-" * 78)
        A("%s" % LABEL[db])
        A("-" * 78)
        A("  %-12s %-18s %8s %18s %6s %5s"
          % ("labels", "CV", "n", "events", "AUC", "95% CI"))
        for lab_name, legacy in (("legacy", True), ("restricted", False)):
            d = npsle_io.load(db, legacy=legacy)
            # Canonical row order. Stays are read back from CSV with whatever
            # order the extraction query happened to return (no ORDER BY), and
            # both splitters are position-dependent: StratifiedKFold(shuffle)
            # permutes positions, GroupKFold assigns groups in order of first
            # appearance. Sorting here makes every cell reproducible from the
            # data alone and removes a silent source of between-run drift.
            d = d.sort_values(["subject_id", "stay_id"], kind="mergesort").reset_index(drop=True)
            cols = [c for c in FEATURES if c in d.columns]
            X = prep(d, cols)
            y = d["npsle_core"].astype(int)
            gcol = "subject_id"
            groups = (d[gcol].astype(str).values if gcol in d.columns
                      else np.arange(len(d)))
            for grouped in (False, True):
                p = fit_predict(X, y, groups, grouped)
                a = auc(y, p)
                lo, hi = boot_ci(y, p, groups, grouped)
                key = "%s|%s|%s" % (db, lab_name,
                                    "patient" if grouped else "stay")
                res[key] = dict(n=int(len(y)), events=int(y.sum()), auc=a,
                                lo=lo, hi=hi)
                A("  %-12s %-18s %8d %18d %8.3f  %.3f-%.3f"
                  % (lab_name, "patient-grouped" if grouped else "stay-level",
                     len(y), int(y.sum()), a, lo, hi))
        A("")

    # the two contrasts the manuscript needs
    A("=" * 78)
    A("CONTRASTS")
    A("=" * 78)
    for db in ("mimiciv", "eicu"):
        base = res[db + "|legacy|stay"]["auc"]
        for lab_name in ("legacy", "restricted"):
            g = res[db + "|" + lab_name + "|patient"]["auc"]
            s = res[db + "|" + lab_name + "|stay"]["auc"]
            A("  %-9s %-12s stay-level %.3f -> patient-grouped %.3f  (CV effect %+.3f)"
              % (LABEL[db], lab_name, s, g, g - s))
        r_stay = res[db + "|restricted|stay"]["auc"] - base
        r_pat = (res[db + "|restricted|patient"]["auc"]
                 - res[db + "|legacy|patient"]["auc"])
        A("  %-9s label effect under stay-level CV      %+.3f" % (LABEL[db], r_stay))
        A("  %-9s label effect under patient-grouped CV %+.3f" % (LABEL[db], r_pat))
        A("")

    (OUT / "v8_cv_label_2x2.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    (OUT / "v8_cv_label_2x2.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
