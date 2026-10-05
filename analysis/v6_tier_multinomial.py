import os
# -*- coding: utf-8 -*-
"""
v6 — formal comparison of the tier-specific sepsis coefficients.

WHY THIS FILE EXISTS
--------------------
v9 reported the sepsis association for Tier C and for Tier A+B as TWO SEPARATE
logistic models that share the same control group.  A reviewer is right that
"C significant, A+B not significant" is not a test of whether the two
tier-specific effects differ.  This script fits a single multinomial model with
a three-level outcome and Wald-tests the equality of the two sepsis
coefficients directly.

    outcome = 0  no recorded core event          (reference)
              1  Tier A+B event   (core & npsle_hi)
              2  Tier C-only event (core & tier_c & not npsle_hi)

Stays with a core event assigned to neither tier ("definite other cause",
n = 6 in eICU-CRD) are excluded and reported.

First-stay cohort: one stay per patient, so no clustering is required.
"""
import io
import json
import math
import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npsle_io

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
DBS = ["mimiciv", "eicu"]
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD"}
ADJ = ["age", "female", "lupus_nephritis", "creat", "plt"]

FIRST = json.loads((OUT / "_first_stays.json").read_text(encoding="utf-8"))
FKEY = {"mimiciv": "mimic_first", "eicu": "eicu_first"}


def num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)


def load(db):
    c = npsle_io.load(db)
    first = set(FIRST[FKEY[db]])
    d = c[c["stay_id"].isin(first)].copy()
    d["_core"] = num(d["npsle_core"]) == 1
    d["_hi"] = num(d["npsle_hi"]) == 1
    d["_tc"] = num(d["tier_c"]) == 1
    return d


def fit(db):
    d = load(db)
    ab = d["_core"] & d["_hi"]
    co = d["_core"] & d["_tc"] & ~d["_hi"]
    un = d["_core"] & ~d["_hi"] & ~d["_tc"]
    no = ~d["_core"]

    y = np.where(ab, 1, np.where(co, 2, np.where(un, 3, 0)))
    d["_y"] = y
    keep = d["_y"] != 3
    n_un = int((d["_y"] == 3).sum())
    sub = d[keep].copy()

    # v9_revision.fit_or drops adjusters that are entirely missing or constant
    # in a given database (e.g. lupus nephritis in eICU-CRD).  Mirror that rule
    # so the multinomial model is fitted on the same per-database covariate set.
    use_adj = [a for a in ADJ
               if a in sub.columns
               and pd.to_numeric(sub[a], errors="coerce").notna().sum() > 0
               and pd.to_numeric(sub[a], errors="coerce").nunique(dropna=True) > 1]
    dropped = [a for a in ADJ if a not in use_adj]
    cols = ["sepsis_dx"] + use_adj
    sub = sub[["_y"] + cols + ["subject_id"]].dropna(subset=cols + ["_y"])
    sub["sepsis_dx"] = num(sub["sepsis_dx"])
    if sub["_y"].nunique() < 3 or sub["sepsis_dx"].nunique() < 2:
        return dict(db=db, fail="levels or exposure degenerate after filtering")

    X = sm.add_constant(sub[cols].astype(float), has_constant="add")
    res = None
    cov_used = "ML (non-robust)"
    try:
        res = sm.MNLogit(sub["_y"].astype(int), X).fit(
            cov_type="cluster", cov_kwds={"groups": sub["subject_id"].astype(str)},
            disp=0, maxiter=200)
        cov_used = "cluster-robust by subject"
    except Exception as e:
        try:
            res = sm.MNLogit(sub["_y"].astype(int), X).fit(disp=0, maxiter=200)
            cov_used = "ML (non-robust); cluster failed: %s" % str(e)[:60]
        except Exception as e2:
            return dict(db=db, fail=str(e2))

    # statsmodels MNLogit: params is (k_vars, J-1) with columns = non-reference
    # outcome levels; cov_params is (k*J, k*J) in EQUATION-MAJOR order, i.e.
    # index of (variable v, equation e) = e*k + v.  Note that
    # res.t_test(R.ravel()) uses the interleaved C order and is therefore NOT
    # valid here (verified empirically) -- build the contrast by hand.
    P = np.asarray(res.params)
    if P.ndim != 2:
        raise RuntimeError("unexpected params shape %r" % (P.shape,))
    k, J = P.shape
    b1, b2 = P[1, 0], P[1, 1]                    # sepsis coefficient, level 1 / 2
    V = np.asarray(res.cov_params())
    i1, i2 = 0 * k + 1, 1 * k + 1
    v1, v2, c12 = V[i1, i1], V[i2, i2], V[i1, i2]
    diff = b2 - b1
    se = math.sqrt(max(v1 + v2 - 2.0 * c12, 0.0))
    z = diff / se if se > 0 else float("nan")
    Pv = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    se1, se2 = math.sqrt(v1), math.sqrt(v2)
    return dict(
        db=db, label=LABEL[db], cov=cov_used, n=int(len(sub)), adj=use_adj,
        dropped=dropped,
        n0=int((sub["_y"] == 0).sum()), n1=int((sub["_y"] == 1).sum()),
        n2=int((sub["_y"] == 2).sum()), n_unassigned=n_un,
        b1=b1, se1=se1, or1=math.exp(b1),
        lo1=math.exp(b1 - 1.96 * se1), hi1=math.exp(b1 + 1.96 * se1),
        p1=2 * (1 - stats.norm.cdf(abs(b1 / se1))),
        b2=b2, se2=se2, or2=math.exp(b2),
        lo2=math.exp(b2 - 1.96 * se2), hi2=math.exp(b2 + 1.96 * se2),
        p2=2 * (1 - stats.norm.cdf(abs(b2 / se2))),
        diff=diff, se_diff=se, ror=math.exp(diff),
        ror_lo=math.exp(diff - 1.96 * se), ror_hi=math.exp(diff + 1.96 * se),
        z=z, P=Pv)


def pool(rows):
    """Fixed-effect inverse-variance pooling of the log ratio of ORs."""
    ok = [r for r in rows if not r.get("fail")]
    if len(ok) < 2:
        return None
    w = np.array([1.0 / r["se_diff"] ** 2 for r in ok])
    th = np.array([r["diff"] for r in ok])
    mu = float((w * th).sum() / w.sum())
    sem = math.sqrt(1.0 / w.sum())
    Q = float((w * (th - mu) ** 2).sum())
    k = len(ok)
    I2 = max(0.0, 100 * (Q - (k - 1)) / Q) if Q > 0 else 0.0
    z = mu / sem
    return dict(k=k, ror=math.exp(mu), lo=math.exp(mu - 1.96 * sem),
                hi=math.exp(mu + 1.96 * sem),
                P=2 * (1 - stats.norm.cdf(abs(z))), I2=I2, Q=Q)


def main():
    rows = [fit(db) for db in DBS]
    pooled = pool(rows)
    W = []
    A = W.append
    A("=" * 92)
    A("Multinomial (3-level) model: sepsis coefficient by diagnostic-confidence tier")
    A("outcome 0 = no recorded core event (ref); 1 = Tier A+B; 2 = Tier C-only")
    A("=" * 92)
    for r in rows:
        if r.get("fail"):
            A("%s : FAILED %s" % (r["db"], r["fail"]))
            continue
        A("")
        A("%s   n = %d  (no event %d | A+B %d | C-only %d; excluded unassigned %d)"
          % (r["label"], r["n"], r["n0"], r["n1"], r["n2"], r["n_unassigned"]))
        A("   SE: %s" % r["cov"])
        A("   adjusters: %s%s" % (", ".join(r["adj"]),
                                  ("  [dropped: %s]" % ", ".join(r["dropped"])
                                   if r["dropped"] else "")))
        A("   Tier A+B   OR = %.2f (95%% CI %.2f-%.2f), P = %.3f" %
          (r["or1"], r["lo1"], r["hi1"], r["p1"]))
        A("   Tier C-oly OR = %.2f (95%% CI %.2f-%.2f), P = %.3f" %
          (r["or2"], r["lo2"], r["hi2"], r["p2"]))
        A("   difference (log OR, C minus A+B) = %+.3f (SE %.3f)" %
          (r["diff"], r["se_diff"]))
        A("   ratio of ORs = %.2f (95%% CI %.2f-%.2f), z = %.2f, P = %.3f" %
          (r["ror"], r["ror_lo"], r["ror_hi"], r["z"], r["P"]))
    if pooled:
        A("")
        A("-" * 92)
        A("Pooled ratio of ORs (fixed effect, k = %d): %.2f (95%% CI %.2f-%.2f), "
          "P = %.3f; I2 = %.1f%%" %
          (pooled["k"], pooled["ror"], pooled["lo"], pooled["hi"],
           pooled["P"], pooled["I2"]))
        A("(k = 2: tau2 and I2 are unstable; descriptive only.)")
    txt = "\n".join(W)
    (OUT / "v6_tier_multinomial.txt").write_text(txt, encoding="utf-8")
    (OUT / "v6_tier_multinomial.json").write_text(
        json.dumps(dict(rows=rows, pooled=pooled), ensure_ascii=False,
                   indent=1, default=str), encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
