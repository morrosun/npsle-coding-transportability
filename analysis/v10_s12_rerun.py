import os
# -*- coding: utf-8 -*-
"""Re-run S12 (all stays + patient-clustered robust SE) on the PRIMARY universe.

The legacy out/t16_sepsis_assoc.csv was produced by positive_analyses.py, which
reads out/cohort_<db>.csv -- the pre-rewrite whole-path universe (eICU-CRD core
= 100).  Every other supplementary table now uses the leaf-restricted universe
(eICU-CRD core = 85).  This script rebuilds S12 from npsle_io.load(db,
legacy=False) with the identical estimator, so S12 and S14/S15 finally describe
the same main model.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
sys.path.insert(0, str(ROOT / "scripts"))
import npsle_io as nio  # noqa: E402

DBS = ["mimiciv", "eicu", "nwicu"]
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
ADJ_BASE = ["age", "female", "lupus_nephritis", "creat", "plt"]
MIN_TOTAL_EV, MIN_EXP_EV = 10, 5


def fit_or(d, y, x, adjust=None, cluster=True):
    adjust = [a for a in (adjust or []) if a in d.columns
              and pd.to_numeric(d[a], errors="coerce").notna().sum() > 0
              and pd.to_numeric(d[a], errors="coerce").nunique(dropna=True) > 1]
    num_cols = [y, x] + adjust
    m = d[num_cols].apply(pd.to_numeric, errors="coerce")
    if cluster and "subject_id" in d.columns:
        m = m.assign(_grp=pd.factorize(d["subject_id"].astype(str))[0])
    m = m.dropna()
    nev = int(m[y].sum()) if len(m) else 0
    if len(m) < 30 or m[y].nunique() < 2 or m[x].nunique() < 2:
        return (np.nan,) * 4 + (len(m), nev)
    X = sm.add_constant(m[[x] + adjust], has_constant="add")
    try:
        mod = sm.Logit(m[y].astype(float), X)
        res = (mod.fit(disp=0, method="bfgs", maxiter=400,
                       cov_type="cluster", cov_kwds={"groups": m["_grp"]})
               if cluster and "_grp" in m.columns else
               mod.fit(disp=0, method="bfgs", maxiter=400))
        b, se, p = res.params[x], res.bse[x], res.pvalues[x]
        if not np.isfinite(se) or se > 5 or abs(b) > 8:
            return (np.nan,) * 4 + (len(m), nev)
        tab = pd.crosstab(m[x], m[y])
        if tab.shape != (2, 2) or (tab.values == 0).any():
            return (np.nan,) * 4 + (len(m), nev)
        return (float(np.exp(b)), float(np.exp(b - 1.96 * se)),
                float(np.exp(b + 1.96 * se)), float(p), len(m), nev)
    except Exception:
        return (np.nan,) * 4 + (len(m), nev)


def dl_meta(logor, se):
    logor, se = np.asarray(logor, float), np.asarray(se, float)
    w = 1 / se ** 2
    fe = (w * logor).sum() / w.sum()
    Q = (w * (logor - fe) ** 2).sum()
    k = len(logor)
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    wr = 1 / (se ** 2 + tau2)
    mu = (wr * logor).sum() / wr.sum()
    semu = np.sqrt(1 / wr.sum())
    I2 = max(0.0, 100 * (Q - (k - 1)) / Q) if Q > 0 else 0.0
    z = mu / semu
    return dict(OR=np.exp(mu), lo=np.exp(mu - 1.96 * semu),
                hi=np.exp(mu + 1.96 * semu),
                P=2 * (1 - stats.norm.cdf(abs(z))), I2=I2, tau2=tau2, k=k)


def se_from_ci(lo, hi):
    return (np.log(hi) - np.log(lo)) / (2 * 1.96)


rows, meta_src = [], []
for db in DBS:
    d = nio.load(db, legacy=False)
    if "subject_id" not in d.columns:
        raise SystemExit("no subject_id in %s" % db)
    y, x = "npsle_core", "sepsis_dx"
    cr = fit_or(d, y, x, adjust=None, cluster=False)
    ad = fit_or(d, y, x, adjust=ADJ_BASE, cluster=True)
    s1 = pd.to_numeric(d.loc[d[y] == 1, x], errors="coerce")
    s0 = pd.to_numeric(d.loc[d[y] == 0, x], errors="coerce")
    c = nio.counts(d)
    rows.append({
        "数据库": LABEL[db],
        "NP事件组脓毒症率": f"{int(s1.sum())}/{s1.notna().sum()} ({100*s1.mean():.1f}%)",
        "无NP事件组脓毒症率": f"{int(s0.sum())}/{s0.notna().sum()} ({100*s0.mean():.1f}%)",
        "粗 OR (95%CI)": f"{cr[0]:.2f} ({cr[1]:.2f}–{cr[2]:.2f})" if pd.notna(cr[0]) else "不可估计",
        "粗 OR P": f"{cr[3]:.3f}" if pd.notna(cr[3]) else "—",
        "校正 OR (95%CI)†": f"{ad[0]:.2f} ({ad[1]:.2f}–{ad[2]:.2f})" if pd.notna(ad[0]) else "不可估计",
        "校正 OR P": f"{ad[3]:.3f}" if pd.notna(ad[3]) else "—",
        "模型 n": ad[4], "事件数": ad[5]})
    print("%-9s core=%d controls=%d  crude %.2f  adj %.2f (%.2f-%.2f) P=%.3f  n=%s"
          % (LABEL[db], c["core"], c["controls"], cr[0], ad[0], ad[1], ad[2],
             ad[3], ad[4]))
    if pd.notna(ad[0]) and ad[5] >= MIN_TOTAL_EV and int(s1.sum()) >= MIN_EXP_EV:
        meta_src.append(dict(db=LABEL[db], logor=np.log(ad[0]),
                             se=se_from_ci(ad[1], ad[2]), OR=ad[0],
                             lo=ad[1], hi=ad[2]))

t16 = pd.DataFrame(rows)
mt = dl_meta([m["logor"] for m in meta_src],
             [m["se"] for m in meta_src]) if len(meta_src) >= 2 else None
if mt:
    t16 = pd.concat([t16, pd.DataFrame([{
        "数据库": f"合并 (随机效应, k={mt['k']})",
        "NP事件组脓毒症率": "—", "无NP事件组脓毒症率": "—",
        "粗 OR (95%CI)": "—", "粗 OR P": "—",
        "校正 OR (95%CI)†": f"{mt['OR']:.2f} ({mt['lo']:.2f}–{mt['hi']:.2f})",
        "校正 OR P": f"{mt['P']:.3f}", "模型 n": "—",
        "事件数": f"I²={mt['I2']:.1f}%"}])], ignore_index=True)
    print("\npooled = %.2f (%.2f-%.2f) P=%.4f  I2=%.1f%%  tau2=%.4f"
          % (mt["OR"], mt["lo"], mt["hi"], mt["P"], mt["I2"], mt["tau2"]))

t16.to_csv(OUT / "t16_sepsis_assoc.csv", index=False, encoding="utf-8-sig")
print("\nwritten out/t16_sepsis_assoc.csv")
print(t16.to_string(index=False))
