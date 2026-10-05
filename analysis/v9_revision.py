# -*- coding: utf-8 -*-
"""
V9 revision (major revision, reviewer round 2):
  M1  -> mutually-exclusive tier counts (A+B / C-only / unassigned) for
         first-stay AND all-stays, with chi-square across databases.
  M3  -> fix S13 control group: no_ev must be npsle_core==0, not npsle_any==0
         (the latter wrongly absorbs the 6 eICU 'definite other cause' stays
         into the no-recorded-event control).
  M10 -> unassigned identity (definite other cause) cross-tab for the note.
  E13 -> S21 stratum-event reconciliation (GCS-verbal analyzable counts).
Output: out/t27_v9.csv (replaces S4), out/t17_sepsis_by_tier.csv (S13 fixed),
        out/_v9_key.json
"""
import os, json, pathlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import warnings
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npsle_io
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
DBS = ["mimiciv", "eicu", "nwicu"]
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
ADJ_BASE = ["age", "female", "lupus_nephritis", "creat", "plt"]
FIRST = json.loads((OUT / "_first_stays.json").read_text(encoding="utf-8"))
FKEY = {"mimiciv": "mimic_first", "eicu": "eicu_first", "nwicu": "nwicu_first"}


def load(db):
    return npsle_io.load(db)


def tier_counts(d):
    """Mutually-exclusive stay-level tier membership among core events."""
    core = pd.to_numeric(d["npsle_core"], errors="coerce").fillna(0) == 1
    hi = pd.to_numeric(d["npsle_hi"], errors="coerce").fillna(0) == 1
    tc = pd.to_numeric(d["tier_c"], errors="coerce").fillna(0) == 1
    ta_raw = d["tier_a"] if "tier_a" in d.columns else pd.Series(np.nan, index=d.index)
    ta = pd.to_numeric(ta_raw, errors="coerce").fillna(0) == 1
    ta_na = ta_raw.isna().all()          # Tier A structurally unavailable (ICD-based DBs)
    n_core = int(core.sum())
    n_ab = int((core & hi).sum())
    n_conly = int((core & tc & ~hi).sum())
    n_un = int((core & ~hi & ~tc).sum())
    n_ta = int(ta.sum()) if not ta_na else 0
    return n_core, n_ab, n_conly, n_un, n_ta, ta_na


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))


# ============================================================ 1. S4 mutually-exclusive table
data = {db: load(db) for db in DBS}
rows = []
for db in DBS:
    for cname, df in [("all", data[db]), ("first", None)]:
        d = df if cname == "all" else None
    # all stays
    n_all = len(data[db])
    n_core_all, n_ab_all, n_conly_all, n_un_all, n_ta_all, ta_na_all = tier_counts(data[db])
    # first stays
    fids = set(pd.Series(FIRST[FKEY[db]]).astype(str))
    df = data[db][data[db]["stay_id"].astype(str).isin(fids)]
    n_f = len(df)
    n_core_f, n_ab_f, n_conly_f, n_un_f, n_ta_f, ta_na_f = tier_counts(df)

    def mk(n, core_n, ab, conly, un, ta_n, ta_na):
        clo, chi = wilson(conly, core_n) if core_n else (np.nan, np.nan)
        return dict(n=n, core=core_n,
                    ab=f"{ab} ({100*ab/core_n:.1f}%)" if core_n else "—",
                    conly=(f"{conly} ({100*conly/core_n:.1f}%, "
                           f"{100*clo:.1f}–{100*chi:.1f})" if core_n else "—"),
                    un=un,
                    ta=("NA (structurally unavailable)" if ta_na
                        else (f"{ta_n} ({100*ta_n/n:.1f}%)" if n else "—")))

    a = mk(n_all, n_core_all, n_ab_all, n_conly_all, n_un_all, n_ta_all, ta_na_all)
    b = mk(n_f, n_core_f, n_ab_f, n_conly_f, n_un_f, n_ta_f, ta_na_f)
    for tag, m in [("All stays", a), ("First stay (primary)", b)]:
        rows.append({"Database": LABEL[db], "Cohort": tag, "ICU stays": m["n"],
                     "Core events n": m["core"],
                     "Tier A n (% of ICU stays)": m["ta"],
                     "Tier A+B n (% of core)": m["ab"],
                     "Tier C only n (%, 95%CI)": m["conly"],
                     "Unassigned to any tier n": m["un"]})

t27v9 = pd.DataFrame(rows)
t27v9.to_csv(OUT / "t27_v9.csv", index=False, encoding="utf-8-sig")
print(t27v9.to_string(index=False))

# cross-database chi-square on C-only proportion (mutually exclusive), first-stay and all-stays
key = {}
for tag in ["first", "all"]:
    mm = {}
    for db in ["mimiciv", "eicu"]:
        if tag == "first":
            fids = set(pd.Series(FIRST[FKEY[db]]).astype(str))
            d = data[db][data[db]["stay_id"].astype(str).isin(fids)]
        else:
            d = data[db]
        n_core, n_ab, n_conly, n_un, n_ta, ta_na = tier_counts(d)
        mm[db] = (n_core, n_conly)
    a, b = mm["mimiciv"], mm["eicu"]
    tbl = [[a[1], a[0] - a[1]], [b[1], b[0] - b[1]]]
    chi2, p = stats.chi2_contingency(tbl, correction=False)[:2]
    chi2y, py = stats.chi2_contingency(tbl, correction=True)[:2]
    key[f"conly_{tag}"] = dict(
        mimiciv=[a[1], a[0]], eicu=[b[1], b[0]],
        pct_m=100 * a[1] / a[0], pct_e=100 * b[1] / b[0],
        chi2=float(chi2), P=float(p), chi2y=float(chi2y), Py=float(py))
    print(f"\n[conly_{tag}] MIMIC {a[1]}/{a[0]}={100*a[1]/a[0]:.1f}% vs "
          f"eICU {b[1]}/{b[0]}={100*b[1]/b[0]:.1f}%  chi2={chi2:.2f} P={p:.4f} "
          f"(Yates chi2={chi2y:.2f} P={py:.4f})")


# ============================================================ 2. M10: unassigned identity
print("\n[unassigned identity]")
for db in DBS:
    d = data[db]
    core = pd.to_numeric(d["npsle_core"], errors="coerce").fillna(0) == 1
    hi = pd.to_numeric(d["npsle_hi"], errors="coerce").fillna(0) == 1
    tc = pd.to_numeric(d["tier_c"], errors="coerce").fillna(0) == 1
    oc = pd.to_numeric(d["other_cause"], errors="coerce").fillna(0) == 1
    un = d[core & ~hi & ~tc]
    print(f"  {db}: unassigned n={len(un)} | all other_cause=1: {int(oc.sum())} | "
          f"unassigned & other_cause: {int((core & ~hi & ~tc & oc).sum())}")


# ============================================================ 3. M3: fix S13 (t17) control
# control = no RECORDED CORE event (npsle_core==0), NOT npsle_any==0
def fit_or(d, y, x, adjust=None, cluster=True):
    adjust = [a for a in (adjust or [])
              if a in d.columns and pd.to_numeric(d[a], errors="coerce").notna().sum() > 0
              and pd.to_numeric(d[a], errors="coerce").nunique(dropna=True) > 1]
    m = d[[y, x] + adjust].apply(pd.to_numeric, errors="coerce")
    if cluster and "subject_id" in d.columns:
        m = m.assign(_grp=pd.factorize(d["subject_id"].astype(str))[0])
    m = m.dropna()
    nev = int(m[y].sum()) if len(m) else 0
    if len(m) < 30 or m[y].nunique() < 2 or m[x].nunique() < 2:
        return (np.nan,) * 4 + (len(m), nev, len(adjust))
    X = sm.add_constant(m[[x] + adjust], has_constant="add")
    try:
        mod = sm.Logit(m[y].astype(float), X)
        res = (mod.fit(disp=0, method="bfgs", maxiter=400,
                       cov_type="cluster", cov_kwds={"groups": m["_grp"]})
               if cluster and "_grp" in m.columns else
               mod.fit(disp=0, method="bfgs", maxiter=400))
        b, se = res.params[x], res.bse[x]
        if not np.isfinite(se) or se > 5 or abs(b) > 8:
            return (np.nan,) * 4 + (len(m), nev, len(adjust))
        tab = pd.crosstab(m[x], m[y])
        if tab.shape != (2, 2) or (tab.values == 0).any():
            return (np.nan,) * 4 + (len(m), nev, len(adjust))
        return (float(np.exp(b)), float(np.exp(b - 1.96 * se)),
                float(np.exp(b + 1.96 * se)), float(res.pvalues[x]), len(m), nev, len(adjust))
    except Exception:
        return (np.nan,) * 4 + (len(m), nev, len(adjust))


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
    return dict(OR=np.exp(mu), lo=np.exp(mu - 1.96 * semu), hi=np.exp(mu + 1.96 * semu),
                P=2 * (1 - stats.norm.cdf(abs(z))), I2=I2, tau2=tau2, k=k)


def se_ci(lo, hi):
    return (np.log(hi) - np.log(lo)) / (2 * 1.96)


rows17, mc17, mh17 = [], [], []
for db in DBS:
    d = data[db]
    no_ev = pd.to_numeric(d["npsle_core"], errors="coerce").fillna(0) == 0   # FIXED
    only_c = (pd.to_numeric(d["tier_c"], errors="coerce").fillna(0) == 1) & \
             (pd.to_numeric(d["npsle_hi"], errors="coerce").fillna(0) == 0)
    hi = pd.to_numeric(d["npsle_hi"], errors="coerce").fillna(0) == 1
    sub_c = d[only_c | no_ev].copy(); sub_c["_y"] = only_c[only_c | no_ev].astype(int)
    sub_h = d[hi | no_ev].copy();     sub_h["_y"] = hi[hi | no_ev].astype(int)
    for tag, sub in [("Tier C 非特异事件", sub_c), ("Tier A+B 高置信事件", sub_h)]:
        r = fit_or(sub, "_y", "sepsis_dx", adjust=ADJ_BASE, cluster=True)
        nexp = int(pd.to_numeric(sub.loc[sub["_y"] == 1, "sepsis_dx"], errors="coerce").sum())
        rows17.append({"数据库": LABEL[db], "对比": f"{tag} vs 无记录事件",
                       "事件数": int(sub["_y"].sum()), "对照数": int((sub["_y"] == 0).sum()),
                       "校正 OR (95%CI)†": (f"{r[0]:.2f} ({r[1]:.2f}–{r[2]:.2f})"
                                            if pd.notna(r[0]) else "不可估计"),
                       "P": f"{r[3]:.3f}" if pd.notna(r[3]) else "—"})
        if pd.notna(r[0]) and int(sub["_y"].sum()) >= 10 and nexp >= 5:
            (mc17 if tag.startswith("Tier C") else mh17).append(
                dict(logor=np.log(r[0]), se=se_ci(r[1], r[2]), OR=r[0], lo=r[1], hi=r[2]))

t17v9 = pd.DataFrame(rows17)
for tag, src in [("Tier C 非特异事件", mc17), ("Tier A+B 高置信事件", mh17)]:
    if len(src) >= 2:
        m = dl_meta([s["logor"] for s in src], [s["se"] for s in src])
        t17v9 = pd.concat([t17v9, pd.DataFrame([{
            "数据库": f"合并 (k={m['k']})", "对比": f"{tag} vs 无记录事件",
            "事件数": "—", "对照数": f"I²={m['I2']:.1f}%",
            "校正 OR (95%CI)†": f"{m['OR']:.2f} ({m['lo']:.2f}–{m['hi']:.2f})",
            "P": f"{m['P']:.3f}"}])], ignore_index=True)
t17v9 = t17v9.sort_values(["对比", "数据库"]).reset_index(drop=True)
t17v9.to_csv(OUT / "t17_sepsis_by_tier.csv", index=False, encoding="utf-8-sig")  # overwrite S13
print("\n[S13 fixed: t17_sepsis_by_tier.csv]")
print(t17v9.to_string(index=False))


# ============================================================ 4. E13: S21 stratum reconciliation
print("\n[E13: S21 stratum event reconciliation]")
e13 = {}
for db in ["mimiciv", "eicu"]:
    d = data[db]
    core = pd.to_numeric(d["npsle_core"], errors="coerce").fillna(0) == 1
    v = pd.to_numeric(d["vent24"], errors="coerce")
    gv = pd.to_numeric(d["gcs_verbal"], errors="coerce")
    n_core = int(core.sum())
    n_core_vent_ok = int((core & v.notna()).sum())
    n_nv = int((core & (v == 0)).sum()); n_v = int((core & (v == 1)).sum())
    # events with analyzable GCS verbal within each stratum (as t24 counts)
    n_nv_gcs = int((core & (v == 0) & gv.notna()).sum())
    n_v_gcs = int((core & (v == 1) & gv.notna()).sum())
    e13[db] = dict(n_core=n_core, nv=n_nv, v=n_v, nv_gcs=n_nv_gcs, v_gcs=n_v_gcs,
                   gcs_missing_nv=n_nv - n_nv_gcs, gcs_missing_v=n_v - n_v_gcs)
    print(f"  {db}: core={n_core}, non-vent core={n_nv} (GCS-analyzable {n_nv_gcs}), "
          f"vent core={n_v} (GCS-analyzable {n_v_gcs}) -> GCS-missing {n_nv-n_nv_gcs}/{n_v-n_v_gcs}")

key["e13"] = e13
key["t27_v9"] = t27v9.to_dict(orient="records")
(OUT / "_v9_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n[ok] -> out/t27_v9.csv, out/t17_sepsis_by_tier.csv (S13 fixed), out/_v9_key.json")
