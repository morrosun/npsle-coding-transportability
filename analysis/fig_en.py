#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
English (Latin-font) regeneration of every manuscript figure.
=============================================================
Reproduces the IDENTICAL computations of

    scripts/fig_revision.py      -> fig/tier_composition.png, fig/label_auc.png
    scripts/positive_analyses.py -> fig/tierC_reprod.png
    scripts/v4_analyses.py       -> fig/forest_sepsis_first.png
    scripts/v5_analyses.py       -> fig/forest_tier_first.png, fig/forest_outcomes_first.png
    scripts/part2_model.py       -> fig/roc_xgb.png, fig/calib_xgb.png, fig/dca_xgb.png,
                                    fig/shap_mimiciv.png, fig/shap_eicu.png

Only the *text labels* and the *font* differ; every statistic, OR, CI, AUC and
point estimate is computed by the same code path with the same seeds.
fig/strobe_flow.png is produced by scripts/make_strobe.py and is left untouched.

No CSV in out/ is written or modified — they are read only.

Usage: python scripts/fig_en.py
"""
import os
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb

warnings.filterwarnings("ignore")
os.environ["LC_ALL"] = "C"
os.environ["LANG"] = "C"

# ---------------------------------------------------------------- Latin font
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DPI = 300
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT, FIG = ROOT / "out", ROOT / "fig"
FIG.mkdir(exist_ok=True)

DBS = ["mimiciv", "eicu", "nwicu"]
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
MIN_TOTAL_EV, MIN_EXP_EV = 10, 5
MIN_EXP_EV_TIER = 3
ADJ_BASE = ["age", "female", "lupus_nephritis", "creat", "plt"]
SEV = {"mimiciv": "sofa24", "eicu": "apache", "nwicu": "sofa24"}
RNG = 20260731

DONE = []


def _save(path, **kw):
    plt.savefig(path, dpi=DPI, **kw)
    plt.close()
    DONE.append(pathlib.Path(path).name)
    print(f"[fig] {pathlib.Path(path).name}")


# ================================================================ shared stats
def load(db):
    c = pd.read_csv(OUT / f"cohort_{db}.csv")
    t = pd.read_csv(OUT / f"tier_{db}.csv")
    d = c.merge(t, on="stay_id", how="left")
    d["prolonged_icu"] = (pd.to_numeric(d["icu_los"], errors="coerce") > 7).astype(float)
    return d


def fit_or(d, y, x, adjust=None, cluster=True):
    """Identical to v4/v5_analyses.fit_or (returns OR, lo, hi, P, n, n_events, n_adj)."""
    adjust = [a for a in (adjust or [])
              if a in d.columns
              and pd.to_numeric(d[a], errors="coerce").notna().sum() > 0
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
                float(np.exp(b + 1.96 * se)), float(res.pvalues[x]),
                len(m), nev, len(adjust))
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
                P=2 * (1 - stats.norm.cdf(abs(z))), I2=I2, tau2=tau2, Q=Q,
                Pq=1 - stats.chi2.cdf(Q, k - 1) if k > 1 else np.nan, k=k)


def se_ci(lo, hi):
    return (np.log(hi) - np.log(lo)) / (2 * 1.96)


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))


data = {db: load(db) for db in DBS}
FIRST = json.loads((OUT / "_first_stays.json").read_text(encoding="utf-8"))
FKEY = {"mimiciv": "mimic_first", "eicu": "eicu_first", "nwicu": "nwicu_first"}
first_ids = {db: set(pd.Series(FIRST[FKEY[db]]).astype(str)) for db in DBS}
data_first = {db: data[db][data[db]["stay_id"].astype(str).isin(first_ids[db])].copy()
              for db in DBS}


# ======================================================= 1. tier_composition.png
# source: scripts/fig_revision.py (lines 22-68)
fig, ax = plt.subplots(figsize=(7, 8))
dbs = list(LABEL)
x = np.arange(len(dbs))
pa, pb, pc = [], [], []
for db in dbs:
    d = data[db]
    a = d["tier_a"].fillna(0).astype(int)
    b = d["tier_b"].astype(int)
    c = d["tier_c"].astype(int)
    only_a = a
    only_b = ((b == 1) & (a == 0)).astype(int)
    only_c = ((c == 1) & (a == 0) & (b == 0)).astype(int)
    pa.append(100 * only_a.mean())
    pb.append(100 * only_b.mean())
    pc.append(100 * only_c.mean())

C = ["#1f6f8b", "#4c9f70", "#d9a441"]
ax.bar(x, pa, 0.55, label="Tier A \u00b7 Attributable (SLE explicitly named)", color=C[0])
ax.bar(x, pb, 0.55, bottom=pa, label="Tier B \u00b7 Specific NP syndrome", color=C[1])
ax.bar(x, pc, 0.55, bottom=np.array(pa) + np.array(pb),
       label="Tier C \u00b7 Non-specific consciousness/behavior change", color=C[2])

for i in range(len(dbs)):
    tot = pa[i] + pb[i] + pc[i]
    if pa[i] > 0.8:
        ax.text(i, pa[i] / 2, f"{pa[i]:.1f}%", ha="center", va="center",
                color="white", fontsize=9.5, fontweight="bold")
    if pb[i] > 0.8:
        ax.text(i, pa[i] + pb[i] / 2, f"{pb[i]:.1f}%", ha="center", va="center",
                color="white", fontsize=9.5, fontweight="bold")
    if pc[i] > 0.8:
        ax.text(i, pa[i] + pb[i] + pc[i] / 2, f"{pc[i]:.1f}%", ha="center", va="center",
                color="white", fontsize=9.5, fontweight="bold")
    ax.text(i, tot + 1.0, f"Total {tot:.1f}%", ha="center", fontsize=10, fontweight="bold")
    if data[dbs[i]]["tier_a"].isna().all():
        ax.text(i, -3.6, "Tier A structurally unavailable", ha="center",
                fontsize=8.6, color="#b03a2e")

ax.set_xticks(x)
ax.set_xticklabels([f"{LABEL[d]}\n(n={len(data[d])})" for d in dbs], fontsize=10.5)
ax.set_ylabel("Percentage of SLE-ICU cohort (%)", fontsize=11)
ax.set_ylim(-5.5, max(np.array(pa) + np.array(pb) + np.array(pc)) * 1.25)
ax.legend(fontsize=9.5, loc="upper left", framealpha=0.95)
ax.grid(axis="y", ls=":", alpha=0.45)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
_save(FIG / "tier_composition.png", bbox_inches="tight", facecolor="white")


# ============================================================= 2. label_auc.png
# source: scripts/fig_revision.py (lines 70-106)
LABDEF_EN = {
    "原核心定义 (A+B+C)": "Original core definition (A+B+C)",
    "高置信 (A+B)": "High-confidence (A+B)",
    "严格 (A+B 去他因)": "Strict (A+B, competing causes excluded)",
    "Tier A 归因明确": "Tier A \u00b7 attributable",
}

t15 = pd.read_csv(OUT / "t15_label_auc.csv")
t15 = t15[~t15["XGBoost CV-AUC (95%CI)"].astype(str).str.contains("不估计")].copy()


def parse(s):
    a, rest = s.split(" (")
    lo, hi = rest.rstrip(")").split("\u2013")
    return float(a), float(lo), float(hi)


vals = t15["XGBoost CV-AUC (95%CI)"].map(parse)
t15["auc"] = [v[0] for v in vals]
t15["lo"] = [v[1] for v in vals]
t15["hi"] = [v[2] for v in vals]
t15 = t15.iloc[::-1].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8.4, 3.9))
cols = {"MIMIC-IV": "#1f6f8b", "eICU-CRD": "#c0392b"}
yy = np.arange(len(t15))
for i, r in t15.iterrows():
    c = cols[r["数据库"]]
    ax.plot([r["lo"], r["hi"]], [i, i], color=c, lw=2.2, solid_capstyle="round")
    ax.plot(r["auc"], i, "o", color=c, ms=7, zorder=3)
    ax.text(r["hi"] + 0.012, i, f'{r["auc"]:.3f} ({r["lo"]:.3f}\u2013{r["hi"]:.3f})',
            va="center", fontsize=9)
ax.axvline(0.5, color="#666", ls="--", lw=1.2)
ax.text(0.503, len(t15) - 0.35, "Chance level", fontsize=8.8, color="#666")
ax.set_yticks(yy)
ax.set_yticklabels(
    [f'{r["数据库"]} \u00b7 {LABDEF_EN.get(r["标签定义"], r["标签定义"])} '
     f'({r["事件数"]} events)' for _, r in t15.iterrows()], fontsize=9.8)
ax.set_xlabel("XGBoost internal 5\u00d75 cross-validated AUC (95% bootstrap CI)", fontsize=10.5)
ax.set_xlim(0.33, 0.90)
ax.grid(axis="x", ls=":", alpha=0.45)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
_save(FIG / "label_auc.png", bbox_inches="tight", facecolor="white")


# ========================================================== 3. tierC_reprod.png
# source: scripts/positive_analyses.py (T18 block + Fig 7 block)
t18_rows = []
for db in DBS:
    d = data[db]
    n_ev = int((d["npsle_any"].fillna(0) == 1).sum())
    has_c = int((d["tier_c"].fillna(0) == 1).sum())
    only_c = int(((d["tier_c"].fillna(0) == 1) & (d["npsle_hi"].fillna(0) == 0)).sum())
    lw, hw = wilson(has_c, n_ev)
    ls, hs = wilson(only_c, n_ev)
    t18_rows.append({"数据库": LABEL[db], "_hasc": has_c, "_onlyc": only_c, "_n": n_ev})
t18 = pd.DataFrame(t18_rows)
_a18 = t18[t18["数据库"] == "MIMIC-IV"].iloc[0]
_b18 = t18[t18["数据库"] == "eICU-CRD"].iloc[0]
chi_w, p_w = stats.chi2_contingency([[_a18["_hasc"], _a18["_n"] - _a18["_hasc"]],
                                     [_b18["_hasc"], _b18["_n"] - _b18["_hasc"]]],
                                    correction=False)[:2]

fig, ax = plt.subplots(figsize=(7, 8))
sub = t18[t18["数据库"].isin(["MIMIC-IV", "eICU-CRD"])].reset_index(drop=True)
xs = np.arange(len(sub))
w = 0.34
for j, (key, nm, col) in enumerate([
        ("_hasc", "Including non-specific codes", "#2e6da4"),
        ("_onlyc", "Non-specific only (no specific/attributable phenotype)", "#c0392b")]):
    ps = [100 * r[key] / r["_n"] for _, r in sub.iterrows()]
    cis = [wilson(r[key], r["_n"]) for _, r in sub.iterrows()]
    err = np.array([[p - 100 * c[0] for p, c in zip(ps, cis)],
                    [100 * c[1] - p for p, c in zip(ps, cis)]])
    pos = xs + (j - 0.5) * w
    ax.bar(pos, ps, width=w, color=col, alpha=.88, zorder=2, label=nm)
    ax.errorbar(pos, ps, yerr=err, fmt="none", ecolor="#333", capsize=5, lw=1.3, zorder=3)
    for xp, p in zip(pos, ps):
        ax.text(xp, p + 5, f"{p:.1f}%", ha="center", fontsize=10.5, fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels(sub["数据库"], fontsize=11)
ax.set_ylim(0, 100)
ax.set_ylabel("Percentage of all recorded NP events (%)", fontsize=10.5)
ax.legend(fontsize=9.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
ax.grid(axis="y", ls=":", alpha=.5, zorder=0)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
plt.tight_layout()
_save(FIG / "tierC_reprod.png", bbox_inches="tight")


# ==================================================== forest helper (v4 == v5)
def forest(items, title, xlab, path, xlim=(0.15, 12), ticks=(0.25, 0.5, 1, 2, 4, 8)):
    """items: [(label, OR, lo, hi, is_pooled), ...]
    Plain-number x-axis (no scientific notation), name labels in left margin
    (do NOT overlap CI bars), OR labels in right margin."""
    fig, ax = plt.subplots(figsize=(7.8, 0.52 * len(items) + 1.7))
    ys = np.arange(len(items))[::-1]
    # generous left/right margins so labels never cross CI whiskers
    fig.subplots_adjust(left=0.36, right=0.72, top=0.90, bottom=0.14)
    for i, (nm, o, lo, hi, pooled) in enumerate(items):
        y = ys[i]
        if not np.isfinite(o):
            ax.text(-0.02, y, nm, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=10, color="#888")
            ax.text(1.03, y, "Not estimable", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=9.5, color="#888")
            continue
        ax.plot([lo, hi], [y, y], color="#1f4e79" if pooled else "#444",
                lw=2.3 if pooled else 1.5, solid_capstyle="butt", zorder=2)
        ax.scatter([o], [y], s=155 if pooled else 78, marker="D" if pooled else "s",
                   color="#c0392b" if pooled else "#1f4e79", zorder=3)
        ax.text(-0.02, y, nm, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=10,
                fontweight="bold" if pooled else "normal")
        ax.text(1.03, y, f"{o:.2f} ({lo:.2f}\u2013{hi:.2f})",
                transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=9.8,
                fontweight="bold" if pooled else "normal")
    ax.axvline(1, color="#888", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    # plain-number ticks (no scientific notation)
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([])
    ax.set_ylim(-0.8, len(items) - 0.2)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_xlabel(xlab, fontsize=10.5)
    _save(path, bbox_inches="tight")


# =================================================== 4a. forest_sepsis_first.png
# source: scripts/v4_analyses.py (T22a build + forest call, lines 219-408)
meta_src_f = []
for db in DBS:
    df = data_first[db]
    y, x = "npsle_core", "sepsis_dx"
    s1 = pd.to_numeric(df.loc[pd.to_numeric(df[y], errors="coerce") == 1, x], errors="coerce")
    ad = fit_or(df, y, x, adjust=ADJ_BASE, cluster=False)
    if pd.notna(ad[0]) and ad[5] >= MIN_TOTAL_EV and int(s1.sum()) >= MIN_EXP_EV:
        meta_src_f.append(dict(db=LABEL[db], logor=np.log(ad[0]), se=se_ci(ad[1], ad[2]),
                               OR=ad[0], lo=ad[1], hi=ad[2]))
mtf = (dl_meta([m["logor"] for m in meta_src_f], [m["se"] for m in meta_src_f])
       if len(meta_src_f) >= 2 else None)

items = [(m["db"], m["OR"], m["lo"], m["hi"], False) for m in meta_src_f]
if mtf:
    items.append(("Pooled (random effects)", mtf["OR"], mtf["lo"], mtf["hi"], True))
forest(items,
       "Sepsis comorbidity association with recorded neuropsychiatric events\n"
       "(first ICU stay cohort)",
       "Adjusted OR", FIG / "forest_sepsis_first.png",
       xlim=(0.5, 6), ticks=(0.5, 1, 2, 4))
if mtf:
    print(f"      pooled OR = {mtf['OR']:.2f} ({mtf['lo']:.2f}-{mtf['hi']:.2f}), "
          f"P = {mtf['P']:.4f}, I2 = {mtf['I2']:.1f}%")


# ===================================================== 4b. forest_tier_first.png
# source: scripts/v5_analyses.py (T25a build lines 143-183 + forest call lines 470-479)
mc, mh = [], []
for db in DBS:
    df = data_first[db]
    no_ev = pd.to_numeric(df["npsle_core"], errors="coerce").fillna(0) == 0
    only_c = (pd.to_numeric(df["tier_c"], errors="coerce").fillna(0) == 1) & \
             (pd.to_numeric(df["npsle_hi"], errors="coerce").fillna(0) == 0)
    hi = pd.to_numeric(df["npsle_hi"], errors="coerce").fillna(0) == 1
    sub_c = df[only_c | no_ev].copy()
    sub_c["_y"] = only_c[only_c | no_ev].astype(int)
    sub_h = df[hi | no_ev].copy()
    sub_h["_y"] = hi[hi | no_ev].astype(int)
    for tag, sub in [("Tier C", sub_c), ("Tier A+B", sub_h)]:
        r = fit_or(sub, "_y", "sepsis_dx", adjust=ADJ_BASE, cluster=False)
        nexp = int(pd.to_numeric(sub.loc[sub["_y"] == 1, "sepsis_dx"], errors="coerce").sum())
        if pd.notna(r[0]) and int(sub["_y"].sum()) >= MIN_TOTAL_EV and nexp >= MIN_EXP_EV_TIER:
            (mc if tag == "Tier C" else mh).append(
                dict(db=LABEL[db], logor=np.log(r[0]), se=se_ci(r[1], r[2]),
                     OR=r[0], lo=r[1], hi=r[2]))

tier_meta = {}
for tag, src in [("Tier C", mc), ("Tier A+B", mh)]:
    if len(src) >= 2:
        tier_meta[tag] = dl_meta([s["logor"] for s in src], [s["se"] for s in src])

ti = []
for tag in ["Tier C", "Tier A+B"]:
    src = mc if tag == "Tier C" else mh
    for s in src:
        ti.append((f"{tag} \u00b7 {s['db']}", s["OR"], s["lo"], s["hi"], False))
    if tag in tier_meta:
        m = tier_meta[tag]
        ti.append((f"{tag} \u00b7 Pooled", m["OR"], m["lo"], m["hi"], True))
forest(ti, "Sepsis comorbidity association by diagnostic-confidence tier\n"
           "(first ICU stay cohort)",
       "Adjusted OR", FIG / "forest_tier_first.png",
       xlim=(0.1, 12), ticks=(0.25, 0.5, 1, 2, 4, 8))
for tag, m in tier_meta.items():
    print(f"      {tag}: pooled OR = {m['OR']:.3f} ({m['lo']:.3f}-{m['hi']:.3f}), "
          f"P = {m['P']:.4f}, I2 = {m['I2']:.1f}%")


# ================================================= 4c. forest_outcomes_first.png
# source: scripts/v5_analyses.py (T25b build lines 190-220 + forest call lines 483-491)
OUTCOMES = [("In-hospital mortality", "hosp_mort"),
            ("Mechanical ventilation within 24 h", "vent24"),
            ("ICU stay > 7 days", "prolonged_icu")]
meta_A = {}
for nm, ycol in OUTCOMES:
    srcA = []
    for db in DBS:
        df = data_first[db]
        if ycol not in df.columns or pd.to_numeric(df[ycol], errors="coerce").notna().sum() == 0:
            continue
        adA = fit_or(df, ycol, "npsle_core", adjust=ADJ_BASE + [SEV[db], "sepsis_dx"],
                     cluster=False)
        if pd.notna(adA[0]) and adA[5] >= MIN_TOTAL_EV:
            srcA.append(dict(logor=np.log(adA[0]), se=se_ci(adA[1], adA[2]),
                             OR=adA[0], lo=adA[1], hi=adA[2], db=LABEL[db]))
    if len(srcA) >= 2:
        meta_A[nm] = dl_meta([s["logor"] for s in srcA], [s["se"] for s in srcA])
        meta_A[nm]["src"] = srcA

oi = []
for nm, _ in OUTCOMES:
    if nm in meta_A:
        for s in meta_A[nm]["src"]:
            oi.append((f"{nm} \u00b7 {s['db']}", s["OR"], s["lo"], s["hi"], False))
        m = meta_A[nm]
        oi.append((f"{nm} \u00b7 Pooled", m["OR"], m["lo"], m["hi"], True))
forest(oi, "Association of recorded neuropsychiatric events with clinical outcomes\n"
           "(first ICU stay cohort, outcome model A2)",
       "Adjusted OR", FIG / "forest_outcomes_first.png",
       xlim=(0.08, 20), ticks=(0.25, 0.5, 1, 2, 4, 8))


# =============================================== 5/6. part2_model.py XGB figures
# source: scripts/part2_model.py (Prep / models / store / plot_roc / plot_calib /
#         plot_dca / SHAP, lines 54-437). Only the XGBoost main-model settings
#         that back the manuscript figures are refitted; identical seeds.
FEATS_MAIN = ["age", "female", "gcs_min", "hr", "map", "temp", "spo2", "wbc",
              "creat", "sepsis_dx"]
BINARY = ["female", "sepsis_dx"]
LOGVAR = ["wbc", "creat"]
RCSVAR = ["age", "gcs_min"]

# English feature names (same keys as the CN dict in part2_model.py)
EN = {"age": "Age", "female": "Female sex", "gcs_min": "GCS minimum",
      "hr": "Heart rate", "map": "Mean arterial pressure", "temp": "Temperature",
      "spo2": "SpO2 minimum", "wbc": "WBC (log)", "creat": "Creatinine (log)",
      "sepsis_dx": "Sepsis"}


def rcs_knots(x, k=3):
    x = pd.to_numeric(x, errors="coerce").dropna()
    qs = {3: [.10, .50, .90], 4: [.05, .35, .65, .95]}[k]
    kn = np.unique(np.quantile(x, qs))
    return kn if len(kn) == k else None


def rcs_basis(x, kn):
    x = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)
    k = len(kn)
    t1, tkm1, tk = kn[0], kn[-2], kn[-1]
    denom = (tk - t1) ** 2
    cols = [x]
    cub = lambda z: np.where(z > 0, z ** 3, 0.0)
    for j in range(k - 2):
        tj = kn[j]
        term = (cub(x - tj)
                - cub(x - tkm1) * (tk - tj) / (tk - tkm1)
                + cub(x - tk) * (tkm1 - tj) / (tk - tkm1)) / denom
        cols.append(term)
    return np.column_stack(cols)


class Prep:
    def __init__(self, feats, use_rcs=True):
        self.feats = list(feats)
        self.use_rcs = use_rcs

    def fit(self, df):
        self.med_, self.knots_ = {}, {}
        for f in self.feats:
            s = pd.to_numeric(df[f], errors="coerce")
            if f in LOGVAR:
                s = np.log(np.clip(s, 1e-2, None))
            self.med_[f] = float(s.median()) if s.notna().any() else 0.0
            if self.use_rcs and f in RCSVAR:
                kn = rcs_knots(s.fillna(self.med_[f]))
                if kn is not None:
                    self.knots_[f] = kn
        return self

    def transform_raw(self, df):
        cols = []
        for f in self.feats:
            s = pd.to_numeric(df[f], errors="coerce")
            if f in LOGVAR:
                s = np.log(np.clip(s, 1e-2, None))
            cols.append(s.fillna(self.med_[f]).values.astype(float))
        return np.column_stack(cols)


def fit_xgb_model(X, y):
    m = xgb.XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, min_child_weight=5,
        eval_metric="logloss", random_state=RNG, n_jobs=4)
    m.fit(X, y)
    return m


cohort = {db: pd.read_csv(OUT / f"cohort_{db}.csv") for db in LABEL}


def internal_cv(df, feats, n_splits=5, n_repeats=5, target="npsle_core"):
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).values.astype(int)
    oof = np.zeros(len(df))
    cnt = np.zeros(len(df))
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RNG)
    for tr, te in cv.split(np.zeros(len(y)), y):
        pp = Prep(feats, use_rcs=False).fit(df.iloc[tr])
        Xtr, Xte = pp.transform_raw(df.iloc[tr]), pp.transform_raw(df.iloc[te])
        pr = fit_xgb_model(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        oof[te] += pr
        cnt[te] += 1
    ok = cnt > 0
    return y[ok], (oof[ok] / cnt[ok])


def external(df_tr, df_te, feats):
    ytr = df_tr["npsle_core"].values.astype(int)
    yte = df_te["npsle_core"].values.astype(int)
    pp = Prep(feats, use_rcs=False).fit(df_tr)
    m = fit_xgb_model(pp.transform_raw(df_tr), ytr)
    return yte, m.predict_proba(pp.transform_raw(df_te))[:, 1]


store = {}
for db in ["mimiciv", "eicu"]:
    store[f"internal_{db}"] = internal_cv(cohort[db], FEATS_MAIN)
for a, b in [("mimiciv", "eicu"), ("eicu", "mimiciv")]:
    store[f"external_{a}_{b}"] = external(cohort[a], cohort[b], FEATS_MAIN)

C_IN, C_EX = "#1f4e79", "#c0392b"


def plot_roc(keys, fn, title):
    fig, ax = plt.subplots(figsize=(5.6, 5.2), facecolor="white")
    for (k, lab, col, ls) in keys:
        if k not in store:
            continue
        y, p = store[k]
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=col, ls=ls, lw=2,
                label=f"{lab} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color="#999", ls=":", lw=1)
    ax.set_xlabel("1 \u2212 Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(FIG / fn, facecolor="white")


def plot_calib(keys, fn, title):
    fig, ax = plt.subplots(figsize=(5.6, 5.2), facecolor="white")
    for (k, lab, col, mk) in keys:
        if k not in store:
            continue
        y, p = store[k]
        q = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=False)
        obs = pd.Series(y).groupby(q).mean()
        exp = pd.Series(p).groupby(q).mean()
        ax.plot(exp, obs, marker=mk, color=col, lw=1.8, ms=7, label=lab)
    ax.plot([0, 1], [0, 1], color="#999", ls=":", lw=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed proportion")
    ax.legend(fontsize=8.5, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(FIG / fn, facecolor="white")


def net_benefit(y, p, pt):
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    out = []
    for t in pt:
        f = p >= t
        tp = np.sum(f & (y == 1))
        fp = np.sum(f & (y == 0))
        out.append(tp / n - (fp / n) * (t / (1 - t)))
    return np.array(out)


def plot_dca(keys, fn, title):
    pt = np.linspace(0.05, 0.75, 60)
    fig, ax = plt.subplots(figsize=(5.9, 5.2), facecolor="white")
    prev = None
    for (k, lab, col, ls) in keys:
        if k not in store:
            continue
        y, p = store[k]
        prev = np.mean(y)
        ax.plot(pt, net_benefit(y, p, pt), color=col, ls=ls, lw=2, label=lab)
    if prev is not None:
        ax.plot(pt, prev - (1 - prev) * (pt / (1 - pt)), color="#7f8c8d", ls="--",
                lw=1.3, label="All treat")
        ax.axhline(0, color="#34495e", lw=1.3, label="None treat")
        ax.set_ylim(-0.12, max(0.05, prev * 1.15))
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.legend(fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(FIG / fn, facecolor="white")


ks_roc = [("internal_mimiciv", "Internal MIMIC-IV", C_IN, "-"),
          ("internal_eicu", "Internal eICU-CRD", C_IN, "--"),
          ("external_mimiciv_eicu", "External MIMIC\u2192eICU", C_EX, "-"),
          ("external_eicu_mimiciv", "External eICU\u2192MIMIC", C_EX, "--")]
plot_roc(ks_roc, "roc_xgb.png", "XGBoost \u00b7 Internal and external ROC")
ks_cal = [(k, l, c, "o" if s == "-" else "s") for k, l, c, s in ks_roc]
plot_calib(ks_cal, "calib_xgb.png", "XGBoost \u00b7 Calibration curve")
plot_dca([ks_roc[0], ks_roc[2]], "dca_xgb.png",
         "XGBoost \u00b7 Decision curve (MIMIC internal vs external eICU)")

# ---------------------------------------------------------------- SHAP
try:
    import shap
    for db in ["mimiciv", "eicu"]:
        pp = Prep(FEATS_MAIN, use_rcs=False).fit(cohort[db])
        X = pp.transform_raw(cohort[db])
        m = fit_xgb_model(X, cohort[db]["npsle_core"].values.astype(int))
        sv = shap.TreeExplainer(m).shap_values(X)
        plt.figure(figsize=(6.4, 4.6), facecolor="white")
        shap.summary_plot(sv, X, feature_names=[EN.get(f, f) for f in FEATS_MAIN],
                          show=False, plot_size=None, max_display=10)
        plt.tight_layout()
        _save(FIG / f"shap_{db}.png", bbox_inches="tight", facecolor="white")
except Exception as ex:
    print("[!] SHAP failed:", ex)


# ================================================================ verification
from PIL import Image

CHECK = ["tier_composition.png", "label_auc.png", "tierC_reprod.png",
         "forest_sepsis_first.png", "forest_tier_first.png", "forest_outcomes_first.png",
         "roc_xgb.png", "calib_xgb.png", "dca_xgb.png",
         "shap_mimiciv.png", "shap_eicu.png", "strobe_flow.png"]

print("\n" + "=" * 72)
print("VERIFICATION  (exists / size / dpi >= 300)")
print("=" * 72)
allok = True
for fn in CHECK:
    p = FIG / fn
    if not p.exists():
        print(f"  [MISSING] {fn}")
        allok = False
        continue
    with Image.open(p) as im:
        d = im.info.get("dpi")
        w, h = im.size
    dv = round(float(d[0])) if d else 0
    ok = dv >= 300
    allok &= ok
    print(f"  [{'OK ' if ok else 'BAD'}] {fn:28s} {w:5d}x{h:<5d} dpi={dv}")
print("=" * 72)
print("ALL FIGURES OK" if allok else "SOME FIGURES FAILED")
