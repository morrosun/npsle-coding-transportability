# -*- coding: utf-8 -*-
"""
v7 — semantically restricted eICU phenotyping, and re-analysis of the core results.

Why
---
The legacy eICU rules matched the COMPLETE diagnosisstring, which in eICU-CRD is
the full hierarchical path.  A category label on the path (for example
"altered mental status / pain") therefore matched leaves such as depression,
anxiety, pain, bipolar disorder and schizophrenia, and those stays were counted
as core Tier C events.  The same syndromes are Tier B in the ICD branch
(F23 / F28 / F29), so the artefact is database-specific and inflates the eICU
Tier C stratum.

This script builds a second, semantically restricted algorithm and re-runs the
headline eICU-dependent quantities under both:

  * core / broad prevalence
  * tier composition (A+B, C-only, unassigned)
  * sepsis association (crude and covariate-adjusted, per database and pooled)
  * the formal tier-coefficient ratio (multinomial)

The match target is the LAST TWO path components, which preserves genuine
sub-specification ("meningitis|acute") while preventing a parent category from
imputing a clinical state to its children.  The competing-etiology (X) rule
stays path-based, because there the parent carries the syndrome and the leaf
carries the aetiology ("encephalopathy|hepatic").

Outputs: out/v7_eicu_algorithm_compare.{txt,csv}, out/v7_restricted_tiers_eicu.csv
"""
import os
import io
import json
import math
import pathlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
CFG = dict(host="localhost", port=5432, user="postgres", password="1314")
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
ADJ_BASE = ["age", "female", "lupus_nephritis", "creat", "plt"]
FIRST = json.loads((OUT / "_first_stays.json").read_text(encoding="utf-8"))
FKEY = {"mimiciv": "mimic_first", "eicu": "eicu_first", "nwicu": "nwicu_first"}

# ----------------------------------------------------------------- legacy rules
LEG_DOMAIN = {                      # extract_cohort.py, whole-string
    "dom_seizure": r"seizure|status epilepticus",
    "dom_enceph": r"encephalopathy|delirium|coma|altered mental|obtund|unresponsive",
    "dom_psych": r"psychosis|psychotic",
    "dom_mening": r"meningitis|encephalitis",
    "dom_demyel": r"myelitis|demyelinat|multiple sclerosis",
    "dom_cvd": r"stroke|cerebral infarct|intracerebral hemorrhage|subarachnoid"
               r"|cerebrovascular|CVA|transient ischemic",
    "dom_pns": r"guillain|myasthen|polyneuropathy|neuropathy|chorea",
}
LEG_AUX = {
    "hx_epilepsy": r"epilepsy",
    "metabolic_enceph": r"metabolic encephalopathy|hepatic encephalopathy|uremic encephalopathy",
}
LEG_TIER = dict(
    A=r"systemic lupus erythematosus",
    X=(r"post-anoxic|hepatic|metabolic|uremic|drug withdrawal|alcohol|narcotic|sedated"
       r"|meningitis\|acute|bacterial|viral|herpes|west nile"
       r"|post craniotomy|post-neurosurgery|brain tumor|meningioma|mass lesion"),
    B=r"seizure|status epilepticus|psychosis|psychotic|myelitis|demyelinat|multiple sclerosis",
    C=(r"change in mental status|altered mental|encephalopath|coma\b"
       r"|delirium|stupor|obtundation|unresponsive"),
)

# ------------------------------------------------- restricted rules (tail-2 match)
RES_DOMAIN = {
    "dom_seizure": r"seizure|status epilepticus",
    "dom_enceph": (r"change in mental status|encephalopath|coma\b|delirium|stupor"
                   r"|obtundation|unresponsive|confusion"),
    "dom_psych": r"psychosis|psychotic|schizophrenia|bipolar disorder|hallucination",
    "dom_mening": r"aseptic meningitis|non-bacterial meningitis|sterile meningitis",
    "dom_demyel": r"myelitis|demyelinat|multiple sclerosis",
    "dom_cvd": r"stroke|cerebral infarct|intracerebral hemorrhage|subarachnoid"
               r"|cerebrovascular|cva|transient ischemic",
    "dom_pns": r"guillain|myasthen|polyneuropathy|neuropathy|chorea",
}
RES_AUX = {
    "hx_epilepsy": r"epilepsy|seizure disorder",
    "metabolic_enceph": r"metabolic encephalopathy|hepatic encephalopathy|uremic encephalopathy",
}
# tier B / C on the tail; X and A stay path-based (see module docstring)
RES_TIER = dict(
    A=LEG_TIER["A"],
    X=LEG_TIER["X"],
    B=r"seizure|status epilepticus|psychosis|psychotic|myelitis|demyelinat|multiple sclerosis",
    C=(r"change in mental status|encephalopath|coma\b|delirium|stupor"
       r"|obtundation|unresponsive|confusion"),
)
CORE = ["dom_seizure", "dom_enceph", "dom_psych", "dom_mening", "dom_demyel"]
BROAD_EXTRA = ["dom_cvd", "dom_pns"]


# ----------------------------------------------------------------- data access
def fetch_eicu_terms():
    ids = pd.read_csv(OUT / "cohort_eicu_legacy.csv")["stay_id"].astype(int).tolist()
    import psycopg2
    with psycopg2.connect(dbname="eicu", **CFG) as cn:
        df = pd.read_sql(
            "SELECT patientunitstayid AS stay_id, diagnosisstring AS term "
            "FROM eicu_crd.diagnosis WHERE patientunitstayid = ANY(%s)",
            cn, params=(ids,))
    df["s"] = df["term"].fillna("").astype(str).str.lower()
    parts = df["s"].str.split("|")
    df["tail2"] = parts.apply(lambda p: "|".join(p[-2:]) if len(p) >= 2 else p[-1])
    df["neuro_path"] = df["s"].str.contains(r"neurologic\||cns infection", regex=True)
    return df


def classify(df, dom_rules, aux_rules, tier_rules, target, tier_target=None):
    d = df.copy()
    tier_target = tier_target or target
    for k, pat in list(dom_rules.items()) + list(aux_rules.items()):
        d[k] = d[target].str.contains(pat, regex=True, na=False).astype(int)
    d["npsle_core"] = (d[CORE].sum(axis=1) > 0).astype(int)
    d["npsle_broad"] = (d[CORE + BROAD_EXTRA].sum(axis=1) > 0).astype(int)
    # NB: A and X require the neurologic / CNS-infections path, exactly as in
    # extract_tier.py -- the term alone is not sufficient.
    d["mA"] = d["s"].str.contains(tier_rules["A"], regex=True, na=False) & d["neuro_path"]
    d["mX"] = d["s"].str.contains(tier_rules["X"], regex=True, na=False) & d["neuro_path"]
    d["mB"] = d[tier_target].str.contains(tier_rules["B"], regex=True, na=False)
    d["mC"] = d[tier_target].str.contains(tier_rules["C"], regex=True, na=False)
    # precedence A > X > B > C
    tier = pd.Series(pd.NA, index=d.index, dtype="object")
    for lab in ("C", "B", "X", "A"):
        tier[tier.isna() & d["m" + lab].fillna(False)] = lab
    d["tier"] = tier
    return d


def stay_flags(d):
    g = d.dropna(subset=["tier"]).assign(v=1).pivot_table(
        index="stay_id", columns="tier", values="v", aggfunc="max", fill_value=0)
    for c in ("A", "B", "C", "X"):
        if c not in g.columns:
            g[c] = 0
    g = g.rename(columns={"A": "tier_a", "B": "tier_b", "C": "tier_c", "X": "other_cause"})
    dom = d.groupby("stay_id")[CORE + BROAD_EXTRA].max()
    return g.join(dom)


# ----------------------------------------------------------------- statistics
def num(s):
    return pd.to_numeric(s, errors="coerce")


def fit_or(d, y, x, adjust=ADJ_BASE, cluster=True):
    adjust = [a for a in adjust if a in d.columns
              and num(d[a]).notna().sum() > 0 and num(d[a]).nunique(dropna=True) > 1]
    m = d[[y, x] + adjust].apply(pd.to_numeric, errors="coerce")
    if cluster and "subject_id" in d.columns:
        m = m.assign(_grp=pd.factorize(d["subject_id"].astype(str))[0])
    m = m.dropna()
    if len(m) < 30 or m[y].nunique() < 2 or m[x].nunique() < 2:
        return (np.nan,) * 4 + (len(m), 0, len(adjust))
    X = sm.add_constant(m[[x] + adjust], has_constant="add")
    try:
        mod = sm.Logit(m[y].astype(float), X)
        res = (mod.fit(disp=0, method="bfgs", maxiter=400,
                       cov_type="cluster", cov_kwds={"groups": m["_grp"]})
               if cluster and "_grp" in m.columns else
               mod.fit(disp=0, method="bfgs", maxiter=400))
        b, se = res.params[x], res.bse[x]
        if not np.isfinite(se) or se > 5 or abs(b) > 8:
            return (np.nan,) * 4 + (len(m), int(m[y].sum()), len(adjust))
        return (float(np.exp(b)), float(np.exp(b - 1.96 * se)),
                float(np.exp(b + 1.96 * se)), float(res.pvalues[x]),
                len(m), int(m[y].sum()), len(adjust))
    except Exception:
        return (np.nan,) * 4 + (len(m), int(m[y].sum()) if len(m) else 0, len(adjust))


def dl_meta(logor, se):
    logor, se = np.asarray(logor, float), np.asarray(se, float)
    if len(logor) < 2:
        return None
    w = 1 / se ** 2
    mu = float((w * logor).sum() / w.sum())
    sem = math.sqrt(1 / w.sum())
    z = mu / sem
    return dict(OR=math.exp(mu), lo=math.exp(mu - 1.96 * sem),
                hi=math.exp(mu + 1.96 * sem),
                P=2 * (1 - stats.norm.cdf(abs(z))), k=len(logor))


def se_ci(lo, hi):
    return (math.log(hi) - math.log(lo)) / (2 * 1.96)


def tier_multinomial(d):
    """d must carry _core,_hi,_tc and covariates; returns ratio of ORs."""
    y = np.where(d["_hi"], 1, np.where(d["_tc"], 2, np.where(d["_core"], 3, 0)))
    d = d.assign(_y=y)
    d = d[d["_y"] != 3]          # drop unassigned, KEEP the no-event reference
    use = [a for a in ADJ_BASE if a in d.columns
           and num(d[a]).notna().sum() > 0 and num(d[a]).nunique(dropna=True) > 1]
    cols = ["sepsis_dx"] + use
    d = d[["_y"] + cols].dropna()
    if d["_y"].nunique() < 3 or len(d) < 40:
        return None
    X = sm.add_constant(d[cols].astype(float), has_constant="add")
    res = sm.MNLogit(d["_y"].astype(int), X).fit(disp=0, maxiter=200)
    P = np.asarray(res.params)
    k = P.shape[0]
    b1, b2 = P[1, 0], P[1, 1]
    V = np.asarray(res.cov_params())
    v1, v2, c12 = V[1, 1], V[k + 1, k + 1], V[1, k + 1]
    diff = b2 - b1
    sed = math.sqrt(v1 + v2 - 2 * c12)
    return dict(or_ab=math.exp(b1), or_c=math.exp(b2), ror=math.exp(diff),
                lo=math.exp(diff - 1.96 * sed), hi=math.exp(diff + 1.96 * sed),
                P=2 * (1 - stats.norm.cdf(abs(diff / sed))), n=int(len(d)))


# ----------------------------------------------------------------- main
def main():
    terms = fetch_eicu_terms()
    W = []
    A = W.append
    A("eICU-CRD cohort stays in term table: %d" % terms["stay_id"].nunique())

    coh = pd.read_csv(OUT / "cohort_eicu_legacy.csv")
    first = set(FIRST["eicu_first"])

    # LEGACY arm = exactly what was published: cohort_eicu.csv as extracted plus
    # tier_eicu.csv from extract_tier.py.  Reading the published files rather
    # than re-deriving them guarantees this arm reproduces the v6 numbers.
    lg = coh.merge(pd.read_csv(OUT / "tier_eicu_legacy.csv"), on="stay_id", how="left")
    rs = classify(terms, RES_DOMAIN, RES_AUX, RES_TIER, "tail2")
    fl = stay_flags(rs)
    # drop the legacy phenotype columns from the cohort file first, otherwise
    # the merge keeps them and the comparison silently reads the old values
    drop = [c for c in coh.columns
            if c in CORE or c in BROAD_EXTRA or c.startswith("npsle_")]
    rr = coh.drop(columns=drop).merge(fl, on="stay_id", how="left")
    fl.to_csv(OUT / "v7_restricted_tiers_eicu.csv", encoding="utf-8")
    variants = {"legacy": lg, "restricted": rr}

    for name in list(variants):
        m = variants[name]
        for c in ("tier_a", "tier_b", "tier_c", "other_cause") + tuple(CORE) + tuple(BROAD_EXTRA):
            m[c] = num(m[c]).fillna(0).astype(int)
        m["npsle_core"] = (m[CORE].sum(axis=1) > 0).astype(int)
        m["npsle_broad"] = (m[CORE + BROAD_EXTRA].sum(axis=1) > 0).astype(int)
        m["_core"] = m["npsle_core"] == 1
        m["_hi"] = (m["tier_a"] == 1) | (m["tier_b"] == 1)
        m["_tc"] = (m["tier_c"] == 1) & ~m["_hi"]
        m["_first"] = m["stay_id"].isin(first)
        if name == "restricted":
            fl.to_csv(OUT / "v7_restricted_tiers_eicu.csv", encoding="utf-8")

    A("")
    A("=" * 104)
    A("A. PHENOTYPE PREVALENCE AND TIER COMPOSITION (eICU-CRD)")
    A("=" * 104)
    A("%-34s %14s %14s %10s" % ("quantity", "legacy", "restricted", "change"))
    rows = []
    for basis, sel in (("first stay", True), ("all stays", False)):
        for name in ("legacy", "restricted"):
            m = variants[name]
            m = m[m["_first"]] if sel else m
            n = len(m)
            core = int(m["npsle_core"].sum())
            ab = int((m["_core"] & m["_hi"]).sum())
            co = int((m["_core"] & m["_tc"]).sum())
            un = int((m["_core"] & ~m["_hi"] & ~m["_tc"]).sum())
            rows.append(dict(basis=basis, algorithm=name, n=n, core=core,
                             prev=100 * core / n, ab=ab, conly=co, unassigned=un,
                             c_share=100 * co / core if core else np.nan,
                             tier_a=int((m["tier_a"] == 1).sum()),
                             x=int((m["other_cause"] == 1).sum())))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "v7_eicu_algorithm_compare.csv", index=False, encoding="utf-8-sig")
    for basis in ("first stay", "all stays"):
        for metric, lab, fmt in (("prev", "core prevalence, %", "%.1f"),
                                 ("c_share", "Tier C share of core, %", "%.1f"),
                                 ("ab", "Tier A+B events", "%d"),
                                 ("conly", "Tier C-only events", "%d"),
                                 ("unassigned", "unassigned", "%d"),
                                 ("tier_a", "Tier A stays", "%d"),
                                 ("x", "competing-etiology (X) stays", "%d")):
            l = R[(R.basis == basis) & (R.algorithm == "legacy")][metric].iloc[0]
            r = R[(R.basis == basis) & (R.algorithm == "restricted")][metric].iloc[0]
            A("%-22s %-28s %14s %14s %10s" %
              (basis, lab, fmt % l, fmt % r, fmt % (r - l)))

    A("")
    A("=" * 104)
    A("B. SEPSIS ASSOCIATION AND FORMAL TIER COEFFICIENT COMPARISON")
    A("=" * 104)
    A("%-12s %-11s %-34s %-26s %s" %
      ("algorithm", "database", "sepsis, any event OR (95% CI) P", "Tier C OR (95% CI) P",
       "ratio of ORs C vs A+B (95% CI) P"))
    summ = []
    for name in ("legacy", "restricted"):
        e = variants[name]
        mm = pd.read_csv(OUT / "cohort_mimiciv.csv").merge(
            pd.read_csv(OUT / "tier_mimiciv.csv"), on="stay_id", how="left",
            suffixes=("", "_tier"))
        mm["_first"] = mm["stay_id"].isin(FIRST["mimic_first"])
        mm["_core"] = num(mm["npsle_core"]) == 1
        mm["_hi"] = num(mm["npsle_hi"]) == 1
        mm["_tc"] = (num(mm["tier_c"]) == 1) & ~mm["_hi"]
        m2 = mm[mm["_first"]]
        per = {}
        for tag, dd in (("MIMIC-IV", m2), ("eICU-CRD", e[e["_first"]])):
            r = fit_or(dd, "_core", "sepsis_dx")
            per[tag] = r
            mt = tier_multinomial(dd)
            per[tag + "_tier"] = mt
            A("%-12s %-11s %-34s %-30s" %
              (name, tag,
               "%.2f (%.2f-%.2f) P=%.3f" % (r[0], r[1], r[2], r[3]) if r[0] == r[0] else "not estimable",
               ("A+B %.2f | C %.2f | ROR %.2f (%.2f-%.2f) P=%.3f" %
                (mt["or_ab"], mt["or_c"], mt["ror"], mt["lo"], mt["hi"], mt["P"]))
               if mt else "not estimable"))
        # pooled sepsis OR (MIMIC + eICU), DerSimonian-Laird
        src = [(math.log(per[t][0]), se_ci(per[t][1], per[t][2]))
               for t in ("MIMIC-IV", "eICU-CRD") if per[t][0] == per[t][0]]
        pm = dl_meta([a for a, _ in src], [b for _, b in src])
        rors = [(math.log(per[t + "_tier"]["ror"]),
                 (math.log(per[t + "_tier"]["hi"]) - math.log(per[t + "_tier"]["lo"])) / 3.92)
                for t in ("MIMIC-IV", "eICU-CRD") if per[t + "_tier"]]
        rm = dl_meta([a for a, _ in rors], [b for _, b in rors])
        A("%-12s %-11s %-34s %-30s" %
          (name, "POOLED",
           "%.2f (%.2f-%.2f) P=%.3f" % (pm["OR"], pm["lo"], pm["hi"], pm["P"]) if pm else "n/a",
           ("ROR %.2f (%.2f-%.2f) P=%.3f" % (rm["OR"], rm["lo"], rm["hi"], rm["P"]))
           if rm else "n/a"))
        summ.append(dict(algorithm=name,
                         pooled_sepsis=pm["OR"] if pm else np.nan,
                         pooled_sepsis_lo=pm["lo"] if pm else np.nan,
                         pooled_sepsis_hi=pm["hi"] if pm else np.nan,
                         pooled_ror=rm["OR"] if rm else np.nan,
                         pooled_ror_lo=rm["lo"] if rm else np.nan,
                         pooled_ror_hi=rm["hi"] if rm else np.nan,
                         pooled_ror_P=rm["P"] if rm else np.nan))
    pd.DataFrame(summ).to_csv(OUT / "v7_pooled_compare.csv", index=False,
                              encoding="utf-8-sig")

    A("")
    A("=" * 104)
    A("C. eICU STAYS THAT CHANGE TIER OR LOSE THEIR ONLY TIER FLAG")
    A("=" * 104)
    lf = variants["legacy"].set_index("stay_id")
    rf = variants["restricted"].set_index("stay_id")
    chg = lf["_core"] != rf["_core"]
    A("stays changing core status: %d of %d" % (int(chg.sum()), len(lf)))
    lost = rf.loc[chg, ["tier_a", "tier_b", "tier_c"]].sum(axis=1) == 0
    A("  of which losing every tier flag: %d" % int(lost.sum()))
    gained = (~lf["_core"]) & rf["_core"]
    A("stays GAINING a core event under the restricted rule: %d" % int(gained.sum()))

    txt = "\n".join(W)
    (OUT / "v7_eicu_algorithm_compare.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
