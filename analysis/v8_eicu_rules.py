import os
# -*- coding: utf-8 -*-
"""Final eICU-CRD phenotype rules, designed from the observed path structure
(out/v8_eicu_paths.txt) rather than from a component count.

Observed structure
------------------
neurologic|altered mental status / pain|<leaf>            depth 3
neurologic|altered mental status / pain|<leaf>|<subleaf>   depth 4
neurologic|infectious disease of nervous system|encephalitis|systemic lupus erythematosus
infectious diseases|cns infections|encephalitis|systemic lupus erythematosus

'altered mental status / pain' is a real dictionary folder. Its DIRECT children
(change in mental status, encephalopathy, coma, delirium, stupor, obtundation)
genuinely denote an acute confusional state, so those must be retained. The
misclassification only occurs when the child itself is subdivided
(|depression|mild, |pain|moderate, |drug withdrawal syndrome|alcohol): the
parent then vouches for a child that denotes something else entirely.

Rule therefore: match the FULL string, but require the matched concept to be a
direct child of the 'altered mental status / pain' folder OR to appear anywhere
in the path as a clinical diagnosis. Explicitly excluded leaves are listed so
that the exclusion is auditable.
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
CFG = dict(host="localhost", port=5432, user="postgres", password="1314")

# ---------------------------------------------------------------- white list
# Tier C -- non-specific acute confusional state. Each entry is an observed
# leaf (or direct child of the AMS folder) that denotes the clinical state.
AMS = r"altered mental status / pain"
C_LEAF = (
    r"\|change in mental status\b"
    r"|\|encephalopathy\b"
    r"|\|coma\b"
    r"|\|delirium\b"
    r"|\|stupor\b"
    r"|\|obtundation\b"
    r"|\|unresponsive\b"
    r"|\|confusion\b"
    r"|\|agitation\b"
)
# Tier B -- syndrome-specific. Includes the affective/psychotic leaves that the
# ICD branch places in F23/F28/F29 (Tier B), so the two branches align.
B_LEAF = (
    r"\|seizures\b|\|status epilepticus\b"
    r"|\|psychosis\b|\|psychotic\b"
    r"|\|schizophrenia\b|\|bipolar disorder\b|\|hallucination"
    r"|\|myelitis\b|\|demyelinat|\|multiple sclerosis"
)
# Meningitis / encephalitis: any depth, because the aetiology sits in the parent
# ('encephalitis|systemic lupus erythematosus', 'meningitis|acute|bacterial').
MENING = r"meningitis|encephalitis|meningoencephalitis"
# Peripheral / neuromuscular
PNS = r"guillain|myasthen|polyneuropathy|neuropathy|chorea"

# ------------------------------------------------------------------ excludes
# Leaves observed under the AMS folder that denote a different clinical entity.
# Listing them explicitly is what makes the rule auditable: a reviewer can check
# the exclusion list against the dictionary instead of trusting the matcher.
EXCLUDE_LEAF = (
    r"\|depression\b|\|anxiety\b|\|pain\b|\|bipolar disorder\b"
    r"|\|schizophrenia\b|\|dementia\b|\|suicidal ideation\b"
    r"|\|drug withdrawal syndrome\b|\|sedated\b"
)
# 'sedated|unresponsive' keeps the AMS meaning, so it is not excluded wholesale:
# the exclusion above requires the leaf itself to be the last component.

# Tier A: the record itself names SLE as the cause. Whole-path, because the
# attribution word is the deepest node of a 4-level path.
TIER_A = r"systemic lupus erythematosus"
TIER_X = (r"post-anoxic|hepatic|metabolic|uremic|drug withdrawal|alcohol"
          r"|narcotic|sedated|bacterial|viral|herpes|west nile"
          r"|post craniotomy|post-neurosurgery|brain tumor|meningioma"
          r"|mass lesion")


def fetch():
    p = pd.read_csv(OUT / "cohort_eicu.csv")
    import psycopg2
    ids = p["stay_id"].astype(int).tolist()
    with psycopg2.connect(dbname="eicu", **CFG) as cn:
        d = pd.read_sql(
            "SELECT patientunitstayid AS stay_id, diagnosisstring AS term "
            "FROM eicu_crd.diagnosis WHERE patientunitstayid = ANY(%s)", cn,
            params=(ids,))
    d["s"] = d["term"].fillna("").astype(str).str.lower()
    return p, d


def classify(d):
    """Assign a per-term tier. Returns a frame with stay_id and tier columns."""
    s = d["s"]
    neuro_path = s.str.contains(r"neurologic\||cns infection", regex=True, na=False)
    parts = s.str.split("|")
    depth = parts.apply(len)
    leaf = parts.apply(lambda p: p[-1] if p else "")

    # a term is dropped when its own leaf is on the exclusion list
    excluded = leaf.str.contains(EXCLUDE_LEAF, regex=True, na=False)

    d = d.copy()
    d["tier"] = None
    # meningitis / encephalitis: any depth (aetiology lives in the parent node)
    m_men = s.str.contains(MENING, regex=True, na=False) & ~excluded
    # seizure / psychosis / demyelination: the concept must be a direct child of
    # the AMS folder or appear as the leaf, never only as a folder above an
    # unrelated child.
    m_b = (s.str.contains(B_LEAF, regex=True, na=False)
           | (s.str.contains(r"seizure|status epilepticus|psychosis|psychotic"
                             r"|myelitis|demyelinat|multiple sclerosis",
                             regex=True, na=False) & leaf.str.contains(
               r"seizure|epilep|psychosis|psychotic|myelitis|demyelinat"
               r"|multiple sclerosis", regex=True, na=False)))
    m_b &= ~excluded
    # acute confusional state
    m_c = (s.str.contains(C_LEAF, regex=True, na=False)
           | (s.str.contains(r"encephalopath|delirium|coma\b|stupor|obtundation"
                             r"|unresponsive|confusion|change in mental status",
                             regex=True, na=False)
              & (leaf.str.contains(r"encephalopath|delirium|coma\b|stupor"
                                   r"|obtundation|unresponsive|confusion"
                                   r"|change in mental status",
                                   regex=True, na=False)
                 | depth.le(3))))
    m_c &= ~excluded

    d.loc[m_c, "tier"] = "C"
    d.loc[m_b, "tier"] = "B"
    d.loc[s.str.contains(TIER_X, regex=True, na=False) & neuro_path, "tier"] = "X"
    d.loc[s.str.contains(TIER_A, regex=True, na=False) & neuro_path, "tier"] = "A"
    return d


def main():
    p, d = fetch()
    lines = []

    def A(s=""):
        lines.append(str(s))
        print(s)

    A("FINAL eICU RULE SET (white list + explicit exclusions)")
    A("=" * 74)
    for name, rx in (("TIER_A", TIER_A), ("TIER_X", TIER_X), ("B_LEAF", B_LEAF),
                     ("C_LEAF", C_LEAF), ("EXCLUDE_LEAF", EXCLUDE_LEAF)):
        A("  %-13s %s" % (name, rx))
    A("")

    cl = classify(d)
    hit = cl.dropna(subset=["tier"])
    A("term-level tiers: %s"
      % hit["tier"].value_counts().to_dict())

    # domain flags -> core phenotype
    stay = hit.assign(v=1).pivot_table(index="stay_id", columns="tier",
                                        values="v", aggfunc="max",
                                        fill_value=0).reset_index()
    for c in ("A", "B", "C", "X"):
        if c not in stay.columns:
            stay[c] = 0
    stay["tier_a"] = stay["A"]
    stay["tier_b"] = stay["B"]
    stay["tier_c"] = stay["C"]
    stay["other_cause"] = stay["X"]

    m = p[["stay_id", "npsle_core"]].merge(
        stay[["stay_id", "tier_a", "tier_b", "tier_c", "other_cause"]],
        on="stay_id", how="left")
    for c in ("tier_a", "tier_b", "tier_c", "other_cause"):
        m[c] = m[c].fillna(0).astype(int)
    core = m["npsle_core"].astype(int)
    m["npsle_hi"] = (((m.tier_a | m.tier_b) > 0).astype(int) & core)
    m["tier_primary"] = np.where(core == 0, "",
                        np.where(m.tier_a > 0, "A",
                          np.where(m.tier_b > 0, "B",
                            np.where(m.tier_c > 0, "C", ""))))
    m["tier_c_only"] = (m.tier_primary == "C").astype(int)
    m["tier_unassigned"] = ((core == 1) & (m.npsle_hi == 0)
                            & (m.tier_c_only == 0)).astype(int)

    A("")
    A("core events (all stays)      = %d" % int(core.sum()))
    A("  Tier A                    = %d" % int((m.tier_primary == "A").sum()))
    A("  Tier B                    = %d" % int((m.tier_primary == "B").sum()))
    A("  Tier C-only               = %d" % int(m.tier_c_only.sum()))
    A("  unassigned                = %d" % int(m.tier_unassigned.sum()))
    A("  A+B                       = %d" % int(m.npsle_hi.sum()))
    A("")
    A("core events that entered ONLY through a tier-C term:")
    A("  %d" % int((core & (m.tier_c_only > 0)).sum()))

    A("")
    A("every term that fires, with stay counts (this is Table S28 content):")
    g = (hit.groupby(["tier", "s"])["stay_id"].nunique()
         .reset_index(name="stays").sort_values(["tier", "stays"],
                                                ascending=[True, False]))
    for t in ("A", "B", "C", "X"):
        sub = g[g.tier == t]
        A("")
        A("  [%s]  %d strings, %d stay-hits" % (t, len(sub), int(sub.stays.sum())))
        for _, r in sub.iterrows():
            A("     %3d  %s" % (r["stays"], r["s"]))

    (hit.assign(db="eicu", icd_code="")[["db", "tier", "icd_code", "term", "stay_id"]]
     .groupby(["db", "tier", "icd_code", "term"], as_index=False)
     .agg(n=("stay_id", "size"), stays=("stay_id", "nunique"))
     .sort_values(["tier", "stays"], ascending=[True, False])
     .to_csv(OUT / "v8_eicu_final_terms.csv", index=False, encoding="utf-8"))
    m.to_csv(OUT / "v8_eicu_final_tiers.csv", index=False, encoding="utf-8")
    (OUT / "v8_eicu_final_rules.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
