import os
# -*- coding: utf-8 -*-
"""Reconcile the core phenotype (extract_cohort domain rules) with the tier
assignment (extract_tier rules). Both are derived from diagnosisstring but they
answer different questions, so they need not agree stay for stay; this script
lists every disagreement with the strings responsible.
"""
import pathlib

import pandas as pd

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
lines = []


def A(s=""):
    lines.append(str(s))


def main():
    p = pd.read_csv(OUT / "cohort_eicu.csv")
    t = pd.read_csv(OUT / "tier_eicu.csv")
    m = p.merge(t[["stay_id", "tier_primary", "tier_a", "tier_b", "tier_c",
                   "other_cause"]], on="stay_id", how="left")
    for c in ("tier_a", "tier_b", "tier_c", "other_cause"):
        m[c] = m[c].fillna(0).astype(int)

    A("core=%d ; tier-assigned (A/B/C)=%d"
      % (int(m.npsle_core.sum()),
         int((m.tier_primary.isin(["A", "B", "C"])).sum())))
    A("")
    A("disagreement 1: core==1 but no tier assigned")
    d1 = m[(m.npsle_core == 1) & (m.tier_primary == "")]
    A("  n = %d" % len(d1))
    if len(d1):
        doms = ["dom_seizure", "dom_enceph", "dom_psych", "dom_mening",
                "dom_demyel", "dom_cvd", "dom_pns"]
        show = ["stay_id"] + doms + ["tier_a", "tier_b", "tier_c", "other_cause"]
        A(d1[[c for c in show if c in d1.columns]].to_string(index=False))
    A("")
    A("disagreement 2: tier assigned but core==0")
    d2 = m[(m.npsle_core == 0) & (m.tier_primary != "")]
    A("  n = %d" % len(d2))
    if len(d2):
        A(d2[[c for c in ["stay_id", "tier_primary", "tier_a", "tier_b",
                          "tier_c", "other_cause"] if c in d2.columns]]
          .to_string(index=False))

    A("")
    A("domain counts (primary rules)")
    for c in ["dom_seizure", "dom_enceph", "dom_psych", "dom_mening",
              "dom_demyel", "dom_cvd", "dom_pns"]:
        A("  %-14s primary=%3d  legacy=%3d"
          % (c, int(m[c].sum()), int(m[c + "_legacy"].sum())))

    A("")
    A("stays that are core only because of a BROAD-only domain:")
    broad_only = ["dom_cvd", "dom_pns"]
    core5 = ["dom_seizure", "dom_enceph", "dom_psych", "dom_mening", "dom_demyel"]
    bo = m[(m[core5].sum(axis=1) == 0) & (m[broad_only].sum(axis=1) > 0)]
    A("  n = %d  (these are broad, not core: %s)"
      % (len(bo), "correct, excluded from core"))

    # which strings drive the core domains
    import psycopg2
    ids = p["stay_id"].astype(int).tolist()
    with psycopg2.connect(dbname="eicu", host="localhost", port=5432,
                          user="postgres", password="1314") as cn:
        d = pd.read_sql(
            "SELECT patientunitstayid AS stay_id, diagnosisstring AS term "
            "FROM eicu_crd.diagnosis WHERE patientunitstayid = ANY(%s)", cn,
            params=(ids,))
    d["s"] = d["term"].fillna("").astype(str).str.lower()
    d["leaf"] = d["s"].str.split("|").apply(lambda p: p[-1] if p else "")
    keep = ~d["leaf"].str.contains(
        r"\|depression\b|\|anxiety\b|\|pain\b|\|bipolar disorder\b"
        r"|\|schizophrenia\b|\|dementia\b|\|suicidal ideation\b"
        r"|\|drug withdrawal syndrome\b|\|sedated\b", regex=True, na=False)
    A("")
    A("strings firing each PRIMARY core domain (leaf-aware):")
    RULES = {
        "dom_seizure": r"\|seizures\b|\|status epilepticus\b",
        "dom_enceph": (r"\|change in mental status\b|\|encephalopathy\b|\|coma\b"
                       r"|\|delirium\b|\|stupor\b|\|obtundation\b"
                       r"|\|unresponsive\b|\|confusion\b|\|agitation\b"),
        "dom_psych": (r"\|psychosis\b|\|psychotic\b|\|schizophrenia\b"
                      r"|\|bipolar disorder\b|\|hallucination"),
        "dom_mening": r"meningitis|encephalitis|meningoencephalitis",
        "dom_demyel": r"myelitis|demyelinat|multiple sclerosis",
    }
    for k, v in RULES.items():
        h = d[keep & d["s"].str.contains(v, regex=True, na=False)]
        g = (h.groupby("s")["stay_id"].nunique().reset_index(name="stays")
             .sort_values("stays", ascending=False))
        A("")
        A("  [%s] %d strings, %d stays" % (k, len(g), int(g.stays.sum())))
        for _, r in g.iterrows():
            A("     %3d  %s" % (r["stays"], r["s"]))

    (OUT / "v8_core_tier_reconcile.txt").write_text("\n".join(lines),
                                                    encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
