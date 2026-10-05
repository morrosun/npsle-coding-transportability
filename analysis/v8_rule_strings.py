import os
# -*- coding: utf-8 -*-
"""Explain the residual 2-stay difference between the promoted restricted
phenotype (75 core) and the earlier v7 promotion (73 core), and list the
diagnosis strings that drive every domain so the rule set is reproducible.
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
    A("restricted core = %d ; legacy core = %d"
      % (int(p.npsle_core.sum()), int(p.npsle_core_legacy.sum())))

    import psycopg2
    ids = p["stay_id"].astype(int).tolist()
    with psycopg2.connect(dbname="eicu", host="localhost", port=5432,
                          user="postgres", password="1314") as cn:
        d = pd.read_sql(
            "SELECT patientunitstayid AS stay_id, diagnosisstring AS term "
            "FROM eicu_crd.diagnosis WHERE patientunitstayid = ANY(%s)", cn,
            params=(ids,))
    d["s"] = d["term"].fillna("").astype(str).str.lower()
    parts = d["s"].str.split("|")
    d["tail2"] = parts.apply(lambda x: "|".join(x[-2:]) if len(x) >= 2 else x[-1])

    # stays that the restricted rules call non-core but legacy calls core
    m = p.loc[p.npsle_core != p.npsle_core_legacy, ["stay_id"]]
    sub = d[d["stay_id"].isin(m["stay_id"])]
    A("")
    A("=== 25 stays dropped by the restricted rules: their matching strings ===")
    A("")
    A("whole-path match but tail-2 no match (the parent was doing the work):")
    A("")
    # find the strings that legacy domains matched but restricted did not
    DOM_LEG = {
        "dom_seizure": r"seizure|status epilepticus",
        "dom_enceph": r"encephalopathy|delirium|coma|altered mental|obtund|unresponsive",
        "dom_psych": r"psychosis|psychotic",
        "dom_mening": r"meningitis|encephalitis",
        "dom_demyel": r"myelitis|demyelinat|multiple sclerosis",
    }
    for k, v in DOM_LEG.items():
        hit = sub[sub["s"].str.contains(v, regex=True, na=False)
                  & ~sub["tail2"].str.contains(v, regex=True, na=False)]
        if hit.empty:
            continue
        g = (hit.groupby(["s", "tail2"])["stay_id"].nunique()
             .reset_index(name="stays").sort_values("stays", ascending=False))
        A("  [%s]  %d distinct strings, %d stays"
          % (k, len(g), int(g["stays"].sum())))
        for _, r in g.head(14).iterrows():
            A("     %-58s -> tail2 %-34s  stays=%d"
              % (r["s"][:58], r["tail2"][:34], r["stays"]))
        A("")

    A("")
    A("=== restricted rules: every string that fires, with stay counts ===")
    A("")
    RES = {
        "dom_seizure": r"seizure|status epilepticus",
        "dom_enceph": (r"change in mental status|encephalopath|coma\b|delirium|stupor"
                       r"|obtundation|unresponsive|confusion"),
        "dom_psych": r"psychosis|psychotic|schizophrenia|bipolar disorder|hallucination",
        "dom_mening": r"aseptic meningitis|non-bacterial meningitis|sterile meningitis",
        "dom_demyel": r"myelitis|demyelinat|multiple sclerosis",
    }
    for k, v in RES.items():
        hit = d[d["tail2"].str.contains(v, regex=True, na=False)]
        g = (hit.groupby("tail2")["stay_id"].nunique()
             .reset_index(name="stays").sort_values("stays", ascending=False))
        A("  [%s] %d strings, %d stays" % (k, len(g), int(g["stays"].sum())))
        for _, r in g.iterrows():
            A("     %-52s stays=%d" % (r["tail2"][:52], r["stays"]))
        A("")

    (OUT / "v8_rule_strings.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
