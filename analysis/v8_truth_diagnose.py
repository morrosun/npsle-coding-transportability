import os
# -*- coding: utf-8 -*-
"""Diagnose why the raw A|B union (51) exceeds the mutually exclusive A∪B
count (42) in eICU-CRD. The difference must not be explained by overlap
counting; identify the actual mechanism stay by stay.
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
TIER_COLS = ["tier_a", "tier_b", "tier_c", "other_cause"]

lines = []


def A(s=""):
    lines.append(str(s))


def num(x):
    return pd.to_numeric(x, errors="coerce")


def main():
    first = json_first()
    for db in ("mimiciv", "eicu", "nwicu"):
        coh = pd.read_csv(OUT / ("cohort_%s.csv" % db))
        drop = [c for c in TIER_COLS if c in coh.columns]
        coh = coh.drop(columns=drop, errors="ignore")
        t = pd.read_csv(OUT / ("tier_%s.csv" % db))
        d = coh.merge(t[["stay_id"] + TIER_COLS], on="stay_id", how="left")
        for c in TIER_COLS:
            d[c] = num(d[c]).fillna(0).astype(int)
        d["core"] = num(d["npsle_core"]).fillna(0).astype(int)
        fk = {"mimiciv": "mimic", "eicu": "eicu", "nwicu": "nwicu"}[db]
        d["first"] = d["stay_id"].isin(first[fk])

        A("=" * 74)
        A("%s   stays=%d" % (db, len(d)))
        A("=" * 74)
        A("")
        A("  4-way cross-tab of core membership x raw tier flags (all stays)")
        d["any_tier"] = ((d["tier_a"] | d["tier_b"] | d["tier_c"]) == 1).astype(int)
        ct = pd.crosstab([d["core"], d["any_tier"]],
                         columns="n")
        A(ct.to_string())
        A("")
        orc = d[(d["tier_a"] == 1) | (d["tier_b"] == 1)]
        A("  raw A|B stays                        : %d" % len(orc))
        A("    of which core==1                    : %d" % int((orc["core"] == 1).sum()))
        A("    of which core==0 (OUTSIDE universe): %d"
          % int((orc["core"] == 0).sum()))
        A("  mutually exclusive A∪B inside core==1 : %d"
          % int(((orc["core"] == 1)).sum()))
        both = int(((d["tier_a"] == 1) & (d["tier_b"] == 1)).sum())
        A("  stays carrying BOTH A and B flags     : %d" % both)
        bc = int(((d["tier_b"] == 1) & (d["tier_c"] == 1)).sum())
        ac = int(((d["tier_a"] == 1) & (d["tier_c"] == 1)).sum())
        A("  stays carrying BOTH B and C flags     : %d" % bc)
        A("  stays carrying BOTH A and C flags     : %d" % ac)

        if db == "eicu":
            bad = orc[orc["core"] == 0]
            A("")
            A("  --- the %d stays with an A or B flag but core==0 ---" % len(bad))
            A("  domains that actually fired:")
            CORE = ["dom_enceph", "dom_psych", "dom_seiz", "dom_cvs",
                    "dom_headache", "dom_mening"]
            for c in CORE:
                if c in d.columns:
                    A("    %-14s %d of %d" % (c, int(num(bad[c]).fillna(0).sum()),
                                               len(bad)))
            A("  npsle_broad==1 among them: %d"
              % int(num(bad.get("npsle_broad")).fillna(0).sum()))
            A("")
            A("  Are they in the BROAD universe (i.e. the covariate-only domains)?")
            extra = [c for c in d.columns
                     if c.startswith("dom_") and c not in CORE]
            for c in extra:
                k = int(num(bad[c]).fillna(0).sum())
                if k:
                    A("    %-18s %d" % (c, k))

        A("")
        f = d[d["first"]]
        orf = f[(f["tier_a"] == 1) | (f["tier_b"] == 1)]
        A("  FIRST STAYS: raw A|B = %d ; core==1 among them = %d ; core==0 = %d"
          % (len(orf), int((orf["core"] == 1).sum()), int((orf["core"] == 0).sum())))
        A("")

    (OUT / "truth_diagnosis.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def json_first():
    import json
    d = json.loads((OUT / "_first_stays.json").read_text(encoding="utf-8"))
    return {k.replace("_first", ""): set(int(x) for x in v)
            for k, v in d.items()}


if __name__ == "__main__":
    main()
