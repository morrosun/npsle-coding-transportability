import os
# -*- coding: utf-8 -*-
"""Confirm the leak mechanism: which precomputed column is universe-unrestricted,
and which stays are affected. This determines whether the fix is a universe
restriction on the tier flags or a change to the phenotype itself.
"""
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
lines = []


def A(s=""):
    lines.append(str(s))


def num(x):
    return pd.to_numeric(x, errors="coerce")


def main():
    first = json.loads((OUT / "_first_stays.json").read_text(encoding="utf-8"))
    fs = {"mimiciv": set(first["mimic_first"]),
          "eicu": set(first["eicu_first"]),
          "nwicu": set(first["nwicu_first"])}

    A("WHICH PRECOMPUTED COLUMN LEAKS OUTSIDE THE CORE UNIVERSE?")
    A("=" * 74)
    for db in ("mimiciv", "eicu", "nwicu"):
        coh = pd.read_csv(OUT / ("cohort_%s.csv" % db))
        t = pd.read_csv(OUT / ("tier_%s.csv" % db))
        d = coh[[c for c in ("stay_id", "npsle_core") if c in coh.columns]].merge(
            t, on="stay_id", how="left")
        d["core"] = num(d["npsle_core"]).fillna(0).astype(int)
        for c in ("tier_a", "tier_b", "tier_c", "other_cause",
                  "npsle_hi", "npsle_hi_str", "npsle_any"):
            if c in d.columns:
                d[c] = num(d[c]).fillna(0).astype(int)
        fk = fs[db]
        A("")
        A("%s (stays=%d)" % (db, len(d)))
        A("  column            n=1   of which core==0   first-stay n=1  of which core==0")
        for c in ("tier_a", "tier_b", "tier_c", "other_cause",
                  "npsle_hi", "npsle_hi_str", "npsle_any"):
            if c not in d.columns:
                continue
            m = d[c].eq(1)
            mf = d["stay_id"].isin(fk) & m
            A("  %-16s %5d %14d %16d %14d"
              % (c, int(m.sum()), int((m & d["core"].eq(0)).sum()),
                 int(mf.sum()),
                 int((mf & d["core"].eq(0)).sum())))

        # the correct, universe-restricted counts
        d["hi_ok"] = (d["tier_a"].eq(1) | d["tier_b"].eq(1)) & d["core"].eq(1)
        d["tc_ok"] = d["tier_c"].eq(1) & d["core"].eq(1) & ~d["hi_ok"]
        A("  universe-restricted: core=%d  A+B=%d  C-only=%d  unassigned=%d"
          % (int(d["core"].sum()), int(d["hi_ok"].sum()), int(d["tc_ok"].sum()),
             int((d["core"].eq(1) & ~d["hi_ok"] & ~d["tc_ok"]).sum())))
        f = d[d["stay_id"].isin(fk)]
        fhi = (f["tier_a"].eq(1) | f["tier_b"].eq(1)) & f["core"].eq(1)
        ftc = f["tier_c"].eq(1) & f["core"].eq(1) & ~fhi
        A("  first-stay restricted: core=%d  A+B=%d  C-only=%d  unassigned=%d"
          % (int(f["core"].sum()), int(fhi.sum()), int(ftc.sum()),
             int((f["core"].eq(1) & ~fhi & ~ftc).sum())))

    # what does the phenotype itself say about the leaking stays?
    A("")
    A("THE LEAKING STAYS IN DETAIL (eICU-CRD, A or B flag but core==0)")
    A("=" * 74)
    coh = pd.read_csv(OUT / "cohort_eicu.csv")
    t = pd.read_csv(OUT / "tier_eicu.csv")
    d = coh.merge(t, on="stay_id", how="left", suffixes=("", "_t"))
    d["core"] = num(d["npsle_core"]).fillna(0).astype(int)
    bad = d[((d["tier_a"].eq(1)) | (d["tier_b"].eq(1))) & d["core"].eq(0)]
    cols = ["stay_id", "tier_a", "tier_b", "tier_c", "other_cause", "core"]
    cols += [c for c in d.columns if c.startswith("dom_")]
    cols += [c for c in ("npsle_core", "npsle_broad", "npsle_any") if c in d.columns]
    A(bad[[c for c in dict.fromkeys(cols) if c in bad.columns]].to_string(index=False))
    A("")
    A("  their diagnosisString-derived terms (from t11 table):")
    t11 = pd.read_csv(OUT / "t11_tier_terms.csv")
    tt = t11[t11["stay_id"].isin(bad["stay_id"])] \
        if "stay_id" in t11.columns else None
    if tt is None:
        A("    t11_tier_terms.csv has no stay_id column: %s" % list(t11.columns))
    else:
        A(tt.to_string(index=False))

    (OUT / "truth_leak.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
