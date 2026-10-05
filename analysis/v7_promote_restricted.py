import os
# -*- coding: utf-8 -*-
"""
v7 — promote the leaf-restricted eICU phenotype to be the primary one.

Writes the restricted cohort and tier files under their canonical names so that
the existing analysis scripts regenerate every eICU-dependent table from the
new labels, and preserves the legacy (whole-path) files alongside so the
comparison stays reproducible.
"""
import pathlib
import shutil

import pandas as pd

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
CORE = ["dom_seizure", "dom_enceph", "dom_psych", "dom_mening", "dom_demyel"]
BROAD = CORE + ["dom_cvd", "dom_pns"]

# ---- preserve the legacy arm
for fn in ("cohort_eicu.csv", "tier_eicu.csv"):
    src, dst = OUT / fn, OUT / fn.replace(".csv", "_legacy.csv")
    if not dst.exists():
        shutil.copy2(src, dst)
        print("legacy kept:", dst.name)
    else:
        print("legacy already kept:", dst.name)

coh = pd.read_csv(OUT / "cohort_eicu.csv")
fl = pd.read_csv(OUT / "v7_restricted_tiers_eicu.csv")

# restricted phenotype columns come from the leaf-based classification
drop = [c for c in coh.columns
        if c in CORE or c in BROAD or c.startswith("npsle_")]
new = coh.drop(columns=drop).merge(fl, on="stay_id", how="left")

for c in CORE + BROAD:
    new[c] = pd.to_numeric(new[c], errors="coerce").fillna(0).astype(int)
for c in ("tier_a", "tier_b", "tier_c", "other_cause"):
    new[c] = pd.to_numeric(new[c], errors="coerce").fillna(0).astype(int)

new["npsle_core"] = (new[CORE].sum(axis=1) > 0).astype(int)
new["npsle_broad"] = (new[BROAD].sum(axis=1) > 0).astype(int)
new["npsle_sens"] = ((new[CORE].sum(axis=1)
                      + pd.to_numeric(new.get("metabolic_enceph", 0),
                                      errors="coerce").fillna(0)) > 0).astype(int)
new["npsle_hi"] = ((new["tier_a"] == 1) | (new["tier_b"] == 1)).astype(int)
new["npsle_hi_str"] = (new["npsle_hi"] == 1) & (new["other_cause"] == 0)
new["npsle_hi_str"] = new["npsle_hi_str"].astype(int)
new["npsle_any"] = ((new["tier_a"] == 1) | (new["tier_b"] == 1)
                    | (new["tier_c"] == 1)).astype(int)

order = (["db", "stay_id", "subject_id", "age", "female", "race"]
         + ["npsle_core", "npsle_broad", "npsle_sens"] + CORE + BROAD
         + ["hx_epilepsy", "lupus_nephritis", "aps", "sepsis_dx"])
order = [c for c in order if c in new.columns]
order += [c for c in new.columns if c not in order]
# the canonical cohort file must NOT carry the tier columns: the analysis scripts
# merge cohort_* with tier_* themselves, and duplicate names break that merge
TIERCOLS = ["tier_a", "tier_b", "tier_c", "other_cause",
            "npsle_hi", "npsle_hi_str", "npsle_any"]
tier = new[["stay_id"] + TIERCOLS].copy()
new = new.drop(columns=TIERCOLS)
new = new[[c for c in order if c in new.columns]]
new.to_csv(OUT / "cohort_eicu.csv", index=False, encoding="utf-8")
tier.to_csv(OUT / "tier_eicu.csv", index=False, encoding="utf-8")

print("\nrestricted eICU cohort written")
n = new
print("  core events   :", int(n["npsle_core"].sum()), "of", len(n))
print("  A+B           :", int(tier["npsle_hi"].sum()))
print("  C-only        :", int(((n["npsle_core"] == 1) & (tier["npsle_hi"] == 0)
                                 & (tier["tier_c"] == 1)).sum()))
print("  unassigned    :", int(((n["npsle_core"] == 1) & (tier["npsle_hi"] == 0)
                                 & (tier["tier_c"] == 0)).sum()))
print("  Tier A stays  :", int((tier["tier_a"] == 1).sum()))
print("  X stays       :", int((tier["other_cause"] == 1).sum()))
print("  strict A+B    :", int(tier["npsle_hi_str"].sum()))
