#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Review-9 · stay-level audit of the eICU-CRD 85 -> 83 exclusion revision.

Question from the reviewer: the arithmetic 85-83 = 33-31 = 2 does not by itself
identify WHICH stays moved or WHICH domain they lost.  This script recomputes
eICU-CRD core membership twice from the raw diagnosis strings -- once with the
exclusion list applied to the whole string (the shipped, "audited" rule) and
once with the exclusion list disabled (the state before the revision, which is
what leaf-only matching had produced because the leading "|" made the list
unreachable) -- and reports, stay by stay, which core domains were lost.
"""
import os, sys, re
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_cohort as EC

OUT = EC.OUT
CORE_DOMS = EC.CORE_DOMS
EXCL = EC.EICU_EXCLUDE_LEAF
RESTRICTED = EC.NP_EICU_RESTRICTED

cn = psycopg2.connect(dbname="eicu", **EC.CONFIG)
base = pd.read_sql("""
    WITH sle AS (SELECT DISTINCT patientunitstayid AS stay_id
                 FROM eicu_crd.diagnosis WHERE diagnosisstring ~* 'lupus')
    SELECT s.stay_id, p.uniquepid AS subject_id
    FROM sle s JOIN eicu_crd.patient p ON p.patientunitstayid = s.stay_id
""", cn)
ids = ",".join(str(int(x)) for x in base["stay_id"].unique())
diag = pd.read_sql(f"""
    SELECT patientunitstayid AS stay_id, diagnosisstring AS term
    FROM eicu_crd.diagnosis WHERE patientunitstayid IN ({ids})
""", cn)
cn.close()

diag["s"] = diag["term"].fillna("").astype(str).str.lower()
diag["excl_hit"] = diag["s"].str.contains(EXCL, regex=True, na=False)

# domain flags with and without the exclusion list
for k, pat in RESTRICTED.items():
    hit = diag["s"].str.contains(pat, regex=True, na=False)
    diag[k + "_on"] = hit & (~diag["excl_hit"])      # shipped rule
    diag[k + "_off"] = hit                            # exclusion disabled

on = diag.groupby("stay_id")[[k + "_on" for k in CORE_DOMS]].max()
off = diag.groupby("stay_id")[[k + "_off" for k in CORE_DOMS]].max()
on.columns = CORE_DOMS
off.columns = CORE_DOMS
df = base.set_index("stay_id").join(off.add_suffix("_off")).join(on.add_suffix("_on"))
df["core_off"] = (df[[c + "_off" for c in CORE_DOMS]].sum(axis=1) > 0).astype(int)
df["core_on"] = (df[[c + "_on" for c in CORE_DOMS]].sum(axis=1) > 0).astype(int)

# restrict to the analysis universe actually shipped
coh = pd.read_csv(os.path.join(OUT, "cohort_eicu.csv"))
keep = set(coh["stay_id"])
df = df[df.index.isin(keep)]

print("universe (eICU-CRD stays in cohort_eicu.csv): %d" % len(df))
print("core, exclusion DISABLED (pre-revision state): %d" % int(df.core_off.sum()))
print("core, exclusion APPLIED  (shipped, audited)  : %d" % int(df.core_on.sum()))

lost = df[(df.core_off == 1) & (df.core_on == 0)]
gained = df[(df.core_off == 0) & (df.core_on == 1)]
print("stays losing core membership: %d ; gaining: %d" % (len(lost), len(gained)))

rows = []
for sid, r in lost.iterrows():
    doms_off = [c for c in CORE_DOMS if r[c + "_off"] == 1]
    doms_on = [c for c in CORE_DOMS if r[c + "_on"] == 1]
    terms = diag[(diag.stay_id == sid) & (diag.excl_hit)]
    fired = []
    for _, t in terms.iterrows():
        for c in CORE_DOMS:
            if re.search(RESTRICTED[c], t["s"]):
                fired.append("%s :: %s" % (c, t["term"]))
    rows.append(dict(stay_id=sid, subject_id=r["subject_id"],
                     domains_before=",".join(doms_off) or "-",
                     domains_after=",".join(doms_on) or "-",
                     n_excluded_terms=int(len(terms)),
                     excluded_terms_matching_a_core_domain="; ".join(sorted(set(fired))) or "-"))
res = pd.DataFrame(rows)
p = os.path.join(OUT, "t45_eicu_exclusion_stay_audit.csv")
res.to_csv(p, index=False)
print("\n" + res.to_string(index=False))
print("\nwrote %s" % p)

# ---- domain-level effect: the list can also remove non-psychotic evidence ----
print("\n--- domain-level effect of the exclusion list (230-stay universe) ---")
flip_rows = []
for c in CORE_DOMS:
    offc, onc = df[c + "_off"], df[c + "_on"]
    flip_rows.append(dict(domain=c,
                          stays_with_evidence_off=int(offc.sum()),
                          stays_with_evidence_on=int(onc.sum()),
                          stays_losing_this_domain=int(((offc == 1) & (onc == 0)).sum())))
fl = pd.DataFrame(flip_rows)
print(fl.to_string(index=False))
p2 = os.path.join(OUT, "t46_eicu_exclusion_domain_effect.csv")
fl.to_csv(p2, index=False)
print("wrote %s" % p2)

# which excluded leaves fired across the whole cohort (context, not the audit)
allx = diag[diag.excl_hit]["s"].str.extractall(r"(\|[a-z ]+)")
print("\nexcluded fragments observed in the cohort (top 15):")
print(diag[diag.excl_hit]["term"].value_counts().head(15).to_string())
