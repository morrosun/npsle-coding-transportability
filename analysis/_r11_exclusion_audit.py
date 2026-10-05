#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Review-11 · exclusion-list ON/OFF audit restricted to the new
one-stay-per-patient eICU-CRD cohort (the 186-stay set in _first_stays.json).

Replicates extract_tier.extract_eicu exactly (tier priority A > X > B > C) and
recomputes, for the exclusion list applied (ON, the shipped rule) vs disabled
(OFF, the pre-revision state):
  core  : |CORE_DOMS| > 0
  TierB : stays whose final tier is B
  control: n - core
for the all-stays and the one-stay-per-patient bases.
"""
import json
import os
import sys

import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_cohort as EC
import extract_tier as ET

OUT = EC.OUT
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

s = diag["term"].fillna("").astype(str).str.lower()
parts = s.str.split("|")
depth = parts.apply(len)
leaf = parts.apply(lambda p: p[-1] if p else "")
neuro = s.str.contains(r"neurologic\||cns infections", regex=True)
excluded = s.str.contains(ET.EI_EXCLUDE_LEAF, regex=True, na=False)

# ---- core domains, exclusion applied vs disabled
R = EC.NP_EICU_RESTRICTED
for k in EC.CORE_DOMS:
    hit = s.str.contains(R[k], regex=True, na=False)
    diag[k + "_on"] = hit & (~excluded)
    diag[k + "_off"] = hit

# ---- tier flags exactly as extract_eicu(restricted=True) builds them
m_c = (s.str.contains(ET.EI_C_LEAF, regex=True, na=False)
       | (s.str.contains(ET.EI_C_ANY, regex=True, na=False)
          & (leaf.str.contains(ET.EI_C_ANY, regex=True, na=False) | depth.le(3))))
m_b = (s.str.contains(ET.EI_B_LEAF, regex=True, na=False)
       | (s.str.contains(ET.EI_B_ANY, regex=True, na=False)
          & leaf.str.contains(ET.EI_B_ANY, regex=True, na=False)))
m_x = s.str.contains(ET.EI_X, regex=True, na=False) & neuro
m_a = s.str.contains(ET.EI_A, regex=True, na=False) & neuro


def tier_b_stay(m_b_eff):
    """stay-level Tier B flag exactly as build() derives tier_primary == 'B':
    any term whose per-term tier is B, and no term at all whose per-term tier is
    A (per-term tier obeys the priority C -> B -> X -> A)."""
    t = pd.Series(None, index=s.index, dtype=object)
    t[m_c] = "C"
    t[m_b_eff] = "B"
    t[m_x] = "X"
    t[m_a] = "A"
    h = pd.DataFrame({"stay_id": diag["stay_id"].values, "t": t.values})
    g = h.groupby("stay_id")["t"]
    return (g.apply(lambda x: (x == "B").any()) & ~g.apply(lambda x: (x == "A").any()))


tb = pd.DataFrame({"tb_on": tier_b_stay(m_b & (~excluded)),
                   "tb_off": tier_b_stay(m_b)}).astype(int)

on = diag.groupby("stay_id")[[k + "_on" for k in EC.CORE_DOMS]].max()
off = diag.groupby("stay_id")[[k + "_off" for k in EC.CORE_DOMS]].max()
on.columns = EC.CORE_DOMS
off.columns = EC.CORE_DOMS

df = (base.set_index("stay_id")
      .join(off.add_suffix("_off")).join(on.add_suffix("_on")).join(tb))
df["core_off"] = (df[[c + "_off" for c in EC.CORE_DOMS]].sum(axis=1) > 0).astype(int)
df["core_on"] = (df[[c + "_on" for c in EC.CORE_DOMS]].sum(axis=1) > 0).astype(int)

coh = pd.read_csv(os.path.join(OUT, "cohort_eicu.csv"))
df = df[df.index.isin(set(coh["stay_id"]))]

first = set(int(x) for x in json.loads(
    open(os.path.join(OUT, "_first_stays.json"), encoding="utf-8").read())["eicu_first"])
fs = df[df.index.isin(first)]


def block(d, tag):
    n = len(d)
    c_on, c_off = int(d.core_on.sum()), int(d.core_off.sum())
    tb_on = int(((d.tb_on == 1) & (d.core_on == 1)).sum())
    tb_off = int(((d.tb_off == 1) & (d.core_off == 1)).sum())
    print("%-26s n=%4d | core OFF=%3d ON=%3d | TierB(core) OFF=%2d ON=%2d "
          "| ctrl OFF=%3d ON=%3d"
          % (tag, n, c_off, c_on, tb_off, tb_on, n - c_off, n - c_on))
    return dict(n=n, core_off=c_off, core_on=c_on, tb_off=tb_off, tb_on=tb_on,
                ctrl_off=n - c_off, ctrl_on=n - c_on)


res = {"all_stays": block(df, "eICU all stays"),
       "first_stays": block(fs, "eICU one-stay-per-patient")}
open(os.path.join(OUT, "_r11_exclusion_audit.json"), "w", encoding="utf-8").write(
    json.dumps(res, ensure_ascii=False, indent=2))
print("\nwrote", os.path.join(OUT, "_r11_exclusion_audit.json"))
