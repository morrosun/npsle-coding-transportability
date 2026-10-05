import os
# -*- coding: utf-8 -*-
"""
v7 — verify the two remaining factual questions before any wording is changed.

Q1 (tier rules).  How are the tier flags and the competing-etiology (X) flag
    actually combined in the primary analysis?  Report the A/B/C/X overlap so
    the S3 high-confidence -> strict counts can be explained.

Q2 (missing data).  For every analysis family, which variables are missing, at
    what rate, and what is the final analytic n?  This is the source table for
    the unified accounting table.
"""
import io
import json
import pathlib

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npsle_io

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
DBS = ["mimiciv", "eicu", "nwicu"]
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
FIRST = json.loads((OUT / "_first_stays.json").read_text(encoding="utf-8"))
FKEY = {"mimiciv": "mimic_first", "eicu": "eicu_first", "nwicu": "nwicu_first"}


def num(s):
    return pd.to_numeric(s, errors="coerce")


W = []
A = W.append


def load(db):
    c = npsle_io.load(db)
    t = pd.read_csv(OUT / f"tier_{db}.csv")
    d = c.merge(t, on="stay_id", how="left")
    d = d[d["stay_id"].isin(set(FIRST[FKEY[db]]))].copy()
    d["_core"] = num(d["npsle_core"]) == 1
    d["_hi"] = num(d["npsle_hi"]) == 1
    d["_tc"] = num(d["tier_c"]) == 1
    d["_x"] = num(d["other_cause"]) == 1
    d["_a"] = num(d["tier_a"]) == 1
    d["_b"] = num(d["tier_b"]) == 1
    return d


# ======================================================================= Q1
A("=" * 92)
A("Q1  TIER AND COMPETING-ETIOLOGY (X) FLAG OVERLAP -- first-stay cohort")
A("=" * 92)
A("%-11s %6s %6s %6s %6s %6s %6s | %8s %8s | %9s" %
  ("database", "core", "A", "B", "C", "X", "A+B", "A+B&X", "C-only", "unassign"))
q1 = {}
for db in DBS:
    d = load(db)
    core = int(d["_core"].sum())
    a, b, c, x = (int(d["_a"].sum()), int(d["_b"].sum()), int(d["_tc"].sum()),
                  int(d["_x"].sum()))
    ab = int((d["_core"] & d["_hi"]).sum())
    abx = int((d["_core"] & d["_hi"] & d["_x"]).sum())
    co = int((d["_core"] & d["_tc"] & ~d["_hi"]).sum())
    un = int((d["_core"] & ~d["_hi"] & ~d["_tc"]).sum())
    q1[db] = dict(core=core, a=a, b=b, c=c, x=x, ab=ab, abx=abx, conly=co, un=un)
    A("%-11s %6d %6d %6d %6d %6d %6d | %8d %8d | %9d" %
      (LABEL[db], core, a, b, c, x, ab, abx, co, un))
A("")
A("Reading: A+B is tier_a OR tier_b and does NOT exclude X.  The strict variant")
A("(npsle_hi_str in out/tier_*.csv) is (A or B) and not X.  The difference")
A("between the two is the A+B&X column, which is exactly why the S3/S4")
A("high-confidence and strict counts differ.")
A("")
A("stays with X=1 and no A/B/C flag at all (these are the 'unassigned'):")
for db in DBS:
    d = load(db)
    n = int((d["_core"] & ~d["_hi"] & ~d["_tc"] & d["_x"]).sum())
    A("   %-11s %d of %d unassigned" % (LABEL[db], n, q1[db]["un"]))
hs = {}
for db in DBS:
    d = load(db)
    hs[db] = int((d["_core"] & d["_hi"]).sum())

# ======================================================================= Q2
A("")
A("=" * 92)
A("Q2  MISSING DATA BY ANALYSIS FAMILY -- first-stay cohort")
A("=" * 92)
BASE = ["age", "female", "lupus_nephritis", "creat", "plt"]
SEV = ["sofa24", "sofa_cns", "apache"]
OUTV = ["hosp_mort", "vent24", "icu_los"]
MLF = ["age", "female", "gcs_min", "hr", "map", "temp", "spo2", "wbc", "creat",
       "sepsis_dx"]

q2 = {}
for db in DBS:
    d = load(db)
    n = len(d)
    A("")
    A("%s  n = %d" % (LABEL[db], n))
    A("   %-18s %8s %9s" % ("variable", "n_missing", "pct"))
    for v in BASE + SEV + OUTV:
        s = num(d[v])
        A("   %-18s %8d %8.1f%%" % (v, int(s.isna().sum()), 100 * s.isna().mean()))
    nb = [v for v in BASE if num(d[v]).notna().sum() > 0 and num(d[v]).nunique() > 1]
    ns = [v for v in SEV if num(d[v]).notna().sum() > 0 and num(d[v]).nunique() > 1]
    nm = [v for v in MLF if v in d.columns and num(d[v]).notna().sum() > 0]
    n_base = len(d[["sepsis_dx"] + nb].apply(pd.to_numeric, errors="coerce").dropna())
    n_out = len(d[["sepsis_dx"] + nb + ns].apply(pd.to_numeric, errors="coerce").dropna())
    n_ml = len(d[nm].apply(pd.to_numeric, errors="coerce").dropna())
    A("   covariates used, sepsis model : %s" % ", ".join(nb))
    A("   severity added, outcome models: %s" % (", ".join(ns) if ns else "none"))
    A("   final n  sepsis %d | outcome %d | identification %d" % (n_base, n_out, n_ml))
    q2[db] = dict(n=n, base=nb, sev=ns, ml=nm, n_base=n_base, n_out=n_out, n_ml=n_ml)

A("")
A("=" * 92)
A("HANDLING, as implemented")
A("=" * 92)
A("association models : complete case -- v9_revision.fit_or() applies")
A("                     pd.to_numeric(...).dropna() to outcome, exposure and")
A("                     every retained adjuster (listwise deletion)")
A("identification     : fold-wise median imputation -- part2_model.Prep.fit()")
A("                     learns med_ and spline knots on the training fold only")
A("                     and applies them unchanged to the validation fold")
A("")
A("=> the manuscript sentence 'median imputation in the association models and")
A("   complete-case analysis in the identification models' REVERSES the code.")

txt = "\n".join(W)
(OUT / "v7_rule_and_missingness_audit.txt").write_text(txt, encoding="utf-8")
(OUT / "v7_rule_and_missingness_audit.json").write_text(
    json.dumps(dict(q1=q1, q2=q2), ensure_ascii=False, indent=1, default=str),
    encoding="utf-8")
print(txt)
