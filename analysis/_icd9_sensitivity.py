# -*- coding: utf-8 -*-
"""Review-5 item 6, verifying step: how much of the MIMIC-IV sepsis flag rests on
the non-infectious / unspecified 995.9x subcodes, and does removing those cases
change the pooled first-stay sepsis association?

Re-uses the exact model specification of v4_analyses.py T22a (first stays,
complete case, base covariate set, no clustering).
"""
import os
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import psycopg2
import statsmodels.api as sm
from scipy import stats

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
sys.path.insert(0, str(ROOT / "scripts"))
import npsle_io  # noqa: E402

ADJ_BASE = ["age", "female", "lupus_nephritis", "creat", "plt"]
L = []
def A(s=""):
    L.append(str(s))

cn = psycopg2.connect(dbname="mimiciv", host="localhost", port=5432,
                      user="postgres", password="1314")

sql = """
WITH sle AS (
  SELECT DISTINCT i.stay_id, i.hadm_id
  FROM mimiciv_icu.icustays i
  JOIN mimiciv_hosp.diagnoses_icd d ON d.hadm_id = i.hadm_id
  WHERE left(d.icd_code,3)='M32' OR left(d.icd_code,4)='7100'
),
f AS (
  SELECT s.stay_id,
    max(CASE WHEN d.icd_code ~ '^(A4[01]|R652)' THEN 1 ELSE 0 END) AS i10,
    max(CASE WHEN d.icd_code ~ '^038'           THEN 1 ELSE 0 END) AS i9_038,
    max(CASE WHEN d.icd_code ~ '^99591|^99592'  THEN 1 ELSE 0 END) AS i9_9959_inf,
    max(CASE WHEN d.icd_code ~ '^99593|^99594'  THEN 1 ELSE 0 END) AS i9_9959_noninf,
    max(CASE WHEN d.icd_code ~ '^99590'         THEN 1 ELSE 0 END) AS i9_99590
  FROM sle s JOIN mimiciv_hosp.diagnoses_icd d ON d.hadm_id = s.hadm_id
  GROUP BY 1
)
SELECT * FROM f
"""
f = pd.read_sql(sql, cn)
A("==== MIMIC-IV SLE cohort, sepsis code families (stay level, n=%d) ====" % len(f))
A("  any sepsis flag        : %d" % int((f[["i10","i9_038","i9_9959_inf","i9_9959_noninf","i9_99590"]].sum(axis=1) > 0).sum()))
for c in ["i10", "i9_038", "i9_9959_inf", "i9_9959_noninf", "i9_99590"]:
    A("  %-22s : %d" % (c, int(f[c].sum())))
susp = (f[["i9_9959_noninf", "i9_99590"]].sum(axis=1) > 0) & \
       (f[["i10", "i9_038", "i9_9959_inf"]].sum(axis=1) == 0)
A("  ONLY via 99593/99594/99590 (no infectious-sepsis code): %d  -> stay_id %s"
  % (int(susp.sum()), list(f.loc[susp, "stay_id"].astype(int))))
susp_ids = set(f.loc[susp, "stay_id"].astype(int))

code9 = pd.read_sql("""
WITH sle AS (
  SELECT DISTINCT i.hadm_id FROM mimiciv_icu.icustays i
  JOIN mimiciv_hosp.diagnoses_icd d ON d.hadm_id = i.hadm_id
  WHERE left(d.icd_code,3)='M32' OR left(d.icd_code,4)='7100')
SELECT d.icd_code, count(DISTINCT s.hadm_id) AS hadm
FROM sle s JOIN mimiciv_hosp.diagnoses_icd d ON d.hadm_id=s.hadm_id
WHERE d.icd_code ~ '^9959' GROUP BY 1 ORDER BY 1""", cn)
A("\n  every ^9959 code, admission level:")
A(code9.to_string(index=False))
cn.close()

# ---------------- refit the first-stay model with and without them -----------
FIRST = json.loads((OUT / "_first_stays.json").read_text(encoding="utf-8"))
FKEY = {"mimiciv": "mimic_first", "eicu": "eicu_first", "nwicu": "nwicu_first"}


def fit_or(d, y, x, adjust):
    adjust = [a for a in adjust if a in d.columns
              and pd.to_numeric(d[a], errors="coerce").nunique(dropna=True) > 1]
    m = d[[y, x] + adjust].apply(pd.to_numeric, errors="coerce").dropna()
    X = sm.add_constant(m[[x] + adjust], has_constant="add")
    res = sm.Logit(m[y].astype(float), X).fit(disp=0, method="bfgs", maxiter=400)
    b, se = res.params[x], res.bse[x]
    return (float(np.exp(b)), float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se)),
            float(res.pvalues[x]), len(m), int(m[y].sum()))


def dl(logor, se):
    logor, se = np.asarray(logor, float), np.asarray(se, float)
    w = 1 / se ** 2
    fe = (w * logor).sum() / w.sum()
    Q = (w * (logor - fe) ** 2).sum()
    k = len(logor)
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    wr = 1 / (se ** 2 + tau2)
    mu, semu = (wr * logor).sum() / wr.sum(), np.sqrt(1 / wr.sum())
    I2 = max(0.0, 100 * (Q - (k - 1)) / Q) if Q > 0 else 0.0
    z = mu / semu
    return dict(OR=np.exp(mu), lo=np.exp(mu - 1.96 * semu),
                hi=np.exp(mu + 1.96 * semu),
                P=2 * (1 - stats.norm.cdf(abs(z))), I2=I2)


def first(db):
    d = npsle_io.load(db)
    ids = set(pd.Series(FIRST[FKEY[db]]).astype(str))
    return d[d["stay_id"].astype(str).isin(ids)].copy()


res = {}
# Two arms, both explicit:
#   primary  = the analysis definition, in which the ICD-9-CM 995.9x family is
#              NOT matched as a whole (995.90/93/94 do not establish infection);
#   prefix   = the discarded alternative, in which any ^9959 code is matched.
# The two arms therefore differ by exactly the stays that carry only a
# non-infectious 995.9x code (susp_ids below).
for tag, prefix in (("primary", False), ("prefix", True)):
    rows = []
    for db in ("mimiciv", "eicu"):
        d = first(db)
        if db == "mimiciv" and prefix:
            m = d["stay_id"].astype(int).isin(susp_ids)
            d.loc[m, "sepsis_dx"] = 1
        r = fit_or(d, "npsle_core", "sepsis_dx", ADJ_BASE)
        rows.append(r)
        A("\n  [%s] %s adjusted OR = %.2f (%.2f-%.2f) P=%.3f  n=%d"
          % (tag, db, r[0], r[1], r[2], r[3], r[4]))
    p = dl([np.log(r[0]) for r in rows], [(np.log(r[2]) - np.log(r[1])) / (2 * 1.96) for r in rows])
    A("  [%s] pooled OR = %.2f (%.2f-%.2f) P=%.3f I2=%.1f%%"
      % (tag, p["OR"], p["lo"], p["hi"], p["P"], p["I2"]))
    res[tag] = dict(mimic=dict(or_=rows[0][0], lo=rows[0][1], hi=rows[0][2], p=rows[0][3], n=rows[0][4]),
                    eicu=dict(or_=rows[1][0], lo=rows[1][1], hi=rows[1][2], p=rows[1][3], n=rows[1][4]),
                    pooled=dict(or_=p["OR"], lo=p["lo"], hi=p["hi"], p=p["P"], i2=p["I2"]))

(OUT / "_icd9_sensitivity.txt").write_text("\n".join(L), encoding="utf-8")
(OUT / "_icd9_sensitivity.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n".join(L))
print("\n-> out/_icd9_sensitivity.json")
