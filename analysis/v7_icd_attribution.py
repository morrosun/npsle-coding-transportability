import os
# -*- coding: utf-8 -*-
"""
Check whether lupus neuropsychiatric involvement is expressible in ICD at all.
The manuscript currently claims Tier A is "structurally unavailable" in the
ICD-based databases.  Test that directly: within the MIMIC-IV SLE cohort, how
often do organ-involvement SLE subcodes co-occur with a neurologic/encephalitis
code, and specifically does M32.19 (SLE with other organ involvement) appear
with G05.3 (encephalitis in diseases classified elsewhere)?
"""
import io
import pathlib

import pandas as pd
import psycopg2

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
CFG = dict(host="localhost", port=5432, user="postgres", password="1314")
H, I = "mimiciv_hosp", "mimiciv_icu"

ids = pd.read_csv(OUT / "cohort_mimiciv.csv")["stay_id"].astype("int64").tolist()

SQL = f"""
WITH icu AS (
  SELECT i.stay_id, i.hadm_id, i.subject_id
  FROM {I}.icustays i
  WHERE i.stay_id = ANY(%(ids)s)
), dx AS (
  SELECT d.hadm_id,
         bool_or(d.icd_code LIKE 'M32%%')                       AS sle,
         bool_or(d.icd_code LIKE 'M321%%')                     AS sle_organ,
         bool_or(d.icd_code = 'M3215')                          AS m3215,
         bool_or(d.icd_code = 'M3219')                          AS m3219,
         bool_or(d.icd_code = 'G053')                           AS g053,
         bool_or(d.icd_code LIKE 'G05%%')                        AS g05_any,
         bool_or(d.icd_code LIKE 'G934%%' OR d.icd_code LIKE 'R41%%'
                 OR d.icd_code LIKE 'F05%%' OR d.icd_code LIKE 'G40%%') AS np_generic
  FROM {H}.diagnoses_icd d
  JOIN icu ON icu.hadm_id = d.hadm_id
  GROUP BY d.hadm_id
)
SELECT * FROM dx
"""

with psycopg2.connect(dbname="mimiciv", **CFG) as cn:
    df = pd.read_sql(SQL, cn, params={"ids": ids})

W = []
A = W.append
A("MIMIC-IV cohort hospital admissions with any diagnosis row: %d" % len(df))
A("")
A("%-46s %8s %8s" % ("pattern", "n", "%"))
for lab, col in (("SLE (M32*)", "sle"),
                 ("SLE with organ involvement (M321*)", "sle_organ"),
                 ("M32.15 SLE, other nervous system organs", "m3215"),
                 ("M32.19 SLE, other organ involvement", "m3219"),
                 ("G05.3 encephalitis in diseases elsewhere", "g053"),
                 ("any G05*", "g05_any"),
                 ("M321* + any G05*  [lupus + encephalitis]", None),
                 ("M3215 + any G05*  [lupus NPS + encephalitis]", None)):
    if col is None:
        n = int((df["sle_organ"] & df["g05_any"]).sum()) if "M321*" in lab \
            else int((df["m3215"] & df["g05_any"]).sum())
    else:
        n = int(df[col].sum())
    A("%-46s %8d %7.1f%%" % (lab, n, 100 * n / len(df)))

A("")
A("M32.15 AND G05* in the same admission   : %d" % int((df["m3215"] & df["g05_any"]).sum()))
A("M32.19 AND G05.3 in the same admission  : %d" % int((df["m3219"] & df["g053"]).sum()))
A("M321*  AND G05*      in the same admission : %d"
  % int((df["sle_organ"] & df["g05_any"]).sum()))
A("")
A("Interpretation: a nonzero count means the ICD-10-CM combination coding that")
A("links lupus to organ involvement CAN be used to express a lupus-attributed")
A("neurologic diagnosis, so 'structurally unavailable' is too strong.  A zero")
A("count would support the stronger statement for this cohort only.")

txt = "\n".join(W)
(OUT / "v7_icd_attribution_check.txt").write_text(txt, encoding="utf-8")
print(txt)
