# -*- coding: utf-8 -*-
"""Review-8: two count artifacts the manuscript now quotes.

  * out/t43_eicu_timing_first.csv   eICU-CRD diagnostic-timestamp timing on the
    first-stay basis (Table S20 text). The published Table S20 is all-stays;
    the first-stay sentence previously attached 92.2% to the wrong denominator.
  * out/t44_ab_complete_case.csv    composition of the complete-case Tier A+B
    models behind Table 2 and Table S25. The manuscript previously quoted the
    raw screening subsets (313 and 160 stays) as if they were the model sets.

Both are written so that export_final_numbers.py can read them instead of
re-deriving them, keeping one source of truth per number.
"""
import json
import pathlib
import sys

import pandas as pd
import psycopg2

SCR = pathlib.Path(__file__).resolve().parent
ROOT = SCR.parent
OUT = ROOT / "out"
sys.path.insert(0, str(SCR))
import npsle_io  # noqa: E402

# The core-term pattern is restated from v4_analyses.py, which owns the
# all-stays version of this query, so the two artefacts stay comparable.
NP = (r"seizure|status epilepticus|encephalopathy|delirium|coma|altered mental|"
      r"obtund|unresponsive|psychosis|psychotic|meningitis|encephalitis|"
      r"myelitis|demyelinat|multiple sclerosis")
SEP = r"sepsis|septic shock"
ADJ = ["age", "female", "lupus_nephritis", "creat", "plt"]

# ------------------------------------------------ 1. first-stay timing
_first = json.loads((OUT / "_first_stays.json").read_text(encoding="utf-8"))
fs = set(_first["eicu_first"])
d = npsle_io.load("eicu")
ids = tuple(int(x) for x in d["stay_id"].tolist())
sql = f"""
SELECT patientunitstayid AS stay_id,
       min(CASE WHEN diagnosisstring ~* '{NP}' THEN diagnosisoffset END) AS np_off,
       min(CASE WHEN diagnosisstring ~* '{SEP}' THEN diagnosisoffset END) AS sep_off
FROM eicu_crd.diagnosis WHERE patientunitstayid IN {ids} GROUP BY 1"""
cn = psycopg2.connect(dbname="eicu", host="localhost", port=5432,
                      user="postgres", password="1314")
off = pd.read_sql(sql, cn)
cn.close()
d = d.merge(off, on="stay_id", how="left")
f = d[d["stay_id"].isin(fs)].copy()
npd = f[(pd.to_numeric(f["npsle_core"], errors="coerce") == 1)
        & f["np_off"].notna()]
w = int((npd["np_off"] <= 1440).sum())
b = npd[npd["sep_off"].notna()]
sf = int((b["sep_off"] < b["np_off"]).sum())
sm = int((b["sep_off"] == b["np_off"]).sum())
nf = int((b["sep_off"] > b["np_off"]).sum())
t43 = pd.DataFrame([
    {"指标": "首次住院核心事件且有时间戳的住院数", "结果": f"{len(npd)}"},
    {"指标": "首次神经精神诊断发生于入 ICU 24 h 内, n (%)",
     "结果": f"{w}/{len(npd)} ({100*w/len(npd):.1f}%)"},
    {"指标": "首次神经精神诊断距入 ICU 时间, 中位数 [IQR], 小时",
     "结果": f"{npd['np_off'].median()/60:.1f} "
             f"[{npd['np_off'].quantile(.25)/60:.1f}, {npd['np_off'].quantile(.75)/60:.1f}]"},
    {"指标": "同时具备两类诊断时间戳的住院数", "结果": f"{len(b)}"},
    {"指标": "脓毒症诊断时间早于神经精神诊断, n (%)",
     "结果": f"{sf}/{len(b)} ({100*sf/len(b):.1f}%)"},
    {"指标": "两者记录于同一时间点, n (%)",
     "结果": f"{sm}/{len(b)} ({100*sm/len(b):.1f}%)"},
    {"指标": "神经精神诊断时间早于脓毒症诊断, n (%)",
     "结果": f"{nf}/{len(b)} ({100*nf/len(b):.1f}%)"},
])
t43.to_csv(OUT / "t43_eicu_timing_first.csv", index=False, encoding="utf-8-sig")
print(t43.to_string(index=False))

# --------------------------------- 2. complete-case Tier A+B model sets
rows = []
for db, lab in (("mimiciv", "MIMIC-IV"), ("eicu", "eICU-CRD")):
    m = npsle_io.load(db)
    fids = set(_first[{"mimiciv": "mimic_first", "eicu": "eicu_first"}[db]])
    m = m[m["stay_id"].isin(fids)].copy()
    no_ev = pd.to_numeric(m["npsle_core"], errors="coerce").fillna(0) == 0
    hi = pd.to_numeric(m["npsle_hi"], errors="coerce").fillna(0) == 1
    sub = m[hi | no_ev].copy()
    sub["_y"] = hi[hi | no_ev].astype(int)
    raw_n, raw_ev = len(sub), int(sub["_y"].sum())
    adj = [a for a in ADJ
           if a in sub.columns
           and pd.to_numeric(sub[a], errors="coerce").notna().sum() > 0
           and pd.to_numeric(sub[a], errors="coerce").nunique(dropna=True) > 1]
    mm = sub[["_y", "sepsis_dx"] + adj].apply(pd.to_numeric, errors="coerce").dropna()
    ev, co = mm[mm["_y"] == 1], mm[mm["_y"] == 0]
    rows.append(dict(数据库=lab, raw_subset_n=raw_n, raw_events=raw_ev,
                     raw_controls=raw_n - raw_ev, complete_case_n=len(mm),
                     ab_events=len(ev),
                     ab_sepsis_pos=int(ev["sepsis_dx"].sum()),
                     ab_sepsis_neg=int((ev["sepsis_dx"] == 0).sum()),
                     controls=len(co),
                     controls_sepsis_pos=int(co["sepsis_dx"].sum()),
                     controls_sepsis_neg=int((co["sepsis_dx"] == 0).sum()),
                     dropped_events=raw_ev - len(ev),
                     dropped_controls=(raw_n - raw_ev) - len(co)))
t44 = pd.DataFrame(rows)
t44.to_csv(OUT / "t44_ab_complete_case.csv", index=False, encoding="utf-8-sig")
print()
print(t44.to_string(index=False))
print("\n[ok] out/t43_eicu_timing_first.csv, out/t44_ab_complete_case.csv")
