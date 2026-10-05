# -*- coding: utf-8 -*-
"""
生成 out/_first_stays.json —— 每个受试者的「一人一次 ICU 住院」stay_id 列表。
这是 npsle_io.load() / v4_analyses.py / v7_identification_models.py 等的必需前置输入。

依赖：先运行 extract_cohort.py 生成 out/cohort_<db>.csv（本脚本只在这些 SLE 队列
stay_id 范围内挑选，故产出的选择集必为全队列的子集）。

选择规则（review-11 修正）
------------------------------------------------------------------
* MIMIC-IV / NWICU：按 subject_id 取 intime 最早的 icustay（同 intime 取最小 stay_id）。
  绝对时间戳，可直接确定时间上的首次 ICU 住院。

* eICU-CRD：offsets 以每次 ICU admission 为零点，同一次住院内**首次** ICU stay 的
  hospitalAdmitOffset 更大（更接近 0）；跨住院的 offset 不可直接比较。因此采用
  明确、可复现的「one-stay-per-patient」规则：
      1) 取 hospitaldischargeyear 最小（最早一次住院）；
      2) 其内取 unitvisitnumber 最小（该次住院的首次 ICU stay）；
      3) 再取 hospitalAdmitOffset 最大（最接近住院开始）；
      4) 最后取 patientunitstayid 最小以打破并列。
  同一年内存在多次住院时无法判定先后，此时规则退化为 step 3–4 的确定性选择，
  故 eICU 队列应表述为 one-stay-per-patient，而非严格 first-stay。

产物 JSON 键：mimic_first / eicu_first / nwicu_first（与 npsle_io.FKEY 对齐）。
"""
import os, json
import pandas as pd
import psycopg2

CONFIG = dict(host="localhost", port=5432, user="postgres", password="1314")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)


def q(db, sql):
    with psycopg2.connect(dbname=db, **CONFIG) as cn:
        return pd.read_sql(sql, cn)


def _cohort_ids(cohort_csv):
    coh = pd.read_csv(cohort_csv)
    return tuple(int(x) for x in coh["stay_id"].unique())


def pick_first_icd(db, table, subj_col, stay_col, time_expr, cohort_csv):
    """MIMIC-IV / NWICU：按 subject 取 intime 最早一次（同时间取最小 stay_id）。"""
    ids = _cohort_ids(cohort_csv)
    if not ids:
        return []
    df = q(db, "SELECT {subj} AS subj, {stay} AS stay, {t} AS t "
              "FROM {tbl} WHERE {stay} IN {ids}".format(
                  subj=subj_col, stay=stay_col, t=time_expr, tbl=table, ids=ids))
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["t", "subj", "stay"])
    df = df.sort_values(["t", "stay"]).drop_duplicates(subset=["subj"], keep="first")
    return sorted(int(x) for x in df["stay"].tolist())


def pick_first_eicu(cohort_csv):
    """eICU-CRD：one-stay-per-patient（见文件头规则说明）。"""
    ids = _cohort_ids(cohort_csv)
    if not ids:
        return []
    df = q("eicu", "SELECT uniquepid AS subj, patientunitstayid AS stay, "
                   "unitvisitnumber AS vnum, hospitaladmitoffset AS adm, "
                   "hospitaldischargeyear AS yr "
                   "FROM eicu_crd.patient WHERE patientunitstayid IN {ids}".format(ids=ids))
    for c in ["vnum", "adm", "yr"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["subj", "stay"])
    # earliest hospital admission -> first ICU within it -> closest to admission -> smallest stay id
    df = df.sort_values(["yr", "vnum", "adm", "stay"],
                        ascending=[True, True, False, True], na_position="last")
    df = df.drop_duplicates(subset=["subj"], keep="first")
    return sorted(int(x) for x in df["stay"].tolist())


def main():
    mimic = pick_first_icd("mimiciv", "mimiciv_icu.icustays",
                           "subject_id", "stay_id", "EXTRACT(EPOCH FROM intime)",
                           os.path.join(OUT, "cohort_mimiciv.csv"))
    nwicu = pick_first_icd("nwicu", "icu.icustays",
                           "subject_id", "stay_id", "EXTRACT(EPOCH FROM intime)",
                           os.path.join(OUT, "cohort_nwicu.csv"))
    eicu = pick_first_eicu(os.path.join(OUT, "cohort_eicu.csv"))
    out = {"mimic_first": mimic, "eicu_first": eicu, "nwicu_first": nwicu}
    path = os.path.join(OUT, "_first_stays.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"[ok] wrote {path}: "
          f"mimic={len(mimic)} eicu={len(eicu)} nwicu={len(nwicu)} "
          f"(total={len(mimic)+len(eicu)+len(nwicu)})")


if __name__ == "__main__":
    main()
