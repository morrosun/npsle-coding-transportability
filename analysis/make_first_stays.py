# -*- coding: utf-8 -*-
"""
生成 out/_first_stays.json —— 每个受试者（subject）的「首次 ICU 住院」stay_id 列表。
这是 v4_analyses.py / v7_identification_models.py / npsle_io.load() 的必需前置输入：
主分析队列 = 首次 ICU 住院（每例仅纳入最早一次），下游所有 first-stay 敏感性分析都依赖它。

依赖：先运行 extract_cohort.py 生成 out/cohort_<db>.csv（本脚本只在这些 SLE 队列
stay_id 范围内挑选首次住院，故产出的首次队列必为全队列的子集，与稿件口径一致）。

判定规则（已与原始提取逐集合核对完全一致）：
  * MIMIC-IV / NWICU：按 subject_id 取 intime 最早的 icustay（同 intime 取最小 stay_id）。
  * eICU-CRD：按 uniquepid（队列中的 subject_id）取 hospitaladmitoffset 最早的
    patientunitstayid（同 offset 取最小 patientunitstayid）。

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


def pick_first(db, table, subj_col, stay_col, time_expr, cohort_csv):
    """在 cohort CSV 给定的 SLE 队列 stay_id 范围内，按 subject 取最早一次住院。

    进 DB 的 stay 列统一别名为 `stay`，tie-break 也用 `stay`（同时间取最小 stay_id）。
    """
    coh = pd.read_csv(cohort_csv)
    ids = tuple(int(x) for x in coh["stay_id"].unique())
    if not ids:
        return []
    df = q(db, "SELECT {subj} AS subj, {stay} AS stay, {t} AS t "
              "FROM {tbl} WHERE {stay} IN {ids}".format(
                  subj=subj_col, stay=stay_col, t=time_expr, tbl=table, ids=ids))
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["t", "subj", "stay"])
    df = df.sort_values(["t", "stay"]).drop_duplicates(subset=["subj"], keep="first")
    return sorted(int(x) for x in df["stay"].tolist())


def main():
    mimic = pick_first("mimiciv", "mimiciv_icu.icustays",
                       "subject_id", "stay_id", "EXTRACT(EPOCH FROM intime)",
                       os.path.join(OUT, "cohort_mimiciv.csv"))
    nwicu = pick_first("nwicu", "icu.icustays",
                       "subject_id", "stay_id", "EXTRACT(EPOCH FROM intime)",
                       os.path.join(OUT, "cohort_nwicu.csv"))
    eicu = pick_first("eicu", "eicu_crd.patient",
                      "uniquepid", "patientunitstayid", "hospitaladmitoffset",
                      os.path.join(OUT, "cohort_eicu.csv"))
    out = {"mimic_first": mimic, "eicu_first": eicu, "nwicu_first": nwicu}
    path = os.path.join(OUT, "_first_stays.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"[ok] wrote {path}: "
          f"mimic={len(mimic)} eicu={len(eicu)} nwicu={len(nwicu)} "
          f"(total={len(mimic)+len(eicu)+len(nwicu)})")


if __name__ == "__main__":
    main()
