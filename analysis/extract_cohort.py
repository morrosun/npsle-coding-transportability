#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NPSLE 多库研究 · 统一队列提取
用法:  python extract_cohort.py [mimiciv] [eicu] [nwicu]
产出:  out/cohort_<db>.csv   —— 三库列名完全对齐

设计说明
--------
* SLE 识别:  MIMIC/NWICU = ICD-10 M32* / ICD-9 7100* ; eICU = diagnosisstring ~* 'lupus'
* NPSLE 表型: 按 ACR 1999 十九综合征映射为 7 个域
    core  = seizure + enceph + psychosis + meningitis + demyel   (主分析)
    broad = core + cerebrovascular + peripheral                  (敏感性分析)
* 时间窗:  ICU 入科 -6h ~ +24h (eICU 为 offset -360 ~ +1440 分钟)
* 缺失: 一律留 NaN, 不在提取层插补
"""
import sys, os, re
import pandas as pd
import psycopg2

# ----------------------------- CONFIG -----------------------------
CONFIG = dict(
    host=os.environ.get("NPSLE_DB_HOST", "localhost"),
    port=int(os.environ.get("NPSLE_DB_PORT", "5432")),
    user=os.environ.get("NPSLE_DB_USER", "postgres"),
    password=os.environ.get("NPSLE_DB_PASSWORD", ""),
)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

# -------------------------------------------------------------------
# NPSLE 表型定义  (v2 —— 修正了 v1 的三个误分类, 见 references/phenotype.md)
#
#  v1 的坑:
#   (a) `^R402` 会命中整个 R40.2xxx「昏迷量表」编码族, 其中包含
#       R40.2252「最佳言语反应=定向正常」等 **GCS 正常值** 编码 -> 假阳性
#   (b) G40*/345.x 是 **慢性癫痫**(既往史), 不是急性 NPSLE 事件
#   (c) G93.41 / 348.31 **代谢性脑病** 按定义归因于代谢因素, 非狼疮
#
#  v2 处理:
#   - 急性发作(dom_seizure) 只保留 G41(癫痫持续状态)/R56/780.39/345.3
#   - 慢性癫痫另存为协变量 hx_epilepsy
#   - R40.2 只保留 R4020(昏迷,未特指)
#   - 代谢性脑病另存为 metabolic_enceph, 不计入核心 NPSLE
# -------------------------------------------------------------------
NP_ICD = {
    "dom_seizure":   (r"^(G41|R56)",                            r"^(78039|3453|78031|78032)"),
    "dom_enceph":    (r"^(G9340|G9349|F05|R4020|R410)",         r"^(34830|34839|2930|78001|78009)"),
    "dom_psych":     (r"^(F060|F062|F23|F29|F28)",              r"^(29381|29382|298)"),
    "dom_cvd":       (r"^(I6[0-9]|G45|G46)",                    r"^(43[0-8])"),
    "dom_mening":    (r"^(G03|G02|A87)",                        r"^(047|321|322)"),
    "dom_demyel":    (r"^(G35|G36|G37|G04)",                    r"^(340|341|323)"),
    "dom_pns":       (r"^(G610|G700|G5[123]|H46|G6[23]|G255)",  r"^(3570|3580|35[12]|3773|3335)"),
}
# 不计入 NPSLE 的伴随标志 (作协变量 / 敏感性分析)
AUX_ICD = {
    "hx_epilepsy":      (r"^(G40)",          r"^(345)"),
    "metabolic_enceph": (r"^(G9341)",        r"^(34831)"),
}
# eICU 用 diagnosisstring 关键词
NP_EICU = {
    "dom_seizure": r"seizure|status epilepticus",
    "dom_enceph":  r"encephalopathy|delirium|coma|altered mental|obtund|unresponsive",
    "dom_psych":   r"psychosis|psychotic",
    "dom_cvd":     r"stroke|cerebral infarct|intracerebral hemorrhage|subarachnoid|cerebrovascular|CVA|transient ischemic",
    "dom_mening":  r"meningitis|encephalitis",
    "dom_demyel":  r"myelitis|demyelinat|multiple sclerosis",
    "dom_pns":     r"guillain|myasthen|polyneuropathy|neuropathy|chorea",
}
AUX_EICU = {
    "hx_epilepsy":      r"epilepsy",
    "metabolic_enceph": r"metabolic encephalopathy|hepatic encephalopathy|uremic encephalopathy",
}
CORE_DOMS  = ["dom_seizure", "dom_enceph", "dom_psych", "dom_mening", "dom_demyel"]
ALL_DOMS   = list(NP_ICD.keys())
AUX_COLS   = list(AUX_ICD.keys())

# 糖皮质激素 / 免疫抑制剂
GC_RE   = r"prednis|methylpred|dexameth|hydrocort|solumedrol"
IS_RE   = r"methotrex|azathiopr|mycophen|cyclophosph|cyclospor|tacrolim|rituxim|belimum"


def q(db, sql):
    with psycopg2.connect(dbname=db, **CONFIG) as cn:
        return pd.read_sql(sql, cn)


def np_case(col, key):
    """生成 MIMIC/NWICU 的域标志 SQL 片段"""
    i10, i9 = NP_ICD.get(key) or AUX_ICD[key]
    return (f"max(CASE WHEN {col} ~ '{i10}' OR {col} ~ '{i9}' THEN 1 ELSE 0 END) AS {key}")


# ===================================================================
#  MIMIC-IV  /  NWICU   (同构)
# ===================================================================
def extract_mimic_like(db):
    if db == "mimiciv":
        H, I, D = "mimiciv_hosp", "mimiciv_icu", "mimiciv_derived"
        has_derived = True
    else:                      # nwicu
        H, I, D = "hosp", "icu", None
        has_derived = False

    doms = ",\n           ".join(np_case("icd_code", k) for k in ALL_DOMS + AUX_COLS)

    # ---- 主表: 队列 + 人口学 + 结局 + NPSLE 域 ----
    sql = f"""
    WITH sle AS (
      SELECT DISTINCT i.stay_id, i.hadm_id, i.subject_id, i.intime, i.outtime,
             EXTRACT(EPOCH FROM (i.outtime - i.intime))/86400.0 AS icu_los
      FROM {I}.icustays i
      JOIN {H}.diagnoses_icd d ON d.hadm_id = i.hadm_id
      WHERE left(d.icd_code,3)='M32' OR left(d.icd_code,4)='7100'
    ),
    np AS (
      SELECT hadm_id,
           {doms},
           max(CASE WHEN icd_code ~ '^(M3214|M3215)' OR icd_code ~ '^(58381|58081)' THEN 1 ELSE 0 END) AS lupus_nephritis,
           max(CASE WHEN icd_code ~ '^(D6861|D686)'  OR icd_code ~ '^(28981)'        THEN 1 ELSE 0 END) AS aps,
           max(CASE WHEN icd_code ~ '^(A4[01]|R652)' OR icd_code ~ '^(038|9959)'     THEN 1 ELSE 0 END) AS sepsis_dx
      FROM {H}.diagnoses_icd GROUP BY hadm_id
    )
    SELECT s.stay_id, s.subject_id, s.hadm_id, s.icu_los,
           p.gender, a.race, a.hospital_expire_flag AS hosp_mort,
           EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/86400.0 AS hosp_los,
           DATE_PART('year', s.intime) - p.anchor_year + p.anchor_age AS age,
           np.*
    FROM sle s
    JOIN np ON np.hadm_id = s.hadm_id
    JOIN {H}.patients   p ON p.subject_id = s.subject_id
    JOIN {H}.admissions a ON a.hadm_id    = s.hadm_id
    """
    df = q(db, sql).drop(columns=["hadm_id"], errors="ignore")
    df = df.loc[:, ~df.columns.duplicated()]

    # ---- 严重度 / GCS / 生命体征 (仅 MIMIC 有 derived) ----
    if has_derived:
        sofa = q(db, f"SELECT stay_id, sofa AS sofa24, cns AS sofa_cns FROM {D}.first_day_sofa")
        gcs  = q(db, f"SELECT stay_id, gcs_min, gcs_motor, gcs_verbal, gcs_eyes FROM {D}.first_day_gcs")
        vit  = q(db, f"""SELECT stay_id, heart_rate_mean AS hr, mbp_mean AS map,
                                resp_rate_mean AS rr, temperature_mean AS temp, spo2_min AS spo2
                         FROM {D}.first_day_vitalsign""")
        lab  = q(db, f"""SELECT stay_id, wbc_max AS wbc, abs_lymphocytes_min AS lymph_abs,
                                hemoglobin_min AS hgb, platelets_min AS plt, creatinine_max AS creat,
                                albumin_min AS alb, sodium_min AS na, bilirubin_total_max AS bili,
                                inr_max AS inr FROM {D}.first_day_lab""")
        # 乳酸不在 first_day_lab, 而在 first_day_bg (血气衍生表)
        bg   = q(db, f"SELECT stay_id, lactate_max AS lactate FROM {D}.first_day_bg")
        vent = q(db, f"""SELECT DISTINCT v.stay_id, 1 AS vent24 FROM {D}.ventilation v
                         JOIN {I}.icustays i ON i.stay_id = v.stay_id
                         WHERE v.ventilation_status IN ('InvasiveVent','Tracheostomy')
                           AND v.starttime < i.intime + interval '24 hour'""")
        for t in (sofa, gcs, vit, lab, bg, vent):
            df = df.merge(t, on="stay_id", how="left")
        df["vent24"] = df["vent24"].fillna(0).astype(int)
    else:
        # NWICU: 无 derived, 无 GCS(chartevents 里根本没有该项)
        for c in ["sofa24", "sofa_cns", "gcs_min", "gcs_motor", "gcs_verbal", "gcs_eyes"]:
            df[c] = pd.NA
        # NWICU 数据字典陷阱(已实测核对 icu.d_items):
        #   - 无 MAP 项目, 只有 320179 SBP / 320180 DBP -> 必须按 (SBP+2*DBP)/3 推算,
        #     直接 avg(SBP,DBP) 会得到 (SBP+DBP)/2, 系统性高估约 10 mmHg.
        #   - 323761 TEMPERATURE 的 unitname = 'F' (华氏度) -> 必须换算为 ℃.
        vit = q(db, f"""
            SELECT c.stay_id,
                   avg(CASE WHEN c.itemid=320045 THEN c.valuenum END) AS hr,
                   ( avg(CASE WHEN c.itemid=320179 THEN c.valuenum END)
                     + 2 * avg(CASE WHEN c.itemid=320180 THEN c.valuenum END) ) / 3.0 AS map,
                   avg(CASE WHEN c.itemid=320210 THEN c.valuenum END) AS rr,
                   (avg(CASE WHEN c.itemid=323761 THEN c.valuenum END) - 32) * 5.0/9.0 AS temp,
                   min(CASE WHEN c.itemid=320277 THEN c.valuenum END) AS spo2
            FROM {I}.chartevents c JOIN {I}.icustays i ON i.stay_id=c.stay_id
            WHERE c.charttime BETWEEN i.intime - interval '6 hour' AND i.intime + interval '24 hour'
            GROUP BY c.stay_id""")
        # 100016 = White Blood Cells (100013 是 Total CO2, v2 曾误用 -> WBC 中位数假性 25.0)
        lab = q(db, f"""
            SELECT i.stay_id,
                   max(CASE WHEN l.itemid=100016 THEN l.valuenum END) AS wbc,
                   min(CASE WHEN l.itemid=100037 THEN l.valuenum END) AS lymph_abs,
                   min(CASE WHEN l.itemid=100007 THEN l.valuenum END) AS hgb,
                   min(CASE WHEN l.itemid=100014 THEN l.valuenum END) AS plt,
                   max(CASE WHEN l.itemid=100002 THEN l.valuenum END) AS creat,
                   min(CASE WHEN l.itemid=100021 THEN l.valuenum END) AS alb,
                   min(CASE WHEN l.itemid=100010 THEN l.valuenum END) AS na,
                   max(CASE WHEN l.itemid=100020 THEN l.valuenum END) AS bili,
                   max(CASE WHEN l.itemid=100030 THEN l.valuenum END) AS inr,
                   max(CASE WHEN l.itemid=100031 THEN l.valuenum END) AS lactate
            FROM {I}.icustays i JOIN {H}.labevents l ON l.hadm_id=i.hadm_id
            WHERE l.charttime BETWEEN i.intime - interval '6 hour' AND i.intime + interval '24 hour'
            GROUP BY i.stay_id""")
        df = df.merge(vit, on="stay_id", how="left").merge(lab, on="stay_id", how="left")
        df["vent24"] = pd.NA

    # ---- 用药 ----
    med = q(db, f"""
        SELECT i.stay_id,
               max(CASE WHEN lower(pr.drug) ~ '{GC_RE}' THEN 1 ELSE 0 END) AS steroid_any,
               max(CASE WHEN lower(pr.drug) ~ '{IS_RE}' THEN 1 ELSE 0 END) AS is_any
        FROM {I}.icustays i JOIN {H}.prescriptions pr ON pr.hadm_id = i.hadm_id
        GROUP BY i.stay_id""")
    df = df.merge(med, on="stay_id", how="left")
    df[["steroid_any", "is_any"]] = df[["steroid_any", "is_any"]].fillna(0).astype(int)

    df["female"] = (df["gender"].str.upper() == "F").astype(int)
    df["db"] = db
    return df


# ===================================================================
#  eICU-CRD
# ===================================================================
def extract_eicu():
    doms = ",\n           ".join(
        f"max(CASE WHEN diagnosisstring ~* '{v}' THEN 1 ELSE 0 END) AS {k}"
        for k, v in list(NP_EICU.items()) + list(AUX_EICU.items()))

    base = q("eicu", f"""
    WITH sle AS (SELECT DISTINCT patientunitstayid AS stay_id
                 FROM eicu_crd.diagnosis WHERE diagnosisstring ~* 'lupus'),
    np AS (SELECT patientunitstayid AS stay_id,
           {doms},
           max(CASE WHEN diagnosisstring ~* 'lupus nephritis|glomerulonephritis' THEN 1 ELSE 0 END) AS lupus_nephritis,
           max(CASE WHEN diagnosisstring ~* 'antiphospholipid|anticardiolipin' THEN 1 ELSE 0 END) AS aps,
           max(CASE WHEN diagnosisstring ~* 'sepsis|septic shock' THEN 1 ELSE 0 END) AS sepsis_dx
           FROM eicu_crd.diagnosis GROUP BY 1)
    SELECT s.stay_id, p.uniquepid AS subject_id, p.gender, p.ethnicity AS race,
           CASE WHEN p.age = '> 89' THEN 90
                WHEN p.age ~ '^[0-9]+$' THEN p.age::int END AS age,
           CASE WHEN p.hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END AS hosp_mort,
           p.unitdischargeoffset/1440.0     AS icu_los,
           p.hospitaldischargeoffset/1440.0 AS hosp_los,
           np.*
    FROM sle s JOIN np ON np.stay_id = s.stay_id
    JOIN eicu_crd.patient p ON p.patientunitstayid = s.stay_id
    """)
    base = base.loc[:, ~base.columns.duplicated()]

    # 关键: eICU 的 vitalperiodic / lab / medication 都是亿行级大表,
    #       必须先把队列 stay_id 内联进 WHERE, 否则全表聚合会耗尽内存.
    ids = ",".join(str(int(x)) for x in base["stay_id"].unique())

    aps = q("eicu", f"""
        SELECT patientunitstayid AS stay_id,
               max(NULLIF(eyes,-1) + NULLIF(motor,-1) + NULLIF(verbal,-1)) AS gcs_min,
               max(NULLIF(motor,-1))  AS gcs_motor,
               max(NULLIF(verbal,-1)) AS gcs_verbal,
               max(NULLIF(eyes,-1))   AS gcs_eyes,
               max(GREATEST(COALESCE(vent,0), COALESCE(intubated,0))) AS vent24
        FROM eicu_crd.apacheapsvar
        WHERE patientunitstayid IN ({ids}) GROUP BY 1""")
    apr = q("eicu", f"""SELECT patientunitstayid AS stay_id, max(apachescore) AS apache
                        FROM eicu_crd.apachepatientresult
                        WHERE apachescore > 0 AND patientunitstayid IN ({ids}) GROUP BY 1""")
    # vitalperiodic 的 systemicmean/temperature 仅在有动脉置管/持续测温时才有(<20%),
    # 必须再合并 vitalaperiodic 的无创血压 与 nursecharting 的体温, 否则完整度被严重低估.
    vit = q("eicu", f"""
        SELECT patientunitstayid AS stay_id,
               avg(heartrate) AS hr, avg(systemicmean) AS map_inv, avg(respiration) AS rr,
               avg(temperature) AS temp_inv, min(sao2) AS spo2
        FROM eicu_crd.vitalperiodic
        WHERE patientunitstayid IN ({ids})
          AND observationoffset BETWEEN -360 AND 1440 GROUP BY 1""")
    vit_ni = q("eicu", f"""
        SELECT patientunitstayid AS stay_id, avg(noninvasivemean) AS map_ni
        FROM eicu_crd.vitalaperiodic
        WHERE patientunitstayid IN ({ids})
          AND observationoffset BETWEEN -360 AND 1440 GROUP BY 1""")
    temp_nc = q("eicu", f"""
        SELECT patientunitstayid AS stay_id,
               avg(nursingchartvalue::numeric) AS temp_nc
        FROM eicu_crd.nursecharting
        WHERE patientunitstayid IN ({ids})
          AND nursingchartcelltypevallabel = 'Temperature'
          AND nursingchartcelltypevalname  = 'Temperature (C)'
          AND nursingchartvalue ~ '^[0-9]+(\\.[0-9]+)?$'
          AND nursingchartoffset BETWEEN -360 AND 1440 GROUP BY 1""")
    vit = vit.merge(vit_ni, on="stay_id", how="outer").merge(temp_nc, on="stay_id", how="outer")
    vit["map"]  = vit["map_inv"].combine_first(vit["map_ni"])
    vit["temp"] = vit["temp_inv"].combine_first(vit["temp_nc"])
    vit = vit[["stay_id", "hr", "map", "rr", "temp", "spo2"]]
    lab = q("eicu", f"""
        SELECT patientunitstayid AS stay_id,
               max(CASE WHEN labname='WBC x 1000'        THEN labresult END) AS wbc,
               min(CASE WHEN labname='-lymphs'           THEN labresult END) AS lymph_pct,
               min(CASE WHEN labname='Hgb'               THEN labresult END) AS hgb,
               min(CASE WHEN labname='platelets x 1000'  THEN labresult END) AS plt,
               max(CASE WHEN labname='creatinine'        THEN labresult END) AS creat,
               min(CASE WHEN labname='albumin'           THEN labresult END) AS alb,
               min(CASE WHEN labname='sodium'            THEN labresult END) AS na,
               max(CASE WHEN labname='total bilirubin'   THEN labresult END) AS bili,
               max(CASE WHEN labname='PT - INR'          THEN labresult END) AS inr,
               max(CASE WHEN labname='lactate'           THEN labresult END) AS lactate
        FROM eicu_crd.lab
        WHERE patientunitstayid IN ({ids})
          AND labresultoffset BETWEEN -360 AND 1440 GROUP BY 1""")
    med = q("eicu", f"""
        SELECT patientunitstayid AS stay_id,
               max(CASE WHEN lower(drugname) ~ '{GC_RE}' THEN 1 ELSE 0 END) AS steroid_any,
               max(CASE WHEN lower(drugname) ~ '{IS_RE}' THEN 1 ELSE 0 END) AS is_any
        FROM eicu_crd.medication
        WHERE patientunitstayid IN ({ids}) GROUP BY 1""")

    for t in (aps, apr, vit, lab, med):
        base = base.merge(t, on="stay_id", how="left")

    base["wbc"] = base["wbc"]                       # 已是 x1000 (与 MIMIC 同单位)
    base["lymph_abs"] = base["lymph_pct"] * base["wbc"] / 100.0
    base["steroid_any"] = base["steroid_any"].fillna(0).astype(int)
    base["vent24"] = base["vent24"].fillna(0).astype(int)

    # ---- eICU 结构性不可获得变量: 必须置 NA, 不可置 0 ----------------
    # 实测核查结论(见 references/phenotype.md):
    #  1) lupus_nephritis: eICU diagnosisstring 字典中不存在"lupus nephritis"层级,
    #     肾脏诊断最细只到 renal|disorder of kidney|{ESRD, AKI, CKD ...}, 无病因学标注.
    #  2) is_any: 全库唯一被记录的免疫抑制剂是 tacrolimus(3060 条), SLE 队列命中 0 例;
    #     eICU medication 表不采集院外长期口服免疫抑制剂 -> 0 是"未采集"而非"未使用".
    #  3) hx_epilepsy: eICU 只有急性 status epilepticus, 无"既往癫痫史"概念.
    # 若错误置 0, 在 Part 2 的 MIMIC->eICU 外部验证中会被模型当成真实的零暴露信号,
    # 系统性压低预测概率并伪造校准漂移.
    for c in ["lupus_nephritis", "is_any", "hx_epilepsy"]:
        base[c] = pd.NA
    base["female"] = (base["gender"].str.upper().str.startswith("F")).astype(int)
    base["sofa24"] = pd.NA          # eICU 无 SOFA, 用 apache 代理
    base["sofa_cns"] = pd.NA
    base["db"] = "eicu"
    return base


# ===================================================================
def finalize(df):
    for c in ALL_DOMS + AUX_COLS:
        # 整列 NA = 该库结构性不可获得(如 eICU 的 hx_epilepsy), 保持 NA 不要填 0
        if c in df.columns and not df[c].isna().all():
            df[c] = df[c].fillna(0).astype(int)
    df["npsle_core"]  = (df[CORE_DOMS].sum(axis=1) > 0).astype(int)
    df["npsle_broad"] = (df[ALL_DOMS].sum(axis=1) > 0).astype(int)
    # 敏感性: 把代谢性脑病也算作 NPSLE(文献中"急性意识错乱状态"的宽松版)
    df["npsle_sens"] = ((df[CORE_DOMS].sum(axis=1) + df["metabolic_enceph"]) > 0).astype(int)
    cols = (["db", "stay_id", "subject_id", "age", "female", "race",
             "npsle_core", "npsle_broad", "npsle_sens"] + ALL_DOMS + AUX_COLS +
            ["lupus_nephritis", "aps", "sepsis_dx",
             "sofa24", "sofa_cns", "apache", "gcs_min", "gcs_motor", "gcs_verbal", "gcs_eyes",
             "hr", "map", "rr", "temp", "spo2",
             "wbc", "lymph_abs", "hgb", "plt", "creat", "alb", "na", "bili", "inr", "lactate",
             "vent24", "steroid_any", "is_any",
             "hosp_mort", "icu_los", "hosp_los"])
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols]


if __name__ == "__main__":
    targets = sys.argv[1:] or ["mimiciv", "eicu", "nwicu"]
    for db in targets:
        print(f"[*] extracting {db} ...", flush=True)
        d = extract_eicu() if db == "eicu" else extract_mimic_like(db)
        d = finalize(d)
        p = os.path.join(OUT, f"cohort_{db}.csv")
        d.to_csv(p, index=False)
        print(f"    -> {p}   n={len(d)}  NPSLE_core={int(d.npsle_core.sum())}  "
              f"NPSLE_broad={int(d.npsle_broad.sum())}  deaths={int(d.hosp_mort.sum())}", flush=True)
