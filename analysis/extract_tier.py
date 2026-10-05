#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NPSLE 诊断置信度分层提取  (回应审稿意见 Major 2)
用法:  python extract_tier.py
产出:  out/tier_<db>.csv          —— stay_id + 分层标志
       out/t11_tier_terms.csv     —— 各库每一 Tier 命中的原始术语清单 (供附录)

分层依据 (基于三库诊断字典的实测结构, 非事后拟合)
------------------------------------------------------------------
Tier A  归因明确 : 诊断术语本身把神经精神表现归因于 SLE
        · eICU     = diagnosisstring 含 'systemic lupus erythematosus'
                     且位于 neurologic / CNS infection 路径下
        · MIMIC/NWICU = **结构性不可得**
          ICD-10 M32.1x 仅枚举 心内膜炎/心包炎/肺/肾小球/肾小管间质,
          全部 28 个 lupus 码中含神经精神字样者为 0 -> 编码为 NaN, 不是 0
Tier B  特异表型 : ACR 定义的特异性神经精神综合征, 但无归因信息
        癫痫发作 / 无菌性脑膜炎 / 精神病性障碍 / 脱髓鞘与脊髓炎
Tier C  非特异   : 仅意识或精神状态改变
        脑病(未特指) / 谵妄 / 昏迷 / 意识改变 / 定向障碍
X       明确他因 : 缺氧后/肝性/代谢性/中毒性脑病, 感染性脑膜脑炎,
        药物戒断, 镇静, 开颅术后, 颅内占位  -> 独立标志, 供敏感性分析剔除

衍生定义
    npsle_hi      = TierA or TierB                      (高置信度)
    npsle_hi_str  = (TierA or TierB) and not X          (严格, 剔除他因)
    npsle_any     = TierA or TierB or TierC             (≈ 原 npsle_core)
"""
import os
import numpy as np
import pandas as pd
import psycopg2

CONFIG = dict(host="localhost", port=5432, user="postgres", password="1314")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)


def q(db, sql):
    with psycopg2.connect(dbname=db, **CONFIG) as cn:
        return pd.read_sql(sql, cn)


# ================= ICD 库 (MIMIC-IV / NWICU) 分层正则 =================
# Tier B —— 特异性 NP 综合征
TB_10 = (r"^(G41|R56"                    # 癫痫持续状态 / 惊厥
         r"|G03|G02"                     # 非化脓性(无菌性)脑膜炎
         r"|F060|F062|F23|F29|F28"       # 精神病性障碍
         r"|G35|G36|G37|G04)")           # 脱髓鞘 / 脊髓炎 / 脑脊髓炎
TB_9 = (r"^(78039|3453|78031|78032"
        r"|321|322"
        r"|29381|29382|298"
        r"|340|341|323)")
# Tier C —— 非特异性意识 / 精神状态改变
TC_10 = r"^(G9340|G9349|F05|R4020|R410)"
TC_9 = r"^(34830|34839|2930|78001|78009)"
# X —— 明确他因
TX_10 = (r"^(G9341"        # 代谢性脑病
         r"|G931"          # 缺氧性脑损伤
         r"|G92"           # 中毒性脑病
         r"|K7290|K7291"   # 肝性脑病 / 肝衰竭伴昏迷
         r"|G00|G001|G002|G003|G008|G009"  # 化脓性脑膜炎
         r"|A87|A86|B004|G051"             # 病毒性脑膜脑炎
         r"|F1023|F1123|F1323|F1923)")     # 物质戒断性谵妄
TX_9 = (r"^(34831"         # 代谢性脑病
        r"|3481"           # 缺氧性脑损伤
        r"|34982"          # 中毒性脑病
        r"|5722"           # 肝性脑病
        r"|320"            # 细菌性脑膜炎
        r"|047|0498|0499"  # 病毒性脑膜炎
        r"|2910|29181|2920)")  # 酒精/药物戒断谵妄

ICD_TIER = {"tier_b": (TB_10, TB_9), "tier_c": (TC_10, TC_9), "other_cause": (TX_10, TX_9)}


def extract_icd(db):
    H, I = ("mimiciv_hosp", "mimiciv_icu") if db == "mimiciv" else ("hosp", "icu")
    flags = ",\n           ".join(
        f"max(CASE WHEN icd_code ~ '{a}' OR icd_code ~ '{b}' THEN 1 ELSE 0 END) AS {k}"
        for k, (a, b) in ICD_TIER.items())
    sql = f"""
    WITH sle AS (
      SELECT DISTINCT i.stay_id, i.hadm_id FROM {I}.icustays i
      JOIN {H}.diagnoses_icd d ON d.hadm_id=i.hadm_id
      WHERE left(d.icd_code,3)='M32' OR left(d.icd_code,4)='7100'
    ),
    tt AS (SELECT hadm_id, {flags} FROM {H}.diagnoses_icd GROUP BY hadm_id)
    SELECT s.stay_id, tt.tier_b, tt.tier_c, tt.other_cause
    FROM sle s JOIN tt ON tt.hadm_id=s.hadm_id
    """
    df = q(db, sql)
    # Tier A: ICD 体系无「狼疮神经受累」组合码 -> 结构性不可得
    df["tier_a"] = pd.NA
    # 命中术语清单 (供附录表)
    terms = q(db, f"""
    WITH sle AS (
      SELECT DISTINCT i.hadm_id FROM {I}.icustays i
      JOIN {H}.diagnoses_icd d ON d.hadm_id=i.hadm_id
      WHERE left(d.icd_code,3)='M32' OR left(d.icd_code,4)='7100')
    SELECT d.icd_code,
           CASE WHEN d.icd_code ~ '{TX_10}' OR d.icd_code ~ '{TX_9}' THEN 'X'
                WHEN d.icd_code ~ '{TB_10}' OR d.icd_code ~ '{TB_9}' THEN 'B'
                WHEN d.icd_code ~ '{TC_10}' OR d.icd_code ~ '{TC_9}' THEN 'C' END AS tier,
           dd.long_title AS term, count(DISTINCT d.hadm_id) n
    FROM {H}.diagnoses_icd d JOIN sle ON sle.hadm_id=d.hadm_id
    LEFT JOIN {H}.d_icd_diagnoses dd ON dd.icd_code=d.icd_code AND dd.icd_version=d.icd_version
    WHERE d.icd_code ~ '{TB_10}' OR d.icd_code ~ '{TB_9}'
       OR d.icd_code ~ '{TC_10}' OR d.icd_code ~ '{TC_9}'
       OR d.icd_code ~ '{TX_10}' OR d.icd_code ~ '{TX_9}'
    GROUP BY 1,2,3 ORDER BY tier, n DESC
    """)
    terms.insert(0, "db", db)
    return df, terms


# ========================= eICU 分层 =========================
# 逐条术语优先级: A > X > B > C
EI_A = r"systemic lupus erythematosus"
EI_X = (r"post-anoxic|hepatic|metabolic|uremic"
        r"|drug withdrawal|alcohol|narcotic"
        r"|sedated"
        r"|meningitis\|acute|bacterial|viral|herpes|West Nile"
        r"|post craniotomy|post-neurosurgery|brain tumor|meningioma|mass lesion")
EI_B = r"seizure|status epilepticus|psychosis|psychotic|myelitis|demyelinat|multiple sclerosis"
EI_C = (r"change in mental status|altered mental|encephalopath|coma\b"
        r"|delirium|stupor|obtundation|unresponsive")

# ---- PRIMARY rule set for eICU-CRD ----------------------------------------
# eICU-CRD stores diagnosisstring as a full hierarchical path
# (e.g. 'neurologic|altered mental status / pain|change in mental status').
# "altered mental status / pain" is a real dictionary folder whose DIRECT
# children denote an acute confusional state, so those are retained; the
# misclassification arises only when a child is itself subdivided
# (|depression|mild, |pain|moderate, |drug withdrawal syndrome|alcohol), where
# the folder would otherwise vouch for an unrelated diagnosis.
#
# The rules below therefore use explicit leaf patterns plus an explicit
# exclusion list, so that a reader can check the classification of every string
# against the dictionary rather than trust a component-count heuristic.
EI_B_LEAF = (r"\|seizures\b|\|status epilepticus\b"
             r"|\|psychosis\b|\|psychotic\b"
             r"|\|schizophrenia\b|\|bipolar disorder\b|\|hallucination"
             r"|\|myelitis\b|\|demyelinat|\|multiple sclerosis")
EI_C_LEAF = (r"\|change in mental status\b|\|encephalopathy\b|\|coma\b"
             r"|\|delirium\b|\|stupor\b|\|obtundation\b"
             r"|\|unresponsive\b|\|confusion\b|\|agitation\b")
# concept names accepted anywhere in the path when they are also the leaf
EI_B_ANY = (r"seizure|status epilepticus|psychosis|psychotic|myelitis"
            r"|demyelinat|multiple sclerosis")
EI_C_ANY = (r"encephalopath|delirium|coma\b|stupor|obtundation|unresponsive"
            r"|confusion|change in mental status")
# meningitis / encephalitis are accepted at any depth because the aetiology
# lives in the parent node ('encephalitis|systemic lupus erythematosus',
# 'meningitis|acute|bacterial')
EI_MENING = r"meningitis|encephalitis|meningoencephalitis"

# Leaves observed under the AMS folder that denote a different clinical entity.
# The exclusion is stated explicitly so it can be audited against the dictionary.
EI_EXCLUDE_LEAF = (r"\|depression\b|\|anxiety\b|\|pain\b"
                   r"|\|bipolar disorder\b|\|schizophrenia\b|\|dementia\b"
                   r"|\|suicidal ideation\b|\|drug withdrawal syndrome\b"
                   r"|\|sedated\b")

# Legacy whole-path patterns, retained only for the algorithm comparison.
EI_B_LEGACY = EI_B
EI_C_LEGACY = EI_C


def extract_eicu(restricted=True):
    """Extract eICU-CRD stay-level tier flags from diagnosisstring.

    restricted=True  (PRIMARY) uses the audited leaf patterns and the explicit
                     exclusion list defined at the top of this file.
    restricted=False (LEGACY)  matches every tier on the whole string, which is
                     the algorithm the primary rules replace (Table S30).

    The caller intersects every tier indicator with npsle_core, so tier
    membership can never extend outside the recorded core phenotype.
    """
    raw = q("eicu", """
    WITH sle AS (SELECT DISTINCT patientunitstayid pid FROM eicu_crd.diagnosis
                 WHERE diagnosisstring ~* 'lupus')
    SELECT DISTINCT d.patientunitstayid AS stay_id, d.diagnosisstring AS term
    FROM eicu_crd.diagnosis d JOIN sle ON sle.pid=d.patientunitstayid
    """)
    s = raw["term"].str.lower()
    parts = s.str.split("|")
    depth = parts.apply(len)
    leaf = parts.apply(lambda p: p[-1] if p else "")
    # A and X stay path-based: the attribution / aetiology word is the deepest
    # node of a four-level path, so trimming components would discard it.
    neuro_path = s.str.contains(r"neurologic\||cns infections", regex=True)

    if not restricted:
        m_c = s.str.contains(EI_C_LEGACY, regex=True, na=False)
        m_b = s.str.contains(EI_B_LEGACY, regex=True, na=False)
        excluded = pd.Series(False, index=raw.index)
    else:
        # EI_EXCLUDE_LEAF patterns carry a leading "\|" and are therefore meant
        # for the whole diagnosis string, not for the leaf; leaf is the last
        # component after splitting on "|" and has no leading bar, so testing
        # it against leaf leaves the list inert. Whole-string matching is what
        # the list states and what removes subdivided children such as
        # |depression|mild, |drug withdrawal syndrome|alcohol and
        # |sedated|unresponsive as well as the leaves |bipolar disorder and
        # |schizophrenia (the latter two are also listed in EI_B_LEAF; the
        # exclusion takes precedence, so those include entries are inert).
        excluded = s.str.contains(EI_EXCLUDE_LEAF, regex=True, na=False)
        m_c = (s.str.contains(EI_C_LEAF, regex=True, na=False)
               | (s.str.contains(EI_C_ANY, regex=True, na=False)
                  & (leaf.str.contains(EI_C_ANY, regex=True, na=False)
                     | depth.le(3))))
        m_b = (s.str.contains(EI_B_LEAF, regex=True, na=False)
               | (s.str.contains(EI_B_ANY, regex=True, na=False)
                  & leaf.str.contains(EI_B_ANY, regex=True, na=False)))
        m_c &= ~excluded
        m_b &= ~excluded

    raw["tier"] = None
    raw.loc[m_c, "tier"] = "C"
    raw.loc[m_b, "tier"] = "B"
    raw.loc[s.str.contains(EI_X, regex=True, na=False) & neuro_path, "tier"] = "X"
    raw.loc[s.str.contains(EI_A, regex=True, na=False) & neuro_path, "tier"] = "A"
    hit = raw.dropna(subset=["tier"])

    g = hit.assign(v=1).pivot_table(index="stay_id", columns="tier", values="v",
                                    aggfunc="max", fill_value=0).reset_index()
    for c in ["A", "B", "C", "X"]:
        if c not in g.columns:
            g[c] = 0
    g = g.rename(columns={"A": "tier_a", "B": "tier_b", "C": "tier_c",
                          "X": "other_cause"})
    terms = (hit.groupby(["tier", "term"])["stay_id"].nunique()
             .reset_index(name="n").sort_values(["tier", "n"], ascending=[True, False]))
    terms.insert(0, "db", "eicu")
    terms["icd_code"] = ""
    return g[["stay_id", "tier_a", "tier_b", "tier_c", "other_cause"]], terms


# ============================ 主流程 ============================
def build(m, core, has_a):
    """Apply the universe restriction and derive every tier indicator.

    core : the recorded core phenotype flag, one entry per row of m.
    """
    for c in ["tier_b", "tier_c", "other_cause"]:
        m[c] = m[c].fillna(0).astype(int)
    if has_a:
        m["tier_a"] = m["tier_a"].fillna(0).astype(int)
        a = m["tier_a"].astype(int)
    else:
        m["tier_a"] = pd.NA     # not recoverable by the prespecified algorithm
        a = pd.Series(0, index=m.index)

    # Universe restriction. Without it, tier patterns sitting outside the
    # prespecified core domains (e.g. eICU Tier A strings matching no core
    # domain) leak into npsle_hi / npsle_any, and the A|B union stops equalling
    # the sum of the mutually exclusive tiers.
    m["npsle_core"] = core
    m["npsle_hi"] = (((a | m["tier_b"]) > 0).astype(int) & core)
    m["npsle_hi_str"] = (m["npsle_hi"] & (m["other_cause"] == 0)).astype(int)
    m["npsle_any"] = (((a | m["tier_b"] | m["tier_c"]) > 0).astype(int) & core)
    # mutually exclusive stay-level semantic tier, priority A > B > C
    m["tier_primary"] = np.where(core == 0, "",
                        np.where(a > 0, "A",
                          np.where(m["tier_b"] > 0, "B",
                            np.where(m["tier_c"] > 0, "C", ""))))
    m["tier_c_only"] = ((m["tier_primary"] == "C").astype(int))
    m["tier_unassigned"] = ((core == 1) & (m["npsle_hi"] == 0)
                            & (m["tier_c_only"] == 0)).astype(int)
    return m


def report(tag, db, m):
    ta = ("not recoverable" if m["tier_a"].isna().all()
          else "%3d (%4.1f%%)" % (int(m["tier_a"].sum()), 100 * m["tier_a"].mean()))
    print("[%s] %-8s n=%4d core=%3d TierA=%s TierB=%3d TierC=%3d "
          "A+B=%3d C-only=%3d unassigned=%3d X=%3d"
          % (tag, db, len(m), int(m["npsle_core"].sum()), ta,
             int(m["tier_b"].sum()), int(m["tier_c"].sum()),
             int(m["npsle_hi"].sum()), int(m["tier_c_only"].sum()),
             int(m["tier_unassigned"].sum()), int(m["other_cause"].sum())))


def main():
    all_terms = []
    for db in ["mimiciv", "nwicu", "eicu"]:
        cohort = pd.read_csv(os.path.join(OUT, "cohort_%s.csv" % db))
        core = (cohort.set_index("stay_id")["npsle_core"]
                .reindex(cohort["stay_id"]).fillna(0).astype(int).values)

        if db == "eicu":
            # PRIMARY: leaf-restricted matching. The whole-path algorithm is
            # retained only for the reproducibility comparison (Table S30).
            tt, terms = extract_eicu(restricted=True)
            m = build(cohort[["stay_id"]].merge(tt, on="stay_id", how="left"),
                      core, has_a=True)
            m.to_csv(os.path.join(OUT, "tier_eicu.csv"), index=False)
            report("PRIMARY", db, m)

            tt_l, _ = extract_eicu(restricted=False)
            # The LEGACY arm must be intersected with ITS OWN core universe
            # (npsle_core_legacy), not the primary core, so that it reproduces
            # the legacy universe described in Table 3 / Table S29.
            core_legacy = (cohort.set_index("stay_id")["npsle_core_legacy"]
                           .reindex(cohort["stay_id"]).fillna(0).astype(int).values)
            ml = build(cohort[["stay_id"]].merge(tt_l, on="stay_id", how="left"),
                       core_legacy, has_a=True)
            ml.to_csv(os.path.join(OUT, "tier_eicu_legacy.csv"), index=False)
            report("LEGACY", db, ml)
        else:
            tt, terms = extract_icd(db)
            m = build(cohort[["stay_id"]].merge(tt, on="stay_id", how="left"),
                      core, has_a=False)
            m.to_csv(os.path.join(OUT, "tier_%s.csv" % db), index=False)
            report("PRIMARY", db, m)

        all_terms.append(terms)

    pd.concat(all_terms, ignore_index=True)[["db", "tier", "icd_code", "term", "n"]] \
        .to_csv(os.path.join(OUT, "t11_tier_terms.csv"), index=False)
    print("\n[ok] -> out/tier_*.csv, out/t11_tier_terms.csv")


if __name__ == "__main__":
    main()
