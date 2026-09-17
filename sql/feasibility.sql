-- ============================================================
-- NPSLE 多库研究 · 可行性核查 SQL
-- 用法: psql -h $NPSLE_DB_HOST -U $NPSLE_DB_USER -d <mimiciv|nwicu> -f feasibility.sql
--       (eICU 与 INSPIRE 见文件末尾单独段落)
-- MIMIC-IV schema: mimiciv_hosp / mimiciv_icu ; NWICU schema: hosp / icu
-- 下方以 MIMIC-IV 为例; 跑 NWICU 时把 mimiciv_hosp->hosp, mimiciv_icu->icu
-- ============================================================

-- ---------- 复用定义 ----------
-- SLE:   ICD-10 M32*  |  ICD-9 7100*
-- NPSLE: 按 ACR 1999 十九综合征映射, 分 7 个域

\set ON_ERROR_STOP on

-- [1] SLE ICU 队列规模
WITH sle AS (
  SELECT DISTINCT i.stay_id, i.hadm_id, i.subject_id
  FROM mimiciv_icu.icustays i
  JOIN mimiciv_hosp.diagnoses_icd d ON d.hadm_id = i.hadm_id
  WHERE left(d.icd_code,3) = 'M32' OR left(d.icd_code,4) = '7100'
)
SELECT 'cohort_size' AS metric,
       count(DISTINCT stay_id)    AS sle_icu_stays,
       count(DISTINCT subject_id) AS sle_patients
FROM sle;

-- [2] NPSLE 表型分布 + 广义/核心事件数
WITH sle AS (
  SELECT DISTINCT i.stay_id, i.hadm_id
  FROM mimiciv_icu.icustays i
  JOIN mimiciv_hosp.diagnoses_icd d ON d.hadm_id = i.hadm_id
  WHERE left(d.icd_code,3) = 'M32' OR left(d.icd_code,4) = '7100'
),
np AS (
  SELECT hadm_id,
    max(CASE WHEN icd_code ~ '^(G40|G41|R56)'                      OR icd_code ~ '^(345|7803)'                        THEN 1 ELSE 0 END) AS seizure,
    max(CASE WHEN icd_code ~ '^(G934|F05|R402|R410|R413)'          OR icd_code ~ '^(3483|2930|78001|78009)'           THEN 1 ELSE 0 END) AS enceph,
    max(CASE WHEN icd_code ~ '^(F060|F062|F23|F29|F28)'            OR icd_code ~ '^(29381|29382|298)'                 THEN 1 ELSE 0 END) AS psychosis,
    max(CASE WHEN icd_code ~ '^(I6[0-9]|G45|G46)'                  OR icd_code ~ '^(43[0-8])'                         THEN 1 ELSE 0 END) AS cvd,
    max(CASE WHEN icd_code ~ '^(G03|G02|A87)'                      OR icd_code ~ '^(047|321|322)'                     THEN 1 ELSE 0 END) AS meningitis,
    max(CASE WHEN icd_code ~ '^(G35|G36|G37|G04)'                  OR icd_code ~ '^(340|341|323)'                     THEN 1 ELSE 0 END) AS demyel,
    max(CASE WHEN icd_code ~ '^(G610|G700|G5[123]|H46|G6[23]|G255)' OR icd_code ~ '^(3570|3580|35[12]|3773|3335)'      THEN 1 ELSE 0 END) AS pns
  FROM mimiciv_hosp.diagnoses_icd
  GROUP BY hadm_id
)
SELECT count(*) AS sle_icu_stays,
       sum(seizure) AS seizure, sum(enceph) AS enceph, sum(psychosis) AS psychosis,
       sum(cvd) AS cerebrovascular, sum(meningitis) AS meningitis,
       sum(demyel) AS demyelinating, sum(pns) AS peripheral,
       sum(CASE WHEN seizure+enceph+psychosis+cvd+meningitis+demyel+pns > 0 THEN 1 ELSE 0 END) AS npsle_broad,
       sum(CASE WHEN seizure+enceph+psychosis+meningitis+demyel        > 0 THEN 1 ELSE 0 END) AS npsle_core
FROM sle s JOIN np ON np.hadm_id = s.hadm_id;

-- [3] 按 NPSLE 分层的院内死亡
WITH sle AS (
  SELECT DISTINCT i.stay_id, i.hadm_id
  FROM mimiciv_icu.icustays i
  JOIN mimiciv_hosp.diagnoses_icd d ON d.hadm_id = i.hadm_id
  WHERE left(d.icd_code,3) = 'M32' OR left(d.icd_code,4) = '7100'
),
np AS (
  SELECT hadm_id,
    max(CASE WHEN icd_code ~ '^(G40|G41|R56|G934|F05|R402|R410|R413|F060|F062|F23|F29|I6[0-9]|G45|G46|G03|G02|A87|G35|G36|G37|G04|G610|G700|G5[123]|H46|G6[23]|G255)'
              OR icd_code ~ '^(345|7803|3483|2930|78001|78009|29381|29382|298|43[0-8]|047|321|322|340|341|323|3570|3580|35[12]|3773|3335)'
         THEN 1 ELSE 0 END) AS npsle
  FROM mimiciv_hosp.diagnoses_icd GROUP BY hadm_id
)
SELECT np.npsle,
       count(*) AS n,
       sum(a.hospital_expire_flag) AS died,
       round(100.0*sum(a.hospital_expire_flag)/count(*), 1) AS mort_pct
FROM sle s
JOIN np ON np.hadm_id = s.hadm_id
JOIN mimiciv_hosp.admissions a ON a.hadm_id = s.hadm_id
GROUP BY 1 ORDER BY 1;

-- [4] 关键预测因子 24h 可得率 (仅 MIMIC-IV)
WITH sle AS (
  SELECT DISTINCT i.stay_id, i.hadm_id, i.intime
  FROM mimiciv_icu.icustays i
  JOIN mimiciv_hosp.diagnoses_icd d ON d.hadm_id = i.hadm_id
  WHERE left(d.icd_code,3) = 'M32' OR left(d.icd_code,4) = '7100'
)
SELECT l.itemid, di.label,
       count(DISTINCT CASE WHEN l.charttime BETWEEN s.intime - interval '6 hour'
                                                AND s.intime + interval '24 hour'
                           THEN s.stay_id END) AS n_stays_24h,
       count(DISTINCT s.stay_id) AS n_stays_anytime
FROM sle s
JOIN mimiciv_hosp.labevents l   ON l.hadm_id = s.hadm_id
JOIN mimiciv_hosp.d_labitems di ON di.itemid = l.itemid
WHERE l.itemid IN (
   50890,  -- C3
   50891,  -- C4
   50918,  -- Double Stranded DNA
   51288,  -- Sedimentation Rate
   50873,  -- ANA
   51243,  -- Lupus Anticoagulant
   51138, 51139,             -- Anticardiolipin IgG / IgM
   51222, 51265, 51301,      -- Hgb / Plt / WBC
   50912, 51006, 50862,      -- Creatinine / BUN / Albumin
   50931, 51237, 50813, 50882
)
GROUP BY 1,2 ORDER BY n_stays_24h DESC;


-- ============================================================
-- eICU-CRD  (database: eicu, schema: eicu_crd)
-- ============================================================
-- [E1] SLE 队列 + NPSLE 表型
-- WITH sle AS (SELECT DISTINCT patientunitstayid pid FROM eicu_crd.diagnosis
--              WHERE diagnosisstring ~* 'lupus'),
-- np AS (SELECT patientunitstayid pid,
--   max(CASE WHEN diagnosisstring ~* 'seizure|status epilepticus|epilep' THEN 1 ELSE 0 END) seizure,
--   max(CASE WHEN diagnosisstring ~* 'encephalopathy|delirium|coma|altered mental|obtund|unresponsive' THEN 1 ELSE 0 END) enceph,
--   max(CASE WHEN diagnosisstring ~* 'psychosis|psychotic' THEN 1 ELSE 0 END) psychosis,
--   max(CASE WHEN diagnosisstring ~* 'stroke|cerebral infarct|intracerebral hemorrhage|subarachnoid|cerebrovascular|CVA|TIA|transient ischemic' THEN 1 ELSE 0 END) cvd,
--   max(CASE WHEN diagnosisstring ~* 'meningitis|encephalitis' THEN 1 ELSE 0 END) mening,
--   max(CASE WHEN diagnosisstring ~* 'myelitis|demyelinat|multiple sclerosis' THEN 1 ELSE 0 END) demyel,
--   max(CASE WHEN diagnosisstring ~* 'guillain|myasthen|polyneuropathy|neuropathy|chorea' THEN 1 ELSE 0 END) pns
--  FROM eicu_crd.diagnosis GROUP BY 1)
-- SELECT ... FROM sle JOIN np USING(pid);
--
-- [E2] NPSLE 诊断时序 (关键: 判断"预测"是否成立)
--   min(diagnosisoffset) <= 0      -> 入科时已存在
--   0 < offset <= 1440             -> 入科 24h 内
--   offset > 1440                  -> 24h 后新发 (真正可预测的事件)
--   实测: 1 / 99 / 9  => 仅 8.3% 为新发事件

-- ============================================================
-- INSPIRE  (database: inspire, schema: inspire)
-- ============================================================
-- SELECT count(DISTINCT subject_id) FROM inspire.diagnosis
--  WHERE left(icd10_cm,3) = 'M32';        -- 结果: 0
--
-- 原因 (去标识化时结构性删除):
-- SELECT * FROM inspire.icd10_excluded WHERE causes ~* 'musculoskeletal|mental|neurolog';
--   M06,M08,M30,M31,M32,M33,M34,M35,...  | Rare musculoskeletal diseases
--   F                                    | Mental disorder (F00~F99)
--   G04,G10,...,G35,G36,G40,G41,...      | Rare neurologic diseases
-- => INSPIRE 无法用于 SLE 或 NPSLE 研究
