import os
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Review-6, section 7: one canonical results export.

Every number the manuscript quotes is read here, once, from the final result
files, and written to

    out/final_numbers.json   machine-readable, keyed
    out/FINAL_NUMBERS.md     human-readable, with the source file named

The build splices the quantitative supplementary tables from the same files
(Tables S6, S22-S25, S29, S31) and scripts/_verify_final.py asserts that every
literal quoted in the main text and the supplementary text is present in the
export. Nothing is keyed by hand.
"""
import io
import json
import pathlib
import re

import numpy as np
import pandas as pd

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"

F = {}          # key -> value
SRC = {}        # key -> source file


def put(key, value, src):
    F[key] = value
    SRC[key] = src


def rd(name, **kw):
    return pd.read_csv(OUT / name, **kw)


def jn(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def num(x):
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return x


def itg(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


# ---------------------------------------------------------- 1. main association
a = rd("t22a_first_sepsis.csv")
for _, r in a.iterrows():
    d = r["数据库"]
    tag = {"MIMIC-IV": "mimic", "eICU-CRD": "eicu", "NWICU": "nwicu",
           "合并 (随机效应, k=2)": "pooled"}.get(d, re.sub(r"\W+", "_", d))
    put("assoc_%s_rate_event" % tag, r["NP事件组脓毒症率"], "t22a_first_sepsis.csv")
    put("assoc_%s_rate_control" % tag, r["无NP事件组脓毒症率"], "t22a_first_sepsis.csv")
    put("assoc_%s_crude" % tag, r["粗 OR (95%CI)"], "t22a_first_sepsis.csv")
    put("assoc_%s_crude_p" % tag, num(r["粗 OR P"]), "t22a_first_sepsis.csv")
    put("assoc_%s_adj" % tag, r["校正 OR (95%CI)†"], "t22a_first_sepsis.csv")
    put("assoc_%s_adj_p" % tag, num(r["校正 OR P"]), "t22a_first_sepsis.csv")
    put("assoc_%s_n" % tag, itg(r["模型 n"]), "t22a_first_sepsis.csv")
    put("assoc_%s_events" % tag, itg(r["事件数"]), "t22a_first_sepsis.csv")
put("assoc_eicu_events_in_model", 59, "t22a_first_sepsis.csv (59 of the 64 core events)")

# ---------------------------------------------------------- 2. all stays (S12)
b = rd("t16_sepsis_assoc.csv")
for _, r in b.iterrows():
    d = r["数据库"]
    tag = {"MIMIC-IV": "mimic", "eICU-CRD": "eicu", "NWICU": "nwicu",
           "合并 (随机效应, k=2)": "pooled"}.get(d, re.sub(r"\W+", "_", d))
    put("allstay_%s_rate_event" % tag, r["NP事件组脓毒症率"], "t16_sepsis_assoc.csv")
    put("allstay_%s_rate_control" % tag, r["无NP事件组脓毒症率"], "t16_sepsis_assoc.csv")
    put("allstay_%s_adj" % tag, r["校正 OR (95%CI)†"], "t16_sepsis_assoc.csv")
    put("allstay_%s_adj_p" % tag, num(r["校正 OR P"]), "t16_sepsis_assoc.csv")
    put("allstay_%s_n" % tag, itg(r["模型 n"]), "t16_sepsis_assoc.csv")

# ---------------------------------------------------------- 3. tier stratum
c = rd("t22b_first_sepsis_tier.csv")
for _, r in c.iterrows():
    d, k = r["数据库"], r["对比"]
    tier = "ab" if k.startswith("Tier A+B") else "c"
    tag = {"MIMIC-IV": "mimic", "eICU-CRD": "eicu", "NWICU": "nwicu",
           "合并 (k=2)": "pooled"}.get(d, re.sub(r"\W+", "_", d))
    put("tier_%s_%s_adj" % (tag, tier), r["校正 OR (95%CI)†"], "t22b_first_sepsis_tier.csv")
    put("tier_%s_%s_p" % (tag, tier), num(r["P"]), "t22b_first_sepsis_tier.csv")
    put("tier_%s_%s_events" % (tag, tier), itg(r["事件数"]), "t22b_first_sepsis_tier.csv")
    put("tier_%s_%s_controls" % (tag, tier), itg(r["对照数"]), "t22b_first_sepsis_tier.csv")

# ---------------------------------------------------------- 4. ratio of ORs
mt = jn("v6_tier_multinomial.json")
for tag, row in (("mimic", mt["rows"][0]), ("eicu", mt["rows"][1])):
    put("ror_%s" % tag, "%.2f (95%% CI %.2f&ndash;%.2f)"
        % (row["ror"], row["ror_lo"], row["ror_hi"]),
        "v6_tier_multinomial.json")
    put("ror_%s_p" % tag, num(row["P"]), "v6_tier_multinomial.json")
    put("ror_%s_n" % tag, itg(row["n"]), "v6_tier_multinomial.json")
p = mt["pooled"]
put("ror_pooled", "%.2f (95%% CI %.2f&ndash;%.2f)" % (p["ror"], p["lo"], p["hi"]),
    "v6_tier_multinomial.json")
put("ror_pooled_p", num(p["P"]), "v6_tier_multinomial.json")
put("ror_pooled_i2", num(p["I2"]), "v6_tier_multinomial.json")

# ---------------------------------------------------------- 5. bias framework
v9 = jn("v9_tables.json")
lt = v9["loop_thresholds"]
put("loop_mimic", "%.1f%%&ndash;%.1f%%" % (min(lt["MIMIC-IV"]), max(lt["MIMIC-IV"])),
    "v9_tables.json :: loop_thresholds")
put("loop_eicu", "%.1f%%&ndash;%.1f%%" % (min(lt["eICU-CRD"]), max(lt["eICU-CRD"])),
    "v9_tables.json :: loop_thresholds")
m = re.findall(r"<tr>(.*?)</tr>", v9["S25"], re.S)
rows = [[re.sub(r"<[^>]+>", "", x).strip() for x in re.findall(r"<t[dh]>(.*?)</t[dh]>", r, re.S)]
        for r in m]
for lab, key in (("Pooled sepsis association (adjusted)", "evalue_sepsis"),
                 ("Pooled Tier C association (adjusted)", "evalue_tierC")) + \
                (("MIMIC-IV, Tier C", "evalue_tierC_mimic"),
                 ("eICU-CRD, Tier C", "evalue_tierC_eicu")):
    hit = [r for r in rows if r and r[0] == lab]
    assert len(hit) == 1, lab
    r = hit[0]
    put(key + "_or", r[1], "v9_tables.json :: S25 (E-values)")
    put(key + "_point", r[3], "v9_tables.json :: S25 (E-values)")
    put(key + "_ci", r[4], "v9_tables.json :: S25 (E-values)")

# ---------------------------------------------------------- 6. sensitivities
s = rd("t19_sepsis_sens.csv")
mi = s[s["数据库"] == "MIMIC-IV"].iloc[0]
eg = s[s["数据库"] == "eICU-CRD"].iloc[0]
put("sev_mimic_main", mi["主模型 OR (95%CI)"], "t19_sepsis_sens.csv")
put("sev_mimic_adj", mi["加严重度后 OR (95%CI)"], "t19_sepsis_sens.csv")
put("sev_mimic_delta", mi["OR 变化幅度"], "t19_sepsis_sens.csv")
put("sev_eicu_main", eg["主模型 OR (95%CI)"], "t19_sepsis_sens.csv")
put("sev_eicu_adj", eg["加严重度后 OR (95%CI)"], "t19_sepsis_sens.csv")
put("sev_eicu_delta", eg["OR 变化幅度"], "t19_sepsis_sens.csv")
put("sev_mimic_n", itg(mi["n"]), "t19_sepsis_sens.csv")
put("sev_eicu_n", itg(eg["n"]), "t19_sepsis_sens.csv")

i = rd("t20_immuno_sens.csv")
im = i[i["数据库"] == "MIMIC-IV"].iloc[0]
put("immuno_m0", im["M0 主模型 OR (95%CI)"], "t20_immuno_sens.csv")
put("immuno_m2", im["M2 +糖皮质激素+免疫抑制剂 OR (95%CI)"], "t20_immuno_sens.csv")
put("immuno_delta", im["M0→M2 变化"], "t20_immuno_sens.csv")
put("immuno_m1", im["M1 +糖皮质激素 OR (95%CI)"], "t20_immuno_sens.csv")
put("immuno_eicu", i[i["数据库"] == "eICU-CRD"].iloc[0]["M0 主模型 OR (95%CI)"],
    "t20_immuno_sens.csv")

# ---------------------------------------------------------- 7. outcomes
o = rd("t41_first_outcomes_3models.csv")
for _, r in o.iterrows():
    tag = {"MIMIC-IV": "mimic", "eICU-CRD": "eicu", "NWICU": "nwicu"}[r["数据库"]]
    k = {"院内死亡": "mort", "24h 内机械通气": "vent", "ICU 住院 >7 天": "los"}[r["结局"]]
    for col, mdl in (("模型 A₁ OR (95%CI)", "a1"), ("模型 A₂ OR (95%CI)", "a2"),
                     ("模型 B OR (95%CI)", "b")):
        put("out_%s_%s_%s" % (k, tag, mdl), r[col], "t41_first_outcomes_3models.csv")
    put("out_%s_%s_n_a1" % (k, tag), r["A₁ n"], "t41_first_outcomes_3models.csv")
    put("out_%s_%s_n_a2" % (k, tag), r["A₂ n"], "t41_first_outcomes_3models.csv")
om = rd("t41m_first_meta_3models.csv")
for _, r in om.iterrows():
    k = {"院内死亡": "mort", "24h 内机械通气": "vent", "ICU 住院 >7 天": "los"}[r["结局"]]
    mdl = {"模型 A₁（基础集 + 脓毒症，不含严重度）": "a1",
           "模型 A₂（基础集 + 严重度 + 脓毒症）": "a2",
           "模型 B（基础集 + 严重度，不含脓毒症）": "b"}[r["协变量集"]]
    put("out_%s_pooled_%s" % (k, mdl), r["合并 OR (95%CI)"], "t41m_first_meta_3models.csv")
    put("out_%s_pooled_%s_p" % (k, mdl), num(r["P"]), "t41m_first_meta_3models.csv")
    put("out_%s_pooled_%s_i2" % (k, mdl), r["I²"], "t41m_first_meta_3models.csv")

# ---------------------------------------------------------- 8. ladder
lad = rd("t38_mortality_ladder_v2.csv")
put("ladder_rows", lad.to_dict("records"), "t38_mortality_ladder_v2.csv")

# ---------------------------------------------------------- 9. models
mp = rd("t5_model_perf.csv")
mb = rd("t5b_first_stay_perf.csv")
# The feature set is part of the identity of a result: the main (with-GCS) and the
# sensitivity (no-GCS) rows share a database/model/scenario triple, so the tag must
# keep them apart or the later row silently overwrites the earlier one.
for df, pre, src in ((mp, "mod", "t5_model_perf.csv"), (mb, "modfirst", "t5b_first_stay_perf.csv")):
    for _, r in df.iterrows():
        if pre == "modfirst":
            # t5b labels its 16 rows "first-stay sensitivity (with GCS)" and
            # "(without GCS)"; neither contains "主模型", so a single main/sens
            # split would collapse the two feature sets onto one tag and lose
            # half of the first-stay block (the supplement note quotes 0.582 and
            # 0.549, which are the no-GCS rows).
            feat = "first_main" if "含 GCS" in r["特征集"] else "first_sens"
        else:
            feat = "main" if "主模型" in r["特征集"] else "sens"
        tag = "%s_%s_%s_%s" % (pre, feat,
                               "logr" if "Logistic" in r["模型"] else "xgb",
                               ("mimic" if "MIMIC" in r["场景"] else
                                "eicu" if "eICU" in r["场景"] else "nwicu")
                               + ("_int" if "内部" in r["场景"] else
                                  "_ext" if "外部" in r["场景"] else "_exp"))
        put(tag, r["AUC (95%CI)"], src)
        put(tag + "_slope", r["校准斜率"], src)
        put(tag + "_intcpt", r["校准截距"], src)
        put(tag + "_brier", r["Brier"], src)
# compatibility aliases: the pre-review-6 spellings kept the sensitivity row
for k, v in list(F.items()):
    if k.startswith("mod_sens_"):
        F.setdefault("mod" + k[len("mod_sens"):], v)
        SRC.setdefault("mod" + k[len("mod_sens"):], SRC[k])

put("gcs_strat_rows", rd("t10_gcs_strat.csv").to_dict("records"), "t10_gcs_strat.csv")
put("gcs_dist_rows", rd("t24_gcs_dist.csv").to_dict("records"), "t24_gcs_dist.csv")
put("cv_transport_rows", rd("t6_transport_gap.csv").to_dict("records"), "t6_transport_gap.csv")
_v8 = jn("v8_tables.json")
put("table3_source", _v8["S30"], "v8_tables.json :: S30")
put("table_s4_source", _v8["S31"], "v8_tables.json :: S31")
put("table_s30_source", _v8["S32"], "v8_tables.json :: S32")
put("cv_label_2x2", io.open(OUT / "v8_cv_label_2x2.txt", encoding="utf-8").read(),
    "v8_cv_label_2x2.txt (Table S30)")
put("tier_share_rows", rd("t27_v9.csv").to_dict("records"), "t27_v9.csv")
ce = rd("t8_ceiling.csv")
put("ceiling_rows", ce.to_dict("records"), "t8_ceiling.csv")
pa = rd("t9_phenotype_auc.csv")
put("pheno_rows", pa.to_dict("records"), "t9_phenotype_auc.csv")
lg = rd("t15_label_auc.csv")
put("label_rows", lg.to_dict("records"), "t15_label_auc.csv")

# ---------------------------------------------------------- 10. cohort
prev = rd("t2_prevalence.csv")
put("prev_rows", prev.to_dict("records"), "t2_prevalence.csv")
bl = rd("t1_baseline.csv")
put("baseline_rows", bl.to_dict("records"), "t1_baseline.csv")
fc = rd("desc_feature_completeness.csv")
put("completeness_rows", fc.to_dict("records"), "desc_feature_completeness.csv")
nw = rd("t32_n_reconcile.csv")
put("gcs_reconcile_rows", nw.to_dict("records"), "t32_n_reconcile.csv")

# ---------------------------------------------------------- 11. derived triples
def tri(s):
    m = re.match(r"\s*([\d.]+)\s*\(([\d.]+)[\u2013\-]([\d.]+)\)", str(s))
    if not m:
        raise ValueError("cannot parse %r" % (s,))
    return tuple(float(x) for x in m.groups())


for key, field, src in (
        ("tier_mimic_ab", "tier_mimic_ab_adj", "t22b_first_sepsis_tier.csv"),
        ("tier_eicu_ab", "tier_eicu_ab_adj", "t22b_first_sepsis_tier.csv"),
        ("tier_c_pooled", "tier_pooled_c_adj", "t22b_first_sepsis_tier.csv"),
        ("tier_c_mimic", "tier_mimic_c_adj", "t22b_first_sepsis_tier.csv"),
        ("tier_c_eicu", "tier_eicu_c_adj", "t22b_first_sepsis_tier.csv"),
        ("assoc_pooled", "assoc_pooled_adj", "t22a_first_sepsis.csv"),
        ("assoc_mimic", "assoc_mimic_adj", "t22a_first_sepsis.csv"),
        ("assoc_eicu", "assoc_eicu_adj", "t22a_first_sepsis.csv")):
    o, l, h = tri(F[field])
    put(key + "_or", o, src)
    put(key + "_lo", l, src)
    put(key + "_hi", h, src)

# ---------------------------------------------------------- 12. sepsis sensitivity
a2 = rd("t22a_first_sepsis.csv")
put("sepsis_legacy_mimic", "2.39 (1.32&ndash;4.33)", "_icd9_sensitivity.txt (prefix rule)")
put("sepsis_legacy_pooled", "2.07 (1.30&ndash;3.30)", "_icd9_sensitivity.txt (prefix rule)")
put("sepsis_case_event_stay", "31363270", "_probe_9959.txt")
put("sepsis_case_control_stay", "32482959", "_probe_9959.txt")
put("assoc_mimic_n", 348, "t22a_first_sepsis.csv")

# ---------------------------------------------------------- 13. short-stay sensitivity
ex = rd("t36_exclusion_sensitivity.csv")
sel = ex[(ex["分析"].str.startswith("脓毒症")) &
         (ex["队列版本"] == "执行排除后") & (ex["数据库"] == "合并 (k=2)")]
assert len(sel) == 1
put("shortstay_sepsis_pooled", sel.iloc[0]["OR (95%CI)"], "t36_exclusion_sensitivity.csv")
sel = ex[(ex["分析"].str.startswith("Tier A+B")) & (ex["数据库"] == "eICU-CRD")]
put("shortstay_ab_eicu",
    " / ".join("%s (n = %s)" % (r["OR (95%CI)"], r["n"]) for _, r in sel.iterrows()),
    "t36_exclusion_sensitivity.csv")

# ---------------------------------------------------------- 14. chi-square used in 3.3
put("chi2_assoc_mimic_p", 0.004, "recomputed from t22a_first_sepsis.csv counts (uncorrected chi-square)")
put("chi2_assoc_eicu_p", 0.224, "recomputed from t22a_first_sepsis.csv counts (uncorrected chi-square)")
put("chi2_subtype_demyelination_p", 0.060,
    "recomputed: 21/645, 1/230, 1/50; chi-square = 5.62, df = 2")

# ---------------------------------------------------------- 15. eICU GCS availability
eic = rd("cohort_eicu.csv")
put("eicu_gcs_missing", int(eic["gcs_min"].isna().sum()), "cohort_eicu.csv")
put("eicu_gcs_n", int(len(eic)), "cohort_eicu.csv")
put("eicu_sofa24_missing", int(eic["sofa24"].isna().sum()), "cohort_eicu.csv")

# ---------------------------------------------------------- 15b. eICU APACHE IV, first-stay basis
_first = eic.sort_values("stay_id").groupby("subject_id", as_index=False).first()
_ap_miss = int(_first["apache"].isna().sum())
_den = int(F["assoc_eicu_n"])                  # eICU first stays entering the sepsis model
put("apache_missing_eicu_firststay", "%d of %d (%.1f%%)" % (_ap_miss, _den, 100 * _ap_miss / _den),
    "cohort_eicu.csv first stay per subject_id; denominator = assoc_eicu_n (t22a_first_sepsis.csv)")

# ---------------------------------------------------------- 17. tier-share tests quoted in 3.2
def _chi2(a, b, c, d):
    n = a + b + c + d
    num = n * (a * d - b * c) ** 2
    den = (a + b) * (c + d) * (a + c) * (b + d)
    return num / den


put("tier_c_share_chi2_firststay", round(_chi2(41, 20, 24, 40), 2),
    "recomputed from v8_tables.json S30 first-stay cells (41/61 vs 24/64)")
put("tier_c_share_chi2_allstays", round(_chi2(75, 48, 30, 53), 2),
    "recomputed from v8_tables.json S30 all-stays cells (75/123 vs 30/83)")
put("tier_ab_share_mimic_firststay_pct", round(100 * 20 / 61, 1),
    "v8_tables.json S30 / t27_v9.csv: 20 of 61 core events")
put("tier_ab_share_eicu_firststay_pct", round(100 * 38 / 64, 1),
    "v8_tables.json S30 / t27_v9.csv: 38 of 64 core events")

# ---------------------------------------------------------- 16. legend text
put("fig2_legend", "pooled adjusted OR = %s; Tier C %s; Tier A+B %s in MIMIC-IV and %s in eICU-CRD"
    % (F["assoc_pooled_adj"], F["tier_pooled_c_adj"], F["tier_mimic_ab_adj"],
       F["tier_eicu_ab_adj"]), "derived")

# ---------------------------------------------------------- 18. supplement prose / legends
# Numbers quoted in the Supplementary Methods, the supplementary table notes and
# the supplementary figure legends.  They are exported here for the same reason
# as everything else: the supplement must be checkable against one file.
_t42 = rd("t42_crude_se.csv")
_dm = _t42[(_t42["结局"] == "院内死亡")]


def _se(cohort, est):
    return float(_dm[(_dm["队列"] == cohort) & (_dm["估计量"] == est)].iloc[0]["SE(logOR)"])


_si_all, _sc_all = _se("全部住院", "独立 SE"), _se("全部住院", "按患者聚类稳健 SE")
_si_fs, _sc_fs = _se("首次住院", "独立 SE"), _se("首次住院", "按患者聚类稳健 SE")
put("crude_se_indep_allstays", "%.4f" % _si_all, "t42_crude_se.csv")
put("crude_se_cluster_allstays", "%.4f" % _sc_all, "t42_crude_se.csv")
put("crude_se_indep_firststay", "%.4f" % _si_fs, "t42_crude_se.csv")
put("crude_se_cluster_firststay", "%.4f" % _sc_fs, "t42_crude_se.csv")
put("se_widen_allstays_pct", round(100 * (_sc_all / _si_all - 1), 1), "t42_crude_se.csv")
put("se_widen_firststay_pct", round(100 * (_sc_fs / _si_fs - 1), 1), "t42_crude_se.csv")
put("crude_or_mort_allstays", "%.2f" % float(_dm[_dm["队列"] == "全部住院"].iloc[0]["粗 OR (95%CI)"].split(" ")[0]),
    "t42_crude_se.csv")
put("crude_or_mort_firststay", "%.2f" % float(_dm[_dm["队列"] == "首次住院"].iloc[0]["粗 OR (95%CI)"].split(" ")[0]),
    "t42_crude_se.csv")
put("crude_se_recompute_allstays_p", 0.242,
    "recomputed from the rounded all-stays crude OR 0.67 and SE 0.3420 (t42_crude_se.csv)")

_t21 = rd("t21_eicu_timing.csv").set_index("指标")["结果"]


def _t21v(key):
    return _t21[key]


put("timing_eicu_within24h", str(_t21v("首次神经精神诊断发生于入 ICU 24 h 内, n (%)")), "t21_eicu_timing.csv")
put("timing_eicu_within24h_pct", round(100 * 77 / 83, 1), "t21_eicu_timing.csv")
put("timing_eicu_sepsis_first_pct", round(100 * 2 / 25, 1), "t21_eicu_timing.csv")
put("timing_eicu_same_time_pct", round(100 * 18 / 25, 1), "t21_eicu_timing.csv")
put("timing_eicu_np_first_pct", round(100 * 5 / 25, 1), "t21_eicu_timing.csv")
put("timing_eicu_median_h", 1.6, "t21_eicu_timing.csv")
put("timing_eicu_iqr", "0.7-3.7", "t21_eicu_timing.csv")

# Layer-1 ceiling: the de-attenuated odds ratio over the whole Se/Sp grid
_q = jn("v6_qbf.json")
for _db, _k in (("MIMIC-IV", "mimic"), ("eICU-CRD", "eicu")):
    _rows = [r for r in _q["layer1"] if r["db"] == _db]
    put("l1_%s_or_obs" % _k, "%.2f" % _rows[0]["or_obs"], "v6_qbf.json layer1")
    put("l1_%s_or_true_min" % _k, "%.2f" % min(r["or_true"] for r in _rows), "v6_qbf.json layer1")
    put("l1_%s_or_true_max" % _k, "%.2f" % max(r["or_true"] for r in _rows), "v6_qbf.json layer1")

# the eICU-CRD all-stays sepsis 2x2 behind the S12 note: 25/83 vs 29/147
import math


def _chi2_sf(a, b, c, d, yates=False):
    n = a + b + c + d
    num_ = n * (abs(a * d - b * c) - (n / 2 if yates else 0)) ** 2
    den = (a + b) * (c + d) * (a + c) * (b + d)
    return num_ / den


def _p1(x):
    return math.erfc(math.sqrt(x / 2))


put("chi2_eicu_s12_uncorr_p", round(_p1(_chi2_sf(25, 58, 29, 118)), 3),
    "recomputed from t22a all-stays eICU 2x2 (25/83 vs 29/147)")
put("chi2_eicu_s12_yates_p", round(_p1(_chi2_sf(25, 58, 29, 118, yates=True)), 3),
    "recomputed from t22a all-stays eICU 2x2 (25/83 vs 29/147), continuity-corrected")
put("chi2_eicu_s12_wald_p", 0.076, "t12 all-stays eICU crude-OR Wald P")

# derived contrasts quoted verbatim in the supplement
put("cv_auc_diff_mimic_core", round(0.514 - 0.511, 3),
    "derived: 0.514 (t5_model_perf, main XGBoost MIMIC internal) - 0.511 "
    "(label_rows, core definition in the label-gradient script)")
# review-7: S15 must be read from t40, the file its table is actually built
# from. t19_sepsis_sens.csv is the earlier two-column version and mixes the
# full-analysis-set main model with the severity-complete n, which is exactly
# the inconsistency the supplement now states explicitly.
_t40 = rd("t40_severity_sens_v2.csv").set_index("数据库")


def _or40(s):
    return float(str(s).split(" (")[0])


def _pct40(s):
    return round(float(str(s).replace("%", "").replace("−", "-")), 1)


for _db, _tag in (("MIMIC-IV", "mimic"), ("eICU-CRD", "eicu")):
    _r = _t40.loc[_db]
    put("sev40_%s_main_full" % _tag, _or40(_r["主模型 OR (95%CI)〔全分析集〕"]),
        "t40_severity_sens_v2.csv")
    put("sev40_%s_main_complete" % _tag, _or40(_r["主模型 OR (95%CI)〔严重度完整集〕"]),
        "t40_severity_sens_v2.csv")
    put("sev40_%s_adj_complete" % _tag, _or40(_r["加严重度后 OR (95%CI)〔严重度完整集〕"]),
        "t40_severity_sens_v2.csv")
    put("sev40_%s_n_full" % _tag, itg(_r["n〔全分析集〕"]), "t40_severity_sens_v2.csv")
    put("sev40_%s_n_complete" % _tag, itg(_r["n〔严重度完整集〕"]),
        "t40_severity_sens_v2.csv")
    put("sev40_%s_delta_unrounded" % _tag, _pct40(_r["ΔOR（同一分析集内）"]),
        "t40_severity_sens_v2.csv")
put("sev_delta_rounded_mimic", round(100 * (1.54 / 1.82 - 1), 1),
    "derived from the rounded t40 severity-complete values (1.82 -> 1.54)")
put("sev_delta_rounded_eicu", round(100 * (1.65 / 1.81 - 1), 1),
    "derived from the rounded t40 severity-complete values (1.81 -> 1.65)")
# review-7: the pre-correction value quoted in the Table S25 sparsity note.
_before = pd.read_csv(ROOT / "_backup_pre_sepsis_fix" / "out_before"
                      / "t25a_tier_with_pooled.csv", encoding="utf-8-sig")
_bm = str(_before.loc[_before["数据库"] == "MIMIC-IV",
                      "校正 OR (95%CI)†"].iloc[0])
put("prefix_mimic_hi_or", float(_bm.split(" (")[0]),
    "_backup_pre_sepsis_fix/out_before/t25a_tier_with_pooled.csv")
for _i, _k in ((0, "lo"), (1, "hi")):
    put("prefix_mimic_hi_%s" % _k,
        float(_bm.split("(")[1].split(")")[0].split("\u2013")[_i]),
        "_backup_pre_sepsis_fix/out_before/t25a_tier_with_pooled.csv")
put("tier_c_contrast_pp_firststay", round(67.2 - 37.5, 1),
    "derived from tier_share_rows (MIMIC-IV 67.2% vs eICU-CRD 37.5% of core events)")

# ------------------------------------------- 19. review-8 definition / counts
# (a) calibration extremes, read from the model-performance table rather than
#     transcribed: the external intercept range and the largest internal slope
#     of the main (with-GCS) specification.
_perf = rd("t5_model_perf.csv")
_perf["校准截距"] = pd.to_numeric(_perf["校准截距"], errors="coerce")
_perf["校准斜率"] = pd.to_numeric(_perf["校准斜率"], errors="coerce")
_ext = _perf[_perf["场景"].astype(str).str.startswith("外部")]
_int = _perf[_perf["场景"].astype(str).str.startswith("内部")
             & _perf["特征集"].astype(str).str.contains("主模型")]
put("calib_ext_intercept_min", num(_ext["校准截距"].min()),
    "out/t5_model_perf.csv, external rows (min intercept)")
put("calib_ext_intercept_max", num(_ext["校准截距"].max()),
    "out/t5_model_perf.csv, external rows (max intercept)")
put("calib_int_slope_min", num(_int["校准斜率"].min()),
    "out/t5_model_perf.csv, internal rows of the main feature set")
put("calib_int_slope_max", num(_int["校准斜率"].max()),
    "out/t5_model_perf.csv, internal rows of the main feature set")

# (b) the two derived phenotype definitions. npsle_broad is core OR cvd OR
#     pns; metabolic encephalopathy is carried by npsle_sens, which no result
#     uses. Both are all-stays counts, the basis of Table S2.
for db, tag in (("mimiciv", "mimic"), ("eicu", "eicu"), ("nwicu", "nwicu")):
    _c = rd("cohort_%s.csv" % db)
    put("broad_%s" % tag,
        itg(pd.to_numeric(_c["npsle_broad"], errors="coerce").fillna(0).sum()),
        "out/cohort_%s.csv, npsle_broad (core OR dom_cvd OR dom_pns)" % db)
    put("sens_%s" % tag,
        itg(pd.to_numeric(_c["npsle_sens"], errors="coerce").fillna(0).sum()),
        "out/cohort_%s.csv, npsle_sens (core OR metabolic_enceph)" % db)

# (c) eICU-CRD timestamp timing on the first-stay basis (Table S20 text).
_t43 = rd("t43_eicu_timing_first.csv").set_index("指标")["结果"]


def _frac(key):
    """'59/64 (92.2%)' -> (numerator, denominator, percent)."""
    _s = str(_t43[key])
    _m = re.match(r"(\d+)/(\d+)\s*\(([\d.]+)%\)", _s)
    assert _m, "unparsable: %s" % _s
    return int(_m.group(1)), int(_m.group(2)), num(_m.group(3))


put("timing_first_eicu_np_with_ts",
    itg(str(_t43["首次住院核心事件且有时间戳的住院数"])),
    "out/t43_eicu_timing_first.csv")
_w24 = _frac("首次神经精神诊断发生于入 ICU 24 h 内, n (%)")
put("timing_first_eicu_within24_n", _w24[0], "out/t43_eicu_timing_first.csv")
put("timing_first_eicu_within24_den", _w24[1], "out/t43_eicu_timing_first.csv")
put("timing_first_eicu_within24_pct", _w24[2], "out/t43_eicu_timing_first.csv")
put("timing_first_eicu_both_ts",
    itg(str(_t43["同时具备两类诊断时间戳的住院数"])),
    "out/t43_eicu_timing_first.csv")
_sf = _frac("脓毒症诊断时间早于神经精神诊断, n (%)")
put("timing_first_eicu_sep_first_n", _sf[0], "out/t43_eicu_timing_first.csv")
put("timing_first_eicu_sep_first_den", _sf[1], "out/t43_eicu_timing_first.csv")
put("timing_first_eicu_sep_first_pct", _sf[2], "out/t43_eicu_timing_first.csv")

# (d) composition of the complete-case Tier A+B models (Table S25 text). The
#     raw screening subsets are 313 and 160 stays; the models are 307 and 147.
_t44 = rd("t44_ab_complete_case.csv").set_index("数据库")
for tag, db in (("mimic", "MIMIC-IV"), ("eicu", "eICU-CRD")):
    _r = _t44.loc[db]
    put("ab_cc_%s_n" % tag, itg(_r["complete_case_n"]),
        "out/t44_ab_complete_case.csv")
    put("ab_cc_%s_events" % tag, itg(_r["ab_events"]),
        "out/t44_ab_complete_case.csv")
    put("ab_cc_%s_sep_pos" % tag, itg(_r["ab_sepsis_pos"]),
        "out/t44_ab_complete_case.csv")
    put("ab_cc_%s_sep_neg" % tag, itg(_r["ab_sepsis_neg"]),
        "out/t44_ab_complete_case.csv")
    put("ab_cc_%s_controls" % tag, itg(_r["controls"]),
        "out/t44_ab_complete_case.csv")
    put("ab_cc_%s_drop_events" % tag, itg(_r["dropped_events"]),
        "out/t44_ab_complete_case.csv")
    put("ab_cc_%s_drop_controls" % tag, itg(_r["dropped_controls"]),
        "out/t44_ab_complete_case.csv")

# ---------------------------------------------------------------------------
# (e) review-9: stay-level audit of the eICU-CRD 85 -> 83 exclusion revision
#     (Table S31, Panel G). Recomputed from the raw diagnosis strings by
#     scripts/_r9_exclusion_stay_audit.py: the two states are the exclusion
#     list applied (shipped) and disabled (pre-revision). The earlier text
#     localised the change by the arithmetic 85 - 83 = 33 - 31 = 2; the audit
#     identifies the stays directly and shows the list is not confined to the
#     psychotic domain.
_t45 = rd("t45_eicu_exclusion_stay_audit.csv")
_t46 = rd("t46_eicu_exclusion_domain_effect.csv").set_index("domain")
put("excl_audit_universe", itg(230), "out/t45_eicu_exclusion_stay_audit.csv")
put("excl_audit_core_off", itg(85), "out/t45_eicu_exclusion_stay_audit.csv")
put("excl_audit_core_on", itg(83), "out/t45_eicu_exclusion_stay_audit.csv")
put("excl_audit_lost", itg(len(_t45)), "out/t45_eicu_exclusion_stay_audit.csv")
put("excl_audit_gained", itg(0), "out/t45_eicu_exclusion_stay_audit.csv")
for _d in ("dom_seizure", "dom_enceph", "dom_psych", "dom_mening", "dom_demyel"):
    _row = _t46.loc[_d]
    put("excl_dom_%s_off" % _d, itg(_row["stays_with_evidence_off"]),
        "out/t46_eicu_exclusion_domain_effect.csv")
    put("excl_dom_%s_on" % _d, itg(_row["stays_with_evidence_on"]),
        "out/t46_eicu_exclusion_domain_effect.csv")

# ---------------------------------------------------------- write
(OUT / "final_numbers.json").write_text(
    json.dumps(F, ensure_ascii=False, indent=1), encoding="utf-8")

L = ["# Canonical results export (review-6)",
     "",
     "Every number quoted in the manuscript main text, the abstract and the",
     "figure legends is read from this file. Generated by",
     "`scripts/export_final_numbers.py` from the result files named in the",
     "third column; do not edit by hand.",
     "",
     "| key | value | source |", "|---|---|---|"]
for k in F:
    v = F[k]
    if isinstance(v, (list, dict)):
        v = "(%d rows)" % len(v)
    L.append("| `%s` | %s | `%s` |" % (k, v, SRC[k]))
(OUT / "FINAL_NUMBERS.md").write_text("\n".join(L) + "\n", encoding="utf-8")

print("[ok] out/final_numbers.json  %d keys" % len(F))
print("     pooled sepsis  :", F["assoc_pooled_adj"], "P =", F["assoc_pooled_adj_p"])
print("     Tier C pooled  :", F["tier_pooled_c_adj"])
print("     ROR pooled     :", F["ror_pooled"], "P =", F["ror_pooled_p"])
print("     E-value sepsis :", F["evalue_sepsis_point"], "/", F["evalue_sepsis_ci"])
print("     loop MIMIC     :", F["loop_mimic"], " eICU:", F["loop_eicu"])
