#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NPSLE 多库研究 · 描述性汇总 (供研究方向决策)
用法: python describe_cohort.py
产出: out/desc_feature_completeness.csv
      out/desc_baseline_by_npsle.csv
      out/desc_subtype_outcome.csv
      out/desc_summary.md
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

DBS = ["mimiciv", "eicu", "nwicu"]
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
DOMS = ["dom_seizure", "dom_enceph", "dom_psych", "dom_cvd", "dom_mening", "dom_demyel", "dom_pns"]
DOM_CN = {"dom_seizure": "癫痫发作", "dom_enceph": "急性意识障碍/脑病", "dom_psych": "精神病性障碍",
          "dom_cvd": "脑血管病", "dom_mening": "无菌性脑膜炎/脑炎", "dom_demyel": "脱髓鞘/脊髓炎",
          "dom_pns": "周围神经病变"}

FEATURES = ["age", "female", "sofa24", "apache", "gcs_min", "hr", "map", "rr", "temp", "spo2",
            "wbc", "lymph_abs", "hgb", "plt", "creat", "alb", "na", "bili", "inr", "lactate",
            "vent24", "steroid_any", "is_any", "lupus_nephritis", "aps", "sepsis_dx"]
FEAT_CN = {"age": "年龄", "female": "女性", "sofa24": "SOFA(24h)", "apache": "APACHE IV 分",
           "gcs_min": "GCS 最低值", "hr": "心率", "map": "平均动脉压", "rr": "呼吸频率",
           "temp": "体温", "spo2": "SpO2 最低", "wbc": "白细胞", "lymph_abs": "淋巴细胞绝对值",
           "hgb": "血红蛋白", "plt": "血小板", "creat": "肌酐", "alb": "白蛋白", "na": "血钠",
           "bili": "总胆红素", "inr": "INR", "lactate": "乳酸", "vent24": "24h 内机械通气",
           "steroid_any": "糖皮质激素", "is_any": "免疫抑制剂", "lupus_nephritis": "狼疮肾炎",
           "aps": "抗磷脂综合征", "sepsis_dx": "脓毒症诊断"}

data = {db: pd.read_csv(os.path.join(OUT, f"cohort_{db}.csv")) for db in DBS}

# ---------------------------------------------------------------- 1 特征完整度
rows = []
for f in FEATURES:
    r = {"变量": FEAT_CN.get(f, f), "字段": f}
    for db in DBS:
        d = data[db]
        r[LABEL[db]] = round(100.0 * d[f].notna().mean(), 1) if f in d.columns else 0.0
    r["三库最小完整度"] = min(r[LABEL[db]] for db in DBS)
    r["可跨三库建模"] = "是" if r["三库最小完整度"] >= 50 else "否"
    rows.append(r)
comp = pd.DataFrame(rows).sort_values("三库最小完整度", ascending=False)
comp.to_csv(os.path.join(OUT, "desc_feature_completeness.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 2 基线 (按 NPSLE core 分层)
def fmt_num(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return "—"
    return f"{s.median():.1f} ({s.quantile(.25):.1f}–{s.quantile(.75):.1f})"

def fmt_bin(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return "—"
    return f"{int(s.sum())} ({100*s.mean():.1f}%)"

BINARY = {"female", "vent24", "steroid_any", "is_any", "lupus_nephritis", "aps", "sepsis_dx"}
rows = []
for db in DBS:
    d = data[db]
    for f in FEATURES:
        if f not in d.columns:
            continue
        g0, g1 = d[d.npsle_core == 0][f], d[d.npsle_core == 1][f]
        fmt = fmt_bin if f in BINARY else fmt_num
        # 检验
        p = np.nan
        a = pd.to_numeric(g0, errors="coerce").dropna()
        b = pd.to_numeric(g1, errors="coerce").dropna()
        if len(a) >= 5 and len(b) >= 5:
            if f in BINARY:
                tab = pd.crosstab(d[f], d.npsle_core)
                if tab.shape == (2, 2):
                    p = stats.chi2_contingency(tab)[1]
            else:
                p = stats.mannwhitneyu(a, b, alternative="two-sided")[1]
        rows.append({"数据库": LABEL[db], "变量": FEAT_CN.get(f, f),
                     "非NPSLE": fmt(g0), "NPSLE": fmt(g1),
                     "P": ("<0.001" if p < 0.001 else f"{p:.3f}") if pd.notna(p) else "—"})
base = pd.DataFrame(rows)
base.to_csv(os.path.join(OUT, "desc_baseline_by_npsle.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 3 亚型 × 结局
rows = []
for db in DBS:
    d = data[db]
    for dom in DOMS:
        sub = d[d[dom] == 1]
        if len(sub) == 0:
            rows.append({"数据库": LABEL[db], "NPSLE 亚型": DOM_CN[dom], "n": 0,
                         "占SLE队列%": "0.0", "院内死亡": "—", "机械通气": "—", "ICU LOS 中位": "—"})
            continue
        rows.append({
            "数据库": LABEL[db], "NPSLE 亚型": DOM_CN[dom], "n": len(sub),
            "占SLE队列%": f"{100*len(sub)/len(d):.1f}",
            "院内死亡": f"{int(sub.hosp_mort.sum())} ({100*sub.hosp_mort.mean():.1f}%)",
            "机械通气": (f"{int(sub.vent24.sum())} ({100*sub.vent24.mean():.1f}%)"
                         if sub.vent24.notna().any() else "—"),
            "ICU LOS 中位": f"{sub.icu_los.median():.1f}",
        })
sub_out = pd.DataFrame(rows)
sub_out.to_csv(os.path.join(OUT, "desc_subtype_outcome.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 4 总体结局表
rows = []
for db in DBS:
    d = data[db]
    for lab_, m in [("全队列", d), ("非NPSLE", d[d.npsle_core == 0]), ("NPSLE(核心)", d[d.npsle_core == 1])]:
        rows.append({
            "数据库": LABEL[db], "分组": lab_, "n": len(m),
            "年龄中位": f"{m.age.median():.0f}" if m.age.notna().any() else "—",
            "女性%": f"{100*m.female.mean():.1f}",
            "院内死亡": f"{int(m.hosp_mort.sum())} ({100*m.hosp_mort.mean():.1f}%)",
            "机械通气%": (f"{100*m.vent24.mean():.1f}" if m.vent24.notna().any() else "—"),
            "ICU LOS 中位": f"{m.icu_los.median():.1f}",
            "GCS 中位": (f"{m.gcs_min.median():.0f}" if m.gcs_min.notna().any() else "—"),
        })
overall = pd.DataFrame(rows)

# ---------------------------------------------------------------- 5 Markdown 报告
def md(df):
    return df.to_markdown(index=False)

n_tot = sum(len(data[db]) for db in DBS)
core_tot = sum(int(data[db].npsle_core.sum()) for db in DBS)
broad_tot = sum(int(data[db].npsle_broad.sum()) for db in DBS)
death_tot = sum(int(data[db].hosp_mort.sum()) for db in DBS)
death_np = sum(int(data[db][data[db].npsle_core == 1].hosp_mort.sum()) for db in DBS)

txt = f"""# NPSLE 多库研究 · 描述性汇总（决策依据）

生成日期：2026-07-31 ｜ 数据源：本地 PostgreSQL（mimiciv / eicu / nwicu）
NPSLE 定义：**核心**（癫痫、急性意识障碍/脑病、精神病性、无菌性脑膜炎/脑炎、脱髓鞘/脊髓炎）为主分析；
**广义**（核心 + 脑血管病 + 周围神经病变）留作敏感性分析。

三库合计 **{n_tot}** 例 SLE ICU 住院；核心 NPSLE **{core_tot}** 例，广义 NPSLE **{broad_tot}** 例；
院内死亡合计 **{death_tot}** 例，其中核心 NPSLE 组 **{death_np}** 例。

---

## 一、总体结局

{md(overall)}

---

## 二、候选预测因子的三库完整度（%）

**这张表决定了跨库模型能纳入哪些变量。** 只有「三库最小完整度 ≥50%」的变量才可能进入需要三库共同验证的模型。

{md(comp[["变量", "MIMIC-IV", "eICU-CRD", "NWICU", "三库最小完整度", "可跨三库建模"]])}

---

## 三、NPSLE 亚型 × 结局

{md(sub_out)}

---

## 四、基线特征（按核心 NPSLE 分层，中位数 [IQR] 或 n(%)）

{md(base)}

---

## 五、表型定义的修正（v1 → v2，重要）

初版（v1）表型定义存在三处系统性误分类，已修正：

| 问题 | 具体表现 | v2 处理 |
|---|---|---|
| `^R402` 命中整个「昏迷量表」编码族 | R40.2252「最佳言语反应=定向**正常**」、R40.2362「最佳运动反应=**遵嘱**」被判为 NPSLE | 只保留 `R4020`（昏迷，未特指） |
| 慢性癫痫被当作急性事件 | `34590`/`G40909`「癫痫，未特指」是既往史，占 v1 癫痫域 56/127 | 移出 NPSLE，改存为协变量 `hx_epilepsy` |
| 代谢性脑病被计入 | `G93.41`/`348.31` 按定义归因于代谢因素而非狼疮 | 移出核心定义，存为 `metabolic_enceph`，仅进敏感性分析 `npsle_sens` |

修正后 MIMIC-IV 核心 NPSLE 由 **222 降至 123 例**（假阳性率约 45%）。
v1 中「NPSLE 组 GCS 中位数 = 15」这一反常现象即由上述第一条造成。

---

## 六、读数要点

1. **GCS 是 NPSLE 研究最关键的变量，但 NWICU 完全没有。** NWICU 的 `icu.chartevents`
   实际只填充了 13 个项目（脉搏、血氧、呼吸、血压、体温、身高体重），无任何 GCS / 意识评估条目。
2. **SOFA 只有 MIMIC-IV 有**（derived 表）；eICU 只能用 APACHE IV 替代；NWICU 两者皆无。
   严重度校正无法在三库间统一。
3. **NWICU 0 例院内死亡**，任何以死亡为结局的分析都无法纳入该库。
4. **亚型构成存在极大的跨库异质性**，源于编码体系不同（MIMIC/NWICU 为出院 ICD 账单码，
   eICU 为临床医师录入的结构化诊断串）：
   - 急性意识障碍：eICU 32.6% vs MIMIC 13.6%
   - 无菌性脑膜炎/脑炎：eICU 11.3% vs MIMIC 0.5%
   - 脑血管病：MIMIC 18.8% vs eICU 7.0%
   这意味着**简单合并三库会产生编码体系驱动的伪差异，必须分库报告 + 随机效应合并**。
5. 三库共同可得且完整度高的变量集：**人口学、生命体征（HR/MAP/RR/体温/SpO2）、
   血常规、肌酐、血钠、狼疮肾炎、抗磷脂综合征、脓毒症、激素/免疫抑制剂暴露**。
   —— 不含 GCS、SOFA、机械通气（NWICU 缺）。
"""
with open(os.path.join(OUT, "desc_summary.md"), "w", encoding="utf-8") as f:
    f.write(txt)

print(overall.to_string(index=False))
print()
print(comp[["变量", "MIMIC-IV", "eICU-CRD", "NWICU", "可跨三库建模"]].to_string(index=False))
print()
print(sub_out.to_string(index=False))
print(f"\n[ok] -> {OUT}/desc_summary.md 及 3 个 csv")
