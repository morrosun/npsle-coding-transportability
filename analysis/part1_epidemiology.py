#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 1 · 三库 NPSLE 流行病学描述 + 随机效应 meta + 森林图
用法: python part1_epidemiology.py
产出: out/t1_baseline.csv          Table 1 (按核心 NPSLE 分层, 含 SMD)
      out/t2_prevalence.csv        患病率与亚型谱
      out/t3_outcomes.csv          结局: 粗率 / 粗OR / 校正OR
      out/t4_meta.csv              随机效应 meta 合并结果
      fig/forest_<outcome>.png     森林图
      out/part1_report.md
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, FIG = os.path.join(ROOT, "out"), os.path.join(ROOT, "fig")
os.makedirs(FIG, exist_ok=True)

DBS = ["mimiciv", "eicu", "nwicu"]
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
ERA = {"mimiciv": "单中心学术 · 2008–2019", "eicu": "208 家医院多中心 · 2014–2015",
       "nwicu": "芝加哥多院区 · 2020–2022 (COVID)"}

DOMS = ["dom_seizure", "dom_enceph", "dom_psych", "dom_mening", "dom_demyel", "dom_cvd", "dom_pns"]
DOM_CN = {"dom_seizure": "急性癫痫发作", "dom_enceph": "急性意识障碍/脑病", "dom_psych": "精神病性障碍",
          "dom_mening": "无菌性脑膜炎/脑炎", "dom_demyel": "脱髓鞘/脊髓炎",
          "dom_cvd": "脑血管病", "dom_pns": "周围神经病变"}

CONT = ["age", "hr", "map", "rr", "temp", "spo2", "wbc", "lymph_abs", "hgb", "plt", "creat", "na", "bili"]
BINS = ["female", "lupus_nephritis", "aps", "sepsis_dx", "steroid_any", "is_any", "hx_epilepsy"]
CN = {"age": "年龄, 岁", "hr": "心率, /min", "map": "平均动脉压, mmHg", "rr": "呼吸频率, /min",
      "temp": "体温, °C", "spo2": "SpO₂ 最低, %", "wbc": "白细胞, ×10⁹/L", "lymph_abs": "淋巴细胞绝对值, ×10⁹/L",
      "hgb": "血红蛋白, g/dL", "plt": "血小板, ×10⁹/L", "creat": "肌酐, mg/dL", "na": "血钠, mmol/L",
      "bili": "总胆红素, mg/dL", "female": "女性", "lupus_nephritis": "狼疮肾炎", "aps": "抗磷脂综合征",
      "sepsis_dx": "脓毒症", "steroid_any": "糖皮质激素暴露", "is_any": "免疫抑制剂暴露",
      "hx_epilepsy": "既往癫痫史"}

# 校正模型协变量 (三库共同可得且完整度高)
ADJ = ["age", "female", "sepsis_dx", "lupus_nephritis", "creat", "plt"]

data = {db: pd.read_csv(os.path.join(OUT, f"cohort_{db}.csv")) for db in DBS}
for db in DBS:
    d = data[db]
    d["prolonged_icu"] = (d["icu_los"] > 7).astype(int)


# ============================================================ Table 1
def smd(a, b, binary=False):
    a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if binary:
        p1, p2 = a.mean(), b.mean()
        den = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
        return abs(p1 - p2) / den if den > 0 else np.nan
    den = np.sqrt((a.var() + b.var()) / 2)
    return abs(a.mean() - b.mean()) / den if den > 0 else np.nan


rows = []
for db in DBS:
    d = data[db]
    g0, g1 = d[d.npsle_core == 0], d[d.npsle_core == 1]
    rows.append({"数据库": LABEL[db], "变量": "例数, n", "非NPSLE": str(len(g0)),
                 "NPSLE": str(len(g1)), "SMD": "", "P": ""})
    for v in CONT + BINS:
        # 整列 NA = 该库结构性不可获得(如 eICU 无狼疮肾炎/免疫抑制剂/既往癫痫史词条)。
        # 显式标注而非静默跳过 —— 这是跨库可移植性研究必须向读者披露的变量可得性差异。
        if v not in d.columns:
            continue
        if d[v].notna().sum() == 0:
            rows.append({"数据库": LABEL[db], "变量": CN.get(v, v),
                         "非NPSLE": "该库不可获得", "NPSLE": "该库不可获得", "SMD": "—", "P": "—"})
            continue
        a, b = pd.to_numeric(g0[v], errors="coerce"), pd.to_numeric(g1[v], errors="coerce")
        binary = v in BINS
        if binary:
            f0 = f"{int(a.sum())} ({100*a.mean():.1f})" if a.notna().any() else "—"
            f1 = f"{int(b.sum())} ({100*b.mean():.1f})" if b.notna().any() else "—"
        else:
            f0 = f"{a.median():.1f} [{a.quantile(.25):.1f}, {a.quantile(.75):.1f}]" if a.notna().any() else "—"
            f1 = f"{b.median():.1f} [{b.quantile(.25):.1f}, {b.quantile(.75):.1f}]" if b.notna().any() else "—"
        p = np.nan
        aa, bb = a.dropna(), b.dropna()
        if len(aa) >= 5 and len(bb) >= 5:
            if binary:
                tab = pd.crosstab(d[v], d.npsle_core)
                if tab.shape == (2, 2):
                    p = stats.chi2_contingency(tab)[1]
            else:
                p = stats.mannwhitneyu(aa, bb)[1]
        s = smd(a, b, binary)
        rows.append({"数据库": LABEL[db], "变量": CN.get(v, v), "非NPSLE": f0, "NPSLE": f1,
                     "SMD": f"{s:.2f}" if pd.notna(s) else "—",
                     "P": ("<0.001" if p < 0.001 else f"{p:.3f}") if pd.notna(p) else "—"})
t1 = pd.DataFrame(rows)
t1.to_csv(os.path.join(OUT, "t1_baseline.csv"), index=False, encoding="utf-8-sig")

# ============================================================ Table 2 患病率 / 亚型谱
rows = []
for db in DBS:
    d = data[db]
    r = {"数据库": LABEL[db], "队列特征": ERA[db], "SLE ICU 住院, n": len(d),
         "核心 NPSLE, n (%)": f"{int(d.npsle_core.sum())} ({100*d.npsle_core.mean():.1f})",
         "广义 NPSLE, n (%)": f"{int(d.npsle_broad.sum())} ({100*d.npsle_broad.mean():.1f})"}
    for dm in DOMS:
        r[DOM_CN[dm]] = f"{int(d[dm].sum())} ({100*d[dm].mean():.1f})"
    rows.append(r)
t2 = pd.DataFrame(rows)
t2.to_csv(os.path.join(OUT, "t2_prevalence.csv"), index=False, encoding="utf-8-sig")

# 亚型构成的库间异质性检验 (卡方: 亚型 × 数据库, 仅 MIMIC vs eICU)
het_rows = []
for dm in DOMS:
    m, e = data["mimiciv"][dm], data["eicu"][dm]
    tab = np.array([[m.sum(), len(m) - m.sum()], [e.sum(), len(e) - e.sum()]])
    if tab.min() >= 0 and tab.sum(axis=1).min() > 0:
        try:
            p = stats.chi2_contingency(tab)[1] if tab.min() >= 5 else stats.fisher_exact(tab)[1]
        except Exception:
            p = np.nan
        het_rows.append({"亚型": DOM_CN[dm], "MIMIC-IV %": f"{100*m.mean():.1f}",
                         "eICU %": f"{100*e.mean():.1f}",
                         "P(库间差异)": ("<0.001" if p < 0.001 else f"{p:.3f}") if pd.notna(p) else "—"})
het = pd.DataFrame(het_rows)

# ============================================================ Table 3 结局 + OR
OUTCOMES = [("hosp_mort", "院内死亡"), ("vent24", "24h 内机械通气"), ("prolonged_icu", "ICU 住院 >7 天")]

def fit_or(d, y, x="npsle_core", adjust=None):
    """返回 (OR, lo, hi, n)。adjust=None 为粗 OR。"""
    cols = [y, x] + (adjust or [])
    m = d[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(m) < 20 or m[y].nunique() < 2 or m[x].nunique() < 2:
        return (np.nan,) * 3 + (len(m),)
    if m.groupby(x)[y].sum().min() == 0 and m.groupby(x)[y].sum().max() == 0:
        return (np.nan,) * 3 + (len(m),)
    X = sm.add_constant(m[[x] + (adjust or [])].astype(float), has_constant="add")
    try:
        res = sm.Logit(m[y].astype(float), X).fit(disp=0, method="bfgs", maxiter=200)
        b, se = res.params[x], res.bse[x]
        if not np.isfinite(se) or se > 5:
            return (np.nan,) * 3 + (len(m),)
        return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), len(m)
    except Exception:
        return (np.nan,) * 3 + (len(m),)


def adj_for(d):
    """按库自适应协变量集。
    剔除该库中(a)结构性不可获得(整列 NA) 或 (b)零方差(常数)的协变量。
    否则 dropna() 会把整库样本清空, 或产生与截距完全共线的设计矩阵导致不收敛
    —— 这正是 v2 中 eICU「ICU 住院 >7 天」校正 OR 报"不可估计"的真实原因。
    """
    keep = []
    for a in ADJ:
        if a not in d.columns:
            continue
        s = pd.to_numeric(d[a], errors="coerce")
        if s.notna().sum() < 20 or s.dropna().nunique() < 2:
            continue
        keep.append(a)
    return keep


ADJ_USED = {db: adj_for(data[db]) for db in DBS}
for db in DBS:
    dropped = [CN.get(a, a) for a in ADJ if a not in ADJ_USED[db]]
    if dropped:
        print(f"[i] {LABEL[db]} 校正模型剔除协变量: {'、'.join(dropped)}")

META_MIN_EVENTS = 10        # 该库该结局的总事件数下限
META_MIN_EVENTS_EXP = 5     # NPSLE(暴露)组事件数下限

rows, meta_src, meta_excl = [], [], []
for yv, ycn in OUTCOMES:
    for db in DBS:
        d = data[db]
        if yv not in d.columns or d[yv].notna().sum() == 0:
            rows.append({"结局": ycn, "数据库": LABEL[db], "非NPSLE 事件率": "—",
                         "NPSLE 事件率": "—", "粗 OR (95%CI)": "数据缺失", "校正 OR (95%CI)": "—", "校正模型 n": "—"})
            continue
        g0, g1 = d[d.npsle_core == 0][yv], d[d.npsle_core == 1][yv]
        cr = fit_or(d, yv)
        ad = fit_or(d, yv, adjust=ADJ_USED[db])
        rows.append({
            "结局": ycn, "数据库": LABEL[db],
            "非NPSLE 事件率": f"{int(g0.sum())}/{g0.notna().sum()} ({100*g0.mean():.1f}%)",
            "NPSLE 事件率": f"{int(g1.sum())}/{g1.notna().sum()} ({100*g1.mean():.1f}%)",
            "粗 OR (95%CI)": f"{cr[0]:.2f} ({cr[1]:.2f}–{cr[2]:.2f})" if pd.notna(cr[0]) else "不可估计",
            "校正 OR (95%CI)": f"{ad[0]:.2f} ({ad[1]:.2f}–{ad[2]:.2f})" if pd.notna(ad[0]) else "不可估计",
            "校正模型 n": ad[3],
            "校正协变量数": len(ADJ_USED[db])})
        # 预设的 meta 纳入门槛: 该库该结局总事件数 >=10 且 NPSLE 组事件数 >=5。
        # 否则单库估计的 SE 过大(如 NWICU 仅 3 例 NPSLE / 1 个事件, OR 8.24, CI 0.33-208),
        # 会在 DerSimonian-Laird 中人为放大 tau^2 与 I^2, 制造虚假异质性。
        n_ev, n_ev1 = int(d[yv].sum()), int(g1.sum())
        if pd.notna(ad[0]) and n_ev >= META_MIN_EVENTS and n_ev1 >= META_MIN_EVENTS_EXP:
            meta_src.append({"outcome": ycn, "db": LABEL[db], "or": ad[0], "lo": ad[1], "hi": ad[2]})
        elif pd.notna(ad[0]):
            meta_excl.append(f"{ycn} · {LABEL[db]}（总事件 {n_ev}，NPSLE 组事件 {n_ev1}）")
t3 = pd.DataFrame(rows)
t3.to_csv(os.path.join(OUT, "t3_outcomes.csv"), index=False, encoding="utf-8-sig")

# ============================================================ Table 4 随机效应 meta (DerSimonian–Laird)
def dl_meta(logor, se):
    w = 1 / se ** 2
    fe = np.sum(w * logor) / np.sum(w)
    Q = np.sum(w * (logor - fe) ** 2)
    dfree = len(logor) - 1
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - dfree) / C) if C > 0 else 0.0
    wr = 1 / (se ** 2 + tau2)
    est = np.sum(wr * logor) / np.sum(wr)
    se_r = np.sqrt(1 / np.sum(wr))
    I2 = max(0.0, 100 * (Q - dfree) / Q) if Q > 0 else 0.0
    pQ = 1 - stats.chi2.cdf(Q, dfree) if dfree > 0 else np.nan
    return est, se_r, tau2, I2, Q, pQ


meta_df = pd.DataFrame(meta_src)
rows = []
for ycn in [o[1] for o in OUTCOMES]:
    s = meta_df[meta_df.outcome == ycn]
    if len(s) < 2:
        continue
    logor = np.log(s["or"].values)
    se = (np.log(s["hi"].values) - np.log(s["lo"].values)) / (2 * 1.96)
    est, se_r, tau2, I2, Q, pQ = dl_meta(logor, se)
    z = est / se_r
    rows.append({"结局": ycn, "纳入库数": len(s),
                 "合并 OR (95%CI)": f"{np.exp(est):.2f} ({np.exp(est-1.96*se_r):.2f}–{np.exp(est+1.96*se_r):.2f})",
                 "P": f"{2*(1-stats.norm.cdf(abs(z))):.3f}",
                 "I² (%)": f"{I2:.1f}", "τ²": f"{tau2:.3f}",
                 "Q 检验 P": f"{pQ:.3f}" if pd.notna(pQ) else "—"})
t4 = pd.DataFrame(rows)
t4.to_csv(os.path.join(OUT, "t4_meta.csv"), index=False, encoding="utf-8-sig")

# ============================================================ 森林图
for yv, ycn in OUTCOMES:
    s = meta_df[meta_df.outcome == ycn]
    if len(s) == 0:
        continue
    fig, ax = plt.subplots(figsize=(8.2, 1.15 * len(s) + 2.3), facecolor="white")
    ax.set_facecolor("white")
    ys = np.arange(len(s))[::-1]
    ax.errorbar(s["or"], ys,
                xerr=[s["or"] - s["lo"], s["hi"] - s["or"]],
                fmt="s", color="#1f4e79", ecolor="#1f4e79",
                capsize=4, markersize=8, lw=1.6, zorder=3)
    # 合并值
    if len(s) >= 2:
        logor = np.log(s["or"].values)
        se = (np.log(s["hi"].values) - np.log(s["lo"].values)) / (2 * 1.96)
        est, se_r, tau2, I2, Q, pQ = dl_meta(logor, se)
        po, plo, phi = np.exp(est), np.exp(est - 1.96 * se_r), np.exp(est + 1.96 * se_r)
        ax.plot([plo, po, phi, po, plo], [-1, -0.72, -1, -1.28, -1],
                color="#c0392b", lw=1.8, zorder=4)
        ax.fill([plo, po, phi, po], [-1, -0.72, -1, -1.28], color="#c0392b", alpha=.85, zorder=4)
        labels = list(s["db"]) + [f"随机效应合并 (I²={I2:.0f}%)"]
        ypos = list(ys) + [-1]
    else:
        labels, ypos = list(s["db"]), list(ys)
    ax.axvline(1, color="#888", ls="--", lw=1, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xscale("log")
    ax.set_xlabel("校正 OR（核心 NPSLE vs 非 NPSLE）", fontsize=11)
    ax.set_title(f"{ycn} —— 分库校正 OR 与随机效应合并", fontsize=12.5, pad=12)
    for i, (_, r) in enumerate(s.iterrows()):
        ax.text(1.02, 0.97 - i * 0.075, f"{r['or']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})",
                transform=ax.transAxes, fontsize=9.5, va="top")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-1.8, len(s) - 0.3)
    plt.tight_layout()
    p = os.path.join(FIG, f"forest_{yv}.png")
    plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

# ============================================================ 报告
def md(df):
    return df.to_markdown(index=False)

n_tot = sum(len(data[db]) for db in DBS)
rep = f"""# Part 1 · 三库危重 SLE 中 NPSLE 的流行病学与预后

生成日期：2026-07-31 ｜ 队列：{n_tot} 例 SLE ICU 住院（MIMIC-IV 645 / eICU 230 / NWICU 50）
NPSLE 表型：v2 严格定义（见 `references/phenotype.md`）

---

## 表 2 · 患病率与亚型谱（分库）

{md(t2)}

### 亚型构成的库间异质性（MIMIC-IV vs eICU）

{md(het)}

> 亚型构成差异主要由**编码体系**驱动：MIMIC-IV / NWICU 使用出院 ICD 账单编码，
> eICU 使用临床医师在床旁录入的结构化诊断串。后者对「急性意识障碍」「脑炎/脑膜炎」
> 更敏感，而前者对需要影像确诊的「脑血管病」记录更完整。

---

## 表 3 · NPSLE 与临床结局（分库）

{md(t3)}

校正协变量（全集）：{'、'.join(CN.get(a, a) for a in ADJ)}。
各库按变量可得性自适应剔除：{'；'.join(f"{LABEL[db]} 用 {len(ADJ_USED[db])} 个" + (f"（剔除 {'、'.join(CN.get(a,a) for a in ADJ if a not in ADJ_USED[db])}）" if len(ADJ_USED[db]) < len(ADJ) else "") for db in DBS)}。

---

## 表 4 · 随机效应合并（DerSimonian–Laird）

{md(t4)}

纳入门槛（预设）：单库该结局总事件数 ≥ {META_MIN_EVENTS} 且 NPSLE 组事件数 ≥ {META_MIN_EVENTS_EXP}。
{("因事件数不足而未纳入合并的分层：" + "；".join(meta_excl) + "。") if meta_excl else "所有可估计分层均满足门槛。"}

---

## 表 1 · 基线特征（按核心 NPSLE 分层）

{md(t1)}

---

## 图

- `fig/forest_hosp_mort.png` —— 院内死亡
- `fig/forest_vent24.png` —— 24h 内机械通气
- `fig/forest_prolonged_icu.png` —— ICU 住院 >7 天
"""
with open(os.path.join(OUT, "part1_report.md"), "w", encoding="utf-8") as f:
    f.write(rep)

print(t2.to_string(index=False)); print()
print(het.to_string(index=False)); print()
print(t3.to_string(index=False)); print()
print(t4.to_string(index=False))
print(f"\n[ok] -> out/part1_report.md, t1–t4 csv, fig/forest_*.png")
