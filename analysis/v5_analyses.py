# -*- coding: utf-8 -*-
"""
V5 修订所需补充分析（回应第四轮审稿意见）
==================================================================
问题 2  Tier A+B 合并估计缺失 -> T25a 重算并补合并行（放宽暴露门槛并注明）
问题 4  结局模型协变量集静默更改 -> T25b 双版本（含/不含 sepsis_dx）+ 归因阶梯 T25c
问题 5  队列切换只完成一半 -> T26 首次住院版基线 / T27 首次住院版患病率与 Tier 构成
次要    表 18 标注错误 -> T28 全部住院 + 真·聚类稳健（协变量集与主模型对齐）
次要    表 7 的 30 例来自全部住院 -> T29 首次住院子集时序
次要    GCS n 漂移 -> 在 T24 已用"可分析例数"口径，此处输出口径对照

用法: python scripts/v5_analyses.py
"""
import os, warnings, pathlib, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npsle_io

warnings.filterwarnings("ignore")
os.environ["LC_ALL"] = "C"; os.environ["LANG"] = "C"; os.environ["PGCLIENTENCODING"] = "UTF8"

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)
FIG = ROOT / "fig"; FIG.mkdir(exist_ok=True)

for _f in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

DBS = ["mimiciv", "eicu", "nwicu"]
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
MIN_TOTAL_EV, MIN_EXP_EV = 10, 5
MIN_EXP_EV_TIER = 3          # Tier 分层：暴露组事件数门槛放宽至 3（见方法学说明）
ADJ_BASE = ["age", "female", "lupus_nephritis", "creat", "plt"]
SEV = {"mimiciv": "sofa24", "eicu": "apache", "nwicu": "sofa24"}


def load(db):
    c = npsle_io.load(db)
    d = c
    d["prolonged_icu"] = (pd.to_numeric(d["icu_los"], errors="coerce") > 7).astype(float)
    return d


data = {db: load(db) for db in DBS}
FIRST = json.loads((OUT / "_first_stays.json").read_text(encoding="utf-8"))
FKEY = {"mimiciv": "mimic_first", "eicu": "eicu_first", "nwicu": "nwicu_first"}
first_ids = {db: set(pd.Series(FIRST[FKEY[db]]).astype(str)) for db in DBS}
data_first = {db: data[db][data[db]["stay_id"].astype(str).isin(first_ids[db])].copy() for db in DBS}


def fit_or(d, y, x, adjust=None, cluster=True):
    adjust = [a for a in (adjust or [])
              if a in d.columns
              and pd.to_numeric(d[a], errors="coerce").notna().sum() > 0
              and pd.to_numeric(d[a], errors="coerce").nunique(dropna=True) > 1]
    m = d[[y, x] + adjust].apply(pd.to_numeric, errors="coerce")
    if cluster and "subject_id" in d.columns:
        m = m.assign(_grp=pd.factorize(d["subject_id"].astype(str))[0])
    m = m.dropna()
    nev = int(m[y].sum()) if len(m) else 0
    if len(m) < 30 or m[y].nunique() < 2 or m[x].nunique() < 2:
        return (np.nan,) * 4 + (len(m), nev, len(adjust))
    X = sm.add_constant(m[[x] + adjust], has_constant="add")
    try:
        mod = sm.Logit(m[y].astype(float), X)
        res = (mod.fit(disp=0, method="bfgs", maxiter=400,
                       cov_type="cluster", cov_kwds={"groups": m["_grp"]})
               if cluster and "_grp" in m.columns else
               mod.fit(disp=0, method="bfgs", maxiter=400))
        b, se = res.params[x], res.bse[x]
        if not np.isfinite(se) or se > 5 or abs(b) > 8:
            return (np.nan,) * 4 + (len(m), nev, len(adjust))
        tab = pd.crosstab(m[x], m[y])
        if tab.shape != (2, 2) or (tab.values == 0).any():
            return (np.nan,) * 4 + (len(m), nev, len(adjust))
        return (float(np.exp(b)), float(np.exp(b - 1.96 * se)),
                float(np.exp(b + 1.96 * se)), float(res.pvalues[x]), len(m), nev, len(adjust))
    except Exception:
        return (np.nan,) * 4 + (len(m), nev, len(adjust))


def dl_meta(logor, se):
    logor, se = np.asarray(logor, float), np.asarray(se, float)
    w = 1 / se ** 2
    fe = (w * logor).sum() / w.sum()
    Q = (w * (logor - fe) ** 2).sum()
    k = len(logor)
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    wr = 1 / (se ** 2 + tau2)
    mu = (wr * logor).sum() / wr.sum()
    semu = np.sqrt(1 / wr.sum())
    I2 = max(0.0, 100 * (Q - (k - 1)) / Q) if Q > 0 else 0.0
    z = mu / semu
    return dict(OR=np.exp(mu), lo=np.exp(mu - 1.96 * semu), hi=np.exp(mu + 1.96 * semu),
                P=2 * (1 - stats.norm.cdf(abs(z))), I2=I2, tau2=tau2, Q=Q,
                Pq=1 - stats.chi2.cdf(Q, k - 1) if k > 1 else np.nan, k=k)


def se_ci(lo, hi):
    return (np.log(hi) - np.log(lo)) / (2 * 1.96)


def fmt(r):
    return f"{r[0]:.2f} ({r[1]:.2f}–{r[2]:.2f})" if pd.notna(r[0]) else "不可估计"


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))


def iqr(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{s.median():.1f} [{s.quantile(.25):.1f}, {s.quantile(.75):.1f}]" if len(s) else "—"


def npct(s):
    s = pd.to_numeric(s, errors="coerce")
    return f"{int(s.sum())} ({100*s.mean():.1f})" if s.notna().sum() else "不可获得"


# ==================================================== T25a Tier 分层（补 A+B 合并行）
print("=" * 78)
print("T25a  脓毒症共病按诊断置信度分层（首次住院队列）—— 补 Tier A+B 合并估计")
print("=" * 78)

rows, mc, mh = [], [], []
tier_detail = {}
for db in DBS:
    df = data_first[db]
    no_ev = pd.to_numeric(df["npsle_core"], errors="coerce").fillna(0) == 0
    only_c = (pd.to_numeric(df["tier_c"], errors="coerce").fillna(0) == 1) & \
             (pd.to_numeric(df["npsle_hi"], errors="coerce").fillna(0) == 0)
    hi = pd.to_numeric(df["npsle_hi"], errors="coerce").fillna(0) == 1
    sub_c = df[only_c | no_ev].copy(); sub_c["_y"] = only_c[only_c | no_ev].astype(int)
    sub_h = df[hi | no_ev].copy();     sub_h["_y"] = hi[hi | no_ev].astype(int)
    for tag, sub in [("Tier C 非特异事件", sub_c), ("Tier A+B 高置信事件", sub_h)]:
        r = fit_or(sub, "_y", "sepsis_dx", adjust=ADJ_BASE, cluster=False)
        nexp = int(pd.to_numeric(sub.loc[sub["_y"] == 1, "sepsis_dx"], errors="coerce").sum())
        rows.append({"数据库": LABEL[db], "对比": f"{tag} vs 无记录事件",
                     "事件数": int(sub["_y"].sum()), "其中脓毒症 n": nexp,
                     "对照数": int((sub["_y"] == 0).sum()),
                     "校正 OR (95%CI)†": fmt(r),
                     "P": f"{r[3]:.3f}" if pd.notna(r[3]) else "—"})
        tier_detail[(LABEL[db], tag)] = dict(nexp=nexp, nev=int(sub["_y"].sum()))
        if pd.notna(r[0]) and int(sub["_y"].sum()) >= MIN_TOTAL_EV and nexp >= MIN_EXP_EV_TIER:
            (mc if tag.startswith("Tier C") else mh).append(
                dict(db=LABEL[db], logor=np.log(r[0]), se=se_ci(r[1], r[2]),
                     OR=r[0], lo=r[1], hi=r[2]))

t25a = pd.DataFrame(rows)
tier_meta = {}
for tag, src in [("Tier C 非特异事件", mc), ("Tier A+B 高置信事件", mh)]:
    if len(src) >= 2:
        m = dl_meta([s["logor"] for s in src], [s["se"] for s in src])
        tier_meta[tag] = m
        t25a = pd.concat([t25a, pd.DataFrame([{
            "数据库": f"合并 (随机效应, k={m['k']})", "对比": f"{tag} vs 无记录事件",
            "事件数": "—", "其中脓毒症 n": "—", "对照数": f"I²={m['I2']:.1f}%",
            "校正 OR (95%CI)†": f"{m['OR']:.2f} ({m['lo']:.2f}–{m['hi']:.2f})",
            "P": f"{m['P']:.3f}"}])], ignore_index=True)
t25a = t25a.sort_values(["对比", "数据库"]).reset_index(drop=True)
t25a.to_csv(OUT / "t25a_tier_with_pooled.csv", index=False, encoding="utf-8-sig")
print(t25a.to_string(index=False))
for tag, m in tier_meta.items():
    print(f"  >> {tag}: 合并 OR = {m['OR']:.3f} ({m['lo']:.3f}–{m['hi']:.3f}), "
          f"P = {m['P']:.4f}, I² = {m['I2']:.1f}%")

# ==================================================== T25b 结局模型：含/不含脓毒症双版本
print("\n" + "=" * 78)
print("T25b  结局关联（首次住院队列）—— 协变量集含 / 不含脓毒症双版本")
print("=" * 78)

OUTCOMES = [("院内死亡", "hosp_mort"), ("24h 内机械通气", "vent24"), ("ICU 住院 >7 天", "prolonged_icu")]
b_rows, meta_A, meta_B = [], {}, {}
for nm, ycol in OUTCOMES:
    srcA, srcB = [], []
    for db in DBS:
        df = data_first[db]
        if ycol not in df.columns or pd.to_numeric(df[ycol], errors="coerce").notna().sum() == 0:
            b_rows.append({"结局": nm, "数据库": LABEL[db], "非NPSLE 事件率": "—", "NPSLE 事件率": "—",
                           "粗 OR (95%CI)": "数据缺失", "模型A 校正 OR (95%CI)‡": "—",
                           "模型B 校正 OR (95%CI)§": "—", "模型A n": "—", "模型B n": "—"})
            continue
        e1 = pd.to_numeric(df.loc[pd.to_numeric(df["npsle_core"], errors="coerce") == 1, ycol], errors="coerce")
        e0 = pd.to_numeric(df.loc[pd.to_numeric(df["npsle_core"], errors="coerce") == 0, ycol], errors="coerce")
        cr = fit_or(df, ycol, "npsle_core", adjust=None, cluster=False)
        adA = fit_or(df, ycol, "npsle_core", adjust=ADJ_BASE + [SEV[db], "sepsis_dx"], cluster=False)
        adB = fit_or(df, ycol, "npsle_core", adjust=ADJ_BASE + [SEV[db]], cluster=False)
        b_rows.append({
            "结局": nm, "数据库": LABEL[db],
            "非NPSLE 事件率": f"{int(e0.sum())}/{e0.notna().sum()} ({100*e0.mean():.1f}%)" if e0.notna().sum() else "—",
            "NPSLE 事件率": f"{int(e1.sum())}/{e1.notna().sum()} ({100*e1.mean():.1f}%)" if e1.notna().sum() else "—",
            "粗 OR (95%CI)": fmt(cr),
            "模型A 校正 OR (95%CI)‡": fmt(adA), "模型B 校正 OR (95%CI)§": fmt(adB),
            "模型A n": adA[4], "模型B n": adB[4]})
        if pd.notna(adA[0]) and adA[5] >= MIN_TOTAL_EV:
            srcA.append(dict(logor=np.log(adA[0]), se=se_ci(adA[1], adA[2]), OR=adA[0], lo=adA[1], hi=adA[2], db=LABEL[db]))
        if pd.notna(adB[0]) and adB[5] >= MIN_TOTAL_EV:
            srcB.append(dict(logor=np.log(adB[0]), se=se_ci(adB[1], adB[2]), OR=adB[0], lo=adB[1], hi=adB[2], db=LABEL[db]))
    if len(srcA) >= 2:
        meta_A[nm] = dl_meta([s["logor"] for s in srcA], [s["se"] for s in srcA]); meta_A[nm]["src"] = srcA
    if len(srcB) >= 2:
        meta_B[nm] = dl_meta([s["logor"] for s in srcB], [s["se"] for s in srcB]); meta_B[nm]["src"] = srcB

t25b = pd.DataFrame(b_rows)
t25b.to_csv(OUT / "t25b_first_outcomes_2models.csv", index=False, encoding="utf-8-sig")
print(t25b.to_string(index=False))

t25c = pd.DataFrame([{
    "结局": nm, "协变量集": lab, "纳入库数": m["k"],
    "合并 OR (95%CI)": f"{m['OR']:.2f} ({m['lo']:.2f}–{m['hi']:.2f})",
    "P": f"{m['P']:.3f}", "I² (%)": f"{m['I2']:.1f}", "τ²": f"{m['tau2']:.2f}",
    "Q 检验 P": f"{m['Pq']:.3f}"}
    for lab, mm in [("模型A（含脓毒症）", meta_A), ("模型B（不含脓毒症）", meta_B)]
    for nm, m in mm.items()])
t25c = t25c.sort_values(["结局", "协变量集"]).reset_index(drop=True)
t25c.to_csv(OUT / "t25c_first_meta_2models.csv", index=False, encoding="utf-8-sig")
print("\n[T25c] 随机效应合并（双模型）")
print(t25c.to_string(index=False))

# --- 归因阶梯：MIMIC 死亡 OR 的四步分解
print("\n[归因阶梯] MIMIC-IV 院内死亡 OR：队列 × 协变量集 2×2 分解")
ladder = []
for cname, dd, clus in [("全部住院 (聚类稳健SE)", data["mimiciv"], True),
                        ("首次住院", data_first["mimiciv"], False)]:
    for aname, adj in [("含脓毒症", ADJ_BASE + [SEV["mimiciv"], "sepsis_dx"]),
                       ("不含脓毒症", ADJ_BASE + [SEV["mimiciv"]])]:
        r = fit_or(dd, "hosp_mort", "npsle_core", adjust=adj, cluster=clus)
        ladder.append({"队列": cname, "协变量集": aname, "校正 OR (95%CI)": fmt(r),
                       "P": f"{r[3]:.3f}" if pd.notna(r[3]) else "—", "n": r[4]})
# 粗率对照
for cname, dd in [("全部住院", data["mimiciv"]), ("首次住院", data_first["mimiciv"])]:
    y = pd.to_numeric(dd["npsle_core"], errors="coerce"); m_ = pd.to_numeric(dd["hosp_mort"], errors="coerce")
    r1, r0 = m_[y == 1], m_[y == 0]
    cr = fit_or(dd, "hosp_mort", "npsle_core", adjust=None, cluster=False)
    ladder.append({"队列": cname, "协变量集": "粗率（未校正）",
                   "校正 OR (95%CI)": f"{fmt(cr)}；NP组 {100*r1.mean():.1f}% vs 非NP组 {100*r0.mean():.1f}%",
                   "P": f"{cr[3]:.3f}" if pd.notna(cr[3]) else "—", "n": cr[4]})
t25d = pd.DataFrame(ladder)
t25d.to_csv(OUT / "t25d_mortality_ladder.csv", index=False, encoding="utf-8-sig")
print(t25d.to_string(index=False))

# ==================================================== T26 首次住院版基线
print("\n" + "=" * 78)
print("T26  首次住院队列基线特征")
print("=" * 78)

b26 = []
for db in DBS:
    df = data_first[db]
    y = pd.to_numeric(df["npsle_core"], errors="coerce").fillna(0)
    g1, g0 = df[y == 1], df[y == 0]
    sevcol = SEV[db]
    items = [("例数, n", lambda s: str(len(s))),
             ("年龄, 岁 中位[IQR]", lambda s: iqr(s["age"])),
             ("女性, n (%)", lambda s: npct(s["female"])),
             (f"{'SOFA-24' if sevcol=='sofa24' else 'APACHE IV'}, 中位[IQR]", lambda s: iqr(s[sevcol]) if sevcol in s else "—"),
             ("肌酐, mg/dL 中位[IQR]", lambda s: iqr(s["creat"])),
             ("血小板, ×10⁹/L 中位[IQR]", lambda s: iqr(s["plt"])),
             ("脓毒症, n (%)", lambda s: npct(s["sepsis_dx"])),
             ("狼疮肾炎, n (%)", lambda s: npct(s["lupus_nephritis"])),
             ("糖皮质激素暴露, n (%)", lambda s: npct(s["steroid_any"])),
             ("院内死亡, n (%)", lambda s: npct(s["hosp_mort"])),
             ("24h 内机械通气, n (%)", lambda s: npct(s["vent24"]))]
    for nm, fn in items:
        try:
            v1, v0 = fn(g1), fn(g0)
        except Exception:
            v1 = v0 = "—"
        b26.append({"数据库": LABEL[db], "变量": nm, "无记录事件": v0, "有记录事件": v1})
t26 = pd.DataFrame(b26)
t26.to_csv(OUT / "t26_first_baseline.csv", index=False, encoding="utf-8-sig")
print(t26.to_string(index=False))

# ==================================================== T27 首次住院版患病率与 Tier 构成
print("\n" + "=" * 78)
print("T27  首次住院队列患病率与 Tier 构成（对照全部住院口径）")
print("=" * 78)

r27 = []
for db in DBS:
    for cname, df in [("全部住院", data[db]), ("首次住院（主分析）", data_first[db])]:
        n = len(df)
        core = int(pd.to_numeric(df["npsle_core"], errors="coerce").fillna(0).sum())
        broad = int(pd.to_numeric(df["npsle_broad"], errors="coerce").fillna(0).sum()) \
            if "npsle_broad" in df.columns else np.nan
        c_mask = (pd.to_numeric(df["npsle_core"], errors="coerce").fillna(0) == 1) & \
                 (pd.to_numeric(df["tier_c"], errors="coerce").fillna(0) == 1)
        hi_mask = (pd.to_numeric(df["npsle_core"], errors="coerce").fillna(0) == 1) & \
                  (pd.to_numeric(df["npsle_hi"], errors="coerce").fillna(0) == 1)
        lo_, hi_ = wilson(core, n)
        tlo, thi = wilson(int(c_mask.sum()), core) if core else (np.nan, np.nan)
        r27.append({"数据库": LABEL[db], "队列": cname, "n": n,
                    "核心事件 n (%, 95%CI)": f"{core} ({100*core/n:.1f}%, {100*lo_:.1f}–{100*hi_:.1f})",
                    "广义事件 n (%)": f"{broad} ({100*broad/n:.1f}%)" if broad == broad else "—",
                    "含 Tier C n (占核心 %, 95%CI)":
                        (f"{int(c_mask.sum())} ({100*c_mask.sum()/core:.1f}%, {100*tlo:.1f}–{100*thi:.1f})"
                         if core else "—"),
                    "Tier A+B n (占核心 %)":
                        f"{int(hi_mask.sum())} ({100*hi_mask.sum()/core:.1f}%)" if core else "—"})
t27 = pd.DataFrame(r27)
t27.to_csv(OUT / "t27_first_prevalence_tier.csv", index=False, encoding="utf-8-sig")
print(t27.to_string(index=False))

# 首次住院队列 Tier C 占比跨库一致性检验
A27 = t27[(t27["数据库"] == "MIMIC-IV") & (t27["队列"].str.startswith("首次"))].iloc[0]
B27 = t27[(t27["数据库"] == "eICU-CRD") & (t27["队列"].str.startswith("首次"))].iloc[0]
_a = int(A27["含 Tier C n (占核心 %, 95%CI)"].split(" ")[0]); _an = int(A27["核心事件 n (%, 95%CI)"].split(" ")[0])
_b = int(B27["含 Tier C n (占核心 %, 95%CI)"].split(" ")[0]); _bn = int(B27["核心事件 n (%, 95%CI)"].split(" ")[0])
chi_f, p_f = stats.chi2_contingency([[_a, _an - _a], [_b, _bn - _b]], correction=False)[:2]
print(f"  >> 首次住院队列 Tier C 占比：MIMIC {100*_a/_an:.1f}% ({_a}/{_an}) vs "
      f"eICU {100*_b/_bn:.1f}% ({_b}/{_bn})，χ² = {chi_f:.3f}，P = {p_f:.3f}")

# ==================================================== T28 全部住院 + 真·聚类稳健（协变量集对齐主模型）
print("\n" + "=" * 78)
print("T28  结局关联：全部住院 + 按患者聚类稳健标准误（敏感性分析，协变量集同模型A）")
print("=" * 78)

r28, meta28 = [], {}
for nm, ycol in OUTCOMES:
    src = []
    for db in DBS:
        d = data[db]
        if ycol not in d.columns or pd.to_numeric(d[ycol], errors="coerce").notna().sum() == 0:
            r28.append({"结局": nm, "数据库": LABEL[db], "非NPSLE 事件率": "—", "NPSLE 事件率": "—",
                        "粗 OR (95%CI)": "数据缺失", "校正 OR (95%CI)‖": "—", "模型 n": "—"})
            continue
        e1 = pd.to_numeric(d.loc[pd.to_numeric(d["npsle_core"], errors="coerce") == 1, ycol], errors="coerce")
        e0 = pd.to_numeric(d.loc[pd.to_numeric(d["npsle_core"], errors="coerce") == 0, ycol], errors="coerce")
        cr = fit_or(d, ycol, "npsle_core", adjust=None, cluster=True)
        ad = fit_or(d, ycol, "npsle_core", adjust=ADJ_BASE + [SEV[db], "sepsis_dx"], cluster=True)
        r28.append({"结局": nm, "数据库": LABEL[db],
                    "非NPSLE 事件率": f"{int(e0.sum())}/{e0.notna().sum()} ({100*e0.mean():.1f}%)" if e0.notna().sum() else "—",
                    "NPSLE 事件率": f"{int(e1.sum())}/{e1.notna().sum()} ({100*e1.mean():.1f}%)" if e1.notna().sum() else "—",
                    "粗 OR (95%CI)": fmt(cr), "校正 OR (95%CI)‖": fmt(ad), "模型 n": ad[4]})
        if pd.notna(ad[0]) and ad[5] >= MIN_TOTAL_EV:
            src.append(dict(logor=np.log(ad[0]), se=se_ci(ad[1], ad[2])))
    if len(src) >= 2:
        meta28[nm] = dl_meta([s["logor"] for s in src], [s["se"] for s in src])
t28 = pd.DataFrame(r28)
t28.to_csv(OUT / "t28_allstay_cluster_outcomes.csv", index=False, encoding="utf-8-sig")
print(t28.to_string(index=False))

t28m = pd.DataFrame([{"结局": nm, "纳入库数": m["k"],
                      "合并 OR (95%CI)": f"{m['OR']:.2f} ({m['lo']:.2f}–{m['hi']:.2f})",
                      "P": f"{m['P']:.3f}", "I² (%)": f"{m['I2']:.1f}", "τ²": f"{m['tau2']:.2f}",
                      "Q 检验 P": f"{m['Pq']:.3f}"} for nm, m in meta28.items()])
t28m.to_csv(OUT / "t28m_allstay_cluster_meta.csv", index=False, encoding="utf-8-sig")
print("\n[T28m] 随机效应合并（全部住院 + 聚类稳健）")
print(t28m.to_string(index=False))

# ==================================================== T29 首次住院子集的 eICU 时序
print("\n" + "=" * 78)
print("T29  eICU-CRD 诊断时序：全部住院 vs 首次住院子集")
print("=" * 78)

NP_EICU_CORE = (r"seizure|status epilepticus|encephalopathy|delirium|coma|altered mental|obtund|"
                r"unresponsive|psychosis|psychotic|meningitis|encephalitis|myelitis|demyelinat|"
                r"multiple sclerosis")
SEP_EICU = r"sepsis|septic shock"
t29 = None
t29_note = ""
try:
    import psycopg2
    CFG = dict(host="localhost", port=5432, user="postgres", password="1314")
    ids = tuple(int(x) for x in data["eicu"]["stay_id"].tolist())
    sql = f"""
    SELECT patientunitstayid AS stay_id,
           min(CASE WHEN diagnosisstring ~* '{NP_EICU_CORE}' THEN diagnosisoffset END) AS np_off,
           min(CASE WHEN diagnosisstring ~* '{SEP_EICU}'    THEN diagnosisoffset END) AS sep_off
    FROM eicu_crd.diagnosis
    WHERE patientunitstayid IN {ids}
    GROUP BY 1"""
    with psycopg2.connect(dbname="eicu", **CFG) as cn:
        off = pd.read_sql(sql, cn)
    rows29 = []
    for cname, base in [("全部住院", data["eicu"]), ("首次住院（主分析）", data_first["eicu"])]:
        d = base.merge(off, on="stay_id", how="left")
        npd = d[(pd.to_numeric(d["npsle_core"], errors="coerce") == 1) & d["np_off"].notna()].copy()
        if not len(npd):
            continue
        n_np = len(npd); w24 = int((npd["np_off"] <= 1440).sum())
        med = npd["np_off"].median(); q1, q3 = npd["np_off"].quantile([.25, .75])
        both = npd[npd["sep_off"].notna()].copy()
        sf = int((both["sep_off"] < both["np_off"]).sum())
        sm_ = int((both["sep_off"] == both["np_off"]).sum())
        nf = int((both["sep_off"] > both["np_off"]).sum())
        rows29.append({
            "队列": cname, "有时间戳的神经精神事件住院数": n_np,
            "首次 NP 诊断距入 ICU 时间, 中位 [IQR], h": f"{med/60:.1f} [{q1/60:.1f}, {q3/60:.1f}]",
            "24 h 内记录, n (%)": f"{w24}/{n_np} ({100*w24/n_np:.1f}%)",
            "同具两类时间戳 n": len(both),
            "脓毒症在前, n (%)": f"{sf}/{len(both)} ({100*sf/len(both):.1f}%)" if len(both) else "—",
            "同一时间点, n (%)": f"{sm_}/{len(both)} ({100*sm_/len(both):.1f}%)" if len(both) else "—",
            "神经精神在前, n (%)": f"{nf}/{len(both)} ({100*nf/len(both):.1f}%)" if len(both) else "—"})
    t29 = pd.DataFrame(rows29)
    t29.to_csv(OUT / "t29_timing_two_cohorts.csv", index=False, encoding="utf-8-sig")
    print(t29.to_string(index=False))
except Exception as e:
    t29_note = f"（时序对照未能执行：{e}）"
    print(t29_note)

# ==================================================== GCS 口径对照
print("\n" + "=" * 78)
print("GCS 表 n 口径对照（解释 V3 → V4 的 n 漂移）")
print("=" * 78)
gcs_rows = []
for db in ["mimiciv", "eicu"]:
    d = data[db]
    v = pd.to_numeric(d["gcs_verbal"], errors="coerce")
    vent = pd.to_numeric(d["vent24"], errors="coerce")
    y = pd.to_numeric(d["npsle_core"], errors="coerce")
    for tag, msk in [("未机械通气", vent == 0), ("机械通气", vent == 1)]:
        gcs_rows.append({"数据库": LABEL[db], "分层": tag,
                         "分层总例数": int(msk.sum()),
                         "GCS 言语分非缺失（可分析）": int((msk & v.notna() & y.notna()).sum()),
                         "GCS 缺失 n": int((msk & (v.isna() | y.isna())).sum())})
tgcs = pd.DataFrame(gcs_rows)
tgcs.to_csv(OUT / "t30_gcs_n_reconcile.csv", index=False, encoding="utf-8-sig")
print(tgcs.to_string(index=False))

# ==================================================== 森林图重绘
def forest(items, title, xlab, path, xlim=(0.15, 12), ticks=(0.25, 0.5, 1, 2, 4, 8)):
    fig, ax = plt.subplots(figsize=(7.8, 0.52 * len(items) + 1.7))
    ys = np.arange(len(items))[::-1]
    for i, (nm, o, lo, hi, pooled) in enumerate(items):
        y = ys[i]
        if not np.isfinite(o):
            ax.text(xlim[0] * 1.15, y, nm, ha="left", va="center", fontsize=10, color="#888")
            ax.text(xlim[1] * 1.05, y, "不可估计", ha="left", va="center", fontsize=9.5, color="#888")
            continue
        ax.plot([lo, hi], [y, y], color="#1f4e79" if pooled else "#444",
                lw=2.3 if pooled else 1.5, solid_capstyle="butt", zorder=2)
        ax.scatter([o], [y], s=155 if pooled else 78, marker="D" if pooled else "s",
                   color="#c0392b" if pooled else "#1f4e79", zorder=3)
        ax.text(xlim[0] * 1.15, y, nm, ha="left", va="center", fontsize=10,
                fontweight="bold" if pooled else "normal")
        ax.text(xlim[1] * 1.05, y, f"{o:.2f} ({lo:.2f}–{hi:.2f})", ha="left", va="center",
                fontsize=9.8, fontweight="bold" if pooled else "normal")
    ax.axvline(1, color="#888", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlim(*xlim)
    ax.set_xticks(list(ticks)); ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticks([]); ax.set_ylim(-0.8, len(items) - 0.2)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_xlabel(xlab, fontsize=10.5)
    ax.set_title(title, fontsize=11.3, fontweight="bold", pad=10)
    plt.tight_layout(); plt.subplots_adjust(left=0.03, right=0.70)
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()


# 图 4：Tier 分层（含两条合并菱形）
ti = []
for tag in ["Tier C 非特异事件", "Tier A+B 高置信事件"]:
    src = mc if tag.startswith("Tier C") else mh
    for s in src:
        ti.append((f"{tag[:6]} · {s['db']}", s["OR"], s["lo"], s["hi"], False))
    if tag in tier_meta:
        m = tier_meta[tag]
        ti.append((f"{tag[:6]} · 合并", m["OR"], m["lo"], m["hi"], True))
forest(ti, "脓毒症共病关联按诊断置信度分层（首次住院队列）",
       "校正 OR", FIG / "forest_tier_first.png", xlim=(0.1, 12), ticks=(0.25, 0.5, 1, 2, 4, 8))
print("\n[fig] fig/forest_tier_first.png（已补 Tier A+B 合并菱形）")

# 图 5：结局（模型 A）
oi = []
for nm, _ in OUTCOMES:
    if nm in meta_A:
        for s in meta_A[nm]["src"]:
            oi.append((f"{nm} · {s['db']}", s["OR"], s["lo"], s["hi"], False))
        m = meta_A[nm]
        oi.append((f"{nm} · 合并", m["OR"], m["lo"], m["hi"], True))
forest(oi, "编码记录的神经精神事件与临床结局的关联（首次住院队列，模型 A）",
       "校正 OR", FIG / "forest_outcomes_first.png", xlim=(0.08, 20), ticks=(0.25, 0.5, 1, 2, 4, 8))
print("[fig] fig/forest_outcomes_first.png（模型 A：含脓毒症校正）")

# ==================================================== 导出 KEY
prev = json.loads((OUT / "_v4_key.json").read_text(encoding="utf-8"))
key = dict(prev)
key["tier_meta_v5"] = {k: dict(OR=v["OR"], lo=v["lo"], hi=v["hi"], P=v["P"], I2=v["I2"], k=v["k"])
                       for k, v in tier_meta.items()}
key["outcome_meta_A"] = {k: dict(OR=v["OR"], lo=v["lo"], hi=v["hi"], P=v["P"], I2=v["I2"],
                                 tau2=v["tau2"], k=v["k"]) for k, v in meta_A.items()}
key["outcome_meta_B"] = {k: dict(OR=v["OR"], lo=v["lo"], hi=v["hi"], P=v["P"], I2=v["I2"],
                                 tau2=v["tau2"], k=v["k"]) for k, v in meta_B.items()}
key["allstay_cluster_meta"] = {k: dict(OR=v["OR"], lo=v["lo"], hi=v["hi"], P=v["P"], I2=v["I2"],
                                       k=v["k"]) for k, v in meta28.items()}
key["first_tierC"] = dict(mimic=[_a, _an], eicu=[_b, _bn], chi=float(chi_f), P=float(p_f))
key["tier_detail"] = {f"{k[0]}|{k[1]}": v for k, v in tier_detail.items()}
(OUT / "_v5_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n[ok] -> t25a–t30 csv, out/_v5_key.json")
