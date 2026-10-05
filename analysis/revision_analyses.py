#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
应答审稿意见的补充分析
用法:  python revision_analyses.py

产出
----
out/t11_tier_prev.csv     诊断置信度分层的患病率构成          (Major 2)
out/t12_or_by_tier.csv    三种暴露定义 × 聚类稳健SE × 首次住院 (Major 2 + 4)
out/t13_meta_hi.csv       高置信度定义下的随机效应合并         (Major 2)
out/t14_dca.csv           决策曲线净获益数值表                 (Minor 5)
out/t15_label_auc.csv     标签置信度 vs 判别力                 (Major 2 延伸)
out/revision_report.md    汇总
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy import stats
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npsle_io

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
DBS = list(LABEL)

# Canonical row order. The extraction queries carry no ORDER BY, so the cohort
# CSV comes back in whatever order the database happened to return: the eICU-CRD
# eICU frame was re-ordered for 228 of its 230 stays between two runs of the same
# extraction. Both GroupKFold (which assigns groups in order of first appearance)
# and RepeatedStratifiedKFold (seeded, but position-dependent) are sensitive to
# that order, so without an explicit sort the S11 decision-curve values and the
# S19 label-confidence AUCs are not reproducible from the data alone -- the same
# defect that was fixed for the CV comparison in v8_cv_label_2x2.py. Sorting on
# (subject_id, stay_id) makes every published cell reproducible.
data = {db: npsle_io.load(db)
        .sort_values(["subject_id", "stay_id"], kind="mergesort")
        .reset_index(drop=True) for db in DBS}
# ============================================================ T11 分层构成
rows = []
for db in DBS:
    d = data[db]
    n = len(d)
    ta = (f'{int(d.tier_a.fillna(0).sum())} ({100*d.tier_a.fillna(0).mean():.1f}%)'
          if db == "eicu" else "结构性不可得")
    core = int(d.npsle_core.sum())
    rows.append({
        "数据库": LABEL[db], "队列 n": n,
        "Tier A 归因明确": ta,
        "Tier B 特异表型": f"{int(d.tier_b.sum())} ({100*d.tier_b.mean():.1f}%)",
        "Tier C 非特异": f"{int(d.tier_c.sum())} ({100*d.tier_c.mean():.1f}%)",
        "明确他因": f"{int(d.other_cause.sum())} ({100*d.other_cause.mean():.1f}%)",
        "高置信 A+B": f"{int(d.npsle_hi.sum())} ({100*d.npsle_hi.mean():.1f}%)",
        "严格 A+B 去他因": f"{int(d.npsle_hi_str.sum())} ({100*d.npsle_hi_str.mean():.1f}%)",
        "原核心定义": f"{core} ({100*core/n:.1f}%)",
        "Tier C 占原定义比例": f"{100*int(d.tier_c.sum())/core:.1f}%" if core else "—"})
t11 = pd.DataFrame(rows)
t11.to_csv(os.path.join(OUT, "t11_tier_prev.csv"), index=False, encoding="utf-8-sig")

# ============================================================ 回归工具
ADJ = ["age", "female", "sepsis_dx", "lupus_nephritis", "creat", "plt"]
OUTCOMES = [("hosp_mort", "院内死亡"), ("vent24", "24h内机械通气"), ("prolonged_icu", "ICU住院>7天")]
for db in DBS:
    d = data[db]
    d["prolonged_icu"] = (pd.to_numeric(d["icu_los"], errors="coerce") > 7).astype(float)
    d.loc[pd.to_numeric(d["icu_los"], errors="coerce").isna(), "prolonged_icu"] = np.nan


def adj_for(d):
    keep = []
    for a in ADJ:
        if a not in d.columns:
            continue
        s = pd.to_numeric(d[a], errors="coerce")
        if s.notna().sum() >= 20 and s.dropna().nunique() >= 2:
            keep.append(a)
    return keep


def fit_or(d, y, x, adjust=None, cluster=False):
    """返回 (OR, lo, hi, n, 事件数). cluster=True 时用 subject_id 聚类稳健SE。

    注意: eICU 的 subject_id 为字符串型 uniquepid(形如 '002-41835'),
    不能与数值列一起 to_numeric —— 否则整列变 NaN, dropna 会清空全库。
    此处单独 factorize 为整数编码。
    """
    cols = [y, x] + (adjust or [])
    m = d[[c for c in cols if c in d.columns]].apply(pd.to_numeric, errors="coerce")
    if cluster:
        m["_grp"] = pd.factorize(d["subject_id"].astype(str))[0]
    m = m.dropna()
    if len(m) < 30 or m[y].nunique() < 2 or m[x].nunique() < 2:
        return (np.nan,) * 3 + (len(m), int(m[y].sum()) if len(m) else 0)
    X = sm.add_constant(m[[x] + [a for a in (adjust or []) if a in m.columns]], has_constant="add")
    try:
        mod = sm.Logit(m[y].astype(float), X)
        res = (mod.fit(disp=0, method="bfgs", maxiter=300,
                       cov_type="cluster", cov_kwds={"groups": m["_grp"].astype(int)})
               if cluster else mod.fit(disp=0, method="bfgs", maxiter=300))
        b, se = res.params[x], res.bse[x]
        if not np.isfinite(se) or se > 5:
            return (np.nan,) * 3 + (len(m), int(m[y].sum()))
        return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), len(m), int(m[y].sum())
    except Exception:
        return (np.nan,) * 3 + (len(m), int(m[y].sum()))


fmt = lambda r: f"{r[0]:.2f} ({r[1]:.2f}–{r[2]:.2f})" if pd.notna(r[0]) else "不可估计"

# ============================================================ T12 暴露定义 × 聚类
EXPOS = [("npsle_core", "原核心定义 (A+B+C)"), ("npsle_hi", "高置信 (A+B)"), ("npsle_hi_str", "严格 (A+B 去他因)")]
rows = []
for yv, ycn in OUTCOMES:
    for db in DBS:
        d = data[db]
        A = adj_for(d)
        if yv not in d.columns or d[yv].notna().sum() == 0:
            continue
        for xv, xcn in EXPOS:
            naive = fit_or(d, yv, xv, A, cluster=False)
            clust = fit_or(d, yv, xv, A, cluster=True)
            first = d.sort_values("stay_id").drop_duplicates("subject_id")
            fst = fit_or(first, yv, xv, adj_for(first), cluster=False)
            se_ratio = np.nan
            if pd.notna(naive[0]) and pd.notna(clust[0]):
                sn = (np.log(naive[2]) - np.log(naive[1])) / 3.92
                sc = (np.log(clust[2]) - np.log(clust[1])) / 3.92
                se_ratio = sc / sn if sn > 0 else np.nan
            rows.append({
                "结局": ycn, "数据库": LABEL[db], "暴露定义": xcn,
                "暴露 n": int(pd.to_numeric(d[xv], errors="coerce").sum()),
                "校正 OR (独立SE)": fmt(naive),
                "校正 OR (聚类稳健SE)": fmt(clust),
                "SE 比 (聚类/独立)": f"{se_ratio:.2f}" if pd.notna(se_ratio) else "—",
                "首次住院 OR": fmt(fst), "首次住院 n": fst[3]})
t12 = pd.DataFrame(rows)
t12.to_csv(os.path.join(OUT, "t12_or_by_tier.csv"), index=False, encoding="utf-8-sig")

# ============================================================ T13 meta (高置信度)
def dl_meta(logor, se):
    w = 1 / se ** 2
    fe = np.sum(w * logor) / np.sum(w)
    Q = np.sum(w * (logor - fe) ** 2)
    dfree = len(logor) - 1
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - dfree) / C) if C > 0 else 0.0
    ws = 1 / (se ** 2 + tau2)
    mu = np.sum(ws * logor) / np.sum(ws)
    semu = np.sqrt(1 / np.sum(ws))
    I2 = max(0.0, 100 * (Q - dfree) / Q) if Q > 0 else 0.0
    pQ = 1 - stats.chi2.cdf(Q, dfree) if dfree > 0 else np.nan
    z = mu / semu
    return dict(OR=np.exp(mu), lo=np.exp(mu - 1.96 * semu), hi=np.exp(mu + 1.96 * semu),
                p=2 * (1 - stats.norm.cdf(abs(z))), I2=I2, tau2=tau2, Q=Q, pQ=pQ, k=len(logor))


MIN_EV, MIN_EV1 = 10, 5
rows = []
for yv, ycn in OUTCOMES:
    src = []
    for db in DBS:
        d = data[db]
        if yv not in d.columns or d[yv].notna().sum() == 0:
            continue
        r = fit_or(d, yv, "npsle_hi", adj_for(d), cluster=True)
        n_ev = int(pd.to_numeric(d[yv], errors="coerce").sum())
        n_ev1 = int(pd.to_numeric(d.loc[d.npsle_hi == 1, yv], errors="coerce").sum())
        if pd.notna(r[0]) and n_ev >= MIN_EV and n_ev1 >= MIN_EV1:
            src.append((np.log(r[0]), (np.log(r[2]) - np.log(r[1])) / 3.92, LABEL[db]))
    if len(src) >= 2:
        lo = np.array([s[0] for s in src]); se = np.array([s[1] for s in src])
        m = dl_meta(lo, se)
        rows.append({"结局": ycn, "纳入库数 k": m["k"], "纳入库": "、".join(s[2] for s in src),
                     "合并 OR (95%CI)": f'{m["OR"]:.2f} ({m["lo"]:.2f}–{m["hi"]:.2f})',
                     "P": f'{m["p"]:.3f}', "I²": f'{m["I2"]:.1f}%', "τ²": f'{m["tau2"]:.3f}',
                     "Q": f'{m["Q"]:.2f}', "P(Q)": f'{m["pQ"]:.3f}'})
    else:
        rows.append({"结局": ycn, "纳入库数 k": len(src), "纳入库": "、".join(s[2] for s in src) or "—",
                     "合并 OR (95%CI)": "不足 2 库，未合并", "P": "—", "I²": "—", "τ²": "—", "Q": "—", "P(Q)": "—"})
t13 = pd.DataFrame(rows)
t13.to_csv(os.path.join(OUT, "t13_meta_hi.csv"), index=False, encoding="utf-8-sig")

# ============================================================ 建模工具
FEATS = ["age", "female", "gcs_min", "hr", "map", "temp", "spo2", "wbc", "creat", "sepsis_dx"]


def cv_pred(d, feats, target, model="xgb", seed=7, grouped=True):
    """Out-of-fold probabilities under the primary validation design.

    grouped=True splits by patient, which is the scheme used for the
    identification models (Section 2.7); the AUC intervals in boot_auc_ci
    resample whole patients for the same reason.
    """
    y = pd.to_numeric(d[target], errors="coerce").fillna(0).values.astype(int)
    X = d[feats].apply(pd.to_numeric, errors="coerce").values
    groups = (d["subject_id"].astype(str).values if "subject_id" in d.columns
              else np.arange(len(d)))
    oof, cnt = np.zeros(len(d)), np.zeros(len(d))
    if grouped:
        n_splits = 5
        splitter = GroupKFold(n_splits=n_splits)
        folds = list(splitter.split(X, y, groups=groups))
    else:
        folds = list(RepeatedStratifiedKFold(
            n_splits=5, n_repeats=5, random_state=seed).split(X, y))
    for tr, te in folds:
        med = np.nanmedian(X[tr], axis=0); med = np.where(np.isnan(med), 0, med)
        Xtr = np.where(np.isnan(X[tr]), med, X[tr]); Xte = np.where(np.isnan(X[te]), med, X[te])
        if model == "xgb":
            m = xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, reg_lambda=2, min_child_weight=5,
                                  eval_metric="logloss", random_state=seed, n_jobs=4).fit(Xtr, y[tr])
            p = m.predict_proba(Xte)[:, 1]
        else:
            mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
            r = sm.Logit(y[tr], sm.add_constant((Xtr - mu) / sd, has_constant="add")).fit_regularized(
                disp=0, alpha=1.0, L1_wt=0.0)
            p = r.predict(sm.add_constant((Xte - mu) / sd, has_constant="add"))
        oof[te] += p; cnt[te] += 1
    ok = cnt > 0
    return y[ok], oof[ok] / cnt[ok], (np.asarray(groups)[ok] if grouped else None)


def boot_auc_ci(y, p, groups=None, n=1000, seed=7):
    """Percentile bootstrap; resamples whole patients when groups is given."""
    rng = np.random.default_rng(seed)
    if groups is None:
        i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
        a = []
        for _ in range(n):
            idx = np.concatenate([rng.choice(i1, len(i1), True),
                                  rng.choice(i0, len(i0), True)])
            a.append(roc_auc_score(y[idx], p[idx]))
        return np.percentile(a, [2.5, 97.5])
    uniq = np.unique(groups)
    pos = {g: np.where(groups == g)[0] for g in uniq}
    a = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([pos[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        a.append(roc_auc_score(y[idx], p[idx]))
    if len(a) < 50:
        return np.array([np.nan, np.nan])
    return np.percentile(a, [2.5, 97.5])


# ============================================================ T15 标签置信度 vs 判别力
rows = []
for db in ["mimiciv", "eicu"]:
    d = data[db]
    for tgt, tcn in [("npsle_core", "原核心定义 (A+B+C)"), ("npsle_hi", "高置信 (A+B)"),
                     ("npsle_hi_str", "严格 (A+B 去他因)"), ("tier_a", "Tier A 归因明确")]:
        if tgt not in d.columns:
            continue
        s = pd.to_numeric(d[tgt], errors="coerce").fillna(0)
        if s.sum() < 20:
            rows.append({"数据库": LABEL[db], "标签定义": tcn, "事件数": int(s.sum()),
                         "XGBoost CV-AUC (95%CI)": "事件<20，不估计"})
            continue
        y, p, g = cv_pred(d, FEATS, tgt, "xgb")
        auc = roc_auc_score(y, p); ci = boot_auc_ci(y, p, g)
        rows.append({"数据库": LABEL[db], "标签定义": tcn, "事件数": int(s.sum()),
                     "XGBoost CV-AUC (95%CI)": f"{auc:.3f} ({ci[0]:.3f}–{ci[1]:.3f})"})
t15 = pd.DataFrame(rows)
t15.to_csv(os.path.join(OUT, "t15_label_auc.csv"), index=False, encoding="utf-8-sig")

# ============================================================ T14 DCA 数值
def net_benefit(y, p, pt):
    tp = np.sum((p >= pt) & (y == 1)); fp = np.sum((p >= pt) & (y == 0)); n = len(y)
    return tp / n - fp / n * (pt / (1 - pt))


rows = []
for db in ["mimiciv", "eicu"]:
    d = data[db]
    y, p, _g = cv_pred(d, FEATS, "npsle_core", "xgb")
    prev = y.mean()
    for pt in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        nb = net_benefit(y, p, pt)
        nb_all = prev - (1 - prev) * (pt / (1 - pt))
        rows.append({"数据库": LABEL[db], "阈概率": f"{pt:.0%}", "模型净获益": f"{nb:.4f}",
                     "全部干预净获益": f"{nb_all:.4f}", "不干预净获益": "0.0000",
                     "模型优于两参照": "是" if (nb > nb_all and nb > 0) else "否",
                     "每 100 例净获真阳性": f"{100*nb:.1f}"})
t14 = pd.DataFrame(rows)
t14.to_csv(os.path.join(OUT, "t14_dca.csv"), index=False, encoding="utf-8-sig")

# ============================================================ 报告
md = lambda df: df.to_markdown(index=False)
terms = pd.read_csv(os.path.join(OUT, "t11_tier_terms.csv"))
rep = f"""# 应答审稿意见 · 补充分析结果

## 表 11 · 诊断置信度分层构成（回应 Major 2）

{md(t11)}

**关键观察**

1. **Tier A（归因明确）在 ICD 编码库中结构性不可得。** ICD-10 的 M32.1x 逐项枚举了
   心内膜炎（M32.11）、心包炎（M32.12）、肺（M32.13）���肾小球（M32.14）、
   肾小管间质（M32.15）受累，**唯独没有神经或精神受累的条目**；
   MIMIC-IV 字典中全部 28 个 lupus 相关码含神经精神字样者为 **0**。
   eICU-CRD 的 `diagnosisstring` 则设有
   `neurologic|infectious disease of nervous system|encephalitis|systemic lupus erythematosus`
   与 `infectious diseases|CNS infections|encephalitis|systemic lupus erythematosus`，
   使其成为三库中唯一能表达狼疮归因的数据源（{int(data['eicu'].tier_a.fillna(0).sum())} 例，8.7%）。

2. **非特异 Tier C 的占比在两大库中高度一致。** 占原核心定义的比例
   MIMIC-IV 为 {100*int(data['mimiciv'].tier_c.sum())/int(data['mimiciv'].npsle_core.sum()):.1f}%、
   eICU-CRD 为 {100*int(data['eicu'].tier_c.sum())/int(data['eicu'].npsle_core.sum()):.1f}%。
   这说明"以非特异意识改变为主"并非 eICU 特有的假阳性问题，
   而是 ICU 行政数据描述神经精神事件的**共同属性**。

3. 采用高置信度定义后，患病率由 19.1% / 43.5% / 6.0% 降至
   {100*data['mimiciv'].npsle_hi.mean():.1f}% / {100*data['eicu'].npsle_hi.mean():.1f}% / {100*data['nwicu'].npsle_hi.mean():.1f}%，
   与文献报道的 ICU 内 NPSLE 水平更接近。

---

## 表 12 · 三种暴露定义 × 聚类稳健标准误 × 首次住院（回应 Major 2 与 Major 4）

{md(t12)}

> **聚类结构**：MIMIC-IV {len(data['mimiciv'])} 次 ICU 住院来自
> {data['mimiciv'].subject_id.nunique()} 名患者（单例最多 {int(data['mimiciv'].subject_id.value_counts().max())} 次）；
> eICU-CRD {len(data['eicu'])} / {data['eicu'].subject_id.nunique()}；
> NWICU {len(data['nwicu'])} / {data['nwicu'].subject_id.nunique()}。
> 「SE 比」>1 表示忽略聚类会低估标准误。

---

## 表 13 · 高置信度定义下的随机效应合并（回应 Major 2）

{md(t13)}

> 采用聚类稳健标准误作为 meta 的输入。仅 2 个研究时 τ² 估计不稳定，
> I² 仅作定性参考。

---

## 表 14 · 决策曲线净获益数值（回应 Minor 5）

{md(t14)}

---

## 表 15 · 标签置信度与判别力的关系（Major 2 的延伸检验）

{md(t15)}

> 若提高标签置信度可显著提升 AUC，说明原低判别力主要源于标签噪声；
> 若 AUC 不升反降或持平，则说明瓶颈在于**预测变量缺乏狼疮特异信息**，
> 而非结局定义不精确。这是区分两种失败机制的关键证据。

---

## 附录 · 各 Tier 命中的原始诊断术语全表

共 {len(terms)} 条术语。前 25 条：

{md(terms.head(25))}

（完整清单见 `out/t11_tier_terms.csv`）
"""
with open(os.path.join(OUT, "revision_report.md"), "w", encoding="utf-8") as f:
    f.write(rep)

print("===== T11 分层构成 =====")
print(t11.to_string(index=False))
print("\n===== T13 高置信度 meta =====")
print(t13.to_string(index=False))
print("\n===== T15 标签置信度 vs 判别力 =====")
print(t15.to_string(index=False))
print("\n[ok] -> out/t11..t15 csv, out/revision_report.md")
