#!/usr/bin/env python3
"""V8 补算：回应第七轮（投稿前必修清单）。

T40  表 17（严重度敏感性）重算 —— 修复"主模型列与加严重度列不在同一分析集"的缺陷。
     原实现：主模型在全分析集拟合（eICU n=207），加严重度后在完整病例集拟合（n=188），
     ΔOR(+2.2%) 因此混入了分析集变化。本表分列报告三个估计：
       (a) 主模型 @ 全分析集     (b) 主模型 @ 严重度完整集   (c) 主模型+严重度 @ 严重度完整集
     真正的"过度校正效应"= (c) vs (b)，即同一分析集内的比较。

T41  首次住院队列 三结局 × 三协变量集 —— 为机械通气结局提供"不含严重度评分"口径。
     严重度评分（SOFA-24 含呼吸/氧合分量；APACHE IV 直接含机械通气项）与 24h 机械通气结局
     存在机械性关联，对该结局校正严重度构成过度校正/对撞（collider）风险。
       A1 = 基础集 + 脓毒症            （不含严重度）  <- 机械通气结局的首选解读口径
       A2 = 基础集 + 严重度 + 脓毒症   （= 现表 8 "模型 A"）
       B  = 基础集 + 严重度            （= 现表 8 "模型 B"）
     附 DerSimonian–Laird 随机效应合并。

T42  粗率（未校正）层面的估计量对比 —— 证明 SE 展宽在无任何协变量时同样发生，
     即"区间过窄"源于重复入院的组内相关本身，而非建模选择。

输出 out/t40_severity_sens_v2.csv, out/t41_first_outcomes_3models.csv,
     out/t41m_first_meta_3models.csv, out/t42_crude_se.csv, out/_v8_key.json
"""
import json, os, pathlib, sys, warnings
import numpy as np, pandas as pd, statsmodels.api as sm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import npsle_io

warnings.filterwarnings("ignore")
os.environ["LC_ALL"] = "C"

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

DBS = ["mimiciv", "eicu", "nwicu"]
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
ADJ_BASE = ["age", "female", "lupus_nephritis", "creat", "plt"]
SEV = {"mimiciv": "sofa24", "eicu": "apache", "nwicu": "sofa24"}
SEVLAB = {"mimiciv": "SOFA-24", "eicu": "APACHE IV", "nwicu": "SOFA-24"}
MIN_TOTAL_EV = 10          # 与主分析一致的随机效应合并纳入门槛（总事件数）
OUTCOMES = [("院内死亡", "hosp_mort"), ("24h 内机械通气", "vent24"), ("ICU 住院 >7 天", "prolonged_icu")]


def load(db):
    d = npsle_io.load(db)
    d["prolonged_icu"] = (pd.to_numeric(d["icu_los"], errors="coerce") > 7).astype(float)
    return d


data = {db: load(db) for db in DBS}
FIRST = json.loads((OUT / "_first_stays.json").read_text(encoding="utf-8"))
FKEY = {"mimiciv": "mimic_first", "eicu": "eicu_first", "nwicu": "nwicu_first"}
first_ids = {db: set(pd.Series(FIRST[FKEY[db]]).astype(str)) for db in DBS}
data_first = {db: data[db][data[db]["stay_id"].astype(str).isin(first_ids[db])].copy() for db in DBS}


def usable(d, cols):
    return [a for a in (cols or [])
            if a in d.columns
            and pd.to_numeric(d[a], errors="coerce").notna().sum() > 0
            and pd.to_numeric(d[a], errors="coerce").nunique(dropna=True) > 1]


def fit_or(d, y, x, adjust=None, cluster=True, restrict=None):
    """restrict: 额外要求非缺失的列（用于强制两模型落在同一分析集）"""
    adjust = usable(d, adjust)
    need = [y, x] + adjust + [c for c in (restrict or []) if c in d.columns and c not in adjust]
    m = d[need].apply(pd.to_numeric, errors="coerce")
    if cluster and "subject_id" in d.columns:
        m = m.assign(_grp=pd.factorize(d["subject_id"].astype(str))[0])
    m = m.dropna()
    nev = int(m[y].sum()) if len(m) else 0
    ngrp = int(m["_grp"].nunique()) if "_grp" in m.columns and len(m) else np.nan
    bad = dict(OR=np.nan, lo=np.nan, hi=np.nan, P=np.nan, n=len(m), nev=nev,
               nadj=len(adjust), ngrp=ngrp, se=np.nan, b=np.nan)
    if len(m) < 30 or m[y].nunique() < 2 or m[x].nunique() < 2:
        return bad
    X = sm.add_constant(m[[x] + adjust], has_constant="add")
    try:
        mod = sm.Logit(m[y].astype(float), X)
        res = (mod.fit(disp=0, method="bfgs", maxiter=400,
                       cov_type="cluster", cov_kwds={"groups": m["_grp"]})
               if cluster and "_grp" in m.columns else
               mod.fit(disp=0, method="bfgs", maxiter=400))
        b, se = res.params[x], res.bse[x]
        if not np.isfinite(se) or se > 5 or abs(b) > 8:
            return bad
        return dict(OR=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)),
                    hi=float(np.exp(b + 1.96 * se)), P=float(res.pvalues[x]),
                    n=len(m), nev=nev, nadj=len(adjust), ngrp=ngrp,
                    se=float(se), b=float(b))
    except Exception:
        return bad


def s(r):
    return f"{r['OR']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})" if pd.notna(r["OR"]) else "不可估计"


def dl_pool(rs):
    """DerSimonian–Laird 随机效应合并；rs = [dict(b=, se=), ...]"""
    rs = [r for r in rs if pd.notna(r.get("b")) and pd.notna(r.get("se")) and r["se"] > 0]
    k = len(rs)
    if k == 0:
        return None
    b = np.array([r["b"] for r in rs]); se = np.array([r["se"] for r in rs])
    w = 1.0 / se ** 2
    bf = (w * b).sum() / w.sum()
    Q = float((w * (b - bf) ** 2).sum())
    if k > 1:
        c = w.sum() - (w ** 2).sum() / w.sum()
        tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
        I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    else:
        tau2, I2 = 0.0, 0.0
    ws = 1.0 / (se ** 2 + tau2)
    bp = (ws * b).sum() / ws.sum()
    sep = float(np.sqrt(1.0 / ws.sum()))
    z = bp / sep
    from scipy import stats as st
    p = float(2 * (1 - st.norm.cdf(abs(z))))
    return dict(k=k, OR=float(np.exp(bp)), lo=float(np.exp(bp - 1.96 * sep)),
                hi=float(np.exp(bp + 1.96 * sep)), P=p, I2=float(I2), tau2=float(tau2))


# ══════════════════════════════════════════════════════════ T40
print("=" * 96)
print("T40  表 17 重算：主模型与加严重度模型强制落在同一分析集")
print("=" * 96)

rows40 = []
for db in ["mimiciv", "eicu"]:
    d = data[db]
    sv = SEV[db]
    # (a) 主模型 @ 全分析集（原表 14 口径）
    a = fit_or(d, "npsle_core", "sepsis_dx", adjust=ADJ_BASE, cluster=True)
    # (b) 主模型 @ 严重度完整集（同 n，仅去掉严重度缺失者）
    b = fit_or(d, "npsle_core", "sepsis_dx", adjust=ADJ_BASE, cluster=True, restrict=[sv])
    # (c) 主模型 + 严重度 @ 严重度完整集
    c = fit_or(d, "npsle_core", "sepsis_dx", adjust=ADJ_BASE + [sv], cluster=True)
    dlt = (100 * (c["OR"] - b["OR"]) / b["OR"]) if pd.notna(c["OR"]) and pd.notna(b["OR"]) else np.nan
    dlt_old = (100 * (c["OR"] - a["OR"]) / a["OR"]) if pd.notna(c["OR"]) and pd.notna(a["OR"]) else np.nan
    rows40.append({
        "数据库": LABEL[db], "严重度变量": SEVLAB[db],
        "主模型 OR (95%CI)〔全分析集〕": s(a), "n〔全分析集〕": a["n"],
        "主模型 OR (95%CI)〔严重度完整集〕": s(b),
        "加严重度后 OR (95%CI)〔严重度完整集〕": s(c), "n〔严重度完整集〕": c["n"],
        "ΔOR（同一分析集内）": f"{dlt:+.1f}%" if pd.notna(dlt) else "—",
    })
    print(f"  {LABEL[db]:9s} (a)全分析集 {s(a)} n={a['n']}")
    print(f"            (b)严重度完整集·主模型 {s(b)} n={b['n']}")
    print(f"            (c)严重度完整集·加严重度 {s(c)} n={c['n']}")
    print(f"            ΔOR 同一分析集内 = {dlt:+.1f}%   （旧算法跨分析集 = {dlt_old:+.1f}%）")

t40 = pd.DataFrame(rows40)
t40.to_csv(OUT / "t40_severity_sens_v2.csv", index=False, encoding="utf-8-sig")

# ══════════════════════════════════════════════════════════ T41
print()
print("=" * 96)
print("T41  首次住院队列 三结局 × 三协变量集（A1 无严重度 / A2 含严重度+脓毒症 / B 含严重度不含脓毒症）")
print("=" * 96)

rows41, meta_src = [], {}
for nm, ycol in OUTCOMES:
    for db in DBS:
        d = data_first[db]
        if ycol not in d.columns or pd.to_numeric(d[ycol], errors="coerce").notna().sum() == 0:
            rows41.append({"结局": nm, "数据库": LABEL[db],
                           "模型 A₁ OR (95%CI)": "数据缺失", "A₁ n": "—",
                           "模型 A₂ OR (95%CI)": "数据缺失", "A₂ n": "—",
                           "模型 B OR (95%CI)": "数据缺失", "B n": "—"})
            continue
        sets = {"A1": ADJ_BASE + ["sepsis_dx"],
                "A2": ADJ_BASE + [SEV[db], "sepsis_dx"],
                "B": ADJ_BASE + [SEV[db]]}
        r = {k: fit_or(d, ycol, "npsle_core", adjust=v, cluster=False) for k, v in sets.items()}
        for k in sets:
            # 沿用主分析的 meta 纳入门槛：总事件数 ≥ 10（MIN_TOTAL_EV），否则不入合并
            if r[k]["nev"] >= MIN_TOTAL_EV:
                meta_src.setdefault((nm, k), []).append(r[k])
        rows41.append({
            "结局": nm, "数据库": LABEL[db],
            "模型 A₁ OR (95%CI)": s(r["A1"]), "A₁ n": r["A1"]["n"],
            "模型 A₂ OR (95%CI)": s(r["A2"]), "A₂ n": r["A2"]["n"],
            "模型 B OR (95%CI)": s(r["B"]), "B n": r["B"]["n"]})
        print(f"  {nm:14s} {LABEL[db]:9s}  A1 {s(r['A1']):20s} (adj={r['A1']['nadj']})"
              f"  A2 {s(r['A2']):20s} (adj={r['A2']['nadj']})  B {s(r['B']):20s} (adj={r['B']['nadj']})")

t41 = pd.DataFrame(rows41)
t41.to_csv(OUT / "t41_first_outcomes_3models.csv", index=False, encoding="utf-8-sig")

print()
print("  —— DerSimonian–Laird 随机效应合并（k=2：MIMIC-IV + eICU-CRD）——")
rows41m = []
MLAB = {"A1": "模型 A₁（基础集 + 脓毒症，不含严重度）",
        "A2": "模型 A₂（基础集 + 严重度 + 脓毒症）",
        "B": "模型 B（基础集 + 严重度，不含脓毒症）"}
for nm, _ in OUTCOMES:
    for k in ["A1", "A2", "B"]:
        src = [x for x in meta_src.get((nm, k), []) if pd.notna(x.get("b"))]
        p = dl_pool(src)
        if p is None:
            continue
        rows41m.append({"结局": nm, "协变量集": MLAB[k], "k": p["k"],
                        "合并 OR (95%CI)": f"{p['OR']:.2f} ({p['lo']:.2f}–{p['hi']:.2f})",
                        "P": f"{p['P']:.3f}", "I²": f"{p['I2']:.1f}%",
                        "τ²": f"{p['tau2']:.2f}"})
        print(f"  {nm:14s} {MLAB[k]:32s} {p['OR']:.2f} ({p['lo']:.2f}–{p['hi']:.2f})"
              f"  P={p['P']:.3f}  I²={p['I2']:.1f}%")

t41m = pd.DataFrame(rows41m)
t41m.to_csv(OUT / "t41m_first_meta_3models.csv", index=False, encoding="utf-8-sig")

# ══════════════════════════════════════════════════════════ T42
print()
print("=" * 96)
print("T42  粗率（未校正）层面的估计量对比 —— SE 展宽是否独立于建模选择")
print("=" * 96)

rows42 = []
for nm, ycol in [("院内死亡", "hosp_mort")]:
    for cohort, dd in [("全部住院", data["mimiciv"]), ("首次住院", data_first["mimiciv"])]:
        for est, clus in [("独立 SE", False), ("按患者聚类稳健 SE", True)]:
            r = fit_or(dd, ycol, "npsle_core", adjust=None, cluster=clus)
            rows42.append({"结局": nm, "队列": cohort, "估计量": est,
                           "粗 OR (95%CI)": s(r),
                           "SE(logOR)": f"{r['se']:.4f}" if pd.notna(r["se"]) else "—",
                           "P": f"{r['P']:.3f}" if pd.notna(r["P"]) else "—",
                           "n": r["n"], "事件数": r["nev"],
                           "簇数(患者)": "" if pd.isna(r["ngrp"]) else int(r["ngrp"])})
            print(f"  {cohort:6s} {est:18s} {s(r):22s} SE={r['se']:.4f} P={r['P']:.3f} n={r['n']}")

t42 = pd.DataFrame(rows42)
t42.to_csv(OUT / "t42_crude_se.csv", index=False, encoding="utf-8-sig")

# ══════════════════════════════════════════════════════════ key
def g42(cohort, est, col):
    m = t42[(t42["队列"] == cohort) & (t42["估计量"] == est)]
    return m.iloc[0][col] if len(m) else "—"


key = {
    "t40": {r["数据库"]: r for r in rows40},
    "t41_vent_pooled": {k: v for k, v in
                        {r["协变量集"]: r["合并 OR (95%CI)"] + f" P={r['P']}"
                         for r in rows41m if r["结局"] == "24h 内机械通气"}.items()},
    "t42_crude_se": {
        "全部住院_独立": f"{g42('全部住院','独立 SE','粗 OR (95%CI)')} SE={g42('全部住院','独立 SE','SE(logOR)')}",
        "全部住院_聚类": f"{g42('全部住院','按患者聚类稳健 SE','粗 OR (95%CI)')} SE={g42('全部住院','按患者聚类稳健 SE','SE(logOR)')}",
        "首次住院_独立": f"{g42('首次住院','独立 SE','粗 OR (95%CI)')} SE={g42('首次住院','独立 SE','SE(logOR)')}",
    },
    "naming": {
        "基础集": "年龄、性别、狼疮肾炎、肌酐、血小板（5 项；eICU-CRD 因狼疮肾炎全库不可获得为 4 项）",
        "脓毒症共病主模型": "基础集（不含严重度评分，避免过度校正）",
        "结局模型 A₁": "基础集 + 脓毒症诊断（不含严重度评分）",
        "结局模型 A₂": "基础集 + 严重度评分 + 脓毒症诊断（= 原表 8「模型 A」、表 18）",
        "结局模型 B": "基础集 + 严重度评分（不含脓毒症；= 原表 8「模型 B」）",
    },
}
(OUT / "_v8_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n[OK] 写出 t40_severity_sens_v2.csv / t41_first_outcomes_3models.csv / "
      "t41m_first_meta_3models.csv / t42_crude_se.csv / _v8_key.json")
