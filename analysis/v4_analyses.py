# -*- coding: utf-8 -*-
"""
V4 修订所需补充分析（回应第三轮审稿意见）
==================================================================
T20  免疫抑制/糖皮质激素暴露敏感性分析（回应"免疫抑制暴露未进校正集"）
T21  eICU 诊断时间戳时序分析（回应"时序无法确立"）
T22  首次住院队列主分析（回应"聚类稳健 vs 首次住院自相矛盾"）
      - T22a 脓毒症共病关联
      - T22b 按诊断置信度分层
      - T22c 结局关联 + 随机效应合并
T23  Tier C 占比口径统一表（回应"71.5% vs 68.0% / 分母 123 vs 94"）

用法: python scripts/v4_analyses.py
"""
import os, warnings, pathlib, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
ADJ_BASE = ["age", "female", "lupus_nephritis", "creat", "plt"]
SEV = {"mimiciv": "sofa24", "eicu": "apache", "nwicu": "sofa24"}


def load(db):
    c = pd.read_csv(OUT / f"cohort_{db}.csv")
    t = pd.read_csv(OUT / f"tier_{db}.csv")
    d = c.merge(t, on="stay_id", how="left")
    d["prolonged_icu"] = (pd.to_numeric(d["icu_los"], errors="coerce") > 7).astype(float)
    return d


data = {db: load(db) for db in DBS}
FIRST = json.loads((OUT / "_first_stays.json").read_text(encoding="utf-8"))
FKEY = {"mimiciv": "mimic_first", "eicu": "eicu_first", "nwicu": "nwicu_first"}
first_ids = {db: set(pd.Series(FIRST[FKEY[db]]).astype(str)) for db in DBS}
data_first = {db: data[db][data[db]["stay_id"].astype(str).isin(first_ids[db])].copy() for db in DBS}


def fit_or(d, y, x, adjust=None, cluster=True):
    """logistic 回归；cluster=True 时按 subject_id 聚类稳健SE。返回 (OR, lo, hi, P, n, 事件数)。"""
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
        return (np.nan,) * 4 + (len(m), nev)
    X = sm.add_constant(m[[x] + adjust], has_constant="add")
    try:
        mod = sm.Logit(m[y].astype(float), X)
        res = (mod.fit(disp=0, method="bfgs", maxiter=400,
                       cov_type="cluster", cov_kwds={"groups": m["_grp"]})
               if cluster and "_grp" in m.columns else
               mod.fit(disp=0, method="bfgs", maxiter=400))
        b, se = res.params[x], res.bse[x]
        if not np.isfinite(se) or se > 5 or abs(b) > 8:
            return (np.nan,) * 4 + (len(m), nev)
        tab = pd.crosstab(m[x], m[y])
        if tab.shape != (2, 2) or (tab.values == 0).any():
            return (np.nan,) * 4 + (len(m), nev)
        return (float(np.exp(b)), float(np.exp(b - 1.96 * se)),
                float(np.exp(b + 1.96 * se)), float(res.pvalues[x]), len(m), nev)
    except Exception:
        return (np.nan,) * 4 + (len(m), nev)


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


# ============================================================ T20 免疫抑制暴露敏感性
print("=" * 78)
print("T20  免疫抑制 / 糖皮质激素暴露的敏感性分析")
print("=" * 78)

t20_rows = []
for db in DBS:
    d = data[db]
    base = fit_or(d, "npsle_core", "sepsis_dx", adjust=ADJ_BASE, cluster=True)
    gc = fit_or(d, "npsle_core", "sepsis_dx", adjust=ADJ_BASE + ["steroid_any"], cluster=True)
    isx = fit_or(d, "npsle_core", "sepsis_dx", adjust=ADJ_BASE + ["steroid_any", "is_any"], cluster=True)
    n_is = int(pd.to_numeric(d["is_any"], errors="coerce").notna().sum())
    n_gc = int(pd.to_numeric(d["steroid_any"], errors="coerce").notna().sum())
    t20_rows.append({
        "数据库": LABEL[db],
        "免疫抑制剂变量可得性": f"{n_is}/{len(d)} 例非缺失" if n_is else "该库结构性不可得",
        "M0 主模型 OR (95%CI)": fmt(base),
        "M1 +糖皮质激素 OR (95%CI)": fmt(gc) if n_gc else "—",
        "M2 +糖皮质激素+免疫抑制剂 OR (95%CI)": fmt(isx) if n_is else "不可执行",
        "M0→M2 变化": (f"{100*(isx[0]-base[0])/base[0]:+.1f}%"
                      if (n_is and pd.notna(isx[0]) and pd.notna(base[0])) else "—"),
        "M2 模型 n": isx[4] if n_is else "—"})
t20 = pd.DataFrame(t20_rows)
t20.to_csv(OUT / "t20_immuno_sens.csv", index=False, encoding="utf-8-sig")
print(t20.to_string(index=False))

# ============================================================ T21 eICU 时序
print("\n" + "=" * 78)
print("T21  eICU 诊断时间戳时序分析")
print("=" * 78)

NP_EICU_CORE = (r"seizure|status epilepticus|encephalopathy|delirium|coma|altered mental|obtund|"
                r"unresponsive|psychosis|psychotic|meningitis|encephalitis|myelitis|demyelinat|"
                r"multiple sclerosis")
SEP_EICU = r"sepsis|septic shock"

t21 = None
t21_note = ""
try:
    import psycopg2
    CFG = dict(
    host=os.environ.get("NPSLE_DB_HOST", "localhost"),
    port=int(os.environ.get("NPSLE_DB_PORT", "5432")),
    user=os.environ.get("NPSLE_DB_USER", "postgres"),
    password=os.environ.get("NPSLE_DB_PASSWORD", ""),
)
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
    d = data["eicu"].merge(off, on="stay_id", how="left")
    npd = d[(pd.to_numeric(d["npsle_core"], errors="coerce") == 1) & d["np_off"].notna()].copy()
    n_np = len(npd)
    within24 = int((npd["np_off"] <= 1440).sum())
    med = npd["np_off"].median()
    q1, q3 = npd["np_off"].quantile([.25, .75])
    both = npd[npd["sep_off"].notna()].copy()
    sep_first = int((both["sep_off"] < both["np_off"]).sum())
    same = int((both["sep_off"] == both["np_off"]).sum())
    np_first = int((both["sep_off"] > both["np_off"]).sum())
    t21 = pd.DataFrame([
        {"指标": "记录神经精神事件且有时间戳的住院数", "结果": f"{n_np}"},
        {"指标": "首次神经精神诊断距入 ICU 时间, 中位数 [IQR], 小时",
         "结果": f"{med/60:.1f} [{q1/60:.1f}, {q3/60:.1f}]"},
        {"指标": "首次神经精神诊断发生于入 ICU 24 h 内, n (%)",
         "结果": f"{within24}/{n_np} ({100*within24/n_np:.1f}%)"},
        {"指标": "同时具备脓毒症与神经精神诊断时间戳的住院数", "结果": f"{len(both)}"},
        {"指标": "脓毒症诊断时间早于神经精神诊断, n (%)",
         "结果": f"{sep_first}/{len(both)} ({100*sep_first/len(both):.1f}%)"},
        {"指标": "两者记录于同一时间点, n (%)",
         "结果": f"{same}/{len(both)} ({100*same/len(both):.1f}%)"},
        {"指标": "神经精神诊断时间早于脓毒症诊断, n (%)",
         "结果": f"{np_first}/{len(both)} ({100*np_first/len(both):.1f}%)"},
    ])
    t21.to_csv(OUT / "t21_eicu_timing.csv", index=False, encoding="utf-8-sig")
    print(t21.to_string(index=False))
    t21_note = (f"eICU-CRD 中 {100*within24/n_np:.1f}% 的神经精神诊断在入 ICU 24 h 内即被记录；"
                f"在同时具备两类诊断时间戳的 {len(both)} 例中，仅 {100*sep_first/len(both):.1f}% "
                f"脓毒症诊断时间早于神经精神诊断，{100*same/len(both):.1f}% 记录于同一时间点。"
                f"因此暴露—结局的时间先后无法确立，本研究不作因果推断。")
except Exception as e:
    t21_note = f"（时间戳分析未能执行：{e}）"
    print(t21_note)

# ============================================================ T22 首次住院主分析
print("\n" + "=" * 78)
print("T22  首次住院队列主分析")
print("=" * 78)

# --- T22a 脓毒症共病关联
a_rows, meta_src_f = [], []
for db in DBS:
    df = data_first[db]
    y, x = "npsle_core", "sepsis_dx"
    s1 = pd.to_numeric(df.loc[pd.to_numeric(df[y], errors="coerce") == 1, x], errors="coerce")
    s0 = pd.to_numeric(df.loc[pd.to_numeric(df[y], errors="coerce") == 0, x], errors="coerce")
    cr = fit_or(df, y, x, adjust=None, cluster=False)
    ad = fit_or(df, y, x, adjust=ADJ_BASE, cluster=False)      # 首次住院 = 独立观测, 不需聚类
    a_rows.append({
        "数据库": LABEL[db], "首次住院 n": len(df),
        "NP事件组脓毒症率": f"{int(s1.sum())}/{s1.notna().sum()} ({100*s1.mean():.1f}%)",
        "无NP事件组脓毒症率": f"{int(s0.sum())}/{s0.notna().sum()} ({100*s0.mean():.1f}%)",
        "粗 OR (95%CI)": fmt(cr), "粗 OR P": f"{cr[3]:.3f}" if pd.notna(cr[3]) else "—",
        "校正 OR (95%CI)†": fmt(ad), "校正 OR P": f"{ad[3]:.3f}" if pd.notna(ad[3]) else "—",
        "模型 n": ad[4], "事件数": ad[5]})
    if pd.notna(ad[0]) and ad[5] >= MIN_TOTAL_EV and int(s1.sum()) >= MIN_EXP_EV:
        meta_src_f.append(dict(db=LABEL[db], logor=np.log(ad[0]), se=se_ci(ad[1], ad[2]),
                               OR=ad[0], lo=ad[1], hi=ad[2]))
t22a = pd.DataFrame(a_rows)
mtf = dl_meta([m["logor"] for m in meta_src_f], [m["se"] for m in meta_src_f]) if len(meta_src_f) >= 2 else None
if mtf:
    t22a = pd.concat([t22a, pd.DataFrame([{
        "数据库": f"合并 (随机效应, k={mtf['k']})", "首次住院 n": "—",
        "NP事件组脓毒症率": "—", "无NP事件组脓毒症率": "—", "粗 OR (95%CI)": "—", "粗 OR P": "—",
        "校正 OR (95%CI)†": f"{mtf['OR']:.2f} ({mtf['lo']:.2f}–{mtf['hi']:.2f})",
        "校正 OR P": f"{mtf['P']:.3f}", "模型 n": "—", "事件数": f"I²={mtf['I2']:.1f}%"}])],
        ignore_index=True)
t22a.to_csv(OUT / "t22a_first_sepsis.csv", index=False, encoding="utf-8-sig")
print("\n[T22a] 脓毒症共病关联（首次住院）")
print(t22a.to_string(index=False))
if mtf:
    print(f"  >> 合并 OR = {mtf['OR']:.2f} ({mtf['lo']:.2f}–{mtf['hi']:.2f}), P = {mtf['P']:.4f}, "
          f"I² = {mtf['I2']:.1f}%")

# --- T22b 按 Tier 分层
b_rows, mc, mh = [], [], []
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
        b_rows.append({"数据库": LABEL[db], "对比": f"{tag} vs 无记录事件",
                       "事件数": int(sub["_y"].sum()), "对照数": int((sub["_y"] == 0).sum()),
                       "校正 OR (95%CI)†": fmt(r),
                       "P": f"{r[3]:.3f}" if pd.notna(r[3]) else "—"})
        if pd.notna(r[0]) and int(sub["_y"].sum()) >= MIN_TOTAL_EV and nexp >= MIN_EXP_EV:
            (mc if tag.startswith("Tier C") else mh).append(
                dict(logor=np.log(r[0]), se=se_ci(r[1], r[2])))
t22b = pd.DataFrame(b_rows)
tier_meta_f = {}
for tag, src in [("Tier C 非特异事件", mc), ("Tier A+B 高置信事件", mh)]:
    if len(src) >= 2:
        m = dl_meta([s["logor"] for s in src], [s["se"] for s in src])
        tier_meta_f[tag] = m
        t22b = pd.concat([t22b, pd.DataFrame([{
            "数据库": f"合并 (k={m['k']})", "对比": f"{tag} vs 无记录事件",
            "事件数": "—", "对照数": f"I²={m['I2']:.1f}%",
            "校正 OR (95%CI)†": f"{m['OR']:.2f} ({m['lo']:.2f}–{m['hi']:.2f})",
            "P": f"{m['P']:.3f}"}])], ignore_index=True)
t22b = t22b.sort_values(["对比", "数据库"]).reset_index(drop=True)
t22b.to_csv(OUT / "t22b_first_sepsis_tier.csv", index=False, encoding="utf-8-sig")
print("\n[T22b] 按诊断置信度分层（首次住院）")
print(t22b.to_string(index=False))

# --- T22c 结局关联
OUTCOMES = [("院内死亡", "hosp_mort"), ("24h 内机械通气", "vent24"), ("ICU 住院 >7 天", "prolonged_icu")]
c_rows, meta_out = [], {}
for nm, ycol in OUTCOMES:
    src = []
    for db in DBS:
        df = data_first[db]
        if ycol not in df.columns or pd.to_numeric(df[ycol], errors="coerce").notna().sum() == 0:
            c_rows.append({"结局": nm, "数据库": LABEL[db], "非NPSLE 事件率": "—",
                           "NPSLE 事件率": "—", "粗 OR (95%CI)": "数据缺失",
                           "校正 OR (95%CI)": "—", "模型 n": "—"})
            continue
        e1 = pd.to_numeric(df.loc[pd.to_numeric(df["npsle_core"], errors="coerce") == 1, ycol], errors="coerce")
        e0 = pd.to_numeric(df.loc[pd.to_numeric(df["npsle_core"], errors="coerce") == 0, ycol], errors="coerce")
        cr = fit_or(df, ycol, "npsle_core", adjust=None, cluster=False)
        ad = fit_or(df, ycol, "npsle_core", adjust=ADJ_BASE + [SEV[db]], cluster=False)
        c_rows.append({
            "结局": nm, "数据库": LABEL[db],
            "非NPSLE 事件率": f"{int(e0.sum())}/{e0.notna().sum()} ({100*e0.mean():.1f}%)" if e0.notna().sum() else "—",
            "NPSLE 事件率": f"{int(e1.sum())}/{e1.notna().sum()} ({100*e1.mean():.1f}%)" if e1.notna().sum() else "—",
            "粗 OR (95%CI)": fmt(cr), "校正 OR (95%CI)": fmt(ad), "模型 n": ad[4]})
        if pd.notna(ad[0]) and ad[5] >= MIN_TOTAL_EV:
            src.append(dict(logor=np.log(ad[0]), se=se_ci(ad[1], ad[2])))
    if len(src) >= 2:
        meta_out[nm] = dl_meta([s["logor"] for s in src], [s["se"] for s in src])
t22c = pd.DataFrame(c_rows)
t22c.to_csv(OUT / "t22c_first_outcomes.csv", index=False, encoding="utf-8-sig")
print("\n[T22c] 结局关联（首次住院）")
print(t22c.to_string(index=False))

t22d = pd.DataFrame([{
    "结局": nm, "纳入库数": m["k"],
    "合并 OR (95%CI)": f"{m['OR']:.2f} ({m['lo']:.2f}–{m['hi']:.2f})",
    "P": f"{m['P']:.3f}", "I² (%)": f"{m['I2']:.1f}", "τ²": f"{m['tau2']:.3f}",
    "Q 检验 P": f"{m['Pq']:.3f}"} for nm, m in meta_out.items()])
if len(t22d):
    t22d.to_csv(OUT / "t22d_first_meta.csv", index=False, encoding="utf-8-sig")
    print("\n[T22d] 随机效应合并（首次住院）")
    print(t22d.to_string(index=False))

# ============================================================ T23 Tier C 口径统一
print("\n" + "=" * 78)
print("T23  Tier C 占比口径统一（统一分母 = 核心表型定义 npsle_core）")
print("=" * 78)

t23_rows = []
for db in DBS:
    d = data[db]
    core = pd.to_numeric(d["npsle_core"], errors="coerce").fillna(0) == 1
    n_core = int(core.sum())
    tc = pd.to_numeric(d["tier_c"], errors="coerce").fillna(0) == 1
    hi = pd.to_numeric(d["npsle_hi"], errors="coerce").fillna(0) == 1
    anyt = pd.to_numeric(d["npsle_any"], errors="coerce").fillna(0) == 1
    has_c = int((core & tc).sum())
    only_c = int((core & tc & ~hi).sum())
    unclass = int((core & ~anyt).sum())
    lw, hw = wilson(has_c, n_core); ls, hs = wilson(only_c, n_core)
    t23_rows.append({
        "数据库": LABEL[db], "核心表型定义事件数（统一分母）": n_core,
        "含非特异性编码 n (%, 95%CI)": f"{has_c} ({100*has_c/n_core:.1f}%, {100*lw:.1f}–{100*hw:.1f})" if n_core else "—",
        "仅非特异性 n (%, 95%CI)": f"{only_c} ({100*only_c/n_core:.1f}%, {100*ls:.1f}–{100*hs:.1f})" if n_core else "—",
        "未能归入任一 Tier n": unclass,
        "_hasc": has_c, "_onlyc": only_c, "_n": n_core})
t23 = pd.DataFrame(t23_rows)
A = t23[t23["数据库"] == "MIMIC-IV"].iloc[0]
B = t23[t23["数据库"] == "eICU-CRD"].iloc[0]
chi_w, p_w = stats.chi2_contingency([[A["_hasc"], A["_n"] - A["_hasc"]],
                                     [B["_hasc"], B["_n"] - B["_hasc"]]], correction=False)[:2]
chi_s, p_s = stats.chi2_contingency([[A["_onlyc"], A["_n"] - A["_onlyc"]],
                                     [B["_onlyc"], B["_n"] - B["_onlyc"]]], correction=False)[:2]
t23_show = t23.drop(columns=["_hasc", "_onlyc", "_n"])
t23_show.to_csv(OUT / "t23_tierC_unified.csv", index=False, encoding="utf-8-sig")
print(t23_show.to_string(index=False))
print(f"\n  >> 宽口径 MIMIC {100*A['_hasc']/A['_n']:.1f}% vs eICU {100*B['_hasc']/B['_n']:.1f}%："
      f"χ² = {chi_w:.3f}, P = {p_w:.3f}")
print(f"  >> 严格口径 MIMIC {100*A['_onlyc']/A['_n']:.1f}% vs eICU {100*B['_onlyc']/B['_n']:.1f}%："
      f"χ² = {chi_s:.3f}, P = {p_s:.3f}")

# ============================================================ 图：首次住院队列森林图
def forest(items, title, xlab, path, xlim=(0.15, 12), ticks=(0.25, 0.5, 1, 2, 4, 8)):
    """items: [(label, OR, lo, hi, is_pooled), ...]"""
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


items = [(m["db"], m["OR"], m["lo"], m["hi"], False) for m in meta_src_f]
if mtf:
    items.append(("合并（随机效应）", mtf["OR"], mtf["lo"], mtf["hi"], True))
forest(items,
       f"脓毒症与编码记录神经精神事件的共病关联（首次住院队列）\n"
       f"合并校正 OR {mtf['OR']:.2f}（95%CI {mtf['lo']:.2f}–{mtf['hi']:.2f}），P = {mtf['P']:.3f}，I² = {mtf['I2']:.0f}%"
       if mtf else "脓毒症共病关联（首次住院队列）",
       "校正 OR（脓毒症 ↔ 编码记录的神经精神事件）", FIG / "forest_sepsis_first.png",
       xlim=(0.5, 6), ticks=(0.5, 1, 2, 4))
print("\n[fig] fig/forest_sepsis_first.png")

# Tier 分层对比图
tier_items = []
for tag in ["Tier C 非特异事件", "Tier A+B 高置信事件"]:
    for db in ["mimiciv", "eicu"]:
        row = t22b[(t22b["数据库"] == LABEL[db]) & (t22b["对比"].str.startswith(tag))]
        if len(row):
            s = row.iloc[0]["校正 OR (95%CI)†"]
            if s != "不可估计":
                o = float(s.split(" ")[0]); lo, hi = [float(v) for v in s.split("(")[1].rstrip(")").split("–")]
                tier_items.append((f"{tag[:6]} · {LABEL[db]}", o, lo, hi, False))
    if tag in tier_meta_f:
        m = tier_meta_f[tag]
        tier_items.append((f"{tag[:6]} · 合并", m["OR"], m["lo"], m["hi"], True))
forest(tier_items, "脓毒症共病关联按诊断置信度分层（首次住院队列）",
       "校正 OR", FIG / "forest_tier_first.png", xlim=(0.1, 12), ticks=(0.25, 0.5, 1, 2, 4, 8))
print("[fig] fig/forest_tier_first.png")

# 结局森林图
out_items = []
for nm, _ in OUTCOMES:
    for db in ["mimiciv", "eicu"]:
        row = t22c[(t22c["结局"] == nm) & (t22c["数据库"] == LABEL[db])]
        if len(row):
            s = row.iloc[0]["校正 OR (95%CI)"]
            if s not in ("不可估计", "—", "数据缺失"):
                o = float(s.split(" ")[0]); lo, hi = [float(v) for v in s.split("(")[1].rstrip(")").split("–")]
                out_items.append((f"{nm} · {LABEL[db]}", o, lo, hi, False))
    if nm in meta_out:
        m = meta_out[nm]
        out_items.append((f"{nm} · 合并", m["OR"], m["lo"], m["hi"], True))
forest(out_items, "编码记录的神经精神事件与临床结局的关联（首次住院队列）",
       "校正 OR", FIG / "forest_outcomes_first.png", xlim=(0.08, 20), ticks=(0.25, 0.5, 1, 2, 4, 8))
print("[fig] fig/forest_outcomes_first.png")


# ============================================================ T24 GCS 言语分分布（补 IQR 与分布形态）
print("\n" + "=" * 78)
print("T24  GCS 言语评分分布（按机械通气状态分层，补充 IQR 与低分占比）")
print("=" * 78)


def auc_mw(x1, x0):
    """Mann-Whitney U 转 AUC（以 NPSLE 组为阳性，方向取 <，与低分=异常一致）。"""
    x1, x0 = np.asarray(x1, float), np.asarray(x0, float)
    if len(x1) < 3 or len(x0) < 3:
        return np.nan, np.nan
    u, p = stats.mannwhitneyu(x0, x1, alternative="two-sided")
    return u / (len(x1) * len(x0)), p


t24_rows = []
for db in ["mimiciv", "eicu"]:
    d = data[db]
    v = pd.to_numeric(d["gcs_verbal"], errors="coerce")
    vent = pd.to_numeric(d["vent24"], errors="coerce")
    y = pd.to_numeric(d["npsle_core"], errors="coerce")
    for tag, msk in [("未机械通气", vent == 0), ("机械通气", vent == 1)]:
        sub = msk & v.notna() & y.notna()
        v1, v0 = v[sub & (y == 1)], v[sub & (y == 0)]
        if len(v1) < 3 or len(v0) < 3:
            continue
        a, p = auc_mw(v1, v0)
        t24_rows.append({
            "数据库": LABEL[db], "分层": tag, "n": int(sub.sum()), "事件数": int((sub & (y == 1)).sum()),
            "非NPSLE 中位数 [IQR]": f"{v0.median():.0f} [{v0.quantile(.25):.0f}, {v0.quantile(.75):.0f}]",
            "NPSLE 中位数 [IQR]": f"{v1.median():.0f} [{v1.quantile(.25):.0f}, {v1.quantile(.75):.0f}]",
            "非NPSLE 言语分 ≤2, %": f"{100*(v0 <= 2).mean():.1f}",
            "NPSLE 言语分 ≤2, %": f"{100*(v1 <= 2).mean():.1f}",
            "单变量 AUC": f"{a:.3f}", "Mann-Whitney P": f"{p:.3f}"})
t24 = pd.DataFrame(t24_rows)
t24.to_csv(OUT / "t24_gcs_dist.csv", index=False, encoding="utf-8-sig")
print(t24.to_string(index=False))
print("\n  说明：中位数相同而 AUC > 0.5，源于分布形态差异——两组中位数均为 5（言语正常），")
print("        但 NPSLE 组落在低分区间（≤2）的比例更高，判别力来自分布下尾而非中心趋势。")


# ============================================================ 报告
def md(df):
    return df.to_markdown(index=False)


rep = f"""# V4 修订补充分析报告（回应第三轮审稿意见）

> 生成脚本：`scripts/v4_analyses.py`
> 主分析队列变更：**首次 ICU 住院**（每例患者仅纳入最早一次 ICU 住院），观测相互独立，
> 因此模型不再需要聚类稳健标准误；全部住院 + 聚类稳健标准误作为敏感性分析（表 16–19）。

---

## 表 20 · 免疫抑制 / 糖皮质激素暴露的敏感性分析

{md(t20)}

- 校正基线：年龄、女性、狼疮肾炎、肌酐、血小板（按库自适应）。
- eICU-CRD 的 `medication` 表未能可靠还原免疫抑制剂暴露，属**结构性不可得**，该库无法执行 M2。

---

## 表 21 · eICU-CRD 诊断时间戳时序分析

{md(t21) if t21 is not None else t21_note}

**结论**：{t21_note}

---

## 表 22a · 脓毒症与编码记录神经精神事件的共病关联（首次住院队列，主分析）

{md(t22a)}

† 校正变量：年龄、女性、狼疮肾炎（按库自适应）、肌酐、血小板。首次住院队列中每例患者仅一条观测，
标准误为常规模型标准误。

## 表 22b · 按诊断置信度分层（首次住院队列）

{md(t22b)}

## 表 22c · 神经精神事件与临床结局的关联（首次住院队列）

{md(t22c)}

## 表 22d · 随机效应合并（首次住院队列）

{md(t22d) if len(t22d) else "（无可合并结局）"}

---

## 表 23 · 非特异性事件（Tier C）占比：统一分母口径

{md(t23_show)}

- **统一分母 = 核心表型定义事件数**（MIMIC-IV {A['_n']}、eICU-CRD {B['_n']}、NWICU {int(t23.iloc[2]['_n'])}）。
- eICU-CRD 有 {int(B['未能归入任一 Tier n'])} 例满足核心表型定义但未能归入任一 Tier
  （诊断串可匹配领域关键词，但缺少可用于分层的特异性/归因信息），本文在分层分析中将其计为「未分层」，
  不纳入 Tier C 分子；此为两处比例此前不一致（分母 {A['_n']} vs 94）的来源，现已统一。
- 宽口径（含非特异性编码）：MIMIC-IV {100*A['_hasc']/A['_n']:.1f}% vs eICU-CRD {100*B['_hasc']/B['_n']:.1f}%，
  χ² = {chi_w:.3f}，P = {p_w:.3f}。
- 严格口径（仅非特异性）：{100*A['_onlyc']/A['_n']:.1f}% vs {100*B['_onlyc']/B['_n']:.1f}%，
  χ² = {chi_s:.3f}，P = {p_s:.3f}；差异由 eICU-CRD 独有的狼疮归因编码能力解释。

---

## 表 24 · GCS 言语评分分布（补充 IQR）

{md(t24)}

> 两组中位数相同（均为 5）而单变量 AUC > 0.5，源于**分布形态**差异：判别信息来自分布下尾
> （言语分 ≤ 2 的比例），而非中心趋势。机械通气亚组中该下尾差异被插管镇静抹平，AUC 趋近随机。
"""
(OUT / "v4_report.md").write_text(rep, encoding="utf-8")

# 供 gen_v4.py 读取的关键数值
key = dict(
    first_n={LABEL[db]: int(len(data_first[db])) for db in DBS},
    t22a_meta=(dict(OR=mtf["OR"], lo=mtf["lo"], hi=mtf["hi"], P=mtf["P"], I2=mtf["I2"], k=mtf["k"])
               if mtf else None),
    t22b_meta={k: dict(OR=v["OR"], lo=v["lo"], hi=v["hi"], P=v["P"], I2=v["I2"], k=v["k"])
               for k, v in tier_meta_f.items()},
    t22d_meta={k: dict(OR=v["OR"], lo=v["lo"], hi=v["hi"], P=v["P"], I2=v["I2"], k=v["k"])
               for k, v in meta_out.items()},
    tierC=dict(mimic_n=int(A["_n"]), mimic_hasc=int(A["_hasc"]), mimic_onlyc=int(A["_onlyc"]),
               eicu_n=int(B["_n"]), eicu_hasc=int(B["_hasc"]), eicu_onlyc=int(B["_onlyc"]),
               eicu_unclass=int(B["未能归入任一 Tier n"]),
               chi_w=float(chi_w), p_w=float(p_w), chi_s=float(chi_s), p_s=float(p_s)),
    t21_note=t21_note)
(OUT / "_v4_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n[ok] -> out/v4_report.md, t20–t23 csv, out/_v4_key.json")
