# -*- coding: utf-8 -*-
"""
V6 核查脚本（回应第五轮审稿意见第四节）
==================================================================
核查 1  表18 聚类稳健 CI 的来源：协变量集 × 估计量 全网格重算
        —— 判定 V3 的 0.44 (0.22–0.89) 与 V5 表18 的 0.38 (0.15–0.99) 差异归因
核查 2  三个 n 的对账：表14 (207) / 表18 (188) / 表8 (156)
核查 3  2.2 节 ICU LOS < 4h 排除例数
核查 4  Tier C 口径：首次住院 vs 全部住院 的比例差 + 95%CI 重叠情况

用法: python scripts/v6_verify.py
"""
import os, warnings, pathlib, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
os.environ["LC_ALL"] = "C"; os.environ["LANG"] = "C"; os.environ["PGCLIENTENCODING"] = "UTF8"

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

DBS = ["mimiciv", "eicu", "nwicu"]
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
SEV = {"mimiciv": "sofa24", "eicu": "apache", "nwicu": "sofa24"}

# V3 主表（表14）实际使用的协变量集：6 个，**不含**疾病严重度评分
ADJ_V3 = ["age", "female", "sepsis_dx", "lupus_nephritis", "creat", "plt"]
# V5 模型 A 协变量集：基础 5 个 + 严重度 + 脓毒症 = 7 个
ADJ_BASE = ["age", "female", "lupus_nephritis", "creat", "plt"]


def load(db):
    c = pd.read_csv(OUT / f"cohort_{db}.csv")
    t = pd.read_csv(OUT / f"tier_{db}.csv")
    # `npsle_core` now exists in BOTH tables; keep the cohort-derived column
    # under its own name and carry the tier copy as `*_tier`.
    d = c.merge(t, on="stay_id", how="left", suffixes=("", "_tier"))
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
        return (np.nan,) * 4 + (len(m), nev, len(adjust), np.nan)
    X = sm.add_constant(m[[x] + adjust], has_constant="add")
    try:
        mod = sm.Logit(m[y].astype(float), X)
        res = (mod.fit(disp=0, method="bfgs", maxiter=400,
                       cov_type="cluster", cov_kwds={"groups": m["_grp"]})
               if cluster and "_grp" in m.columns else
               mod.fit(disp=0, method="bfgs", maxiter=400))
        b, se = res.params[x], res.bse[x]
        ngrp = int(m["_grp"].nunique()) if "_grp" in m.columns else np.nan
        return (float(np.exp(b)), float(np.exp(b - 1.96 * se)),
                float(np.exp(b + 1.96 * se)), float(res.pvalues[x]),
                len(m), nev, len(adjust), ngrp, float(se))
    except Exception as e:
        return (np.nan,) * 4 + (len(m), nev, len(adjust), np.nan, np.nan)


print("=" * 96)
print("核查 1  表18 CI 溯源：协变量集 × 估计量 2×2 全网格（全部住院队列）")
print("=" * 96)

grid = []
for db in ["mimiciv", "eicu"]:
    d = data[db]
    for aname, adj in [("V3 六协变量（无严重度评分）", ADJ_V3),
                       ("模型A 七协变量（+严重度评分）", ADJ_BASE + [SEV[db], "sepsis_dx"])]:
        for sname, clus in [("独立 SE", False), ("按患者聚类稳健 SE", True)]:
            for oname, ycol in [("院内死亡", "hosp_mort"), ("ICU 住院 >7 天", "prolonged_icu"),
                                ("24h 内机械通气", "vent24")]:
                r = fit_or(d, ycol, "npsle_core", adjust=adj, cluster=clus)
                grid.append({
                    "数据库": LABEL[db], "结局": oname, "协变量集": aname, "估计量": sname,
                    "OR (95%CI)": (f"{r[0]:.2f} ({r[1]:.2f}–{r[2]:.2f})"
                                   if pd.notna(r[0]) else "不可估计"),
                    "P": f"{r[3]:.3f}" if pd.notna(r[3]) else "—",
                    "n": r[4], "事件数": r[5], "协变量数": r[6],
                    "簇数(患者)": r[7] if pd.notna(r[7]) else "—",
                    "SE(logOR)": f"{r[8]:.4f}" if len(r) > 8 and pd.notna(r[8]) else "—"})
g = pd.DataFrame(grid)
g.to_csv(OUT / "t31_table18_provenance.csv", index=False, encoding="utf-8-sig")
for oname in ["院内死亡", "ICU 住院 >7 天", "24h 内机械通气"]:
    print(f"\n--- {oname} ---")
    print(g[g["结局"] == oname].drop(columns=["结局"]).to_string(index=False))

print("\n>> 关键判定（MIMIC-IV 院内死亡）：")
sub = g[(g["数据库"] == "MIMIC-IV") & (g["结局"] == "院内死亡")]
for _, r in sub.iterrows():
    print(f"   {r['协变量集']:<28} | {r['估计量']:<18} | {r['OR (95%CI)']:<22} | n={r['n']}")

print("\n" + "=" * 96)
print("核查 2  三个 n 的对账（eICU-CRD 院内死亡）")
print("=" * 96)
d = data["eicu"]; df1 = data_first["eicu"]
recon = []
for cname, dd in [("全部住院", d), ("首次住院", df1)]:
    for aname, adj in [("V3 六协变量（无 APACHE）", ADJ_V3),
                       ("模型A 七协变量（+APACHE）", ADJ_BASE + ["apache", "sepsis_dx"])]:
        r = fit_or(dd, "hosp_mort", "npsle_core", adjust=adj, cluster=False)
        recon.append({"队列": cname, "协变量集": aname, "模型 n": r[4], "事件数": r[5],
                      "OR (95%CI)": f"{r[0]:.2f} ({r[1]:.2f}–{r[2]:.2f})" if pd.notna(r[0]) else "不可估计"})
rc = pd.DataFrame(recon)
print(rc.to_string(index=False))

# 逐变量缺失贡献
print("\n>> eICU 全部住院队列各协变量缺失例数（总 n = %d）：" % len(d))
miss = []
for c in ["hosp_mort", "npsle_core"] + ADJ_BASE + ["apache", "sepsis_dx"]:
    if c in d.columns:
        nm = int(pd.to_numeric(d[c], errors="coerce").isna().sum())
        miss.append({"变量": c, "缺失 n": nm, "非缺失 n": len(d) - nm})
mm = pd.DataFrame(miss)
print(mm.to_string(index=False))
rc.to_csv(OUT / "t32_n_reconcile.csv", index=False, encoding="utf-8-sig")
mm.to_csv(OUT / "t32b_eicu_missing.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 96)
print("核查 3  ICU 住院时间 < 4 h 的排除例数")
print("=" * 96)
ex = []
for db in DBS:
    try:
        raw = pd.read_csv(OUT / f"raw_{db}.csv")
    except Exception:
        raw = None
    d = data[db]
    los = pd.to_numeric(d["icu_los"], errors="coerce")
    ex.append({"数据库": LABEL[db], "最终队列 n": len(d),
               "队列内 ICU LOS 最小值 (天)": f"{los.min():.3f}" if los.notna().any() else "—",
               "队列内 <4h (0.167d) 例数": int((los < 4 / 24).sum()),
               "原始表可得": "是" if raw is not None else "否（排除在建队列阶段完成，见 build_cohort 日志）"})
exd = pd.DataFrame(ex)
print(exd.to_string(index=False))
exd.to_csv(OUT / "t33_los_exclusion.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 96)
print("核查 4  Tier C 占比：两口径的比例差与 95%CI 重叠")
print("=" * 96)


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))


tc = []
for cname, dset in [("全部住院", data), ("首次住院（主分析）", data_first)]:
    vals = {}
    for db in ["mimiciv", "eicu"]:
        df = dset[db]
        core = pd.to_numeric(df["npsle_core"], errors="coerce").fillna(0) == 1
        cmask = core & (pd.to_numeric(df["tier_c"], errors="coerce").fillna(0) == 1)
        k, n = int(cmask.sum()), int(core.sum())
        lo, hi = wilson(k, n)
        vals[db] = (k, n, lo, hi)
        tc.append({"队列": cname, "数据库": LABEL[db], "Tier C / 核心事件": f"{k}/{n}",
                   "占比 %": f"{100*k/n:.1f}", "95%CI (Wilson)": f"{100*lo:.1f}–{100*hi:.1f}"})
    (ka, na, loa, hia) = vals["mimiciv"]; (kb, nb, lob, hib) = vals["eicu"]
    chi, p = stats.chi2_contingency([[ka, na - ka], [kb, nb - kb]], correction=False)[:2]
    diff = 100 * (ka / na - kb / nb)
    # 两比例差的 95%CI（Wald）
    pa, pb = ka / na, kb / nb
    sed = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    tc.append({"队列": cname, "数据库": "— 组间比较 —",
               "Tier C / 核心事件": f"χ²={chi:.3f}",
               "占比 %": f"差 {diff:+.1f}",
               "95%CI (Wilson)": f"差值95%CI {100*(pa-pb-1.96*sed):+.1f}–{100*(pa-pb+1.96*sed):+.1f}；P={p:.3f}"})
    print(f"\n[{cname}] MIMIC {100*pa:.1f}% ({ka}/{na}, CI {100*loa:.1f}–{100*hia:.1f}) vs "
          f"eICU {100*pb:.1f}% ({kb}/{nb}, CI {100*lob:.1f}–{100*hib:.1f})")
    print(f"          比例差 {diff:+.1f} 个百分点，95%CI {100*(pa-pb-1.96*sed):+.1f}–{100*(pa-pb+1.96*sed):+.1f}，"
          f"χ² = {chi:.3f}，P = {p:.3f}")
    print(f"          两库 Wilson CI 是否重叠：{'是' if min(hia, hib) >= max(loa, lob) else '否'}"
          f"（重叠区间 {100*max(loa,lob):.1f}–{100*min(hia,hib):.1f}）")

tcd = pd.DataFrame(tc)
tcd.to_csv(OUT / "t34_tierC_two_cohorts.csv", index=False, encoding="utf-8-sig")

# 事后功效（首次住院口径，检验两比例差）
pa_, pb_ = 47 / 61, 50 / 77
from scipy.stats import norm
n1, n2 = 61, 77
pbar = (47 + 50) / (61 + 77)
se0 = np.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
se1 = np.sqrt(pa_ * (1 - pa_) / n1 + pb_ * (1 - pb_) / n2)
zc = norm.ppf(0.975)
pw = 1 - norm.cdf((zc * se0 - abs(pa_ - pb_)) / se1) + norm.cdf((-zc * se0 - abs(pa_ - pb_)) / se1)
print(f"\n>> 首次住院口径事后功效（检出 {100*abs(pa_-pb_):.1f} 个百分点差异，α=0.05 双侧）≈ {100*pw:.1f}%")
print(f">> 达到 80% 功效所需每组样本量 ≈ "
      f"{int(np.ceil(2*((zc+norm.ppf(0.8))**2)*pbar*(1-pbar)/(pa_-pb_)**2))} 例/组")

json.dump({"post_hoc_power_first": float(pw)},
          open(OUT / "_v6_power.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n[ok] -> t31–t34 csv, out/_v6_power.json")
