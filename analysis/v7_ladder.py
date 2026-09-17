#!/usr/bin/env python3
"""V7 补算：回应第六轮审稿意见。

T38  归因阶梯重建 —— 每一步只变一个因素（队列 / 估计量 / 协变量集）
     关键核实：表 8 模型 A 的真实协变量集 = ADJ_BASE(5) + 严重度评分 + 脓毒症 = 7 项
     （审稿人以为是 6 项，实为正文 2.6 节漏写严重度评分；表 18 脚注声明反而是正确的）
     补齐此前缺失的一格：首次住院 + 聚类稳健 SE + 模型 A

T39  表 6（Tier 分层脓毒症关联）补模型 n 列，交代 41+293=334 → n=328 的协变量缺失

输出 out/t38_mortality_ladder_v2.csv, out/t39_table6_with_n.csv, out/_v7_key.json
"""
import json, os, pathlib, warnings
import numpy as np, pandas as pd, statsmodels.api as sm

warnings.filterwarnings("ignore")
os.environ["LC_ALL"] = "C"

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

DBS = ["mimiciv", "eicu", "nwicu"]
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
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
    adjust = [a for a in (adjust or [])
              if a in d.columns
              and pd.to_numeric(d[a], errors="coerce").notna().sum() > 0
              and pd.to_numeric(d[a], errors="coerce").nunique(dropna=True) > 1]
    m = d[[y, x] + adjust].apply(pd.to_numeric, errors="coerce")
    if cluster and "subject_id" in d.columns:
        m = m.assign(_grp=pd.factorize(d["subject_id"].astype(str))[0])
    m = m.dropna()
    nev = int(m[y].sum()) if len(m) else 0
    ngrp = int(m["_grp"].nunique()) if "_grp" in m.columns and len(m) else np.nan
    bad = dict(OR=np.nan, lo=np.nan, hi=np.nan, P=np.nan, n=len(m), nev=nev,
               nadj=len(adjust), ngrp=ngrp, se=np.nan)
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
                    n=len(m), nev=nev, nadj=len(adjust), ngrp=ngrp, se=float(se))
    except Exception:
        return bad


def s(r):
    return f"{r['OR']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})" if pd.notna(r["OR"]) else "不可估计"


print("=" * 92)
print("T38  院内死亡归因阶梯重建（MIMIC-IV，npsle_core）—— 每步只变一个因素")
print("=" * 92)

db = "mimiciv"
d_all, d_fst = data[db], data_first[db]
COV6 = ADJ_BASE + ["sepsis_dx"]                       # 六协变量（无严重度）—— V3 口径
COV7A = ADJ_BASE + [SEV[db], "sepsis_dx"]             # 七协变量 = 表8 模型 A 真实口径
COV6B = ADJ_BASE + [SEV[db]]                          # 六协变量 = 表8 模型 B 真实口径

steps = [
    ("① 起点：全部住院 / 独立 SE / 六协变量（无严重度）",  d_all, COV6,  False, "—"),
    ("② 仅换估计量 → 按患者聚类稳健 SE",                  d_all, COV6,  True,  "估计量"),
    ("③ 仅换协变量集 → 七协变量（+严重度，= 模型 A）",     d_all, COV7A, True,  "协变量集"),
    ("④ 仅换队列 → 首次住院（每例 1 条观测）",             d_fst, COV7A, True,  "队列"),
    ("⑤ 首次住院 / 独立 SE（表 8 报告值，对照）",          d_fst, COV7A, False, "估计量"),
]

rows = []
for tag, dd, cov, clus, changed in steps:
    r = fit_or(dd, "hosp_mort", "npsle_core", adjust=cov, cluster=clus)
    rows.append({
        "步骤": tag,
        "本步改变的因素": changed,
        "队列": "全部住院" if dd is d_all else "首次住院",
        "估计量": "按患者聚类稳健 SE" if clus else "独立 SE",
        "协变量数": r["nadj"],
        "校正 OR (95%CI)": s(r),
        "P": f"{r['P']:.3f}" if pd.notna(r["P"]) else "—",
        "n": r["n"], "事件数": r["nev"],
        "簇数(患者)": "" if pd.isna(r["ngrp"]) else int(r["ngrp"]),
    })
    print(f"  {tag}\n      → {s(r)}  P={r['P']:.3f}  n={r['n']}  ev={r['nev']}  "
          f"adj={r['nadj']}  grp={r['ngrp']}")

# 模型 B（不含脓毒症）对照两行
for tag, dd, clus in [("全部住院 / 聚类稳健 / 模型 B（不含脓毒症）", d_all, True),
                      ("首次住院 / 独立 SE / 模型 B（不含脓毒症）", d_fst, False)]:
    r = fit_or(dd, "hosp_mort", "npsle_core", adjust=COV6B, cluster=clus)
    rows.append({
        "步骤": "对照：" + tag, "本步改变的因素": "协变量集（去脓毒症）",
        "队列": "全部住院" if dd is d_all else "首次住院",
        "估计量": "按患者聚类稳健 SE" if clus else "独立 SE",
        "协变量数": r["nadj"], "校正 OR (95%CI)": s(r),
        "P": f"{r['P']:.3f}" if pd.notna(r["P"]) else "—",
        "n": r["n"], "事件数": r["nev"],
        "簇数(患者)": "" if pd.isna(r["ngrp"]) else int(r["ngrp"])})
    print(f"  {tag}\n      → {s(r)}  P={r['P']:.3f}  n={r['n']}")

# 粗率两行
for tag, dd in [("全部住院", d_all), ("首次住院", d_fst)]:
    r = fit_or(dd, "hosp_mort", "npsle_core", adjust=None, cluster=False)
    mm = dd[["hosp_mort", "npsle_core"]].apply(pd.to_numeric, errors="coerce").dropna()
    p1 = mm.loc[mm["npsle_core"] == 1, "hosp_mort"].mean()
    p0 = mm.loc[mm["npsle_core"] == 0, "hosp_mort"].mean()
    rows.append({"步骤": f"粗率（未校正）：{tag}", "本步改变的因素": "—", "队列": tag,
                 "估计量": "独立 SE", "协变量数": 0,
                 "校正 OR (95%CI)": f"{s(r)}；NP组 {p1*100:.1f}% vs 非NP组 {p0*100:.1f}%",
                 "P": f"{r['P']:.3f}" if pd.notna(r["P"]) else "—",
                 "n": r["n"], "事件数": r["nev"], "簇数(患者)": ""})
    print(f"  粗率 {tag}: {s(r)}  NP {p1*100:.1f}% vs 非NP {p0*100:.1f}%")

t38 = pd.DataFrame(rows)
t38.to_csv(OUT / "t38_mortality_ladder_v2.csv", index=False, encoding="utf-8-sig")

print()
print("=" * 92)
print("T39  表 6（Tier 分层脓毒症共病）补模型 n —— 交代事件数+对照数 与 模型 n 的差")
print("=" * 92)

rows6 = []
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
        nev, nctl = int(sub["_y"].sum()), int((sub["_y"] == 0).sum())
        rows6.append({"数据库": LABEL[db], "对比": f"{tag} vs 无记录事件",
                      "事件数": nev, "对照数": nctl,
                      "亚队列合计": nev + nctl, "模型 n‡": r["n"],
                      "校正 OR (95%CI)†": s(r),
                      "P": f"{r['P']:.3f}" if pd.notna(r["P"]) else "—"})
        print(f"  {LABEL[db]:9s} {tag:18s} ev={nev:3d} ctl={nctl:3d} "
              f"合计={nev+nctl:3d} 模型n={r['n']:3d} 缺失={nev+nctl-r['n']:2d}  {s(r)}")

t39 = pd.DataFrame(rows6)
# 合并行沿用既有 t25a 结果（不重算 meta，避免与主文数字漂移）
t25a = pd.read_csv(OUT / "t25a_tier_with_pooled.csv")
pool = t25a[t25a["数据库"].astype(str).str.startswith("合并")].copy()
if len(pool):
    pr = []
    for _, x in pool.iterrows():
        pr.append({"数据库": x["数据库"], "对比": x["对比"], "事件数": "—", "对照数": "—",
                   "亚队列合计": "—", "模型 n‡": "—",
                   "校正 OR (95%CI)†": x["校正 OR (95%CI)†"], "P": x["P"]})
    t39 = pd.concat([t39, pd.DataFrame(pr)], ignore_index=True)
t39 = t39.sort_values(["对比", "数据库"]).reset_index(drop=True)
t39.to_csv(OUT / "t39_table6_with_n.csv", index=False, encoding="utf-8-sig")
print()
print(t39.to_string(index=False))

key = {
    "ladder": {r["步骤"]: r["校正 OR (95%CI)"] + f" P={r['P']}" for r in rows},
    "model_A_covariates": "年龄、性别、狼疮肾炎、肌酐、血小板、SOFA-24/APACHE IV、脓毒症诊断（MIMIC/NWICU 7 项；eICU 因狼疮肾炎不可获得为 6 项）",
    "model_B_covariates": "同模型 A 但不含脓毒症诊断（MIMIC/NWICU 6 项；eICU 5 项）",
}
(OUT / "_v7_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n[OK] 写出 t38_mortality_ladder_v2.csv / t39_table6_with_n.csv / _v7_key.json")
