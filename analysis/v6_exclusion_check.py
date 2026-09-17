# -*- coding: utf-8 -*-
"""
V6 追加核查：2.2 节纳入/排除标准与代码的一致性
==================================================================
发现：extract_cohort.py 从未实现「年龄 < 18 岁」与「ICU 住院 < 4 h」两条排除。
本脚本：
  (1) 量化两条标准若真正执行会剔除多少例；
  (2) 在「执行排除后」的队列上重跑全部主结果，验证结论是否改变；
  (3) 输出可直接写入稿件的敏感性分析表。

用法: python scripts/v6_exclusion_check.py
"""
import os, warnings, pathlib, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
os.environ["LC_ALL"] = "C"; os.environ["LANG"] = "C"

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

DBS = ["mimiciv", "eicu", "nwicu"]
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
SEV = {"mimiciv": "sofa24", "eicu": "apache", "nwicu": "sofa24"}
ADJ_BASE = ["age", "female", "lupus_nephritis", "creat", "plt"]
MIN_TOTAL_EV, MIN_EXP_EV_TIER = 10, 3


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


def fit_or(d, y, x, adjust=None, cluster=False):
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
        res = (mod.fit(disp=0, method="bfgs", maxiter=400, cov_type="cluster",
                       cov_kwds={"groups": m["_grp"]})
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
    Q = (w * (logor - fe) ** 2).sum(); k = len(logor)
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    wr = 1 / (se ** 2 + tau2)
    mu = (wr * logor).sum() / wr.sum(); semu = np.sqrt(1 / wr.sum())
    I2 = max(0.0, 100 * (Q - (k - 1)) / Q) if Q > 0 else 0.0
    z = mu / semu
    return dict(OR=np.exp(mu), lo=np.exp(mu - 1.96 * semu), hi=np.exp(mu + 1.96 * semu),
                P=2 * (1 - stats.norm.cdf(abs(z))), I2=I2, k=k)


def se_ci(lo, hi):
    return (np.log(hi) - np.log(lo)) / (2 * 1.96)


def fmt(r):
    return f"{r[0]:.2f} ({r[1]:.2f}–{r[2]:.2f})" if pd.notna(r[0]) else "不可估计"


print("=" * 92)
print("步骤 1  两条排除标准若真正执行的影响量化")
print("=" * 92)
imp = []
for db in DBS:
    d = data[db]
    age = pd.to_numeric(d["age"], errors="coerce")
    los = pd.to_numeric(d["icu_los"], errors="coerce")
    n_minor = int((age < 18).sum())
    n_short = int((los < 4 / 24).sum())
    n_either = int(((age < 18) | (los < 4 / 24)).sum())
    imp.append({"数据库": LABEL[db], "现有队列 n": len(d),
                "年龄 <18 岁 n": n_minor, "ICU LOS <4 h n": n_short,
                "满足任一排除条件 n": n_either, "执行排除后 n": len(d) - n_either,
                "年龄最小值": f"{age.min():.0f}" if age.notna().any() else "—",
                "LOS 最小值 (h)": f"{24*los.min():.2f}" if los.notna().any() else "—"})
impd = pd.DataFrame(imp)
print(impd.to_string(index=False))
impd.to_csv(OUT / "t35_exclusion_impact.csv", index=False, encoding="utf-8-sig")

# ---- 构造「执行排除后」队列
data_x = {}
for db in DBS:
    d = data[db]
    age = pd.to_numeric(d["age"], errors="coerce")
    los = pd.to_numeric(d["icu_los"], errors="coerce")
    keep = ~((age < 18) | (los < 4 / 24))
    data_x[db] = d[keep].copy()
first_x = {db: data_x[db][data_x[db]["stay_id"].astype(str).isin(first_ids[db])].copy() for db in DBS}
data_first = {db: data[db][data[db]["stay_id"].astype(str).isin(first_ids[db])].copy() for db in DBS}

print("\n执行排除后首次住院队列例数：",
      {LABEL[db]: len(first_x[db]) for db in DBS},
      "（原：", {LABEL[db]: len(data_first[db]) for db in DBS}, "）")

print("\n" + "=" * 92)
print("步骤 2  主结果在「执行排除后」队列的重算（首次住院口径）")
print("=" * 92)

rows = []

# --- 2.1 脓毒症 × 核心事件
for tag, dset in [("原队列（未执行排除）", data_first), ("执行排除后", first_x)]:
    src = []
    for db in DBS:
        r = fit_or(dset[db], "npsle_core", "sepsis_dx", adjust=ADJ_BASE, cluster=False)
        rows.append({"分析": "脓毒症 × 编码记录的神经精神事件", "队列版本": tag,
                     "数据库": LABEL[db], "OR (95%CI)": fmt(r),
                     "P": f"{r[3]:.3f}" if pd.notna(r[3]) else "—", "n": r[4]})
        if pd.notna(r[0]) and r[5] >= MIN_TOTAL_EV:
            src.append(dict(logor=np.log(r[0]), se=se_ci(r[1], r[2])))
    if len(src) >= 2:
        m = dl_meta([s["logor"] for s in src], [s["se"] for s in src])
        rows.append({"分析": "脓毒症 × 编码记录的神经精神事件", "队列版本": tag,
                     "数据库": f"合并 (k={m['k']})",
                     "OR (95%CI)": f"{m['OR']:.2f} ({m['lo']:.2f}–{m['hi']:.2f})",
                     "P": f"{m['P']:.3f}", "n": f"I²={m['I2']:.1f}%"})

# --- 2.2 Tier C / Tier A+B
for tag, dset in [("原队列（未执行排除）", data_first), ("执行排除后", first_x)]:
    for tname in ["Tier C 非特异事件", "Tier A+B 高置信事件"]:
        src = []
        for db in DBS:
            df = dset[db]
            no_ev = pd.to_numeric(df["npsle_core"], errors="coerce").fillna(0) == 0
            only_c = (pd.to_numeric(df["tier_c"], errors="coerce").fillna(0) == 1) & \
                     (pd.to_numeric(df["npsle_hi"], errors="coerce").fillna(0) == 0)
            hi = pd.to_numeric(df["npsle_hi"], errors="coerce").fillna(0) == 1
            msk = only_c if tname.startswith("Tier C") else hi
            sub = df[msk | no_ev].copy(); sub["_y"] = msk[msk | no_ev].astype(int)
            r = fit_or(sub, "_y", "sepsis_dx", adjust=ADJ_BASE, cluster=False)
            nexp = int(pd.to_numeric(sub.loc[sub["_y"] == 1, "sepsis_dx"], errors="coerce").sum())
            rows.append({"分析": f"{tname} × 脓毒症", "队列版本": tag, "数据库": LABEL[db],
                         "OR (95%CI)": fmt(r), "P": f"{r[3]:.3f}" if pd.notna(r[3]) else "—",
                         "n": r[4]})
            if pd.notna(r[0]) and int(sub["_y"].sum()) >= MIN_TOTAL_EV and nexp >= MIN_EXP_EV_TIER:
                src.append(dict(logor=np.log(r[0]), se=se_ci(r[1], r[2])))
        if len(src) >= 2:
            m = dl_meta([s["logor"] for s in src], [s["se"] for s in src])
            rows.append({"分析": f"{tname} × 脓毒症", "队列版本": tag, "数据库": f"合并 (k={m['k']})",
                         "OR (95%CI)": f"{m['OR']:.2f} ({m['lo']:.2f}–{m['hi']:.2f})",
                         "P": f"{m['P']:.3f}", "n": f"I²={m['I2']:.1f}%"})

# --- 2.3 结局（模型 A）
for tag, dset in [("原队列（未执行排除）", data_first), ("执行排除后", first_x)]:
    for onm, ycol in [("院内死亡", "hosp_mort"), ("24h 内机械通气", "vent24"),
                      ("ICU 住院 >7 天", "prolonged_icu")]:
        src = []
        for db in DBS:
            df = dset[db]
            if ycol not in df.columns or pd.to_numeric(df[ycol], errors="coerce").notna().sum() == 0:
                continue
            r = fit_or(df, ycol, "npsle_core", adjust=ADJ_BASE + [SEV[db], "sepsis_dx"], cluster=False)
            rows.append({"分析": f"结局：{onm}（模型A）", "队列版本": tag, "数据库": LABEL[db],
                         "OR (95%CI)": fmt(r), "P": f"{r[3]:.3f}" if pd.notna(r[3]) else "—",
                         "n": r[4]})
            if pd.notna(r[0]) and r[5] >= MIN_TOTAL_EV:
                src.append(dict(logor=np.log(r[0]), se=se_ci(r[1], r[2])))
        if len(src) >= 2:
            m = dl_meta([s["logor"] for s in src], [s["se"] for s in src])
            rows.append({"分析": f"结局：{onm}（模型A）", "队列版本": tag, "数据库": f"合并 (k={m['k']})",
                         "OR (95%CI)": f"{m['OR']:.2f} ({m['lo']:.2f}–{m['hi']:.2f})",
                         "P": f"{m['P']:.3f}", "n": f"I²={m['I2']:.1f}%"})

t36 = pd.DataFrame(rows)
t36.to_csv(OUT / "t36_exclusion_sensitivity.csv", index=False, encoding="utf-8-sig")
for a in t36["分析"].unique():
    print(f"\n--- {a} ---")
    print(t36[t36["分析"] == a].drop(columns=["分析"]).to_string(index=False))

# --- 2.4 Tier C 占比（执行排除后）
print("\n" + "=" * 92)
print("步骤 3  Tier C 占比在「执行排除后」队列")
print("=" * 92)
tc = []
for tag, dset in [("原队列（未执行排除）", data_first), ("执行排除后", first_x)]:
    v = {}
    for db in ["mimiciv", "eicu"]:
        df = dset[db]
        core = pd.to_numeric(df["npsle_core"], errors="coerce").fillna(0) == 1
        cm = core & (pd.to_numeric(df["tier_c"], errors="coerce").fillna(0) == 1)
        v[db] = (int(cm.sum()), int(core.sum()))
        tc.append({"队列版本": tag, "数据库": LABEL[db],
                   "Tier C/核心": f"{int(cm.sum())}/{int(core.sum())}",
                   "占比 %": f"{100*cm.sum()/core.sum():.1f}" if core.sum() else "—"})
    (ka, na), (kb, nb) = v["mimiciv"], v["eicu"]
    chi, p = stats.chi2_contingency([[ka, na - ka], [kb, nb - kb]], correction=False)[:2]
    tc.append({"队列版本": tag, "数据库": "— 组间比较 —", "Tier C/核心": f"χ²={chi:.3f}",
               "占比 %": f"P={p:.3f}"})
    print(f"[{tag}] MIMIC {100*ka/na:.1f}% ({ka}/{na}) vs eICU {100*kb/nb:.1f}% ({kb}/{nb})，P = {p:.3f}")
pd.DataFrame(tc).to_csv(OUT / "t37_tierC_exclusion.csv", index=False, encoding="utf-8-sig")

print("\n[ok] -> t35–t37 csv")
