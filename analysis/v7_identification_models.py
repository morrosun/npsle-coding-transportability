# -*- coding: utf-8 -*-
"""
v7 — identification models, re-run with the clustering structure respected.

Changes relative to part2_model.py
----------------------------------
1. Cross-validation is GROUPED BY PATIENT: five repeats of five-fold
   StratifiedGroupKFold, each repeat with a prespecified random seed.  The
   v6 run used RepeatedStratifiedKFold on a cohort in which one patient can
   contribute several ICU stays, so the same patient could appear in both the
   training and the validation fold.
2. AUC confidence intervals come from a PATIENT-level bootstrap: whole patients
   are resampled with replacement and all of their stays travel together.
3. A first-stay sensitivity analysis is added, fitted on the same per-database
   variable set, so that the clustering question and the cohort question can be
   separated.

Writes the same t5-t10 files as before, so the manuscript generator picks them
up unchanged, plus t5b_first_stay.csv and t11_grouping_audit.csv.
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
import statsmodels.api as sm
import xgboost as xgb
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npsle_io

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcfamily = plt.rcParams["font.sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, FIG = ROOT + "/out", ROOT + "/fig"
os.makedirs(FIG, exist_ok=True)
RNG = 20260731
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
FIRST = json.load(open(OUT + "/_first_stays.json", encoding="utf-8"))
FKEY = {"mimiciv": "mimic_first", "eicu": "eicu_first", "nwicu": "nwicu_first"}

# Grouped cross-validation.  scikit-learn provides StratifiedGroupKFold but has
# no RepeatedStratifiedGroupKFold class, so repetition is implemented explicitly
# as n_repeats independent StratifiedGroupKFold splits, each with a prespecified
# random seed, always grouped by patient.
from sklearn.model_selection import StratifiedGroupKFold
_GROUPED = True

FEATS_MAIN = ["age", "female", "gcs_min", "hr", "map", "temp", "spo2", "wbc",
              "creat", "sepsis_dx"]
FEATS_NOGCS = [f for f in FEATS_MAIN if f != "gcs_min"]
BINARY = ["female", "sepsis_dx"]
LOGVAR = ["wbc", "creat"]
RCSVAR = ["age", "gcs_min"]
CN = {"age": "年龄", "female": "女性", "gcs_min": "GCS 最低值", "hr": "心率",
      "map": "平均动脉压", "temp": "体温", "spo2": "SpO₂ 最低",
      "wbc": "白细胞(对数)", "creat": "肌酐(对数)", "sepsis_dx": "脓毒症"}


def rcs_knots(x, k=3):
    x = pd.to_numeric(x, errors="coerce").dropna()
    qs = {3: [.10, .50, .90], 4: [.05, .35, .65, .95]}[k]
    kn = np.unique(np.quantile(x, qs))
    return kn if len(kn) == k else None


def rcs_basis(x, kn):
    x = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)
    k = len(kn)
    t1, tkm1, tk = kn[0], kn[-2], kn[-1]
    denom = (tk - t1) ** 2
    cols = [x]
    cub = lambda z: np.where(z > 0, z ** 3, 0.0)
    for j in range(k - 2):
        tj = kn[j]
        cols.append((cub(x - tj) - cub(x - tkm1) * (tk - tj) / (tk - tkm1)
                     + cub(x - tk) * (tkm1 - tj) / (tk - tkm1)) / denom)
    return np.column_stack(cols)


class Prep:
    """Medians and spline knots are learned on the TRAINING FOLD only."""

    def __init__(self, feats, use_rcs=True):
        self.feats, self.use_rcs = list(feats), use_rcs

    def fit(self, df):
        self.med_, self.knots_ = {}, {}
        for f in self.feats:
            s = pd.to_numeric(df[f], errors="coerce")
            if f in LOGVAR:
                s = np.log(np.clip(s, 1e-2, None))
            self.med_[f] = float(s.median()) if s.notna().any() else 0.0
            if self.use_rcs and f in RCSVAR:
                kn = rcs_knots(s.fillna(self.med_[f]))
                if kn is not None:
                    self.knots_[f] = kn
        return self

    def transform(self, df):
        mats, names = [], []
        for f in self.feats:
            s = pd.to_numeric(df[f], errors="coerce")
            if f in LOGVAR:
                s = np.log(np.clip(s, 1e-2, None))
            s = s.fillna(self.med_[f]).values.astype(float)
            if f in self.knots_:
                B = rcs_basis(s, self.knots_[f])
                mats.append(B)
                names += [CN.get(f, f)] + ["%s′(样条)" % CN.get(f, f)
                                            for _ in range(B.shape[1] - 1)]
            else:
                mats.append(s.reshape(-1, 1))
                names.append(CN.get(f, f))
        self.names_ = names
        return np.column_stack(mats)

    def transform_raw(self, df):
        cols = []
        for f in self.feats:
            s = pd.to_numeric(df[f], errors="coerce")
            if f in LOGVAR:
                s = np.log(np.clip(s, 1e-2, None))
            cols.append(s.fillna(self.med_[f]).values.astype(float))
        return np.column_stack(cols)


def fit_logit(X, y):
    Xc = sm.add_constant(X, has_constant="add")
    try:
        return sm.Logit(y, Xc).fit(disp=0, method="bfgs", maxiter=500)
    except Exception:
        return None


def pred_logit(res, X):
    return None if res is None else res.predict(
        sm.add_constant(X, has_constant="add"))


def fit_xgb_model(X, y):
    m = xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                          min_child_weight=5, eval_metric="logloss",
                          random_state=RNG, n_jobs=4)
    m.fit(X, y)
    return m


# ------------------------------------------------- patient-level bootstrap AUC CI
def auc_ci_patient(y, p, grp, n_boot=1000, seed=RNG):
    y, p = np.asarray(y), np.asarray(p)
    grp = np.asarray(grp)
    a = roc_auc_score(y, p)
    rs = np.random.RandomState(seed)
    uniq = np.unique(grp)
    idx_by_g = {g: np.where(grp == g)[0] for g in uniq}
    bs = []
    for _ in range(n_boot):
        pick = rs.choice(uniq, len(uniq), replace=True)
        i = np.concatenate([idx_by_g[g] for g in pick])
        if len(np.unique(y[i])) < 2:
            continue
        bs.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return a, lo, hi


def calibration(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    try:
        sl = sm.Logit(y, sm.add_constant(lp, has_constant="add")).fit(
            disp=0, method="bfgs").params[1]
    except Exception:
        sl = np.nan
    try:
        it = sm.Logit(y, np.ones((len(y), 1)), offset=lp).fit(
            disp=0, method="bfgs").params[0]
    except Exception:
        it = np.nan
    return sl, it


def eval_all(y, p, grp):
    a, lo, hi = auc_ci_patient(y, p, grp)
    sl, it = calibration(y, p)
    return {"auc": a, "auc_lo": lo, "auc_hi": hi,
            "brier": brier_score_loss(y, p), "slope": sl, "citl": it}


data = {db: npsle_io.load(db) for db in LABEL}
data_first = {db: d[d["stay_id"].isin(set(FIRST[FKEY[db]]))].copy()
              for db, d in data.items()}


def internal_cv_grouped(df, feats, model="logit", n_splits=5, n_repeats=5,
                        target="npsle_core"):
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).values.astype(int)
    grp = df["subject_id"].astype(str).values
    oof, cnt = np.zeros(len(df)), np.zeros(len(df))
    X0 = np.zeros(len(df))
    # n_repeats independent five-fold StratifiedGroupKFold splits, grouped by
    # patient, each repeat using a prespecified random seed.
    splits = []
    for r in range(n_repeats):
        c2 = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                  random_state=RNG + r)
        splits += list(c2.split(X0, y, groups=grp))
    for tr, te in splits:
        pp = Prep(feats, use_rcs=(model == "logit")).fit(df.iloc[tr])
        if model == "logit":
            res = fit_logit(pp.transform(df.iloc[tr]), y[tr])
            pr = pred_logit(res, pp.transform(df.iloc[te]))
            if pr is None:
                continue
        else:
            pr = fit_xgb_model(pp.transform_raw(df.iloc[tr]),
                               y[tr]).predict_proba(
                                   pp.transform_raw(df.iloc[te]))[:, 1]
        oof[te] += pr
        cnt[te] += 1
    ok = cnt > 0
    return y[ok], (oof[ok] / cnt[ok]), grp[ok]


def external(df_tr, df_te, feats, model="logit"):
    ytr = df_tr["npsle_core"].values.astype(int)
    yte = df_te["npsle_core"].values.astype(int)
    pp = Prep(feats, use_rcs=(model == "logit")).fit(df_tr)
    if model == "logit":
        res = fit_logit(pp.transform(df_tr), ytr)
        return yte, pred_logit(res, pp.transform(df_te)), res, pp
    m = fit_xgb_model(pp.transform_raw(df_tr), ytr)
    return yte, m.predict_proba(pp.transform_raw(df_te))[:, 1], m, pp


rows, store, rows_first = [], {}, []
SETTINGS = [("主模型(含 GCS)", FEATS_MAIN, data, rows, "全部住院"),
            ("敏感性(无 GCS)", FEATS_NOGCS, data, rows, "全部住院"),
            ("首次住院敏感性(无 GCS)", FEATS_NOGCS, data_first, rows_first, "首次住院"),
            ("首次住院敏感性(含 GCS)", FEATS_MAIN, data_first, rows_first, "首次住院")]

for fname, feats, src, sink, basis in SETTINGS:
    for model, mcn in [("logit", "Logistic+RCS"), ("xgb", "XGBoost")]:
        for db in ["mimiciv", "eicu"]:
            y, p, g = internal_cv_grouped(src[db], feats, model)
            r = eval_all(y, p, g)
            store[(fname, mcn, "内部 · %s" % LABEL[db])] = (y, p, g)
            sink.append(dict(特征集=fname, 模型=mcn, 场景="内部 · %s" % LABEL[db],
                             队列口径=basis, 类型="内部", n=len(y),
                             患者数=len(np.unique(g)), 事件=int(y.sum()), **r))
        for a, b in [("mimiciv", "eicu"), ("eicu", "mimiciv")]:
            y, p, _, _ = external(src[a], src[b], feats, model)
            if p is None:
                continue
            g = src[b]["subject_id"].astype(str).values
            r = eval_all(y, p, g)
            store[(fname, mcn, "外部 · %s→%s" % (LABEL[a], LABEL[b]))] = (y, p, g)
            sink.append(dict(特征集=fname, 模型=mcn,
                             场景="外部 · %s→%s" % (LABEL[a], LABEL[b]),
                             队列口径=basis, 类型="外部", n=len(y),
                             患者数=len(np.unique(g)), 事件=int(y.sum()), **r))
        if fname == "敏感性(无 GCS)":
            for a in ["mimiciv", "eicu"]:
                y, p, _, _ = external(src[a], src["nwicu"], feats, model)
                if p is None or y.sum() < 2:
                    continue
                sink.append(dict(特征集=fname, 模型=mcn,
                                 场景="探索 · %s→NWICU" % LABEL[a], 队列口径=basis,
                                 类型="探索", n=len(y),
                                 患者数=len(np.unique(src["nwicu"]["subject_id"]
                                                      .astype(str))),
                                 事件=int(y.sum()), auc=roc_auc_score(y, p),
                                 auc_lo=np.nan, auc_hi=np.nan,
                                 brier=brier_score_loss(y, p), slope=np.nan,
                                 citl=np.nan))

t5 = pd.DataFrame(rows)
t5b = pd.DataFrame(rows_first)
fmt = lambda r: ("%.3f (%.3f–%.3f)" % (r.auc, r.auc_lo, r.auc_hi)
                 if pd.notna(r.auc_lo) else "%.3f (事件过少, 不估 CI)" % r.auc)
for t in (t5, t5b):
    t["AUC (95%CI)"] = t.apply(fmt, axis=1)
    t["Brier"] = t.brier.map(lambda v: "%.3f" % v)
    t["校准斜率"] = t.slope.map(lambda v: "%.2f" % v if pd.notna(v) else "—")
    t["校准截距"] = t.citl.map(lambda v: "%+.2f" % v if pd.notna(v) else "—")
cols = ["特征集", "模型", "场景", "n", "患者数", "事件", "AUC (95%CI)", "Brier",
        "校准斜率", "校准截距"]
t5[cols].to_csv(OUT + "/t5_model_perf.csv", index=False, encoding="utf-8-sig")
t5b[cols].to_csv(OUT + "/t5b_first_stay_perf.csv", index=False, encoding="utf-8-sig")

gap = []
for fname in t5.特征集.unique():
    for mcn in t5.模型.unique():
        s = t5[(t5.特征集 == fname) & (t5.模型 == mcn)]
        for a, b in [("MIMIC-IV", "eICU-CRD"), ("eICU-CRD", "MIMIC-IV")]:
            ii = s[s.场景 == "内部 · %s" % a]
            ee = s[s.场景 == "外部 · %s→%s" % (a, b)]
            if not len(ii) or not len(ee):
                continue
            gap.append({"特征集": fname, "模型": mcn, "训练库": a, "验证库": b,
                        "内部 AUC": "%.3f" % ii.auc.iloc[0],
                        "外部 AUC": "%.3f" % ee.auc.iloc[0],
                        "AUC 落差": "%+.3f" % (ii.auc.iloc[0] - ee.auc.iloc[0]),
                        "外部校准斜率": ("%.2f" % ee.slope.iloc[0]
                                        if pd.notna(ee.slope.iloc[0]) else "—"),
                        "外部校准截距": ("%+.2f" % ee.citl.iloc[0]
                                        if pd.notna(ee.citl.iloc[0]) else "—")})
pd.DataFrame(gap).to_csv(OUT + "/t6_transport_gap.csv", index=False,
                         encoding="utf-8-sig")

# ---- coefficients
cr = []
for db in ["mimiciv", "eicu"]:
    pp = Prep(FEATS_MAIN, use_rcs=True).fit(data[db])
    res = fit_logit(pp.transform(data[db]), data[db]["npsle_core"].values.astype(int))
    if res is None:
        continue
    names = ["截距"] + pp.names_
    for i, nm in enumerate(names):
        b, se = res.params[i], res.bse[i]
        if nm == "截距" or not np.isfinite(se) or se > 5:
            continue
        cr.append({"数据库": LABEL[db], "变量": nm,
                   "OR (95%CI)": "%.3f (%.3f–%.3f)" % (np.exp(b),
                                                        np.exp(b - 1.96 * se),
                                                        np.exp(b + 1.96 * se)),
                   "P": ("<0.001" if res.pvalues[i] < 0.001 else "%.3f" % res.pvalues[i])})
pd.DataFrame(cr).to_csv(OUT + "/t7_coef.csv", index=False, encoding="utf-8-sig")

# ---- ceiling, phenotype specificity, GCS stratum (all on grouped CV)
ALL_CAND = ["age", "female", "hr", "map", "rr", "temp", "spo2", "wbc", "lymph_abs",
            "hgb", "plt", "creat", "alb", "na", "bili", "inr", "lactate", "gcs_min",
            "gcs_motor", "gcs_verbal", "gcs_eyes", "sofa24", "sofa_cns", "apache",
            "aps", "sepsis_dx", "lupus_nephritis", "steroid_any", "is_any",
            "hx_epilepsy", "vent24"]
ceil = []
for db in ["mimiciv", "eicu"]:
    d = data[db]
    allf = [c for c in ALL_CAND if c in d.columns
            and pd.to_numeric(d[c], errors="coerce").notna().sum() > len(d) * 0.3]
    ya, pa, _ = internal_cv_grouped(d, allf, "xgb")
    yb, pb, _ = internal_cv_grouped(d, FEATS_MAIN, "xgb")
    a_all, a_10 = roc_auc_score(ya, pa), roc_auc_score(yb, pb)
    ceil.append({"数据库": LABEL[db], "简约模型变量数": len(FEATS_MAIN),
                 "简约模型 AUC": "%.3f" % a_10, "全变量数": len(allf),
                 "全变量 AUC": "%.3f" % a_all, "增益": "%+.3f" % (a_all - a_10)})
pd.DataFrame(ceil).to_csv(OUT + "/t8_ceiling.csv", index=False, encoding="utf-8-sig")

PH = {"npsle_core": "核心 NPSLE（混合）", "dom_enceph": "急性意识障碍/脑病",
      "dom_seizure": "急性癫痫发作", "dom_cvd": "脑血管病"}
ph = []
for db in ["mimiciv", "eicu"]:
    d = data[db]
    for tgt in PH:
        if tgt not in d.columns:
            continue
        ev = int(pd.to_numeric(d[tgt], errors="coerce").fillna(0).sum())
        if ev < 20:
            ph.append({"数据库": LABEL[db], "目标表型": PH[tgt], "事件数": ev,
                       "CV-AUC": "事件过少，未评估"})
            continue
        yy, pp_, _ = internal_cv_grouped(d, FEATS_MAIN, "xgb", target=tgt)
        ph.append({"数据库": LABEL[db], "目标表型": PH[tgt], "事件数": ev,
                   "CV-AUC": "%.3f" % roc_auc_score(yy, pp_)})
pd.DataFrame(ph).to_csv(OUT + "/t9_phenotype_auc.csv", index=False, encoding="utf-8-sig")

gcs = []
for db in ["mimiciv", "eicu"]:
    d = data[db]
    if "vent24" not in d.columns or pd.to_numeric(d.vent24, errors="coerce").notna().sum() == 0:
        continue
    for v, vcn in [(0, "未机械通气"), (1, "机械通气")]:
        s = d[pd.to_numeric(d.vent24, errors="coerce") == v]
        gv = pd.to_numeric(s.get("gcs_verbal"), errors="coerce")
        ok = gv.notna()
        if ok.sum() < 30 or s.npsle_core[ok].nunique() < 2:
            continue
        a = roc_auc_score(s.npsle_core[ok], gv[ok])
        gcs.append({"数据库": LABEL[db], "分层": vcn, "n": len(s),
                    "事件": int(s.npsle_core.sum()),
                    "GCS-言语 中位（非NPSLE）": "%.0f" % gv[ok & (s.npsle_core == 0)].median(),
                    "GCS-言语 中位（NPSLE）": "%.0f" % gv[ok & (s.npsle_core == 1)].median(),
                    "GCS-言语=0 例数": int((gv[ok] == 0).sum()),
                    "单变量 AUC": "%.3f" % max(a, 1 - a)})
pd.DataFrame(gcs).to_csv(OUT + "/t10_gcs_strat.csv", index=False, encoding="utf-8-sig")

# ---- figures
C_IN, C_EX = "#1f4e79", "#c0392b"
F = "主模型(含 GCS)"


def plot_roc(keys, fn, title):
    fig, ax = plt.subplots(figsize=(5.6, 5.2), facecolor="white")
    for k, lab, col, ls in keys:
        if k not in store:
            continue
        y, p, g = store[k]
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=col, ls=ls, lw=2,
                label="%s (AUC=%.3f)" % (lab, roc_auc_score(y, p)))
    ax.plot([0, 1], [0, 1], color="#999", ls=":", lw=1)
    ax.set_xlabel("1 − 特异度")
    ax.set_ylabel("敏感度")
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG + "/" + fn, dpi=300, facecolor="white")
    plt.close()


def plot_calib(keys, fn, title):
    fig, ax = plt.subplots(figsize=(5.6, 5.2), facecolor="white")
    for k, lab, col, mk in keys:
        if k not in store:
            continue
        y, p, g = store[k]
        q = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=False)
        obs, exp = pd.Series(y).groupby(q).mean(), pd.Series(p).groupby(q).mean()
        ax.plot(exp, obs, marker=mk, color=col, lw=1.8, ms=7, label=lab)
    ax.plot([0, 1], [0, 1], color="#999", ls=":", lw=1)
    ax.set_xlabel("预测概率")
    ax.set_ylabel("实际观察比例")
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(fontsize=8.5, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG + "/" + fn, dpi=300, facecolor="white")
    plt.close()


def net_benefit(y, p, pt):
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    out = []
    for t in pt:
        f = p >= t
        tp, fp = np.sum(f & (y == 1)), np.sum(f & (y == 0))
        out.append(tp / n - (fp / n) * (t / (1 - t)))
    return np.array(out)


def plot_dca(keys, fn, title):
    pt = np.linspace(0.05, 0.75, 60)
    fig, ax = plt.subplots(figsize=(5.9, 5.2), facecolor="white")
    prev = None
    for k, lab, col, ls in keys:
        if k not in store:
            continue
        y, p, g = store[k]
        prev = np.mean(y)
        ax.plot(pt, net_benefit(y, p, pt), color=col, ls=ls, lw=2, label=lab)
    if prev is not None:
        ax.plot(pt, prev - (1 - prev) * (pt / (1 - pt)), color="#7f8c8d",
                ls="--", lw=1.3, label="全部干预")
    ax.axhline(0, color="#34495e", lw=1.3, label="全不干预")
    ax.set_ylim(-0.12, max(0.05, prev * 1.15) if prev else 0.05)
    ax.set_xlabel("阈值概率")
    ax.set_ylabel("净获益")
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG + "/" + fn, dpi=300, facecolor="white")
    plt.close()


for mcn, tag in [("Logistic+RCS", "logit"), ("XGBoost", "xgb")]:
    ks = [((F, mcn, "内部 · MIMIC-IV"), "内部 MIMIC-IV", C_IN, "-"),
          ((F, mcn, "内部 · eICU-CRD"), "内部 eICU-CRD", C_IN, "--"),
          ((F, mcn, "外部 · MIMIC-IV→eICU-CRD"), "外部 MIMIC→eICU", C_EX, "-"),
          ((F, mcn, "外部 · eICU-CRD→MIMIC-IV"), "外部 eICU→MIMIC", C_EX, "--")]
    plot_roc(ks, "roc_%s.png" % tag, "%s · 内部与外部 ROC" % mcn)
    plot_calib([(k, l, c, "o" if s == "-" else "s") for k, l, c, s in ks],
               "calib_%s.png" % tag, "%s · 校准曲线" % mcn)
    plot_dca([ks[0], ks[2]], "dca_%s.png" % tag,
             "%s · 决策曲线（MIMIC 内部 vs 外推 eICU）" % mcn)

try:
    import shap
    for db in ["mimiciv", "eicu"]:
        pp = Prep(FEATS_MAIN, use_rcs=False).fit(data[db])
        X = pp.transform_raw(data[db])
        m = fit_xgb_model(X, data[db]["npsle_core"].values.astype(int))
        sv = shap.TreeExplainer(m).shap_values(X)
        plt.figure(figsize=(6.4, 4.6), facecolor="white")
        shap.summary_plot(sv, X, feature_names=[CN.get(f, f) for f in FEATS_MAIN],
                          show=False, plot_size=None, max_display=10)
        plt.title("SHAP · %s 训练的 XGBoost" % LABEL[db], fontsize=11.5)
        plt.tight_layout()
        plt.savefig(FIG + "/shap_%s.png" % db, dpi=300, bbox_inches="tight",
                    facecolor="white")
        plt.close()
    shap_ok = True
except Exception as ex:
    print("[!] SHAP failed:", ex)
    shap_ok = False

# ---- grouping audit: how much did grouping matter?
aud = []
for db in ["mimiciv", "eicu"]:
    d = data[db]
    g = d["subject_id"].astype(str)
    aud.append(dict(数据库=LABEL[db], 住院数=len(d), 患者数=int(g.nunique()),
                    每院平均住院数=round(len(d) / max(g.nunique(), 1), 2),
                    多住院患者数=int((g.value_counts() > 1).sum()),
                    分组CV=("是" if _GROUPED else "否(回退)"),
                    bootstrap层级="患者"))
pd.DataFrame(aud).to_csv(OUT + "/t11_grouping_audit.csv", index=False,
                         encoding="utf-8-sig")

print("grouped CV available:", _GROUPED)
print(pd.DataFrame(aud).to_string(index=False))
print()
print(t5[cols].to_string(index=False))
print()
print("--- first-stay sensitivity ---")
print(t5b[cols].to_string(index=False))
print()
print(pd.DataFrame(ceil).to_string(index=False))
print(pd.DataFrame(gcs).to_string(index=False))
print("[ok] -> t5, t5b, t6-t11 csv, fig/*.png  shap=%s" % shap_ok)
