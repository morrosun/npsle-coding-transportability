#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 2 · NPSLE 诊断识别模型与双向外部验证
用法: python part2_model.py

设计
----
目标变量 : npsle_core (ACR 核心 5 域, 排除脑血管病与周围神经病变)
建模人群 : MIMIC-IV (n=645, 事件 123) 与 eICU-CRD (n=230, 事件 100)
特征集   : 仅取两库均 >=93% 完整度的入科 24h 变量, 共 10 个
           -> EPV: MIMIC 123/12=10.3, eICU 100/12=8.3 (自由度 12, 见 report)
模型     : (A) Logistic + 限制性立方样条(RCS, 3 节点)  (B) XGBoost
验证     : 内部 5x5 重复分层交叉验证
           外部 MIMIC -> eICU 与 eICU -> MIMIC 双向
           NWICU 仅用于「无 GCS」敏感性模型的描述性应用 (事件仅 3, 不可靠)

核心问题: 内部性能与外部性能的落差有多大 = 跨库可移植性的量化

产出
----
out/t5_model_perf.csv        内部/外部 AUC, Brier, 校准斜率/截距
out/t6_transport_gap.csv     可移植性落差 (内部 - 外部)
out/t7_coef.csv              Logistic 模型系数 (OR)
fig/roc_<setting>.png        ROC 曲线
fig/calib_<setting>.png      校准曲线
fig/dca_<setting>.png        决策曲线
fig/shap_<db>.png            SHAP 重要性
out/part2_report.md
"""
import os, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
import statsmodels.api as sm
import xgboost as xgb

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, FIG = os.path.join(ROOT, "out"), os.path.join(ROOT, "fig")
os.makedirs(FIG, exist_ok=True)
RNG = 20260731

LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}

# ---------------------------------------------------------------- 特征定义
# 主模型: 两库完整度均 >=93%
FEATS_MAIN = ["age", "female", "gcs_min", "hr", "map", "temp", "spo2", "wbc", "creat", "sepsis_dx"]
# 敏感性模型: 去 GCS -> 唯一能应用到 NWICU 的版本 (NWICU chartevents 无 GCS 项目)
FEATS_NOGCS = [f for f in FEATS_MAIN if f != "gcs_min"]

BINARY = ["female", "sepsis_dx"]
LOGVAR = ["wbc", "creat"]           # 右偏, 取对数
RCSVAR = ["age", "gcs_min"]         # 施加限制性立方样条

CN = {"age": "年龄", "female": "女性", "gcs_min": "GCS 最低值", "hr": "心率",
      "map": "平均动脉压", "temp": "体温", "spo2": "SpO₂ 最低", "wbc": "白细胞(对数)",
      "creat": "肌酐(对数)", "sepsis_dx": "脓毒症"}


# ================================================================ RCS 基函数
def rcs_knots(x, k=3):
    """Harrell 推荐分位: 3 节点取 10/50/90 百分位。"""
    x = pd.to_numeric(x, errors="coerce").dropna()
    qs = {3: [.10, .50, .90], 4: [.05, .35, .65, .95]}[k]
    kn = np.unique(np.quantile(x, qs))
    return kn if len(kn) == k else None


def rcs_basis(x, kn):
    """限制性立方样条基 (Harrell RMS 公式), k 节点 -> k-1 列 (含线性项)。"""
    x = np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)
    k = len(kn)
    t1, tkm1, tk = kn[0], kn[-2], kn[-1]
    denom = (tk - t1) ** 2
    cols = [x]
    cub = lambda z: np.where(z > 0, z ** 3, 0.0)
    for j in range(k - 2):
        tj = kn[j]
        term = (cub(x - tj)
                - cub(x - tkm1) * (tk - tj) / (tk - tkm1)
                + cub(x - tk) * (tkm1 - tj) / (tk - tkm1)) / denom
        cols.append(term)
    return np.column_stack(cols)


# ================================================================ 预处理器
class Prep:
    """在训练集上学习: 中位数(插补) + RCS 节点。再套用到验证集 —— 严禁用验证集信息。"""

    def __init__(self, feats, use_rcs=True):
        self.feats = list(feats)
        self.use_rcs = use_rcs

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
                names += [CN.get(f, f)] + [f"{CN.get(f,f)}′(样条)" for _ in range(B.shape[1] - 1)]
            else:
                mats.append(s.reshape(-1, 1))
                names.append(CN.get(f, f))
        self.names_ = names
        return np.column_stack(mats)

    def transform_raw(self, df):
        """XGBoost 用: 不做样条展开, 只做对数变换与插补。"""
        cols = []
        for f in self.feats:
            s = pd.to_numeric(df[f], errors="coerce")
            if f in LOGVAR:
                s = np.log(np.clip(s, 1e-2, None))
            cols.append(s.fillna(self.med_[f]).values.astype(float))
        return np.column_stack(cols)


# ================================================================ 模型
def fit_logit(X, y):
    Xc = sm.add_constant(X, has_constant="add")
    try:
        return sm.Logit(y, Xc).fit(disp=0, method="bfgs", maxiter=500)
    except Exception:
        return None


def pred_logit(res, X):
    if res is None:
        return None
    return res.predict(sm.add_constant(X, has_constant="add"))


def fit_xgb_model(X, y):
    m = xgb.XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, min_child_weight=5,
        eval_metric="logloss", random_state=RNG, n_jobs=4)
    m.fit(X, y)
    return m


# ================================================================ 评价指标
def auc_ci(y, p, n_boot=1000, seed=RNG):
    y, p = np.asarray(y), np.asarray(p)
    a = roc_auc_score(y, p)
    rs = np.random.RandomState(seed)
    bs = []
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    for _ in range(n_boot):
        i = np.concatenate([rs.choice(idx0, len(idx0), True), rs.choice(idx1, len(idx1), True)])
        if len(np.unique(y[i])) < 2:
            continue
        bs.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return a, lo, hi


def calibration(y, p):
    """返回 (斜率, 截距/CITL)。斜率 1 且截距 0 为完美校准。"""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    try:
        sl = sm.Logit(y, sm.add_constant(lp, has_constant="add")).fit(disp=0, method="bfgs").params[1]
    except Exception:
        sl = np.nan
    try:
        it = sm.Logit(y, np.ones((len(y), 1)), offset=lp).fit(disp=0, method="bfgs").params[0]
    except Exception:
        it = np.nan
    return sl, it


def eval_all(y, p):
    a, lo, hi = auc_ci(y, p)
    sl, it = calibration(y, p)
    return {"auc": a, "auc_lo": lo, "auc_hi": hi,
            "brier": brier_score_loss(y, p), "slope": sl, "citl": it}


# ================================================================ 载入
data = {db: pd.read_csv(os.path.join(OUT, f"cohort_{db}.csv")) for db in LABEL}
for db, d in data.items():
    print(f"[i] {LABEL[db]:9s} n={len(d):4d}  npsle_core={int(d.npsle_core.sum()):3d}")


def internal_cv(df, feats, model="logit", n_splits=5, n_repeats=5, target="npsle_core"):
    """内部 5x5 重复分层 CV: 每折内独立拟合预处理器, 杜绝信息泄漏。"""
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).values.astype(int)
    oof = np.zeros(len(df))
    cnt = np.zeros(len(df))
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RNG)
    for tr, te in cv.split(np.zeros(len(y)), y):
        pp = Prep(feats, use_rcs=(model == "logit")).fit(df.iloc[tr])
        if model == "logit":
            Xtr, Xte = pp.transform(df.iloc[tr]), pp.transform(df.iloc[te])
            res = fit_logit(Xtr, y[tr])
            pr = pred_logit(res, Xte)
            if pr is None:
                continue
        else:
            Xtr, Xte = pp.transform_raw(df.iloc[tr]), pp.transform_raw(df.iloc[te])
            pr = fit_xgb_model(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        oof[te] += pr
        cnt[te] += 1
    ok = cnt > 0
    return y[ok], (oof[ok] / cnt[ok])


def external(df_tr, df_te, feats, model="logit"):
    ytr = df_tr["npsle_core"].values.astype(int)
    yte = df_te["npsle_core"].values.astype(int)
    pp = Prep(feats, use_rcs=(model == "logit")).fit(df_tr)
    if model == "logit":
        res = fit_logit(pp.transform(df_tr), ytr)
        p = pred_logit(res, pp.transform(df_te))
        return yte, p, res, pp
    m = fit_xgb_model(pp.transform_raw(df_tr), ytr)
    return yte, m.predict_proba(pp.transform_raw(df_te))[:, 1], m, pp


# ================================================================ 主流程
SETTINGS = []   # (标签, 场景类型, 特征集名, y, p)
rows = []
store = {}      # 供绘图

for fname, feats in [("主模型(含 GCS)", FEATS_MAIN), ("敏感性(无 GCS)", FEATS_NOGCS)]:
    for model, mcn in [("logit", "Logistic+RCS"), ("xgb", "XGBoost")]:
        # ---- 内部 ----
        for db in ["mimiciv", "eicu"]:
            y, p = internal_cv(data[db], feats, model)
            r = eval_all(y, p)
            key = (fname, mcn, f"内部 · {LABEL[db]}")
            store[key] = (y, p)
            rows.append(dict(特征集=fname, 模型=mcn, 场景=f"内部 · {LABEL[db]}",
                             类型="内部", n=len(y), 事件=int(y.sum()), **r))
        # ---- 外部双向 ----
        for a, b in [("mimiciv", "eicu"), ("eicu", "mimiciv")]:
            y, p, _, _ = external(data[a], data[b], feats, model)
            if p is None:
                continue
            r = eval_all(y, p)
            key = (fname, mcn, f"外部 · {LABEL[a]}→{LABEL[b]}")
            store[key] = (y, p)
            rows.append(dict(特征集=fname, 模型=mcn, 场景=f"外部 · {LABEL[a]}→{LABEL[b]}",
                             类型="外部", n=len(y), 事件=int(y.sum()), **r))
        # ---- NWICU: 仅无 GCS 模型可应用 ----
        if fname == "敏感性(无 GCS)":
            for a in ["mimiciv", "eicu"]:
                y, p, _, _ = external(data[a], data["nwicu"], feats, model)
                if p is None or y.sum() < 2:
                    continue
                try:
                    auc = roc_auc_score(y, p)
                except Exception:
                    continue
                rows.append(dict(特征集=fname, 模型=mcn, 场景=f"探索 · {LABEL[a]}→NWICU",
                                 类型="探索", n=len(y), 事件=int(y.sum()),
                                 auc=auc, auc_lo=np.nan, auc_hi=np.nan,
                                 brier=brier_score_loss(y, p), slope=np.nan, citl=np.nan))

t5 = pd.DataFrame(rows)
fmt = lambda r: (f"{r.auc:.3f} ({r.auc_lo:.3f}–{r.auc_hi:.3f})"
                 if pd.notna(r.auc_lo) else f"{r.auc:.3f} (事件过少, 不估 CI)")
t5["AUC (95%CI)"] = t5.apply(fmt, axis=1)
t5["Brier"] = t5.brier.map(lambda v: f"{v:.3f}")
t5["校准斜率"] = t5.slope.map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
t5["校准截距"] = t5.citl.map(lambda v: f"{v:+.2f}" if pd.notna(v) else "—")
t5_out = t5[["特征集", "模型", "场景", "n", "事件", "AUC (95%CI)", "Brier", "校准斜率", "校准截距"]]
t5_out.to_csv(os.path.join(OUT, "t5_model_perf.csv"), index=False, encoding="utf-8-sig")

# ================================================================ 可移植性落差
gap_rows = []
for fname in t5.特征集.unique():
    for mcn in t5.模型.unique():
        s = t5[(t5.特征集 == fname) & (t5.模型 == mcn)]
        for a, b in [("MIMIC-IV", "eICU-CRD"), ("eICU-CRD", "MIMIC-IV")]:
            ii = s[s.场景 == f"内部 · {a}"]
            ee = s[s.场景 == f"外部 · {a}→{b}"]
            if len(ii) == 0 or len(ee) == 0:
                continue
            gap_rows.append({"特征集": fname, "模型": mcn, "训练库": a, "验证库": b,
                             "内部 AUC": f"{ii.auc.iloc[0]:.3f}",
                             "外部 AUC": f"{ee.auc.iloc[0]:.3f}",
                             "AUC 落差": f"{ii.auc.iloc[0] - ee.auc.iloc[0]:+.3f}",
                             "外部校准斜率": f"{ee.slope.iloc[0]:.2f}" if pd.notna(ee.slope.iloc[0]) else "—",
                             "外部校准截距": f"{ee.citl.iloc[0]:+.2f}" if pd.notna(ee.citl.iloc[0]) else "—"})
t6 = pd.DataFrame(gap_rows)
t6.to_csv(os.path.join(OUT, "t6_transport_gap.csv"), index=False, encoding="utf-8-sig")

# ================================================================ 系数表 (主模型 Logistic)
coef_rows = []
for db in ["mimiciv", "eicu"]:
    pp = Prep(FEATS_MAIN, use_rcs=True).fit(data[db])
    X = pp.transform(data[db])
    res = fit_logit(X, data[db]["npsle_core"].values.astype(int))
    if res is None:
        continue
    names = ["截距"] + pp.names_
    for i, nm in enumerate(names):
        b, se = res.params[i], res.bse[i]
        if nm == "截距" or not np.isfinite(se) or se > 5:
            continue
        coef_rows.append({"数据库": LABEL[db], "变量": nm,
                          "OR (95%CI)": f"{np.exp(b):.3f} ({np.exp(b-1.96*se):.3f}–{np.exp(b+1.96*se):.3f})",
                          "P": ("<0.001" if res.pvalues[i] < 0.001 else f"{res.pvalues[i]:.3f}")})
t7 = pd.DataFrame(coef_rows)
t7.to_csv(os.path.join(OUT, "t7_coef.csv"), index=False, encoding="utf-8-sig")

# ================================================================ 绘图
C_IN, C_EX = "#1f4e79", "#c0392b"


def plot_roc(keys, fn, title):
    fig, ax = plt.subplots(figsize=(5.6, 5.2), facecolor="white")
    for (k, lab, col, ls) in keys:
        if k not in store:
            continue
        y, p = store[k]
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=col, ls=ls, lw=2,
                label=f"{lab} (AUC={roc_auc_score(y,p):.3f})")
    ax.plot([0, 1], [0, 1], color="#999", ls=":", lw=1)
    ax.set_xlabel("1 − 特异度"); ax.set_ylabel("敏感度")
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, fn), dpi=300, facecolor="white"); plt.close()


def plot_calib(keys, fn, title):
    fig, ax = plt.subplots(figsize=(5.6, 5.2), facecolor="white")
    for (k, lab, col, mk) in keys:
        if k not in store:
            continue
        y, p = store[k]
        q = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=False)
        obs = pd.Series(y).groupby(q).mean()
        exp = pd.Series(p).groupby(q).mean()
        ax.plot(exp, obs, marker=mk, color=col, lw=1.8, ms=7, label=lab)
    ax.plot([0, 1], [0, 1], color="#999", ls=":", lw=1)
    ax.set_xlabel("预测概率"); ax.set_ylabel("实际观察比例")
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(fontsize=8.5, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, fn), dpi=300, facecolor="white"); plt.close()


def net_benefit(y, p, pt):
    y = np.asarray(y); p = np.asarray(p)
    n = len(y)
    out = []
    for t in pt:
        f = p >= t
        tp = np.sum(f & (y == 1)); fp = np.sum(f & (y == 0))
        out.append(tp / n - (fp / n) * (t / (1 - t)))
    return np.array(out)


def plot_dca(keys, fn, title):
    pt = np.linspace(0.05, 0.75, 60)
    fig, ax = plt.subplots(figsize=(5.9, 5.2), facecolor="white")
    prev = None
    for (k, lab, col, ls) in keys:
        if k not in store:
            continue
        y, p = store[k]
        prev = np.mean(y)
        ax.plot(pt, net_benefit(y, p, pt), color=col, ls=ls, lw=2, label=lab)
    if prev is not None:
        ax.plot(pt, prev - (1 - prev) * (pt / (1 - pt)), color="#7f8c8d", ls="--", lw=1.3, label="全部干预")
        ax.axhline(0, color="#34495e", lw=1.3, label="全不干预")
        ax.set_ylim(-0.12, max(0.05, prev * 1.15))
    ax.set_xlabel("阈值概率"); ax.set_ylabel("净获益")
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, fn), dpi=300, facecolor="white"); plt.close()


F = "主模型(含 GCS)"
for mcn, tag in [("Logistic+RCS", "logit"), ("XGBoost", "xgb")]:
    ks_roc = [((F, mcn, "内部 · MIMIC-IV"), "内部 MIMIC-IV", C_IN, "-"),
              ((F, mcn, "内部 · eICU-CRD"), "内部 eICU-CRD", C_IN, "--"),
              ((F, mcn, "外部 · MIMIC-IV→eICU-CRD"), "外部 MIMIC→eICU", C_EX, "-"),
              ((F, mcn, "外部 · eICU-CRD→MIMIC-IV"), "外部 eICU→MIMIC", C_EX, "--")]
    plot_roc(ks_roc, f"roc_{tag}.png", f"{mcn} · 内部与外部 ROC")
    ks_cal = [(k, l, c, "o" if s == "-" else "s") for k, l, c, s in ks_roc]
    plot_calib(ks_cal, f"calib_{tag}.png", f"{mcn} · 校准曲线")
    plot_dca([ks_roc[0], ks_roc[2]], f"dca_{tag}.png", f"{mcn} · 决策曲线（MIMIC 内部 vs 外推 eICU）")

# ---------------- SHAP ----------------
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
        plt.title(f"SHAP · {LABEL[db]} 训练的 XGBoost", fontsize=11.5)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"shap_{db}.png"), dpi=300,
                    bbox_inches="tight", facecolor="white")
        plt.close()
    shap_ok = True
except Exception as ex:
    print("[!] SHAP 失败:", ex)
    shap_ok = False


# ================================================================ 判别力天花板 / 表型特异性 / GCS 分层
# 这三项是本研究的核心论证: 低 AUC 究竟是「特征选得不好」还是「数据本身没有信号」。
ALL_CAND = ["age", "female", "hr", "map", "rr", "temp", "spo2", "wbc", "lymph_abs", "hgb",
            "plt", "creat", "alb", "na", "bili", "inr", "lactate", "gcs_min", "gcs_motor",
            "gcs_verbal", "gcs_eyes", "sofa24", "sofa_cns", "apache", "aps", "sepsis_dx",
            "lupus_nephritis", "steroid_any", "is_any", "hx_epilepsy", "vent24"]

ceil_rows = []
for db in ["mimiciv", "eicu"]:
    d = data[db]
    allf = [c for c in ALL_CAND if c in d.columns
            and pd.to_numeric(d[c], errors="coerce").notna().sum() > len(d) * 0.3]
    y_a, p_a = internal_cv(d, allf, "xgb")
    y_b, p_b = internal_cv(d, FEATS_MAIN, "xgb")
    a_all, a_10 = roc_auc_score(y_a, p_a), roc_auc_score(y_b, p_b)
    ceil_rows.append({"数据库": LABEL[db],
                      "简约模型变量数": len(FEATS_MAIN), "简约模型 AUC": f"{a_10:.3f}",
                      "全变量数": len(allf), "全变量 AUC": f"{a_all:.3f}",
                      "增益": f"{a_all - a_10:+.3f}"})
t8 = pd.DataFrame(ceil_rows)
t8.to_csv(os.path.join(OUT, "t8_ceiling.csv"), index=False, encoding="utf-8-sig")

PH_CN = {"npsle_core": "核心 NPSLE（混合）", "dom_enceph": "急性意识障碍/脑病",
         "dom_seizure": "急性癫痫发作", "dom_cvd": "脑血管病"}
ph_rows = []
for db in ["mimiciv", "eicu"]:
    d = data[db]
    for tgt in ["npsle_core", "dom_enceph", "dom_seizure", "dom_cvd"]:
        if tgt not in d.columns:
            continue
        ev = int(pd.to_numeric(d[tgt], errors="coerce").fillna(0).sum())
        if ev < 20:
            ph_rows.append({"数据库": LABEL[db], "目标表型": PH_CN[tgt], "事件数": ev,
                            "CV-AUC": "事件过少，未评估"})
            continue
        yy, pp_ = internal_cv(d, FEATS_MAIN, "xgb", target=tgt)
        ph_rows.append({"数据库": LABEL[db], "目标表型": PH_CN[tgt], "事件数": ev,
                        "CV-AUC": f"{roc_auc_score(yy, pp_):.3f}"})
t9 = pd.DataFrame(ph_rows)
t9.to_csv(os.path.join(OUT, "t9_phenotype_auc.csv"), index=False, encoding="utf-8-sig")

gcs_rows = []
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
        gcs_rows.append({"数据库": LABEL[db], "分层": vcn, "n": len(s),
                         "事件": int(s.npsle_core.sum()),
                         "GCS-言语 中位（非NPSLE）": f"{gv[ok & (s.npsle_core==0)].median():.0f}",
                         "GCS-言语 中位（NPSLE）": f"{gv[ok & (s.npsle_core==1)].median():.0f}",
                         "单变量 AUC": f"{max(a, 1-a):.3f}"})
t10 = pd.DataFrame(gcs_rows)
t10.to_csv(os.path.join(OUT, "t10_gcs_strat.csv"), index=False, encoding="utf-8-sig")


# ================================================================ 报告
def md(df):
    if df is None or len(df) == 0:
        return "_（无数据）_"
    h = "| " + " | ".join(map(str, df.columns)) + " |"
    s = "| " + " | ".join("---" for _ in df.columns) + " |"
    b = "\n".join("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
                  for r in df.values)
    return "\n".join([h, s, b])


main = t5_out[t5_out.特征集 == "主模型(含 GCS)"].drop(columns=["特征集"])
sens = t5_out[t5_out.特征集 == "敏感性(无 GCS)"].drop(columns=["特征集"])

rep = f"""# Part 2 · NPSLE 诊断识别模型与双向外部验证

> 研究主线：**跨库可移植性**。本部分回答——一个在某库表现良好的 NPSLE 识别模型，
> 换到另一个编码体系、另一批医院时还剩下多少性能？

## 1. 设计与样本

| 项目 | 内容 |
| --- | --- |
| 目标变量 | `npsle_core`（ACR 核心 5 域；排除脑血管病、周围神经病变） |
| 建模库 | MIMIC-IV（n={len(data['mimiciv'])}，事件 {int(data['mimiciv'].npsle_core.sum())}）、eICU-CRD（n={len(data['eicu'])}，事件 {int(data['eicu'].npsle_core.sum())}） |
| 特征集 | {len(FEATS_MAIN)} 个入科 24h 变量（两库完整度均 ≥93%）：{'、'.join(CN.get(f, f) for f in FEATS_MAIN)} |
| 模型自由度 | 12（`年龄`与`GCS`各含 1 个样条项） |
| EPV | MIMIC-IV 10.3；eICU-CRD 8.3 |
| 内部验证 | 5×5 重复分层交叉验证，**每折内独立拟合插补与样条节点** |
| 外部验证 | MIMIC-IV ⇄ eICU-CRD 双向全量 |
| NWICU | 仅用于「无 GCS」模型的探索性应用（事件仅 3，不作推断） |

**变量可得性即移植障碍**：NWICU 的 `icu.chartevents` 未记录 GCS，
因此含 GCS 的主模型在结构上无法迁移到该库。这不是缺失值问题，而是
**特征在目标库不存在** —— 本研究将其作为可移植性的第一类障碍单独报告。

---

## 2. 主模型性能（含 GCS）

{md(main)}

---

## 3. 跨库可移植性落差

{md(t6)}

**校准指标读法**：斜率 <1 表示外推时预测值过于极端（离散度过大）；
截距（CITL）为正表示系统性低估、为负表示系统性高估。

---

## 4. 低判别力的归因：是特征不足，还是数据本身没有信号？

这是本研究最关键的论证。若不排除「特征选择不当」，低 AUC 无法解释为数据局限。

### 4.1 判别力天花板：加满变量是否有增益

{md(t8)}

> 把变量从 {len(FEATS_MAIN)} 个扩展到全部可用变量（含 SOFA、APACHE、GCS 全部分量、
> 狼疮肾炎、免疫抑制剂暴露、机械通气等），AUC 增益仅约 +0.01。
> **判别力天花板由数据内容决定，而非特征选择或模型复杂度。**

### 4.2 表型特异性：混合表型是否稀释了信号

{md(t9)}

> eICU 中单一表型（癫痫、脑病）的判别力明显高于混合的核心 NPSLE，
> 说明 ACR 七域合并确实稀释了信号；但即便聚焦单一表型，上限仍在 0.70 左右。
> MIMIC-IV 中无此提升，与其亚型构成以脑血管病为主（ICD 账单编码特征）一致。

### 4.3 GCS 的判别力被机械通气掩盖

{md(t10)}

> GCS-言语评分是全部变量中最强的单一信号，但该判别力**几乎完全来自未机械通气亚组**。
> 插管患者的言语评分被结构性地压至最低值，组间中位数完全重合，AUC 趋近 0.50。
> 这意味着：在最需要早期识别的重症亚组中，最强的可用信号恰恰失效。

---

## 5. 敏感性分析（去除 GCS）

{md(sens)}

---

## 6. Logistic 模型系数（主模型，各库分别拟合）

{md(t7)}

> 同一变量在两库中方向或量级的差异，直接反映了「编码体系 + 收治人群」的双重异质性。

---

## 7. 图

| 文件 | 内容 |
| --- | --- |
| `fig/roc_logit.png` / `fig/roc_xgb.png` | 内部与外部 ROC |
| `fig/calib_logit.png` / `fig/calib_xgb.png` | 校准曲线（五分位） |
| `fig/dca_logit.png` / `fig/dca_xgb.png` | 决策曲线 |
| `fig/shap_mimiciv.png` / `fig/shap_eicu.png` | SHAP 变量重要性{'' if shap_ok else '（本次未生成）'} |

---

## 8. 主要结论

1. **判别力不足是数据的属性，不是模型的缺陷。**
   入科 24h 的常规生命体征与化验对核心 NPSLE 的判别上限约 AUC 0.60–0.65；
   扩充到全部可用变量后增益 ≈ +0.01（表 8）。

2. **最强信号在最需要它的人群中失效。**
   GCS-言语是唯一稳定信号，但其判别力仅存在于未机械通气亚组；
   插管后组间完全重合（表 10）。

3. **跨库外推使模型丧失临床可用性。**
   外部 AUC 降至 0.50–0.60，且校准彻底失效：斜率 0.01–0.35（远小于 1），
   截距 ±1.5（MIMIC→eICU 系统性低估，eICU→MIMIC 系统性高估）。
   这与 Part 1 中患病率 19.1% vs 43.5% 的巨大差异直接对应。

4. **变量可得性本身构成第一类移植障碍。**
   NWICU 未记录 GCS，含 GCS 的模型在结构上无法迁移；
   eICU 缺少狼疮肾炎、免疫抑制剂、既往癫痫史三类词条。

> **对领域的含义**：现有公开 ICU 数据库缺少 NPSLE 归因诊断所必需的
> 免疫学（C3/C4、抗 dsDNA、抗磷脂抗体谱）、脑脊液与神经影像变量
> （Part 1 可行性核查显示缺失率 >90%）。在补齐这些变量之前，
> 基于此类数据库开发 NPSLE 临床预测模型难以达到可部署的性能。

---

## 9. 方法学声明

1. **无信息泄漏**：所有插补中位数、样条节点、模型参数均仅在训练集上估计。
2. **样本量约束**：eICU 的 EPV=8.3 略低于常用的 10，模型系数的稳定性有限；
   XGBoost 已采用保守超参（`max_depth=3`、`min_child_weight=5`、`reg_lambda=2`）抑制过拟合。
3. **AUC 的 95%CI** 由分层 bootstrap（1000 次重抽样）给出。
4. **NWICU 结果为探索性**：仅 3 例核心 NPSLE，任何判别指标都不具备统计意义，
   列出仅为展示「变量可得性」这一移植障碍。
5. **本研究为阴性/警示性结果**，其价值在于界定现有数据基础设施的能力边界，
   而非提供可部署的预测工具。
"""
with open(os.path.join(OUT, "part2_report.md"), "w", encoding="utf-8") as f:
    f.write(rep)

pd.set_option("display.width", 250)
pd.set_option("display.unicode.east_asian_width", True)
print("\n===== 主模型 =====")
print(main.to_string(index=False))
print("\n===== 可移植性落差 =====")
print(t6.to_string(index=False))
print("\n===== 敏感性(无 GCS) =====")
print(sens.to_string(index=False))
print("\n===== 表8 判别力天花板 =====")
print(t8.to_string(index=False))
print("\n===== 表9 表型特异性 =====")
print(t9.to_string(index=False))
print("\n===== 表10 GCS×机械通气 分层 =====")
print(t10.to_string(index=False))
print("\n[ok] -> out/part2_report.md, t5–t10 csv, fig/*.png")
