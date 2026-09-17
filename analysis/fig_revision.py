#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""修订新增图件:  fig/tier_composition.png   fig/label_auc.png"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, FIG = os.path.join(ROOT, "out"), os.path.join(ROOT, "fig")
os.makedirs(FIG, exist_ok=True)
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
DPI = 300

data = {db: pd.read_csv(os.path.join(OUT, f"cohort_{db}.csv"))
        .merge(pd.read_csv(os.path.join(OUT, f"tier_{db}.csv")), on="stay_id") for db in LABEL}

# ---------------------------------------------------- 图 6 分层构成
fig, ax = plt.subplots(figsize=(8.2, 4.6))
dbs = list(LABEL)
x = np.arange(len(dbs))
# 互斥分解: A 优先, 其次 B(非A), 其次 C(非A非B)
pa, pb, pc = [], [], []
for db in dbs:
    d = data[db]
    n = len(d)
    a = d["tier_a"].fillna(0).astype(int)
    b = d["tier_b"].astype(int)
    c = d["tier_c"].astype(int)
    only_a = a
    only_b = ((b == 1) & (a == 0)).astype(int)
    only_c = ((c == 1) & (a == 0) & (b == 0)).astype(int)
    pa.append(100 * only_a.mean()); pb.append(100 * only_b.mean()); pc.append(100 * only_c.mean())

C = ["#1f6f8b", "#4c9f70", "#d9a441"]
ax.bar(x, pa, 0.55, label="Tier A · 归因明确（术语明写 SLE）", color=C[0])
ax.bar(x, pb, 0.55, bottom=pa, label="Tier B · 特异性 NP 综合征", color=C[1])
ax.bar(x, pc, 0.55, bottom=np.array(pa) + np.array(pb), label="Tier C · 非特异性意识/精神改变", color=C[2])

for i in range(len(dbs)):
    tot = pa[i] + pb[i] + pc[i]
    if pa[i] > 0.8:
        ax.text(i, pa[i] / 2, f"{pa[i]:.1f}%", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    if pb[i] > 0.8:
        ax.text(i, pa[i] + pb[i] / 2, f"{pb[i]:.1f}%", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    if pc[i] > 0.8:
        ax.text(i, pa[i] + pb[i] + pc[i] / 2, f"{pc[i]:.1f}%", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    ax.text(i, tot + 1.0, f"合计 {tot:.1f}%", ha="center", fontsize=10, fontweight="bold")
    if data[dbs[i]]["tier_a"].isna().all():
        ax.text(i, -3.6, "Tier A 结构性不可得", ha="center", fontsize=8.6, color="#b03a2e")

ax.set_xticks(x)
ax.set_xticklabels([f"{LABEL[d]}\n(n={len(data[d])})" for d in dbs], fontsize=10.5)
ax.set_ylabel("占 SLE-ICU 队列的百分比 (%)", fontsize=11)
ax.set_title("图 6　编码记录的神经精神事件按诊断置信度分层的构成", fontsize=12.5, pad=12)
ax.set_ylim(-5.5, max(np.array(pa) + np.array(pb) + np.array(pc)) * 1.25)
ax.legend(fontsize=9.5, loc="upper left", framealpha=0.95)
ax.grid(axis="y", ls=":", alpha=0.45)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "tier_composition.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()

# ---------------------------------------------------- 图 7 标签置信度 vs AUC
t15 = pd.read_csv(os.path.join(OUT, "t15_label_auc.csv"))
t15 = t15[~t15["XGBoost CV-AUC (95%CI)"].astype(str).str.contains("不估计")].copy()


def parse(s):
    a, rest = s.split(" (")
    lo, hi = rest.rstrip(")").split("–")
    return float(a), float(lo), float(hi)


vals = t15["XGBoost CV-AUC (95%CI)"].map(parse)
t15["auc"] = [v[0] for v in vals]; t15["lo"] = [v[1] for v in vals]; t15["hi"] = [v[2] for v in vals]
t15 = t15.iloc[::-1].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8.4, 3.9))
cols = {"MIMIC-IV": "#1f6f8b", "eICU-CRD": "#c0392b"}
yy = np.arange(len(t15))
for i, r in t15.iterrows():
    c = cols[r["数据库"]]
    ax.plot([r["lo"], r["hi"]], [i, i], color=c, lw=2.2, solid_capstyle="round")
    ax.plot(r["auc"], i, "o", color=c, ms=7, zorder=3)
    ax.text(r["hi"] + 0.012, i, f'{r["auc"]:.3f} ({r["lo"]:.3f}–{r["hi"]:.3f})', va="center", fontsize=9)
ax.axvline(0.5, color="#666", ls="--", lw=1.2)
ax.text(0.503, len(t15) - 0.35, "随机水平", fontsize=8.8, color="#666")
ax.set_yticks(yy)
ax.set_yticklabels([f'{r["数据库"]} · {r["标签定义"]}（事件 {r["事件数"]}）' for _, r in t15.iterrows()], fontsize=9.8)
ax.set_xlabel("XGBoost 内部 5×5 交叉验证 AUC（95% bootstrap CI）", fontsize=10.5)
ax.set_title("图 7　提高结局标签置信度并未带来判别力提升", fontsize=12.5, pad=12)
ax.set_xlim(0.33, 0.90)
ax.grid(axis="x", ls=":", alpha=0.45)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "label_auc.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()

print("[ok] -> fig/tier_composition.png, fig/label_auc.png (dpi=%d)" % DPI)
