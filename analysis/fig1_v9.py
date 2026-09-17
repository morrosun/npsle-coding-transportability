# -*- coding: utf-8 -*-
"""V9: redraw Figure 1 panels with mutually-exclusive tier membership.
(a) tier_composition.png: mutually-exclusive stacked composition as % of all
    ICU stays (Tier A / B / C-only; 'unassigned' labelled separately).
(b) tierC_reprod.png:  C-only proportion of CORE events under both cohort
    bases (first-stay vs all-stays) with Wilson 95% CI; unassigned n shown.
Then compose fig1_tier_dual.png at 300 dpi.
"""
import os, json, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repository root, resolved relative to this file so the pipeline runs
# from a fresh clone on any platform.
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out"; FIG = ROOT / "fig"
DBS = ["mimiciv", "eicu", "nwicu"]
LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
FIRST = json.loads((OUT / "_first_stays.json").read_text(encoding="utf-8"))
FKEY = {"mimiciv": "mimic_first", "eicu": "eicu_first", "nwicu": "nwicu_first"}


def load(db):
    c = pd.read_csv(OUT / f"cohort_{db}.csv")
    t = pd.read_csv(OUT / f"tier_{db}.csv")
    return c.merge(t, on="stay_id", how="left")


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))


data = {db: load(db) for db in DBS}

# ================================================= (a) stacked composition, all stays
fig, ax = plt.subplots(figsize=(7, 8))
dbs = list(LABEL)
x = np.arange(len(dbs))
pa, pb, pc, pu = [], [], [], []
for db in dbs:
    d = data[db]
    a = pd.to_numeric(d["tier_a"], errors="coerce").fillna(0).astype(int)
    b = pd.to_numeric(d["tier_b"], errors="coerce").fillna(0).astype(int)
    c = pd.to_numeric(d["tier_c"], errors="coerce").fillna(0).astype(int)
    hi = pd.to_numeric(d["npsle_hi"], errors="coerce").fillna(0).astype(int)
    core = pd.to_numeric(d["npsle_core"], errors="coerce").fillna(0).astype(int)
    un = ((core == 1) & (hi == 0) & (c == 0)).astype(int)
    pa.append(100 * a.mean())                      # Tier A (attributable)
    pb.append(100 * ((b == 1) & (a == 0)).mean())  # Tier B only
    pc.append(100 * ((c == 1) & (hi == 0)).mean()) # Tier C only (mutually exclusive)
    pu.append(100 * un.mean())

C = ["#1f6f8b", "#4c9f70", "#d9a441", "#999999"]
ax.bar(x, pa, 0.55, label="Tier A \u00b7 attributable (SLE explicitly named)", color=C[0])
ax.bar(x, pb, 0.55, bottom=pa, label="Tier B only \u00b7 specific NP syndrome", color=C[1])
ax.bar(x, pc, 0.55, bottom=np.array(pa) + np.array(pb),
       label="Tier C only \u00b7 non-specific (mutually exclusive)", color=C[2])
ax.bar(x, pu, 0.55, bottom=np.array(pa) + np.array(pb) + np.array(pc),
       label="Core event, unassigned to any tier", color=C[3])

for i in range(len(dbs)):
    stack = [pa[i], pb[i], pc[i]]
    acc = 0
    for j, v in enumerate(stack):
        if v > 0.8:
            ax.text(i, acc + v / 2, f"{v:.1f}%", ha="center", va="center",
                    color="white", fontsize=9.5, fontweight="bold")
        acc += v
    if pu[i] > 0.8:
        ax.text(i, acc + pu[i] / 2, f"{pu[i]:.1f}%", ha="center", va="center",
                color="white", fontsize=9.5, fontweight="bold")
        acc += pu[i]
    ax.text(i, acc + 1.0, f"{acc:.1f}%", ha="center", fontsize=10, fontweight="bold")
    if data[dbs[i]]["tier_a"].isna().all():
        ax.text(i, -3.6, "Tier A structurally unavailable", ha="center",
                fontsize=8.6, color="#b03a2e")

ax.set_xticks(x)
ax.set_xticklabels([f"{LABEL[d]}\n(n={len(data[d])})" for d in dbs], fontsize=10.5)
ax.set_ylabel("Percentage of all SLE-ICU stays (%)", fontsize=11)
ax.set_ylim(-5.5, 115)
ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
ax.grid(axis="y", ls=":", alpha=0.45)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / "tier_composition.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("wrote tier_composition.png")

# ================================================= (b) C-only of core events, both bases
rows = []
for db in ["mimiciv", "eicu"]:
    for cname, df in [("First stay (primary)", None), ("All stays", None)]:
        pass
# build per-database / per-cohort C-only proportion
sub = []
for db in ["mimiciv", "eicu"]:
    d_all = data[db]
    fids = set(pd.Series(FIRST[FKEY[db]]).astype(str))
    d_first = d_all[d_all["stay_id"].astype(str).isin(fids)]
    for tag, d in [("First stay", d_first), ("All stays", d_all)]:
        core = pd.to_numeric(d["npsle_core"], errors="coerce").fillna(0) == 1
        hi = pd.to_numeric(d["npsle_hi"], errors="coerce").fillna(0) == 1
        tc = pd.to_numeric(d["tier_c"], errors="coerce").fillna(0) == 1
        n_core = int(core.sum())
        n_conly = int((core & tc & ~hi).sum())
        n_un = int((core & ~hi & ~tc).sum())
        lo, hi2 = wilson(n_conly, n_core) if n_core else (np.nan, np.nan)
        sub.append(dict(db=LABEL[db], tag=tag, n_core=n_core, n=n_conly, un=n_un,
                        pct=100 * n_conly / n_core, lo=100 * lo, hi=100 * hi2))
S = pd.DataFrame(sub)

fig, ax = plt.subplots(figsize=(7, 8))
xs = np.arange(2)
w = 0.32
for j, tag in enumerate(["First stay", "All stays"]):
    part = S[S["tag"] == tag]
    pos = xs + (j - 0.5) * w
    for k, (_, r) in enumerate(part.iterrows()):
        col = "#1f6f8b" if r["db"] == "MIMIC-IV" else "#c0392b"
        ax.bar(pos[k], r["pct"], width=w, color=col, alpha=0.9, zorder=2,
               label=(tag if k == 0 else None))
        ax.errorbar(pos[k], r["pct"], yerr=[[r["pct"] - r["lo"]], [r["hi"] - r["pct"]]],
                    fmt="none", ecolor="#333", capsize=5, lw=1.3, zorder=3)
        ax.text(pos[k], r["pct"] + 4, f"{r['pct']:.1f}%  ({r['n']}/{r['n_core']})",
                ha="center", fontsize=10.5, fontweight="bold")
        if r["un"]:
            ax.text(pos[k], r["pct"] - 12, f"unassigned n={r['un']}", ha="center",
                    fontsize=8.6, color="#555", style="italic")
ax.set_xticks(xs)
ax.set_xticklabels(["MIMIC-IV", "eICU-CRD"], fontsize=11.5)
ax.set_ylim(0, 105)
ax.set_ylabel("Tier C only as % of core recorded events", fontsize=10.5)
ax.legend(fontsize=9.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
ax.grid(axis="y", ls=":", alpha=.5, zorder=0)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / "tierC_reprod.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("wrote tierC_reprod.png (mutually-exclusive C-only, both bases)")

# ================================================= compose fig1_tier_dual.png
from PIL import Image, ImageDraw, ImageFont
# Font: first existing candidate is used (Windows / Linux / macOS).
# Override with the NPSLE_FONT environment variable if needed.
FONT = next((p for p in (
    os.environ.get("NPSLE_FONT", ""),
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
) if p and os.path.exists(p)), None)
def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except Exception:
        return ImageFont.load_default()
A = Image.open(FIG / "tier_composition.png").convert("RGB")
B = Image.open(FIG / "tierC_reprod.png").convert("RGB")
target_h = 1700
def fit(im, h):
    if im.height == h:
        return im
    w = int(round(im.width * h / im.height))
    return im.resize((w, h), Image.LANCZOS)
A = fit(A, target_h); B = fit(B, target_h)
gap = 80; pad_top = 130
W = A.width + B.width + gap
canvas = Image.new("RGB", (W, target_h + pad_top), "white")
canvas.paste(A, (0, pad_top)); canvas.paste(B, (A.width + gap, pad_top))
d = ImageDraw.Draw(canvas)
fs = max(56, int(target_h * 0.05)); f = font(fs)
d.text((22, 22), "(a)", font=f, fill=(20, 20, 20))
d.text((A.width + gap + 22, 22), "(b)", font=f, fill=(20, 20, 20))
canvas.save(FIG / "fig1_tier_dual.png", dpi=(300, 300))
print(f"wrote fig1_tier_dual.png  {canvas.width}x{canvas.height}")
