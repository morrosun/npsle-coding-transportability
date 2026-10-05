import os
"""Generate Figure 5: attribution ladder (MIMIC-IV in-hospital mortality).
Each step changes one factor; OR on a log axis shows the 0.44 -> 0.96 climb.
No on-plot title (per user request). Saved at fig/fig5_ladder.png (300 dpi).
"""
import re, pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = ROOT / "fig"
FIG.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(ROOT / "out/t38_mortality_ladder_v2.csv")
d = d[d["步骤"].str.match(r"^[①②③④⑤]")].reset_index(drop=True)

rows = []
for _, r in d.iterrows():
    m = re.search(r"([\d.]+)\s*\(([\d.]+)\u2013([\d.]+)\)", str(r["校正 OR (95%CI)"]))
    if not m:
        continue
    step = r["步骤"][0]  # circled number
    label = {"①": "① Baseline", "②": "② Estimator", "③": "③ Covariate set",
             "④": "④ Cohort", "⑤": "⑤ Reference"}[step]
    rows.append((label, float(m.group(1)), float(m.group(2)), float(m.group(3))))

# order: ① at bottom (y=1) ... ⑤ at top (y=5) so OR "climbs" upward as steps progress
rows = rows[::-1]  # ①..⑤ -> reversed to put ① at bottom
ys = list(range(1, len(rows) + 1))
markers = ["o", "s", "D", "^", "*"]
colors = ["#1f4e79" if orr < 1 else "#c0392b" for (_, orr, _, _) in rows]

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.axvline(1, color="#888", ls="--", lw=1)
for y, (lab, orr, lo, hi), mk, col in zip(ys, rows, markers, colors):
    ax.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle="butt", zorder=2)
    ax.scatter([orr], [y], s=170 if mk != "*" else 360, marker=mk, color=col,
               edgecolor="white", linewidth=1.2, zorder=3)
    ax.annotate(f"{orr:.2f}", (orr, y), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=9.5, fontweight="bold", color=col)

ax.set_yscale("linear")
ax.set_yticks(ys)
ax.set_yticklabels([lab for (lab, _, _, _) in rows], fontsize=10)
# remove y-axis tick marks (avoid the small dash that appears next to each label
# when the left spine is hidden but tick marks are still drawn)
ax.tick_params(axis="y", which="both", length=0)
ax.tick_params(axis="x", which="both", length=4)
ax.set_xscale("log")
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
ax.xaxis.set_major_locator(FixedLocator([0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2]))
ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
ax.set_xlim(0.14, 2.1)
ax.set_ylim(0.6, len(rows) + 0.4)
ax.set_xlabel("Adjusted odds ratio (log scale)", fontsize=11)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="x", ls=":", color="#dddddd", lw=0.8)
# NOTE: no in-figure legend. The blue (OR < 1) / red (OR >= 1) color coding is
# explained in the Figure 4 caption in the manuscript body.
plt.tight_layout(pad=0.8)
out = FIG / "fig5_ladder.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out, out.stat().st_size, "bytes")
