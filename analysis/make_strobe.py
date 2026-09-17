"""Generate a publication-grade STROBE participant-flow diagram (>=300 dpi, English labels).

925 total SLE-related ICU stays (MIMIC-IV 645, eICU-CRD 230, NWICU 50)
 -> exclude repeat ICU stays within the same patient (342)
 -> 583 first ICU stay cohort (MIMIC-IV 354, eICU-CRD 186, NWICU 43)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import os, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
FIG = REPO / "fig"
FIG.mkdir(exist_ok=True)
OUT = str(FIG / "strobe_flow.png")

fig, ax = plt.subplots(figsize=(8.0, 6.4), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

def box(x, y, w, h, lines, fc="#eaf2fb", ec="#2b6cb0"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=1.2,rounding_size=2.5",
                 fc=fc, ec=ec, lw=1.6))
    cy = y + h - 0.5 - (len(lines) - 1) * 4.0
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, cy + i * 8.0, ln, ha="center", va="center",
                fontsize=10.5, color="#10243e",
                fontweight=("bold" if i == 0 else "normal"),
                fontfamily="DejaVu Sans")

# Top box: identified stays
box(14, 70, 72, 22,
    ["SLE-related ICU stays identified across three databases",
     "MIMIC-IV 645   ·   eICU-CRD 230   ·   NWICU 50",
     "Total = 925"])

# Exclusion label between boxes
ax.text(50, 52, "Exclude repeat ICU stays within the same patient (n = 342)",
        ha="center", va="center", fontsize=9.5, color="#7a3b00",
        fontstyle="italic", fontfamily="DejaVu Sans")

# Bottom box: first-stay cohort
box(14, 18, 72, 24,
    ["First ICU stay cohort (primary analysis)",
     "MIMIC-IV 354   ·   eICU-CRD 186   ·   NWICU 43",
     "Total = 583 unique patients"])

# Arrows
ax.add_patch(FancyArrowPatch((50, 70), (50, 58), arrowstyle="-|>",
             mutation_scale=16, lw=1.8, color="#2b6cb0"))
ax.add_patch(FancyArrowPatch((50, 44), (50, 42), arrowstyle="-|>",
             mutation_scale=16, lw=1.8, color="#2b6cb0"))

# Side notes (inclusion/exclusion criteria)
ax.text(2, 64, "Inclusion:\nICU stay with ICD-9-CM 710.0*\nor ICD-10 M32.* (SLE)",
        ha="left", va="top", fontsize=7.6, color="#444",
        fontfamily="DejaVu Sans")
ax.text(98, 30, "Prespecified exclusions\n(age < 18 y; ICU LOS < 4 h)\nmet or exempted; not applied\nretroactively (see S4)",
        ha="right", va="top", fontsize=7.6, color="#444",
        fontfamily="DejaVu Sans")

ax.text(50, 95, "STROBE Flow Diagram — Participant Selection",
        ha="center", va="center", fontsize=12.5, fontweight="bold",
        color="#10243e", fontfamily="DejaVu Sans")

fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03)
fig.savefig(OUT, dpi=300)
print("saved", OUT)
