import os
# -*- coding: utf-8 -*-
"""v7 — what is GCS verbal = 0?  Recorded value, intubation artefact, or fill?"""
import pathlib

import pandas as pd

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
W = []
A = W.append
A("%-9s %-8s %6s %8s %8s %8s %8s %8s" %
  ("db", "vent24", "n", "gv=0", "gv=1", "gv=2", "gv=3/4/5", "gv missing"))
for db in ("mimiciv", "eicu", "nwicu"):
    d = pd.read_csv(OUT / ("cohort_%s.csv" % db))
    v = pd.to_numeric(d["vent24"], errors="coerce")
    g = pd.to_numeric(d["gcs_verbal"], errors="coerce")
    for lab, sel in (("ventilated", v == 1), ("not vent", v == 0), ("all", v.notna())):
        s = sel & v.notna()
        gg = g[s]
        A("%-9s %-8s %6d %8d %8d %8d %8d %8d" %
          (db, lab, int(s.sum()), int((gg == 0).sum()), int((gg == 1).sum()),
           int((gg == 2).sum()), int(gg.isin([3, 4, 5]).sum()), int(gg.isna().sum())))
A("")
A("If a database records 0 for intubated patients while another leaves the")
A("item absent, the single-variable AUC of the verbal score is not comparable")
A("across databases and the structural zeros must be stated, not averaged in.")
txt = "\n".join(W)
(OUT / "v7_gcs_verbal_check.txt").write_text(txt, encoding="utf-8")
print(txt)
