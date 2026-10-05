import os
# -*- coding: utf-8 -*-
"""Print the full hierarchy of every eICU diagnosisstring that carries a
neurologic / CNS path segment, so the tail-2 rule is designed from the observed
paths rather than guessed.
"""
import pathlib

import pandas as pd

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
lines = []


def A(s=""):
    lines.append(str(s))


def main():
    p = pd.read_csv(OUT / "cohort_eicu.csv")
    import psycopg2
    ids = p["stay_id"].astype(int).tolist()
    with psycopg2.connect(dbname="eicu", host="localhost", port=5432,
                          user="postgres", password="1314") as cn:
        d = pd.read_sql(
            "SELECT patientunitstayid AS stay_id, diagnosisstring AS term "
            "FROM eicu_crd.diagnosis WHERE patientunitstayid = ANY(%s)", cn,
            params=(ids,))
    d["s"] = d["term"].fillna("").astype(str).str.lower()
    n = d["s"].str.count(r"\|") + 1
    A("path depth distribution (number of '|' separated components):")
    A(n.value_counts().sort_index().to_string())
    A("")

    neuro = d[d["s"].str.contains(r"neurologic\||cns infection|due to neurological",
                                  regex=True, na=False)]
    A("CNS-path strings: %d rows, %d stays, %d distinct"
      % (len(neuro), neuro["stay_id"].nunique(), neuro["s"].nunique()))
    A("")
    g = (neuro.groupby("s")["stay_id"].nunique()
         .reset_index(name="stays").sort_values(["stays", "s"], ascending=[False, True]))
    for _, r in g.iterrows():
        A("  %3d  %s" % (r["stays"], r["s"]))

    A("")
    A("=== all strings with depth >= 3 (tail-2 discards the first component) ===")
    deep = d[n >= 3]
    gd = (deep.groupby("s")["stay_id"].nunique()
          .reset_index(name="stays").sort_values("stays", ascending=False))
    A("distinct = %d ; stays = %d" % (len(gd), int(gd["stays"].sum())))
    for _, r in gd.iterrows():
        A("  %3d  %s" % (r["stays"], r["s"]))

    (OUT / "v8_eicu_paths.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
