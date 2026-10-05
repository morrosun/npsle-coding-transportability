import os
# -*- coding: utf-8 -*-
"""Build one canonical stay-level truth table (primary + legacy) and audit
every tier count against it. All downstream tables must be derived from this.

Emits:
  out/truth_stays.csv        stay-level, both label sets, per database
  out/truth_consistency.txt  human-readable audit of the invariants
"""
import io
import json
import math
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"

TIER_COLS = ["tier_a", "tier_b", "tier_c", "other_cause"]
CORE_COLS = ["npsle_core", "npsle_broad"]

lines = []


def A(s=""):
    lines.append(str(s))


def num(x):
    return pd.to_numeric(x, errors="coerce")


def load(db):
    """Return primary and legacy stay-level frames for one database.

    Core phenotype membership lives in cohort_<db>.csv; the stay-level semantic
    tier flags (A/B/C and the competing-etiology flag) live in tier_<db>.csv.
    Both must be joined on stay_id. Only eICU-CRD has two label sets: the
    MIMIC-IV and NWICU phenotypes are built from ICD/diagnosis rows that the
    leaf-restricted rewrite never touched, so for those databases
    primary == legacy by construction.
    """
    coh = pd.read_csv(OUT / ("cohort_%s.csv" % db))
    coh = coh.drop(columns=[c for c in TIER_COLS if c in coh.columns],
                   errors="ignore")
    t_p = pd.read_csv(OUT / ("tier_%s.csv" % db))
    leg_t = OUT / ("tier_%s_legacy.csv" % db)
    t_l = pd.read_csv(leg_t) if leg_t.exists() else t_p.copy()

    p = coh.merge(t_p[["stay_id"] + TIER_COLS], on="stay_id", how="left")
    l = coh.merge(t_l[["stay_id"] + TIER_COLS], on="stay_id", how="left")
    return p, l


def normalise(df, tag):
    d = df.copy()
    for c in TIER_COLS:
        if c not in d.columns:
            d[c] = 0
        d[c] = num(d[c]).fillna(0).astype(int)
    if "npsle_core" not in d.columns:
        raise SystemExit("%s: npsle_core missing" % tag)
    d["npsle_core"] = num(d["npsle_core"]).fillna(0).astype(int)
    d["npsle_broad"] = num(d.get("npsle_broad")).fillna(0).astype(int) \
        if "npsle_broad" in d.columns else 0
    return d


def derive(d):
    """Mutually exclusive stay-level tier + the A|B union, from raw flags."""
    d = d.copy()
    d["_a"] = d["tier_a"].eq(1)
    d["_b"] = d["tier_b"].eq(1)
    d["_c"] = d["tier_c"].eq(1)
    d["_x"] = d["other_cause"].eq(1)
    d["_core"] = d["npsle_core"].eq(1)
    # mutually exclusive semantic tier: A > B > C
    d["tier_primary"] = np.where(d["_a"], "A",
                          np.where(d["_b"], "B",
                            np.where(d["_c"], "C", "")))
    d["tier_primary"] = np.where(d["_core"], d["tier_primary"], "")
    d["competing_flag"] = d["_x"].astype(int)
    d["core_primary"] = d["_core"].astype(int)
    d["core_legacy"] = d["_core"].astype(int)
    d["tier_legacy"] = d["tier_primary"]
    d["hi_primary"] = (d["tier_primary"].isin(["A", "B"])).astype(int)
    d["hi_legacy"] = d["hi_primary"]
    d["tc_primary"] = (d["tier_primary"].eq("C")).astype(int)
    d["tc_legacy"] = d["tc_primary"]
    d["unassigned_primary"] = (d["core_primary"].eq(1) & d["hi_primary"].eq(0)
                              & d["tc_primary"].eq(0)).astype(int)
    return d


def main():
    first = json.loads((OUT / "_first_stays.json").read_text(encoding="utf-8"))
    allp = {}
    for db in ("mimiciv", "eicu", "nwicu"):
        p, l = load(db)
        p = normalise(p, db + "/primary")
        l = normalise(l, db + "/legacy")
        key = "mimiciv" if db == "mimiciv" else db
        fk = {k: v for k, v in first.items() if k.startswith(db[:5]) or db in k}
        p["_first"] = p["stay_id"].isin(_first_set(first, db))
        l["_first"] = l["stay_id"].isin(_first_set(first, db))
        p = derive(p)
        l = derive(l)
        allp[db] = (p, l)

    # ---------------- canonical output ----------------
    keep = ["stay_id", "subject_id", "core_primary", "tier_primary",
            "hi_primary", "tc_primary", "unassigned_primary",
            "competing_flag", "core_legacy", "tier_legacy",
            "hi_legacy", "tc_legacy", "unassigned_primary",
            "_first", "npsle_broad"]
    frames = []
    for db, (p, l) in allp.items():
        m = p[[c for c in keep if c in p.columns]].copy()
        for c in ("core_legacy", "tier_legacy", "hi_legacy", "tc_legacy"):
            if c not in l.columns:
                m[c] = 0
        m["core_legacy"] = l["core_primary"].values
        m["tier_legacy"] = l["tier_primary"].values
        m["hi_legacy"] = l["hi_primary"].values
        m["tc_legacy"] = l["tc_primary"].values
        m["unassigned_legacy"] = l["unassigned_primary"].values
        m["db"] = db
        frames.append(m)
    truth = pd.concat(frames, ignore_index=True)
    truth.to_csv(OUT / "truth_stays.csv", index=False, encoding="utf-8")

    # ---------------- invariants ----------------
    A("CANONICAL STAY-LEVEL TRUTH TABLE")
    A("  out/truth_stays.csv   rows = %d" % len(truth))
    A("")
    fails = []

    def chk(label, ok, detail=""):
        A("  [%s] %s %s" % ("OK " if ok else "FAIL", label, detail))
        if not ok:
            fails.append(label)

    for db, (p, l) in allp.items():
        for tag, d in (("primary", p), ("legacy", l)):
            for scope, sub in (("all", d), ("first", d[d["_first"]])):
                nA = int((sub["tier_primary"] == "A").sum())
                nB = int((sub["tier_primary"] == "B").sum())
                nC = int((sub["tier_primary"] == "C").sum())
                nu = int(sub["unassigned_primary"].sum())
                nc = int(sub["core_primary"].sum())
                A("")
                A("  %-8s %-8s %-6s stays=%4d core=%4d  A=%3d B=%3d C=%3d unassigned=%2d"
                  % (db, tag, scope, len(sub), nc, nA, nB, nC, nu))
                chk("%s/%s/%s core == A+B+C+unassigned" % (db, tag, scope),
                    nc == nA + nB + nC + nu,
                    "(%d vs %d+%d+%d+%d=%d)" % (nc, nA, nB, nC, nu,
                                                nA + nB + nC + nu))
                # A|B union from the mutually exclusive tier
                hi = nA + nB
                chk("%s/%s/%s hi == A+B" % (db, tag, scope),
                    hi == int(sub["hi_primary"].sum()))
                ctrl = int((sub["core_primary"] == 0).sum())
                A("           raw controls (core==0) = %d" % ctrl)
                # overlap accounting must never be used to explain the union
                rawAB = int(((sub["_a"]) | (sub["_b"])).sum())
                A("           A OR B from raw flags = %d (identical to %d: %s)"
                  % (rawAB, hi, rawAB == hi))

    # label transition
    A("")
    A("LABEL TRANSITION  primary -> legacy")
    for db in allp:
        p, l = allp[db]
        m = p[["stay_id", "core_primary", "tier_primary"]].merge(
            l[["stay_id", "core_primary", "tier_primary"]],
            on="stay_id", suffixes=("_p", "_l"))
        ct = pd.crosstab(m["tier_primary_p"].replace("", "(none)"),
                         m["tier_primary_l"].replace("", "(none)"))
        A("")
        A("  %s" % db)
        A(ct.to_string())
        A("  core: primary %d -> legacy %d ; changed %d stays"
          % (int(m["core_primary_p"].sum()), int(m["core_primary_l"].sum()),
             int((m["core_primary_p"] != m["core_primary_l"]).sum())))
        lost = int(((m["core_primary_p"] == 1) & (m["tier_primary_p"] == ""))
                   .sum())
        A("  primary core stays with no tier: %d" % lost)

    A("")
    A("INVARIANT FAILURES: %d" % len(fails))
    for f in fails:
        A("  - %s" % f)

    (OUT / "truth_consistency.txt").write_text("\n".join(lines),
                                               encoding="utf-8")
    print("\n".join(lines))


def _first_set(first, db):
    pref = {"mimiciv": "mimic", "eicu": "eicu", "nwicu": "nwicu"}[db]
    for k, v in first.items():
        if k.startswith(pref):
            return set(int(x) for x in v)
    raise SystemExit("no first-stay key for %s (have %s)" % (db, list(first)))


if __name__ == "__main__":
    main()
