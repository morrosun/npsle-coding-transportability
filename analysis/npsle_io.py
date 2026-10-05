import os
# -*- coding: utf-8 -*-
"""Single loader for the analysis chain.

Every analysis script must build its stay-level frame through load(), so that

  * the recorded core phenotype (cohort_<db>.csv) and the semantic tier
    (tier_<db>.csv) are joined exactly once and cannot drift apart,
  * npsle_core is taken from the tier file, which is where the universe
    restriction is applied, so no merge can produce _x/_y duplicates,
  * the legacy label set is available under a uniform prefix for the
    algorithm comparison.
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
FKEY = {"mimiciv": "mimic_first", "eicu": "eicu_first", "nwicu": "nwicu_first"}

TIER_COLS = ["tier_a", "tier_b", "tier_c", "other_cause",
             "npsle_core", "npsle_hi", "npsle_hi_str", "npsle_any",
             "tier_primary", "tier_c_only", "tier_unassigned"]
# tier_primary is a label, not a number: it must survive the CSV round trip
# with its empty cells intact, otherwise every unassigned stay reads as 0.
STR_COLS = ["tier_primary"]

_first = None


def first_stays():
    global _first
    if _first is None:
        d = json.loads((OUT / "_first_stays.json").read_text(encoding="utf-8"))
        _first = {k: set(int(x) for x in v) for k, v in d.items()}
    return _first


def num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)


def load(db, legacy=False):
    """Stay-level frame for one database.

    legacy=False : primary phenotype (audited rules for eICU-CRD).
    legacy=True  : legacy whole-path phenotype, including its own core
                   membership (npsle_core_legacy), so that the label contrast
                   reflects BOTH the domain rules and the tier rules. Pass
                   legacy_tier_only=True to hold the core universe fixed and
                   vary only the tier assignment.
    """
    c = pd.read_csv(OUT / ("cohort_%s.csv" % db))
    legacy_tier_only = getattr(load, "_tier_only", False)
    legacy_core = None
    if legacy and not legacy_tier_only and "npsle_core_legacy" in c.columns:
        legacy_core = num(c["npsle_core_legacy"]).astype(int)
    tier_path = OUT / ("tier_%s%s.csv" % (db, "_legacy" if legacy else ""))
    if not tier_path.exists():
        # only eICU-CRD has two label sets; for the ICD-based databases the
        # leaf-restricted rewrite changed nothing, so primary == legacy
        if not legacy:
            raise SystemExit("missing %s" % tier_path)
        tier_path = OUT / ("tier_%s.csv" % db)
    t = pd.read_csv(tier_path)
    keep = ["stay_id"] + [x for x in TIER_COLS if x in t.columns]
    # drop the phenotype columns from the cohort frame: the tier file is their
    # authoritative, universe-restricted version
    drop = [x for x in ("npsle_core", "npsle_broad", "npsle_sens",
                        "npsle_hi", "npsle_hi_str", "npsle_any") if x in c.columns]
    m = c.drop(columns=drop).merge(t[keep], on="stay_id", how="left")
    if legacy_core is not None:
        # the legacy label set carries its own core universe
        m["npsle_core"] = legacy_core.reindex(m.index).fillna(0).astype(int)
    for col in TIER_COLS:
        if col in m.columns and col not in STR_COLS:
            m[col] = num(m[col]).astype(int)
    for col in STR_COLS:
        if col in m.columns:
            m[col] = m[col].fillna("").astype(str).str.strip()
    if legacy:
        m["tier_a"] = num(m.get("tier_a")).astype(int)
        m["tier_b"] = num(m.get("tier_b")).astype(int)
        m["tier_c"] = num(m.get("tier_c")).astype(int)
        # the legacy tier file carries the legacy tier flags but the core
        # universe differs, so the mutually exclusive decomposition must be
        # recomputed against this frame's own npsle_core
        m["npsle_core"] = num(m.get("npsle_core")).astype(int)
        hi = ((m["tier_a"] | m["tier_b"]) & m["npsle_core"]).astype(int)
        m["npsle_hi"] = hi
        m["tier_c_only"] = ((m["tier_c"].eq(1)) & m["npsle_core"].eq(1)
                            & hi.eq(0)).astype(int)
        m["tier_unassigned"] = ((m["npsle_core"].eq(1)) & hi.eq(0)
                                & m["tier_c_only"].eq(0)).astype(int)
        m["tier_primary"] = np.where(
            m["npsle_core"].eq(0), "",
            np.where(m["tier_a"].eq(1), "A",
              np.where(m["tier_b"].eq(1), "B",
                np.where(m["tier_c"].eq(1), "C", ""))))
    # Canonical row order, applied in the one loader every analysis script goes
    # through. The extraction queries carry no ORDER BY, so the cohort CSV comes
    # back in whatever order the database happened to return: the eICU-CRD frame
    # was re-ordered for 228 of its 230 stays between two runs of the same
    # extraction, with no cell value changed. GroupKFold assigns groups in order
    # of first appearance and the stratified splitters are seeded but
    # position-dependent, so every cross-validated quantity silently changed
    # with the row order -- which is exactly what happened to the
    # label-confidence AUCs behind Table S19 and the decision-curve values
    # behind Table S11. Ordering here makes every downstream result reproducible
    # from the data alone, and matches the convention already used in
    # v8_cv_label_2x2.py.
    m = (m.sort_values(["subject_id", "stay_id"], kind="mergesort")
           .reset_index(drop=True))
    m["_first"] = m["stay_id"].isin(first_stays()[FKEY[db]])
    return m


def counts(d):
    """The one place where the mutually exclusive decomposition is computed."""
    core = d["npsle_core"].eq(1)
    hi = d["npsle_hi"].eq(1)
    return dict(
        n=len(d),
        core=int(core.sum()),
        A=int((d["tier_primary"] == "A").sum()),
        B=int((d["tier_primary"] == "B").sum()),
        C=int(d["tier_c_only"].sum()),
        unassigned=int(d["tier_unassigned"].sum()),
        AB=int(hi.sum()),
        controls=int((~core).sum()),
        X=int(d["other_cause"].sum()),
    )
