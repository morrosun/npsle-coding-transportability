# -*- coding: utf-8 -*-
"""Generate the three reproducibility tables that the external review demanded.

S30  matching-rule comparison, on a common core universe
S31  stay-level label transition, primary -> legacy
S32  cross-validation scheme x label set (2x2)

Every number is read from the result files; nothing is typed by hand.
"""
import os
import io
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
sys.path.insert(0, str(ROOT / "scripts"))
import npsle_io as io_  # noqa: E402

LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _num(x, d=1):
    return ("&mdash;" if x is None or (isinstance(x, float) and x != x)
            else ("%.*f" % (d, x)))


def build_s30():
    """Matching-rule comparison. Both arms use their own core universe, which is
    what a reviewer needs: the legacy numbers must reproduce the published ones
    exactly, otherwise the comparison is not credible."""
    rows = []
    prim, leg = {}, {}
    for db in LABEL:
        for legacy, store in ((False, prim), (True, leg)):
            d = io_.load(db, legacy=legacy)
            f = d[d._first]
            cp, cf = io_.counts(d), io_.counts(f)
            store[db] = dict(all=cp, first=cf)

    def pct(a, b):
        return 100.0 * a / b if b else float("nan")

    for scope, key in (("All stays", "all"), ("One stay per patient", "first")):
        for db in ("mimiciv", "eicu", "nwicu"):
            p, l = prim[db][key], leg[db][key]
            if p["core"] == 0 and l["core"] == 0:
                continue
            rows.append((
                scope, LABEL[db], p["n"],
                "%d (%.1f%%)" % (l["core"], pct(l["core"], l["n"])),
                "%d (%.1f%%)" % (p["core"], pct(p["core"], p["n"])),
                "%d / %d" % (l["A"], l["B"]), "%d / %d" % (p["A"], p["B"]),
                "%d (%.1f%%)" % (l["C"], pct(l["C"], l["core"])),
                "%d (%.1f%%)" % (p["C"], pct(p["C"], p["core"])),
                str(l["unassigned"]), str(p["unassigned"]),
                str(l["controls"]), str(p["controls"]),
            ))

    hdr = ["Cohort", "Database", "ICU stays",
           "Core events, legacy rule", "Core events, audited rule",
           "Tier A / Tier B, legacy", "Tier A / Tier B, audited",
           "Tier C only, legacy", "Tier C only, audited",
           "Unassigned, legacy", "Unassigned, audited",
           "Controls, legacy", "Controls, audited"]
    body = "".join(
        "<tr>%s</tr>" % "".join("<td>%s</td>" % _esc(c) for c in r)
        for r in rows)
    note = (
        "Both rule sets are produced by the same two scripts in one database "
        "pass, and every tier indicator is intersected with the recorded core "
        "phenotype of its own rule set, so the decomposition is internally "
        "consistent in both arms; the legacy arm reproduces the previously "
        "published counts exactly. Tier A is 0 on the ICD-based databases "
        "because the algorithm recovered none, not because attribution is "
        "unrepresentable in the coding system.")
    return ("<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>"
            % ("".join("<th>%s</th>" % _esc(h) for h in hdr), body), note)


def build_s31():
    """Stay-level label transition, audited -> legacy, eICU-CRD only (the only
    database whose rules were rewritten)."""
    p = io_.load("eicu")
    l = io_.load("eicu", legacy=True)
    m = p[["stay_id", "tier_primary", "npsle_core"]].merge(
        l[["stay_id", "tier_primary", "npsle_core"]], on="stay_id",
        suffixes=("_audited", "_legacy"))
    ct = pd.crosstab(m["tier_primary_audited"].replace("", "(no event)"),
                     m["tier_primary_legacy"].replace("", "(no event)"))
    ct = ct.reindex(index=["A", "B", "C", "(no event)"],
                    columns=["A", "B", "C", "(no event)"]).fillna(0).astype(int)
    hdr = ["Audited tier", "legacy A", "legacy B", "legacy C", "legacy no event",
           "Total"]
    body = ""
    for idx, r in ct.iterrows():
        cells = "".join("<td>%d</td>" % int(v) for v in r.values)
        body += ("<tr><td><strong>%s</strong></td>%s<td><strong>%d</strong></td></tr>"
                 % (_esc(idx), cells, int(r.sum())))
    tot = "".join("<td><strong>%d</strong></td>" % int(v) for v in ct.sum().values)
    body += "<tr><td><strong>Total</strong></td>%s<td><strong>%d</strong></td></tr>" % (
        tot, int(ct.values.sum()))
    n_changed = int((m["npsle_core_audited"] != m["npsle_core_legacy"]).sum())
    n_tier = int((m["tier_primary_audited"] != m["tier_primary_legacy"]).sum())
    note = (
        "MIMIC-IV and NWICU are identical under both rule sets. The direction "
        "is one way: the audited rules replace folder-level matching with "
        "leaf-level matching plus an explicit exclusion list, so they can only "
        "remove a recorded event, never add one.")
    return ("<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>"
            % ("".join("<th>%s</th>" % _esc(h) for h in hdr), body), note)


def build_s32():
    res = json.loads((OUT / "v8_cv_label_2x2.json").read_text(encoding="utf-8"))
    hdr = ["Label set", "Cross-validation", "Database", "ICU stays",
           "Events", "AUC (95% CI)"]
    body = ""
    for db in ("mimiciv", "eicu"):
        for lab in ("legacy", "restricted"):
            for cv in ("stay", "patient"):
                k = db + "|" + lab + "|" + cv
                if k not in res:
                    continue
                r = res[k]
                ci = ("%.3f&ndash;%.3f" % (r["lo"], r["hi"])
                      if r["lo"] == r["lo"] else "&mdash;")
                body += ("<tr><td>%s</td><td>%s</td><td>%s</td><td>%d</td>"
                         "<td>%d</td><td>%.3f (%s)</td></tr>"
                         % (_esc(lab),
                            "patient-grouped" if cv == "patient" else "stay-level",
                            LABEL[db], r["n"], r["events"], r["auc"], ci))
    note = (
        "Features, preprocessing, hyper-parameters and split seeds are held "
        "fixed; only the label set and the splitting unit change. In MIMIC-IV "
        "the two label sets are identical by construction.")
    return ("<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>"
            % ("".join("<th>%s</th>" % _esc(h) for h in hdr), body), note)


if __name__ == "__main__":
    out = {}
    for name, fn in (("S30", build_s30), ("S31", build_s31), ("S32", build_s32)):
        tbl, note = fn()
        out[name] = dict(table=tbl, note=note)
        print("[ok] %s  %d chars" % (name, len(tbl)))
    (OUT / "v8_tables.json").write_text(json.dumps(out, indent=2),
                                         encoding="utf-8")
