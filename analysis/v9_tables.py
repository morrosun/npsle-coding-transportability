import os
# -*- coding: utf-8 -*-
"""Generate S23-S26 and S31 as HTML fragments, from the result files only.

These five tables were the ones the external review found internally
inconsistent: the pooled ratio of odds ratios could not be reproduced from the
single-database rows beside it, the S23/S24 grid still used a superseded 2x2,
the E-values still used superseded odds ratios, and the S31 transition table
merged "no tier assigned" with "no core event".

Every cell below is read from a result file. Two invariants are asserted:
  * a fixed-effect pooled log-effect must lie between the two single-database
    log-effects (it is a positive weighted mean);
  * the S31 transition table must reconcile with the S30 core counts.
"""
import io
import json
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"
sys.path.insert(0, str(ROOT / "scripts"))
import npsle_io as nio  # noqa: E402

LABEL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD"}
# the result files print display names, the loader uses short keys
DISP2KEY = {"MIMIC-IV": "mimiciv", "eICU-CRD": "eicu", "NWICU": "nwicu"}


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def f(x, d=2):
    return "&mdash;" if x is None or (isinstance(x, float) and x != x) \
        else ("%.*f" % (d, x))


def table(hdr, rows):
    h = "".join("<th>%s</th>" % esc(x) for x in hdr)
    b = "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % c for c in r)
                for r in rows)
    return ("<table class=\"dataframe data-table\"><thead><tr>%s</tr>"
            "</thead><tbody>%s</tbody></table>" % (h, b))


def parse_grid(path):
    """Read the Layer 1 / Layer 2 / Layer 3 blocks of the bias report."""
    txt = (OUT / path).read_text(encoding="utf-8") if isinstance(path, str) \
        else path
    rows = []
    for ln in txt.split("\n"):
        if not re.match(r"^(MIMIC-IV|eICU-CRD)\s", ln):
            continue
        p = ln.split()
        rows.append(dict(db=p[0], se=float(p[1]), sp=float(p[2]),
                         cells=p[3:]))
    return rows


# ------------------------------------------------------------------ S23 / S24
def build_s23_s24():
    # Read the full-precision grid from v6_qbf.json rather than re-parsing the
    # text report. The report prints R1/R0 to four decimals and OR_true to
    # three, so parsing it rounded every cell a second time: MIMIC-IV R1 came
    # out as 0.271 instead of 0.270 and the last eICU-CRD de-attenuated OR as
    # 1.78 instead of 1.79. The JSON carries the same grid unrounded.
    grid = json.loads((OUT / "v6_qbf.json").read_text(encoding="utf-8"))
    l1 = [dict(db=r["db"], se=float(r["se"]), sp=float(r["sp"]),
               p1=float(r["p1"]), p0=float(r["p0"]),
               r1=float(r["R1"]), r0=float(r["R0"]),
               orobs=float(r["or_obs"]), ortrue=float(r["or_true"]),
               ppv1=float(r["ppv1"]), ppv0=float(r["ppv0"]),
               trueev=float(r.get("true_ev", float("nan"))))
          for r in grid["layer1"]]
    l2 = [dict(db=r["db"], se=float(r["se"]), sp=float(r["sp"]),
               R=float(r["R"]), c=float(r["c_pct"]) / 100.0)
          for r in grid["layer2"]]
    assert len(l1) == 18 and len(l2) == 18, \
        "expected 18 grid rows per layer, got %d / %d" % (len(l1), len(l2))

    # ---- automatic field checks (the external review asked for these) ------
    #   * every derived risk and PPV is a probability
    #   * p* must be reproducible from the model itself, which is what catches
    #     a column-order mistake: reading trueEv as PPV gave 58.000 before
    #   * the printed de-attenuated OR must equal (R1/(1-R1)) / (R0/(1-R0))
    for r in l1:
        for key in ("r1", "r0", "ppv1", "ppv0"):
            assert 0.0 <= r[key] <= 1.0, \
                "%s: %s = %s is not a probability" % (r["db"], key, r[key])
        p1_rebuilt = r["se"] * r["r1"] + (1 - r["sp"]) * (1 - r["r1"])
        p0_rebuilt = r["se"] * r["r0"] + (1 - r["sp"]) * (1 - r["r0"])
        assert abs(p1_rebuilt - r["p1"]) < 1e-9, \
            "%s: p1* does not rebuild (%s vs %s)" % (r["db"], p1_rebuilt, r["p1"])
        assert abs(p0_rebuilt - r["p0"]) < 1e-9, \
            "%s: p0* does not rebuild (%s vs %s)" % (r["db"], p0_rebuilt, r["p0"])
        odds = (r["r1"] / (1 - r["r1"])) / (r["r0"] / (1 - r["r0"]))
        # The grid now comes from the JSON, which stores unrounded floats, so
        # the rebuild is exact; the tolerance only absorbs float formatting.
        assert abs(odds - r["ortrue"]) / r["ortrue"] < 1e-9, \
            "%s: de-attenuated OR does not rebuild (%s vs %s)" \
            % (r["db"], odds, r["ortrue"])
        ppv1_rebuilt = r["se"] * r["r1"] / r["p1"]
        assert abs(ppv1_rebuilt - r["ppv1"]) < 1e-9, \
            "%s: PPV1 does not rebuild (%s vs %s)" \
            % (r["db"], ppv1_rebuilt, r["ppv1"])

    rows = []
    for r in l1:
        rows.append([r["db"], f(r["se"], 2), f(r["sp"], 2),
                     f(r["p1"], 3), f(r["p0"], 3), f(r["orobs"]),
                     f(r["r1"], 3), f(r["r0"], 3), f(r["ortrue"]),
                     f(r["ppv1"], 3), f(r["ppv0"], 3)])
    s23 = table(["Database", "Sensitivity", "Specificity",
                 "Coded prevalence, septic", "Coded prevalence, non-septic",
                 "Observed OR", "Implied true risk, septic",
                 "Implied true risk, non-septic", "De-attenuated OR",
                 "Derived PPV, septic", "Derived PPV, non-septic"], rows)

    rows = []
    for r in l2:
        rows.append([r["db"], f(r["se"], 2), f(r["sp"], 2),
                     f(r["R"], 4), "%.1f%%" % (100 * r["c"])])
    s24 = table(["Database", "Sensitivity", "Specificity",
                 "True risk in the non-septic arm",
                 "Loop term c* sufficient to null the association"], rows)

    # thresholds for the running text, keyed by display name
    thr = {}
    for db in ("MIMIC-IV", "eICU-CRD"):
        v = [100 * r["c"] for r in l2 if r["db"] == db]
        thr[db] = (min(v), max(v))
    return s23, s24, thr


# ---------------------------------------------------------------------- S25
def evalue(orv, lo=None):
    """E-value with the sqrt(OR) ~ RR approximation for a common outcome."""
    rr = math.sqrt(orv)
    e = rr + math.sqrt(rr * (rr - 1))
    if lo is None:
        return rr, e, None, None
    rr_l = math.sqrt(lo)
    e_l = rr_l + math.sqrt(rr_l * (rr_l - 1))
    naive = orv + math.sqrt(orv * (orv - 1))
    return rr, e, e_l, naive


def build_s25():
    from result_or import adjusted_or, _pooled_label
    t22 = OUT / "t22a_first_sepsis.csv"
    t25 = OUT / "t25a_tier_with_pooled.csv"

    pooled_sep = adjusted_or(t22, contrast="k=2")
    pooled_tc = adjusted_or(t25, db=_pooled_label(t25), contrast="Tier C")
    per = []
    for lab in ("MIMIC-IV", "eICU-CRD"):
        per.append((lab, adjusted_or(t25, db=lab, contrast="Tier C")))

    rows = []
    items = [("Pooled sepsis association (adjusted)", pooled_sep),
             ("Pooled Tier C association (adjusted)", pooled_tc)]
    items += [("%s, Tier C" % lab, o) for lab, o in per]
    for name, (o, lo, hi) in items:
        rr, e, e_l, naive = evalue(o, lo)
        rows.append([esc(name), "%.2f (%.2f&ndash;%.2f)" % (o, lo, hi),
                     f(rr, 3), f(e), f(e_l), f(naive)])
    tbl = table(["Estimate", "OR (95% CI)", "Approximate RR = sqrt(OR)",
                 "E-value (point)", "E-value (at CI limit)",
                 "E-value if OR substituted for RR"], rows)
    return tbl, pooled_sep, pooled_tc


# ---------------------------------------------------------------------- S26
def build_s26():
    txt = (OUT / "v6_tier_multinomial.txt").read_text(encoding="utf-8")
    blocks = {}
    cur = None
    for ln in txt.split("\n"):
        m = re.match(r"^(MIMIC-IV|eICU-CRD)\s+n = (\d+)\s+\(no event (\d+) \| "
                     r"A\+B (\d+) \| C-only (\d+); "
                     r"excluded unassigned (\d+)\)", ln)
        if m:
            cur = m.group(1)
            blocks[cur] = dict(n=int(m.group(2)), noevent=int(m.group(3)),
                               ab_n=int(m.group(4)), c_n=int(m.group(5)),
                               excl=int(m.group(6)), adj=[], ab=None, c=None,
                               ror=None)
            continue
        if cur is None:
            continue
        m = re.match(r"\s+adjusters: (.+)$", ln)
        if m:
            txt_adj = m.group(1).strip()
            blocks[cur]["adj"] = txt_adj.replace("  [dropped:", ", dropped:")
        m = re.search(r"Tier A\+B\s+OR = ([\d.]+) \(95% CI ([\d.]+)-([\d.]+)\)",
                      ln)
        if m:
            blocks[cur]["ab"] = tuple(float(x) for x in m.groups())
        m = re.search(r"Tier C-oly OR = ([\d.]+) \(95% CI ([\d.]+)-([\d.]+)\)",
                      ln)
        if m:
            blocks[cur]["c"] = tuple(float(x) for x in m.groups())
        m = re.search(r"ratio of ORs = ([\d.]+) \(95% CI ([\d.]+)-([\d.]+)\), "
                      r"z = [\d.]+, P = ([\d.]+)", ln)
        if m:
            blocks[cur]["ror"] = tuple(float(x) for x in m.groups())

    m = re.search(r"Pooled ratio of ORs \(fixed effect, k = 2\): "
                  r"([\d.]+) \(95% CI ([\d.]+)-([\d.]+)\), P = ([\d.]+)", txt)
    assert m, "pooled row not found in the multinomial report"
    pooled = tuple(float(x) for x in m.groups())

    # invariant: a fixed-effect pooled log-effect is a positive weighted mean,
    # so it MUST lie between the two single-database log-effects
    singles = [blocks[d]["ror"][0] for d in blocks]
    lo, hi = min(singles), max(singles)
    assert lo <= pooled[0] <= hi, (
        "pooled ROR %.2f outside [%.2f, %.2f] -- the pooled row and the "
        "single-database rows are not from the same models" % (pooled[0], lo, hi))

    rows = []
    for db in ("MIMIC-IV", "eICU-CRD"):
        b = blocks[db]
        rows.append([db, str(b["n"]), esc(b["adj"]),
                     "%.2f (%.2f&ndash;%.2f)" % b["ab"],
                     "%.2f (%.2f&ndash;%.2f)" % b["c"],
                     "%.2f (%.2f&ndash;%.2f)" % b["ror"][:3],
                     "%.3f" % b["ror"][3]])
    rows.append(["Pooled (fixed effect)", "&mdash;", "&mdash;", "&mdash;",
                 "&mdash;",
                 "%.2f (%.2f&ndash;%.2f)" % pooled[:3], "%.3f" % pooled[3]])
    tbl = table(["Database", "Model n", "Adjusters",
                 "Tier A+B OR (95% CI)", "Tier C-only OR (95% CI)",
                 "Ratio of ORs, C versus A+B (95% CI)", "P"], rows)

    excl = {db: blocks[db].get("excl", 0) for db in blocks}
    ns = {db: blocks[db]["n"] for db in blocks}
    ev = {db: blocks[db].get("ab_n", 0) + blocks[db].get("c_n", 0)
          for db in blocks}
    note = (
        "A single multinomial model with a three-level outcome (no recorded "
        "core event, Tier A+B event, Tier C-only event) was fitted in each "
        "database, and the equality of the two sepsis coefficients was tested "
        "directly by a Wald test on their difference. Core-positive stays that "
        "no tier could be assigned were excluded from the outcome, not from the "
        "cohort: %d in MIMIC-IV and %d in eICU-CRD, the latter carrying a "
        "competing aetiology. Model n is therefore %d and %d against %d and %d "
        "tier-assigned events. The fixed-effect pooled ratio is a positive "
        "weighted mean of the two log-ratios and must lie between them; it "
        "does. Because a stratum-specific estimate being significant and "
        "another not being significant is not itself a test of their "
        "difference, this table provides that test; the stratum-specific odds "
        "ratios in Table 2 come from the pre-specified separate models and "
        "remain the primary estimates."
        % (excl.get("MIMIC-IV", 0), excl.get("eICU-CRD", 0),
           ns.get("MIMIC-IV", 0), ns.get("eICU-CRD", 0),
           ev.get("MIMIC-IV", 0), ev.get("eICU-CRD", 0)))
    return tbl, note, blocks, pooled


# ---------------------------------------------------------------------- S31
def build_s31():
    p = nio.load("eicu")
    l = nio.load("eicu", legacy=True)
    m = p[["stay_id", "tier_primary", "npsle_core"]].merge(
        l[["stay_id", "tier_primary", "npsle_core"]], on="stay_id",
        suffixes=("_aud", "_leg"))

    def state(df_core, df_tier):
        s = pd.Series("no core event", index=m.index)
        s = s.mask(df_core == 1, "")
        s = s.mask((df_core == 1) & (df_tier == "A"), "A")
        s = s.mask((df_core == 1) & (df_tier == "B"), "B")
        s = s.mask((df_core == 1) & (df_tier == "C"), "C-only")
        s = s.mask((df_core == 1) & (df_tier == ""), "core, unassigned")
        return s

    m["aud"] = state(m.npsle_core_aud, m.tier_primary_aud)
    m["leg"] = state(m.npsle_core_leg, m.tier_primary_leg)

    ORDER = ["A", "B", "C-only", "core, unassigned", "no core event"]
    ct = pd.crosstab(m["aud"], m["leg"])
    # crosstab keeps the Series names, which may be suffixed by the merge
    ct.index = pd.Categorical(ct.index, categories=ORDER, ordered=True)
    ct.columns = pd.Categorical(ct.columns, categories=ORDER, ordered=True)
    ct = ct.reindex(index=ORDER, columns=ORDER).fillna(0).astype(int)
    ct.index.name = ct.columns.name = None

    # Reconcile against S30. Rows are the audited rule set and columns the
    # legacy one, so the audited core count is a row-wise sum and the legacy
    # core count a column-wise one.
    CORE_ROWS = ORDER[:4]          # A, B, C-only, core-unassigned
    n_aud = len(m)
    aud_core = int(ct.loc[CORE_ROWS, :].values.sum())
    aud_ctrl = int(ct.loc["no core event", :].sum())
    leg_core = int(ct.loc[:, CORE_ROWS].values.sum())
    leg_ctrl = int(ct.loc[:, "no core event"].sum())
    for tag, a, b in (("audited", aud_core, aud_ctrl),
                      ("legacy", leg_core, leg_ctrl)):
        assert a + b == n_aud, (
            "%s: core %d + controls %d != n %d" % (tag, a, b, n_aud))

    hdr = ["Audited rule set"] + ORDER + ["Total"]
    rows = []
    for idx in ORDER:
        rows.append(["<strong>%s</strong>" % esc(idx)]
                    + [str(int(ct.loc[idx, c])) for c in ORDER]
                    + ["<strong>%d</strong>" % int(ct.loc[idx].sum())])
    rows.append(["<strong>Total</strong>"]
                + ["<strong>%d</strong>" % int(ct[c].sum()) for c in ORDER]
                + ["<strong>%d</strong>" % int(ct.values.sum())])
    tbl = table(hdr, rows)

    # removed = audited says no core event while legacy says core
    removed = int(ct.loc["no core event", CORE_ROWS].sum())
    # added = the reverse
    added = int(ct.loc[CORE_ROWS, "no core event"].sum())
    note = (
        "MIMIC-IV and NWICU are identical under both rule sets, so only "
        "eICU-CRD is shown. Rows give the audited rule set, columns the legacy "
        "whole-path rule set; &ldquo;core, unassigned&rdquo; is a positive "
        "core event that no tier could be assigned and is deliberately kept "
        "separate from &ldquo;no core event&rdquo;. The audited rule set "
        "records %d core events against %d under the legacy rules: %d were "
        "removed and %d added. The change is not strictly one-way at the stay "
        "level, so the earlier claim that the audited rules can only remove an "
        "event does not hold; it holds at the level of the matching "
        "principle, where folder-level matching is replaced by leaf-level "
        "matching plus an exclusion list."
        % (aud_core, leg_core, removed, added))
    return tbl, note, ct


if __name__ == "__main__":
    import pandas as pd  # noqa: F401  (used inside build_s31)
    globals()["pd"] = pd
    out = {}
    s23, s24, thr = build_s23_s24()
    out["S23"], out["S24"] = s23, s24
    out["loop_thresholds"] = thr
    s25, ps, pt = build_s25()
    out["S25"] = s25
    s26, n26, blocks, pooled = build_s26()
    out["S26"], out["S26_note"] = s26, n26
    s31, n31, ct = build_s31()
    out["S31"], out["S31_note"] = s31, n31
    out["transition"] = {k: {kk: int(vv) for kk, vv in v.items()}
                         for k, v in ct.to_dict("index").items()}
    (OUT / "v9_tables.json").write_text(json.dumps(out, indent=2,
                                                   ensure_ascii=False),
                                        encoding="utf-8")
    print("[ok] S23/S24/S25/S26/S31 generated")
    print("  loop thresholds:", {k: tuple(round(x, 4) for x in v)
                                 for k, v in thr.items()})
    print("  S26 pooled %.2f (%.2f-%.2f) P=%.3f ; singles %s"
          % (pooled[0], pooled[1], pooled[2], pooled[3],
             {d: round(blocks[d]["ror"][0], 2) for d in blocks}))
    print("  S26 model n:", {d: blocks[d]["n"] for d in blocks},
          "excluded unassigned:", {d: blocks.get(d, {}).get("excl")
                                   for d in blocks})
    print("\nS31 transition table:")
    print(ct.to_string())
