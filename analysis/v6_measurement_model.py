import os
# -*- coding: utf-8 -*-
"""
v6 — rebuild the quantitative bias framework on an explicit measurement model.

WHY THIS FILE EXISTS
--------------------
v2-v5 used  R = p*/PPV  (i.e. it assumed p* = R x PPV).  That is wrong for
PPV = P(Y=1 | Y*=1):

        PPV * p*  =  P(Y=1, Y*=1)  =  Se * R

so with Se < 1 nothing is identified from (PPV, p*) alone; with Se = 1 one gets
R = PPV * p*  (true prevalence BELOW the coded prevalence), not p*/PPV.

This rebuild does it properly:

    p*_s = Se * R_s + (1 - Sp) * (1 - R_s) + c_s * (1 - R_s)

    Y   = clinically attributed NPSLE (latent)
    Y*  = recorded neuropsychiatric event (observed)
    S   = sepsis status; c_1 = definitional-loop term, c_0 = 0

  Layer 1  c = 0, Se/Sp equal across arms  -> de-attenuated OR, derived PPV
  Layer 2  true OR forced to 1             -> loop term c* that would null the
                                              observed association
  Layer 3  E-value with the sqrt(OR) ~ RR approximation for a COMMON outcome

PPV is now an OUTPUT of each (Se, Sp) scenario, not an input.
No new data: every number is derived from the published first-stay 2x2 tables.
"""
import io
import json
import math
import pathlib

import pandas as pd

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ROOT / "out"

# first-stay counts (identical to qbf_bias_framework.py / main Table 1)
#   ev = recorded events; ev_sep = recorded events among septic
#   nev = no recorded event; nev_sep = no recorded event among septic
# first-stay counts are parsed from out/t22a_first_sepsis.csv so that the
# framework always follows the phenotype definition currently in force
def _parse(txt):
    a, b = txt.split("/")[0].split()[0], txt.split("/")[1].split()[0]
    return int(a), int(b)


COH = {}
for _lab, _fn in (("MIMIC-IV", "MIMIC-IV"), ("eICU-CRD", "eICU-CRD")):
    _r = pd.read_csv(ROOT / "out" / "t22a_first_sepsis.csv")
    _row = _r[_r["数据库"] == _fn].iloc[0]
    _ev_sep, _ev = _parse(_row["NP事件组脓毒症率"])
    _nev_sep, _nev = _parse(_row["无NP事件组脓毒症率"])
    COH[_lab] = dict(n=int(_row["首次住院 n"]), ev=_ev, ev_sep=_ev_sep,
                     nev=_nev, nev_sep=_nev_sep)


def _or(text):
    """Parse '2.44 (1.36-3.35)' into a dict of floats.

    The interval separator in the result tables is an en dash (U+2013) as
    written by the analysis scripts, so the pattern accepts '-' as well as the
    two unicode dash characters instead of only the ASCII hyphen.
    """
    import re as _re
    m = _re.search(r"([0-9.]+)\s*\(([0-9.]+)\s*[-\u2013\u2014]\s*([0-9.]+)\)", str(text))
    if not m:
        raise SystemExit("cannot parse OR %r" % text)
    return dict(orv=float(m.group(1)), lo=float(m.group(2)),
                hi=float(m.group(3)))


def _orcol(df):
    """The last column whose header names an odds ratio with a 95% interval.

    Columns used to be addressed positionally, which silently broke as soon as
    the result tables gained trailing columns (n, event count): r.columns[-3]
    stopped being the odds ratio and became a P value or a denominator.
    """
    cand = [c for c in df.columns
            if "OR" in str(c) and ("95" in str(c) or "CI" in str(c))]
    if not cand:
        raise SystemExit("no odds-ratio column in %s" % list(df.columns))
    return cand[-1]


def _pooled(fn, label_contains, or_col=None):
    """Pooled adjusted OR for the row carrying the marker.

    The marker is not always in the first column: in t25a the first column is
    the database and the marker ("Tier C") is the second, so every column is
    searched. When the table carries a pooled row (marked "k=") that row is
    preferred, because it is the estimate the framework is built on.
    """
    r = pd.read_csv(ROOT / "out" / fn)
    c0 = r.columns[0]
    mask = pd.Series(False, index=r.index)
    for c in r.columns:
        mask |= r[c].astype(str).str.contains(label_contains, regex=False)
    hit = r[mask]
    if hit.empty:
        raise SystemExit("no row matching %r in %s (col0=%r)"
                         % (label_contains, fn, c0))
    pooled = hit[hit[c0].astype(str).str.contains("k=", regex=False)]
    row = (pooled if not pooled.empty else hit).iloc[0]
    col = or_col or _orcol(r)
    return _or(row[col])


# Every estimate below is read from the result tables, so a change in the
# phenotype definition propagates here instead of leaving a stale constant.
POOLED_SEPSIS = _pooled("t22a_first_sepsis.csv", "k=2")
POOLED_TIERC = _pooled("t25a_tier_with_pooled.csv", "Tier C")
_t22b = pd.read_csv(ROOT / "out" / "t25a_tier_with_pooled.csv")
TIERC_DB = {}
_c0, _c1, _orc = _t22b.columns[0], _t22b.columns[1], _orcol(_t22b)
for _lab in ("MIMIC-IV", "eICU-CRD"):
    _hit = _t22b[(_t22b[_c0] == _lab)
                 & (_t22b[_c1].astype(str).str.contains("Tier C", regex=False))]
    if not _hit.empty:
        TIERC_DB[_lab] = _or(_hit.iloc[0][_orc])

SE_GRID = (1.00, 0.90, 0.80)
SP_GRID = (0.99, 0.95, 0.90)


def arms(c):
    n_sep = c["ev_sep"] + c["nev_sep"]
    n_non = c["n"] - n_sep
    return n_sep, c["ev_sep"], n_non, c["ev"] - c["ev_sep"]


def odds(p):
    return p / (1.0 - p)


def or_from(p1, p0):
    return odds(p1) / odds(p0)


def back_solve(pstar, se, sp):
    """R = (p* - (1-Sp)) / (Se + Sp - 1);  None if outside [0, 1]."""
    denom = se + sp - 1.0
    if denom <= 0:
        return None
    r = (pstar - (1.0 - sp)) / denom
    if r <= 0.0 or r >= 1.0:
        return None
    return r


# ------------------------------------------------------------------ layer 1
def layer1():
    rows = []
    for db, c in COH.items():
        n_sep, a_sep, n_non, a_non = arms(c)
        p1, p0 = a_sep / n_sep, a_non / n_non
        or_obs = or_from(p1, p0)
        for se in SE_GRID:
            for sp in SP_GRID:
                R1 = back_solve(p1, se, sp)
                R0 = back_solve(p0, se, sp)
                if R1 is None or R0 is None:
                    rows.append(dict(db=db, se=se, sp=sp, ok=False, or_obs=or_obs,
                                     p1=p1, p0=p0))
                    continue
                or_true = or_from(R1, R0)
                # PPV in each arm, derived
                ppv1, ppv0 = se * R1 / p1, se * R0 / p0
                # true events captured among the coded events
                true_ev = se * R1 * n_sep + se * R0 * n_non
                rows.append(dict(db=db, se=se, sp=sp, ok=True, or_obs=or_obs,
                                 p1=p1, p0=p0, R1=R1, R0=R0, or_true=or_true,
                                 ppv1=ppv1, ppv0=ppv0,
                                 coded_ev=c["ev"], true_ev=true_ev))
    return rows


# ------------------------------------------------------------------ layer 2
def layer2():
    rows = []
    for db, c in COH.items():
        n_sep, a_sep, n_non, a_non = arms(c)
        p1, p0 = a_sep / n_sep, a_non / n_non
        for se in SE_GRID:
            for sp in SP_GRID:
                R = back_solve(p0, se, sp)
                if R is None:
                    rows.append(dict(db=db, se=se, sp=sp, ok=False))
                    continue
                fp = 1.0 - sp
                cstar = (p1 - se * R) / (1.0 - R) - fp
                rows.append(dict(db=db, se=se, sp=sp, ok=True, R=R,
                                 c=cstar, c_pct=cstar * 100.0,
                                 n_loop=cstar * (1.0 - R) * n_sep))
    return rows


# ------------------------------------------------------------------ layer 3
def evalue(rr):
    """VanderWeele & Ding E-value, risk-ratio scale."""
    if rr is None or rr != rr:
        return float("nan")
    if rr >= 1:
        return rr + math.sqrt(rr * (rr - 1.0))
    inv = 1.0 / rr
    return inv + math.sqrt(inv * (inv - 1.0))


def evalue_or(orr):
    """OR -> E-value via the sqrt(OR) ~ RR approximation (common outcome)."""
    return evalue(math.sqrt(orr))


def layer3():
    out = []
    for label, d in (("Pooled sepsis association (adjusted)", POOLED_SEPSIS),
                     ("Pooled Tier C association (adjusted)", POOLED_TIERC)):
        out.append(dict(est=label, orr=d["orv"], rr=math.sqrt(d["orv"]),
                        ev=evalue_or(d["orv"]),
                        lo=d["lo"], ev_lo=evalue_or(d["lo"]),
                        ev_naive=evalue(d["orv"])))
    for db, d in TIERC_DB.items():
        out.append(dict(est="Tier C, %s" % db, orr=d["orv"], rr=math.sqrt(d["orv"]),
                        ev=evalue_or(d["orv"]), lo=d["lo"],
                        ev_lo=evalue_or(d["lo"]), ev_naive=evalue(d["orv"])))
    return out


def main():
    L1, L2, L3 = layer1(), layer2(), layer3()
    W = []
    A = W.append

    A("=" * 96)
    A("LAYER 1  Explicit measurement model, non-differential (c = 0)")
    A("  p* = Se*R + (1-Sp)*(1-R)   ->   R = (p* - (1-Sp)) / (Se + Sp - 1)")
    A("=" * 96)
    A("%-9s %5s %5s | %7s %7s | %7s %7s | %8s %8s | %6s %6s | %6s" %
      ("db", "Se", "Sp", "p1*", "p0*", "R1", "R0", "OR_obs", "OR_true",
       "PPV1", "PPV0", "trueEv"))
    for r in L1:
        if not r["ok"]:
            A("%-9s %5.2f %5.2f | %7.4f %7.4f |  infeasible (R outside 0-1)" %
              (r["db"], r["se"], r["sp"], r["p1"], r["p0"]))
            continue
        A("%-9s %5.2f %5.2f | %7.4f %7.4f | %7.4f %7.4f | %8.3f %8.3f | %6.3f %6.3f | %6.1f" %
          (r["db"], r["se"], r["sp"], r["p1"], r["p0"], r["R1"], r["R0"],
           r["or_obs"], r["or_true"], r["ppv1"], r["ppv0"], r["true_ev"]))

    A("")
    A("=" * 96)
    A("LAYER 2  Definitional loop c* sufficient to null the association (true OR = 1)")
    A("  c* = (p1* - Se*R) / (1 - R) - (1 - Sp)")
    A("=" * 96)
    A("%-9s %5s %5s | %8s | %9s | %10s" % ("db", "Se", "Sp", "R", "c*", "loop n"))
    for r in L2:
        if not r["ok"]:
            A("%-9s %5.2f %5.2f | infeasible" % (r["db"], r["se"], r["sp"]))
            continue
        A("%-9s %5.2f %5.2f | %8.4f | %8.1f%% | %10.1f" %
          (r["db"], r["se"], r["sp"], r["R"], r["c_pct"], r["n_loop"]))
    ok2 = [r for r in L2 if r["ok"]]
    for db in COH:
        v = [r["c_pct"] for r in ok2 if r["db"] == db]
        if v:
            A("  -> %s: c* range %.1f%% - %.1f%%" % (db, min(v), max(v)))

    A("")
    A("=" * 96)
    A("LAYER 3  E-values, sqrt(OR) ~ RR approximation for a common outcome")
    A("=" * 96)
    A("%-38s %7s %7s | %9s %9s | %12s" %
      ("estimate", "OR", "RR~", "E-value", "E at CI lo", "E if OR naive"))
    for r in L3:
        A("%-38s %7.2f %7.3f | %9.2f %9.2f | %12.2f" %
          (r["est"], r["orr"], r["rr"], r["ev"], r["ev_lo"], r["ev_naive"]))

    txt = "\n".join(W)
    (OUT / "v6_qbf_report.txt").write_text(txt, encoding="utf-8")
    (OUT / "v6_qbf.json").write_text(
        json.dumps(dict(layer1=L1, layer2=L2, layer3=L3),
                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
