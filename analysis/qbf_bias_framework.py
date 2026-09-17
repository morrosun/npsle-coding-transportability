# -*- coding: utf-8 -*-
"""
Quantitative bias framework for administrative neuropsychiatric phenotyping.

Three layers, all derived from the first-stay (primary) 2x2 tables already
reported in the manuscript (Supplementary Table S22a / main Table 1):

  Layer 1  Non-differential PPV deficiency      -> bias vs. precision
  Layer 2  Differential PPV + definitional loop -> OR inflation, threshold c
  Layer 3  E-value for unmeasured confounding
  Layer 4  Effect homogeneity vs. measurement heterogeneity

No new data are generated; every number is an algebraic transform of
published counts.
"""
import io
import json
import math
import pathlib

# Repository root, resolved relative to this file so the pipeline runs
# from a fresh clone on any platform.
ROOT = pathlib.Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- input
# first-stay counts, reconstructed from out/t22a_first_sepsis.csv
#   NP-event group: sepsis / no sepsis ;  no-NP group: sepsis / no sepsis
COH = {
    "MIMIC-IV": dict(n=354, ev=61, ev_sep=25, nev=293, nev_sep=65),
    "eICU-CRD": dict(n=186, ev=77, ev_sep=20, nev=109, nev_sep=20),
}
# Tier C (non-specific) vs no recorded event, first stay  (out/t22b)
TIERC = {
    "MIMIC-IV": dict(k=41, n0=293, orv=3.99, lo=2.00, hi=7.95),
    "eICU-CRD": dict(k=33, n0=109, orv=3.42, lo=1.25, hi=9.33),
}
POOLED_SEPSIS = dict(orv=2.11, lo=1.33, hi=3.36, i2=0.0)
POOLED_TIERC = dict(orv=3.80, lo=2.15, hi=6.70, i2=0.0)


def build_2x2(c):
    """Return (n_sep, a_sep_np, n_nonsep, a_nonsep_np) from the reported rates."""
    n_sep = c["ev_sep"] + c["nev_sep"]
    a_sep = c["ev_sep"]                      # NP events among septic
    n_non = c["n"] - n_sep
    a_non = c["ev"] - c["ev_sep"]            # NP events among non-septic
    return n_sep, a_sep, n_non, a_non


def or_from(a, b, c, d):
    """OR for exposed (a/(a+b)) vs unexposed (c/(c+d)) -> (a/c)/(b/d)."""
    return (a / b) / (c / d)


# ---------------------------------------------------------------- layer 1
def layer1():
    """Non-differential PPV: OR distortion and effective-event loss."""
    rows = []
    for db, c in COH.items():
        n_sep, a_sep, n_non, a_non = build_2x2(c)
        p1 = a_sep / n_sep
        p0 = a_non / n_non
        or_obs = or_from(a_sep, n_sep - a_sep, a_non, n_non - a_non)
        for ppv in (1.00, 0.80, 0.60, 0.50, 0.40):
            # de-attenuate assuming p* = R * PPV with equal PPV in both arms
            R1 = p1 / ppv
            R0 = p0 / ppv
            if R1 >= 1 or R0 >= 1:
                or_true = float("nan")
            else:
                or_true = (R1 / (1 - R1)) / (R0 / (1 - R0))
            rows.append(dict(db=db, ppv=ppv, p1=p1, p0=p0, or_obs=or_obs,
                             R1=R1, R0=R0, or_true=or_true,
                             eff_ev=c["ev"] * ppv))
    return rows


# ---------------------------------------------------------------- layer 2
def layer2():
    """
    Differential model:
        unexposed:  p0* = R * PPV                     (no loop term)
        exposed:    p1* = R * PPV + (1 - R) * c
    Solve for c such that the true OR equals 1 (i.e. the entire observed
    association is produced by the definitional loop).
    """
    rows = []
    for db, c in COH.items():
        n_sep, a_sep, n_non, a_non = build_2x2(c)
        p1 = a_sep / n_sep
        p0 = a_non / n_non
        for ppv in (1.00, 0.80, 0.60, 0.50):
            R = p0 / ppv
            if R >= 1:
                rows.append(dict(db=db, ppv=ppv, R=float("nan"),
                                 c=float("nan"), n_loop=float("nan")))
                continue
            cstar = (p1 - p0) / (1 - R)
            rows.append(dict(db=db, ppv=ppv, R=R, c=cstar,
                             n_loop=cstar * (1 - R) * n_sep))
    return rows


def layer2_reverse():
    """
    Forward check: with a loop term c fixed at literature values for
    sepsis-associated encephalopathy, what OR survives?
    True risk R is back-solved from the unexposed arm at each PPV.
    """
    rows = []
    for db, c in COH.items():
        n_sep, a_sep, n_non, a_non = build_2x2(c)
        p1 = a_sep / n_sep
        p0 = a_non / n_non
        for ppv in (1.00, 0.80, 0.60):
            R = p0 / ppv
            if R >= 1:
                continue
            for c_loop in (0.00, 0.10, 0.20, 0.30, 0.40):
                # true OR assumed = 1 -> R identical in both arms
                q1 = R * ppv + (1 - R) * c_loop
                q0 = R * ppv
                if q1 >= 1 or q0 >= 1:
                    continue
                rows.append(dict(db=db, ppv=ppv, c_loop=c_loop,
                                 or_obs=(q1 / (1 - q1)) / (q0 / (1 - q0))))
    return rows


# ---------------------------------------------------------------- layer 3
def evalue(orr):
    """VanderWeele & Ding E-value for an OR>1 (RR approximation)."""
    if orr is None or orr != orr or orr < 1:
        return float("nan")
    return orr + math.sqrt(orr * (orr - 1.0))


def layer3():
    out = []
    for name, d in (("Pooled sepsis association (adjusted)", POOLED_SEPSIS),
                    ("Pooled Tier C association (adjusted)", POOLED_TIERC)):
        out.append(dict(est=name, point=d["orv"], ev_point=evalue(d["orv"]),
                        ci_lo=d["lo"], ev_lo=evalue(d["lo"])))
    for db, t in TIERC.items():
        out.append(dict(est="Tier C, %s" % db, point=t["orv"],
                        ev_point=evalue(t["orv"]), ci_lo=t["lo"],
                        ev_lo=evalue(t["lo"])))
    return out


# ---------------------------------------------------------------- layer 4
def layer4():
    k = json.load(io.open(ROOT / "out" / "_v9_key.json", encoding="utf-8"))["conly_first"]
    pm, pe = k["pct_m"], k["pct_e"]
    # Wald CI for the difference of two proportions
    n1, n2 = k["mimiciv"][1], k["eicu"][1]
    p1, p2 = k["mimiciv"][0] / n1, k["eicu"][0] / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    return dict(pct_m=pm, pct_e=pe, diff=diff * 100,
                lo=(diff - 1.96 * se) * 100, hi=(diff + 1.96 * se) * 100,
                chi2=k["chi2"], P=k["P"],
                or_m=TIERC["MIMIC-IV"]["orv"], or_e=TIERC["eICU-CRD"]["orv"],
                i2=POOLED_TIERC["i2"])


def main():
    L1, L2, L2r, L3, L4 = layer1(), layer2(), layer2_reverse(), layer3(), layer4()
    res = dict(layer1=L1, layer2=L2, layer2_reverse=L2r, layer3=L3, layer4=L4)

    lines = []
    W = lines.append
    W("=" * 78)
    W("LAYER 1  Non-differential PPV deficiency")
    W("=" * 78)
    W("%-10s %5s %8s %8s %8s %8s %8s %7s" %
      ("db", "PPV", "p1*", "p0*", "OR_obs", "R1", "R0", "OR_true"))
    for r in L1:
        W("%-10s %5.2f %8.4f %8.4f %8.3f %8.4f %8.4f %7s" %
          (r["db"], r["ppv"], r["p1"], r["p0"], r["or_obs"], r["R1"], r["R0"],
           ("%.3f" % r["or_true"]) if r["or_true"] == r["or_true"] else "n/a"))
    W("")
    W("effective events at PPV=0.5: " +
      ", ".join("%s %.1f (of %d)" %
                (r["db"], r["eff_ev"], COH[r["db"]]["ev"])
                for r in L1 if r["ppv"] == 0.50))

    W("")
    W("=" * 78)
    W("LAYER 2  Definitional loop: c needed to explain away the association")
    W("=" * 78)
    W("%-10s %5s %8s %10s %10s" % ("db", "PPV", "R", "c*", "loop n"))
    for r in L2:
        W("%-10s %5.2f %8s %10s %10s" %
          (r["db"], r["ppv"],
           "%.4f" % r["R"] if r["R"] == r["R"] else "n/a",
           "%.1f%%" % (r["c"] * 100) if r["c"] == r["c"] else "n/a",
           "%.1f" % r["n_loop"] if r["n_loop"] == r["n_loop"] else "n/a"))

    W("")
    W("--- forward check: OR observed if true OR=1 and loop term is c ---")
    W("%-10s %5s %8s %10s" % ("db", "PPV", "c", "OR_obs"))
    for r in L2r:
        W("%-10s %5.2f %8.2f %10.3f" % (r["db"], r["ppv"], r["c_loop"], r["or_obs"]))

    W("")
    W("=" * 78)
    W("LAYER 3  E-values (VanderWeele & Ding)")
    W("=" * 78)
    W("%-40s %8s %10s %8s %10s" % ("estimate", "OR", "E-value", "CI lo", "E-value lo"))
    for r in L3:
        W("%-40s %8.2f %10.2f %8.2f %10.2f" %
          (r["est"], r["point"], r["ev_point"], r["ci_lo"], r["ev_lo"]))

    W("")
    W("=" * 78)
    W("LAYER 4  Effect homogeneity vs. measurement heterogeneity")
    W("=" * 78)
    W("Tier C-only share: MIMIC %.1f%% vs eICU %.1f%%" % (L4["pct_m"], L4["pct_e"]))
    W("difference %.1f pp (95%% CI %.1f to %.1f); chi2=%.2f, P=%.4f" %
      (L4["diff"], L4["lo"], L4["hi"], L4["chi2"], L4["P"]))
    W("Tier C OR: MIMIC %.2f vs eICU %.2f; pooled I2=%.1f%%" %
      (L4["or_m"], L4["or_e"], L4["i2"]))

    txt = "\n".join(lines)
    io.open(ROOT / "out" / "qbf_report.txt", "w", encoding="utf-8").write(txt)
    io.open(ROOT / "out" / "qbf_results.json", "w", encoding="utf-8").write(
        json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(txt)


if __name__ == "__main__":
    main()
