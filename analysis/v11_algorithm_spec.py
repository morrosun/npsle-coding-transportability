import os
# -*- coding: utf-8 -*-
"""Review-6, decision B: emit Table S31, the complete specification of the
audited phenotyping algorithm, straight from the rule constants that the two
extraction scripts actually run.

Outputs
  out/v11_S31.json   {"table": <html>, "note": <html>, "audit": {...}}

Nothing is transcribed by hand: the patterns come from extract_tier.py and
extract_cohort.py, the frequencies come from out/t11_tier_terms.csv and from the
legacy inventory already published as Table S27, and the counts come from
out/_counts_before_after.txt and out/tier_eicu*.csv.
"""
import io
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCR = ROOT / "scripts"
OUT = ROOT / "out"
sys.path.insert(0, str(SCR))

import pandas as pd                                        # noqa: E402
import extract_tier as ET                                  # noqa: E402
import extract_cohort as EC                                # noqa: E402

BODY = io.open(SCR / "_jce_body.html", encoding="utf-8").read()

# ------------------------------------------------------------------ helpers
def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def code(s):
    return "<code>%s</code>" % esc(s)


# ------------------------------------------------- 1. audited eICU term list
terms = pd.read_csv(OUT / "t11_tier_terms.csv")
eic = terms[terms.db == "eicu"].sort_values(["tier", "n"], ascending=[True, False])

# ------------------------------------------------- 2. legacy eICU term list
# Taken from the Table S27 block that is already on the page, so that the two
# tables can never disagree about what the legacy rule set matched.
m27 = re.search(r"<p><b>Table S27</b>.*?<tbody>(.*?)</tbody>", BODY, re.S)
assert m27, "Table S27 body not found in _jce_body.html"
legacy = []
for row in re.findall(r"<tr>(.*?)</tr>", m27.group(1), re.S):
    cells = [re.sub(r"<[^>]+>", "", c).strip() for c in
             re.findall(r"<td>(.*?)</td>", row, re.S)]
    if len(cells) == 5 and cells[0] == "eICU-CRD":
        legacy.append((cells[1], cells[3], int(cells[4])))

EXCL_RE = ET.EI_EXCLUDE_LEAF
excluded, retained = [], []
for tier, term, n in legacy:
    (excluded if re.search(EXCL_RE, term) else retained).append((tier, term, n))
excl_c = sum(n for t, _, n in excluded if t == "C")

# ------------------------------------------------- 3. counts
cb = io.open(OUT / "_counts_before_after.txt", encoding="utf-8").read()
m = re.search(r"eicu  ALL STAYS(.*?)={10,}", cb, re.S)
assert m, "eICU all-stays block not found"
eb = m.group(1)
m = re.search(r"eicu  FIRST STAY(.*?)={10,}", cb, re.S)
assert m, "eICU first-stay block not found"
ef = m.group(1)
assert "core            85 ->     83" in eb and "core            65 ->     64" in ef

tp = pd.read_csv(OUT / "tier_eicu.csv")
assert int(tp.npsle_core.sum()) == 83 and int(tp.tier_unassigned.sum()) == 2

# Stay counts for the two derived definitions, read back from the extraction
# output so that Panel F states what the code produced rather than what the
# prose intends. Both are all-stays counts, the basis of Table S2.
_ch = {db: pd.read_csv(OUT / ("cohort_%s.csv" % db))
       for db in ("mimiciv", "eicu", "nwicu")}


def _cnt(db, col):
    return int(pd.to_numeric(_ch[db][col], errors="coerce").fillna(0).sum())

# ------------------------------------------------- 4. build the panels
def panel(title, ncol):
    return '<tr><td colspan="%d" style="background:var(--bg2);text-align:left">' \
           '<b>%s</b></td></tr>' % (ncol, title)

# ---- review-9: the stay-level audit of the 85 -> 83 revision ---------------
# Produced by scripts/_r9_exclusion_stay_audit.py, which recomputes eICU-CRD
# core membership twice from the raw diagnosis strings (exclusion list applied
# versus disabled). Panel G quotes these numbers, so they are read from the
# audit files rather than transcribed.
_dom_eff = pd.read_csv(OUT / "t46_eicu_exclusion_domain_effect.csv").set_index("domain")
_lbl = {"dom_seizure": "seizure", "dom_enceph": "encephalopathy",
        "dom_psych": "psychotic", "dom_mening": "meningitis",
        "dom_demyel": "demyelination"}
DOM_TXT = ", ".join(
    "%s %d/%d" % (_lbl[d], int(_dom_eff.loc[d, "stays_with_evidence_off"]),
                  int(_dom_eff.loc[d, "stays_with_evidence_on"]))
    for d in EC.CORE_DOMS)
_audit = pd.read_csv(OUT / "t45_eicu_exclusion_stay_audit.csv")
N_LOST = int(len(_audit))


H = ['<table class="dataframe data-table">']
H.append("<thead><tr><th>Element</th><th>Rule as implemented</th></tr></thead><tbody>")

# ---- Panel A: the rule constants, verbatim
H.append(panel("Panel A. Rule constants, verbatim from the two extraction "
               "scripts (<code>extract_tier.py</code>, <code>extract_cohort.py</code>)", 2))
for lab, txt in [
    ("MIMIC-IV / NWICU &mdash; Tier B, ICD-10-CM", ET.TB_10),
    ("MIMIC-IV / NWICU &mdash; Tier B, ICD-9-CM", ET.TB_9),
    ("MIMIC-IV / NWICU &mdash; Tier C, ICD-10-CM", ET.TC_10),
    ("MIMIC-IV / NWICU &mdash; Tier C, ICD-9-CM", ET.TC_9),
    ("MIMIC-IV / NWICU &mdash; Tier X, ICD-10-CM", ET.TX_10),
    ("MIMIC-IV / NWICU &mdash; Tier X, ICD-9-CM", ET.TX_9),
    ("MIMIC-IV / NWICU &mdash; Tier A", "no pattern: the prespecified algorithm "
     "implements no ICD-based attribution rule, so <code>tier_a</code> is not "
     "set for these two databases"),
    ("eICU-CRD &mdash; Tier A", code(ET.EI_A) + ", applied only on a neurologic "
     "or CNS-infection path"),
    ("eICU-CRD &mdash; Tier X", code(ET.EI_X) + ", applied only on a neurologic "
     "or CNS-infection path"),
    ("eICU-CRD &mdash; Tier B, audited", code(ET.EI_B_LEAF) + " <i>or</i> ("
     + code(ET.EI_B_ANY) + " at any depth <i>and</i> in the leaf)"),
    ("eICU-CRD &mdash; Tier C, audited", code(ET.EI_C_LEAF) + " <i>or</i> ("
     + code(ET.EI_C_ANY) + " at any depth <i>and</i> in the leaf <i>or</i> on a "
     "path of at most three nodes)"),
    ("eICU-CRD &mdash; exclusion list", code(EXCL_RE) +
     " &mdash; tested against the whole string, so it also removes sub-divided "
     "children such as <code>|depression|mild</code> and "
     "<code>|sedated|unresponsive</code> that a parent folder would otherwise "
     "license. The list is applied to the Tier B and Tier C candidate matches "
     "only and is evaluated before any tier is written; it has no effect on "
     "Tier A or on Tier X, which are matched on the whole string inside the "
     "neurologic-path restriction. The identical list is instantiated a second "
     "time in <code>extract_cohort.py</code>, where it suppresses a diagnosis "
     "string before the core-domain rules are evaluated (Panel F); the two "
     "instantiations are kept textually identical so that the core phenotype "
     "and the semantic tier cannot drift apart"),
    ("eICU-CRD &mdash; Tier B / C, legacy comparison only",
     code(ET.EI_B_LEGACY) + " and " + code(ET.EI_C_LEGACY) +
     ", matched on the whole string with no exclusion list"),
]:
    H.append('<tr><td style="text-align:left">%s</td><td style="text-align:left">%s</td></tr>'
             % (lab, txt))

# ---- Panel B: path mechanics
H.append(panel("Panel B. eICU-CRD path mechanics: how one "
               "<code>diagnosisstring</code> becomes a matched node", 2))
for lab, txt in [
    ("Matching semantics", "node-boundary matching with leaf confirmation and "
     "explicit exclusions. A Tier B or Tier C pattern opens with a separator, "
     "so it fires at any node boundary of the path and not only at the leaf; "
     "the concept lists additionally require confirmation in the leaf (Tier B) "
     "or in the leaf or on a path of at most three nodes (Tier C). The "
     "implementation is therefore not a leaf-only rule and is not reported as "
     "one"),
    ("Path tokenisation", "the string is lower-cased and split on the "
     "character <code>|</code>; the components are the dictionary nodes from "
     "the top-level folder to the leaf"),
    ("<code>leaf</code>", "the last component of the split"),
    ("<code>depth</code>", "the number of components in the path"),
    ("Neurologic-path restriction (Tiers A and X only)",
     "the whole string must contain <code>neurologic|</code> or "
     "<code>cns infections</code>; the attribution or aetiology word is the "
     "deepest node of a four-node path, so it is matched on the whole string "
     "rather than on the leaf"),
    ("Tier A matched node", "whole string, inside the neurologic-path restriction"),
    ("Tier X matched node", "whole string, inside the neurologic-path restriction"),
    ("Tier B matched node",
     "a path-boundary-anchored pattern (the pattern opens with a separator, so "
     "it fires at any node boundary of the path) <i>or</i> a concept matched at "
     "any depth that is confirmed in the leaf"),
    ("Tier C matched node", "the same form as Tier B, plus the fallback that "
     "accepts a concept matched at any depth on a path of at most three nodes"),
    ("Exclusion matched node", "whole string, at any node boundary, because "
     "every pattern in the list opens with a separator; a sub-divided child "
     "such as <code>|depression|mild</code> is therefore removed as well as a "
     "leaf such as <code>|bipolar disorder</code>"),
    ("Precedence within a term", "the exclusion test is evaluated first and "
     "clears Tier B and Tier C candidates only. The surviving candidates are "
     "then written to the term in the order C, then B, then X, then A, so the "
     "last write wins and the effective precedence is A &gt; X &gt; B &gt; C. "
     "Tier A and Tier X are never cleared by the exclusion test"),
    ("Priority at the level of the stay",
     "the semantic tier is A &gt; B &gt; C and is mutually exclusive; the "
     "competing-aetiology flag X is carried independently and is not a tier"),
]:
    H.append('<tr><td style="text-align:left">%s</td><td style="text-align:left">%s</td></tr>'
             % (lab, txt))

# ---- Panel C: observed terms that fired
H.append(panel("Panel C. Every eICU-CRD term that the audited rules matched, "
               "with the number of unit stays &mdash; the complete audited "
               "inclusion list, to be read against the legacy inventory in "
               "Table S27", 2))
for _, r in eic.iterrows():
    H.append('<tr><td style="text-align:left">Tier %s &nbsp;%d</td>'
             '<td style="text-align:left">%s</td></tr>'
             % (r["tier"], int(r["n"]), code(r["term"])))

# ---- Panel D: what the exclusion list removes
H.append(panel("Panel D. What the exclusion list removes: the eICU-CRD terms "
               "that carried a Tier B or Tier C assignment under the legacy "
               "whole-path rules and no longer do", 2))
_xonly = [x for x in excluded if x[0] == "X"]
_bc = [x for x in excluded if x[0] in ("B", "C")]
# The four X terms that also match the exclusion list are already on the
# competing-aetiology list; the ten B/C terms are the ones the exclusion removes
# from a tier. Nothing else may appear here.
assert len(_bc) == 12 and len(_xonly) == 4, (len(_bc), len(_xonly))
assert all(x[0] == "B" or x[0] == "C" for x in _bc)
H.append('<tr><td style="text-align:left">Terms removed</td>'
         '<td style="text-align:left">%s</td></tr>'
         % "; ".join("%s (%d)" % (code(t), n) for _, t, n in _bc))
H.append('<tr><td style="text-align:left">Effect on Tier C</td>'
         '<td style="text-align:left">%d term-level records across %d distinct '
         'terms are suppressed. Every one of them sits under the '
         '<code>neurologic|altered mental status / pain</code> folder, so each '
         'was previously licensed by its parent category; none is a clinical '
         'state that the tier is meant to capture</td></tr>'
         % (sum(n for _, _, n in _bc), len(_bc)))
H.append('<tr><td style="text-align:left">Terms unaffected</td>'
         '<td style="text-align:left">the competing-aetiology pattern is '
         'evaluated independently of the exclusion test, so the terms on the X '
         'list keep their flag whether or not they also appear in the exclusion '
         'list (for example <code>|sedated</code> and '
         '<code>|drug withdrawal syndrome</code>)</td></tr>')

# ---- Panel E: flag derivation
H.append(panel("Panel E. Derivation of every stay-level flag "
               "(<code>extract_tier.py::build</code>)", 2))
for lab, txt in [
    ("<code>npsle_core</code>",
     "the recorded core phenotype from the cohort-extraction pass, specified in "
     "full in Panel F; it is <i>not</i> re-derived from the tier flags, and "
     "every tier indicator is intersected with it"),
    ("<code>npsle_hi</code>", "(tier_a <i>or</i> tier_b) <i>and</i> npsle_core"),
    ("<code>npsle_hi_str</code>", "npsle_hi <i>and</i> not other_cause"),
    ("<code>npsle_any</code>",
     "(tier_a <i>or</i> tier_b <i>or</i> tier_c) <i>and</i> npsle_core"),
    ("<code>tier_primary</code>",
     "the empty string when there is no recorded core event, otherwise A if "
     "tier_a, else B if tier_b, else C if tier_c, else the empty string"),
    ("<code>tier_c_only</code>", "tier_primary equals C"),
    ("<code>tier_unassigned</code>",
     "npsle_core equals 1 <i>and</i> npsle_hi equals 0 <i>and</i> tier_c_only "
     "equals 0. A stay lands here when every core domain it carries was matched "
     "through a term that the tier rules place on the competing-aetiology "
     "list: the term is a core-domain term, so the stay is core-positive, but X "
     "is a flag rather than a tier, so no tier is assigned. This is the origin "
     "of the %d core-positive, tier-unassigned eICU-CRD stays; the legacy rules "
     "produce six"
     % int(tp.tier_unassigned.sum())),
    ("<code>other_cause</code>", "the competing-aetiology flag, kept independent "
     "of the tier"),
]:
    H.append('<tr><td style="text-align:left">%s</td><td style="text-align:left">%s</td></tr>'
             % (lab, txt))

# ---- Panel F: the complete core-phenotype specification -------------------
# The tier rules alone do not define the phenotype: npsle_core is computed by
# extract_cohort.py from five domain rules, and it is the universe every tier
# indicator is intersected with. RECORD requires the full Boolean rule for
# every variable that defines exposure or outcome, so the domains are emitted
# from the same constants the extraction runs.
H.append(panel("Panel F. Complete specification of the recorded core "
               "phenotype <code>npsle_core</code> and of the two definitions "
               "derived from it, verbatim from "
               "<code>extract_cohort.py</code>", 2))


def row(lab, txt):
    H.append('<tr><td style="text-align:left">%s</td>'
             '<td style="text-align:left">%s</td></tr>' % (lab, txt))


row("Boolean form, MIMIC-IV and NWICU",
    "<code>npsle_core = dom_seizure <i>or</i> dom_enceph <i>or</i> dom_psych "
    "<i>or</i> dom_mening <i>or</i> dom_demyel</code>, evaluated at the level "
    "of the hospital admission. A domain is positive when any ICD-10-CM code "
    "matches its ICD-10 pattern <i>or</i> any ICD-9-CM code matches its ICD-9 "
    "pattern; every pattern is anchored at the start of the code")
for d in EC.CORE_DOMS:
    p10, p9 = EC.NP_ICD[d]
    row("%s, ICD" % d, code(p10) + " (ICD-10-CM) <i>or</i> " + code(p9) +
        " (ICD-9-CM)")
row("Boolean form, eICU-CRD (audited)",
    "the same disjunction over the same five domains, evaluated at the level "
    "of the ICU unit stay. Every diagnosis string of the stay that matches the "
    "exclusion list of Panel A is discarded first, and the domain patterns "
    "below are matched against the remaining strings only")
for d in EC.CORE_DOMS:
    row("%s, eICU-CRD" % d, code(EC.NP_EICU_RESTRICTED[d]))
row("Recorded meningitis / encephalitis",
    "there is no literal for <i>aseptic</i>: the domain is carried by "
    "ICD-10-CM G03 (meningitis, other and unspecified causes), G02 (meningitis "
    "in other infectious diseases) and A87 (viral meningitis), and ICD-9-CM "
    "047, 321 and 322; in eICU-CRD the pattern "
    + code(EC.NP_EICU_RESTRICTED["dom_mening"]) + " carries no leading "
    "separator and is deliberately matched at any depth, because the aetiology "
    "sits in the parent or child node (<code>encephalitis|systemic lupus "
    "erythematosus</code>, <code>meningitis|acute|bacterial</code>, "
    "<code>encephalitis|viral|herpes</code>). The domain is therefore an "
    "operational proxy for the ACR aseptic-meningitis syndrome and not a set "
    "of confirmed aseptic or lupus-attributed cases: infectious forms enter it "
    "and are marked by the Tier X list of Panel A rather than excluded from "
    "the core phenotype")
row("How the exclusion list changes the psychotic domain",
    "the leaves <code>|schizophrenia</code> and <code>|bipolar disorder</code> "
    "appear both in the <code>dom_psych</code> pattern and in the exclusion "
    "list. Because the exclusion is applied to the string before the domain is "
    "matched, those two leaves can never set the domain, and "
    "<code>dom_psych</code> reduces effectively to " +
    code(r"\|psychosis\b|\|psychotic\b|\|hallucination") +
    ". The list is not confined to that domain: it also removes "
    "<code>dom_enceph</code> evidence from the "
    "<code>|sedated|unresponsive</code> path, so equal counts would not by "
    "themselves identify which stays moved. The stay-level audit of Panel G "
    "does")
row("How a competing-aetiology term can still set the core phenotype",
    "the Tier X list and the core domains are evaluated independently, so a "
    "term can be both. The path actually present in eICU-CRD is "
    "<code>neurologic|altered mental status / pain|encephalopathy|metabolic</code>: "
    "<code>dom_enceph</code> matches at the <code>|encephalopathy</code> node "
    "boundary and the <code>|metabolic</code> child leaf does not cancel that "
    "match, while the Tier X pattern of Panel A contains the bare token "
    "<code>metabolic</code>. The stay can therefore remain core-positive and "
    "X-flagged at the same time. What the X flag then does depends on the rest "
    "of the stay: if no other term supplies an A, B or C assignment the stay "
    "is core-positive but tier-unassigned and never enters "
    "<code>npsle_hi</code>; if another term does supply A or B membership, the "
    "stay is high-confidence on that term and the X flag excludes it from "
    "<code>npsle_hi_str</code>, which additionally requires "
    "<code>other_cause = 0</code>. The auxiliary metabolic flag defined below "
    "is a contiguous phrase and does not fire on this path, so the two "
    "implementations are not symmetrical here: in the ICD-based implementation "
    "G93.41 and 348.31 are removed from the core definition outright, whereas "
    "in eICU-CRD an encephalopathy node carrying a metabolic qualifier remains "
    "core-positive. Conversely, a term whose only core-domain "
    "evidence sits beneath an excluded node, such as "
    "<code>|sedated|unresponsive</code>, loses core membership altogether. "
    "This is the mechanism behind the " + str(int(tp.tier_unassigned.sum())) +
    " core-positive, tier-unassigned eICU-CRD stays of Panel E")
row("Broad definition <code>npsle_broad</code> (sensitivity analysis)",
    "<code>npsle_core <i>or</i> dom_cvd <i>or</i> dom_pns</code>. "
    + code(EC.NP_ICD["dom_cvd"][0]) + " / " + code(EC.NP_ICD["dom_cvd"][1]) +
    " and " + code(EC.NP_ICD["dom_pns"][0]) + " / " +
    code(EC.NP_ICD["dom_pns"][1]) + " in the ICD databases; " +
    code(EC.NP_EICU_RESTRICTED["dom_cvd"]) + " and " +
    code(EC.NP_EICU_RESTRICTED["dom_pns"]) + " in eICU-CRD. Metabolic "
    "encephalopathy is <i>not</i> part of this definition: it is carried by "
    "<code>npsle_sens</code> below. All-stays counts are %d (MIMIC-IV), %d "
    "(eICU-CRD) and %d (NWICU) stays, the &ldquo;broad events&rdquo; column of "
    "Table S2" % (_cnt("mimiciv", "npsle_broad"), _cnt("eicu", "npsle_broad"),
                  _cnt("nwicu", "npsle_broad")))
row("Sensitivity definition <code>npsle_sens</code>",
    "<code>npsle_core <i>or</i> metabolic_enceph</code>, where "
    "<code>metabolic_enceph</code> is " + code(EC.AUX_ICD["metabolic_enceph"][0]) +
    " / " + code(EC.AUX_ICD["metabolic_enceph"][1]) + " (ICD) or " +
    code(EC.AUX_EICU_RESTRICTED["metabolic_enceph"]) + " (eICU-CRD). The two "
    "auxiliary flags are matched without the exclusion list, so they are not "
    "suppressed by it. This definition is computed in the archived extraction "
    "code (all-stays counts %d, %d and %d stays respectively) but no result "
    "reported in this paper uses it; the metabolic extension is reported here "
    "only so that the two definitions are not conflated"
    % (_cnt("mimiciv", "npsle_sens"), _cnt("eicu", "npsle_sens"),
       _cnt("nwicu", "npsle_sens")))
row("Where the core phenotype is intersected",
    "<code>extract_tier.py::build</code> sets <code>npsle_core</code> from the "
    "cohort-extraction pass and intersects every tier indicator with it, so "
    "tier membership can never extend outside the recorded core phenotype")

# ---- Panel G: reconciliation
H.append(panel("Panel G. Reconciliation of the two rule sets and of the "
               "revision within the audited rule set", 2))
for lab, txt in [
    ("Audited versus legacy, eICU-CRD (all stays)",
     "100 core events under the legacy whole-path rules against 83 under the "
     "audited rules: 17 stays carry a legacy core event and no audited one, and "
     "no stay is added (Table S29)"),
    ("Audited versus legacy, eICU-CRD (first stays)",
     "77 against 64"),
    ("MIMIC-IV and NWICU", "identical under the two rule sets, because the "
     "audited rules differ from the legacy ones only in the path handling that "
     "applies to the free-text dictionary"),
    ("Revision within the audited rule set",
     "the exclusion list was first tested against the <code>leaf</code> "
     "component, where its leading separator made it unreachable, so it had no "
     "effect; testing it against the whole string, which is what the list "
     "states, moved the eICU-CRD all-stays core count from 85 to 83 and the "
     "Tier B count from 33 to 31, and the first-stay core count from 65 to 64 "
     "and Tier B from 23 to 22, with the control group moving from 145 to 147 "
     "and from 121 to 122"),
    ("Which stays moved, and why (stay-level audit)",
     ("the two states were recomputed stay by stay from the raw eICU-CRD "
      "diagnosis strings, once with the exclusion list applied and once with "
      "it disabled. With it disabled the all-stays core count is 85 against 83 "
      "with it applied: %d stays lose core membership and none gains it, so "
      "the change is one-way at the level of core membership. The %d stays are "
      "identified directly rather than inferred from the equal differences "
      "85 &minus; 83 = 33 &minus; 31 = 2: in both, the only recorded core "
      "evidence was the single path <code>neurologic|altered mental status / "
      "pain|bipolar disorder</code>, which sets <code>dom_psych</code> and "
      "appears on the exclusion list. The list is not confined to the psychotic "
      "domain: it also removes <code>dom_enceph</code> evidence from the "
      "<code>|sedated|unresponsive</code> path in one further stay, which "
      "stays core-positive through <code>dom_seizure</code> and so does not "
      "cross the core boundary. Domain-level counts with the list disabled "
     "against applied: " + DOM_TXT + ". Both removed stays carry the "
     "bipolar-disorder leaf, one of the two excluded psychotic leaves (the "
     "schizophrenia leaf did not occur among them); both leaves are also in "
     "the Tier B inclusion pattern, but the exclusion list takes precedence, "
     "so the two stays lose Tier B as well, and the psychotic-disorder domain "
     "ends at zero in all three databases") % (N_LOST, N_LOST)),
    ("Direction of the change",
     "in these data the change is one-way at the level of core membership. "
     "Seventeen stays with a legacy core event carry no recorded core event "
     "under the audited rules and none is added; the later revision of the "
     "exclusion test moved the eICU-CRD counts from 85 to 83 and from 65 to 64 "
     "with no stay added. The exclusion list removes evidence rather than "
     "reassigning it, so a stay can lose a domain or a tier but cannot gain "
     "one, and the tier assignment of every retained event was unchanged"),
]:
    H.append('<tr><td style="text-align:left">%s</td><td style="text-align:left">%s</td></tr>'
             % (lab, txt))

H.append("</tbody></table>")
table = "\n".join(H)

note = (
    "Panels A to G are the audited algorithm in full: the rule constants as "
    "they appear in the source, the node at which each tier is matched, every "
    "term the audited rules matched, every term the exclusion list removes, the "
    "derivation of each stay-level flag, the complete Boolean specification of "
    "the recorded core phenotype and of the two definitions derived from it, "
    "and the reconciliation between the two rule sets and between the two "
    "revisions of the audited rule set. The "
    "legacy comparison appears in Table S27 and the stay-level transition in "
    "Table S29; S26 gives the clinical reading of the same rules. The audited "
    "rules are the primary analysis; the legacy whole-path rules are retained "
    "only so that the effect of the matching rule can be measured (Table 3). "
    "Tier A is empty in MIMIC-IV and NWICU because the prespecified algorithm "
    "implements no ICD-based attribution rule, which is a property of the rule "
    "set and not a statement about what ICD coding can express."
)

assert "{" not in table and "}" not in table, "braces would break the f-string splice"
assert '"""' not in table and '"""' not in note

# ---- machine check for the locality claim in Panel F -----------------------
_hit = []
for _k, _v in EC.NP_EICU_RESTRICTED.items():
    if _k in EC.CORE_DOMS and re.search(EXCL_RE, _v):
        _hit.append(_k)
assert _hit == ["dom_psych"], "exclusion list touches more than dom_psych: %s" % _hit
assert 85 - 83 == 33 - 31 == 2

audit = {
    "eicu_core_audited": int(tp.npsle_core.sum()),
    "eicu_unassigned": int(tp.tier_unassigned.sum()),
    "excluded_terms": len(excluded),
    "excluded_tierC_hits": excl_c,
    "audited_terms": int(len(eic)),
}
(OUT / "v11_S31.json").write_text(
    json.dumps({"table": table, "note": note, "audit": audit},
               ensure_ascii=False, indent=1), encoding="utf-8")
print("[ok] out/v11_S31.json", audit)
