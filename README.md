# NPSLE Coding Transportability

**Reproducible analysis code** for the study:

*Implementation dependence and cross-database evaluation of administrative
neuropsychiatric phenotyping in critically ill patients with systemic lupus
erythematosus.*

This repository contains **analysis code only**. No patient-level data are
included; all cohorts are derived from PhysioNet-credentialed databases
(MIMIC-IV, eICU-CRD, NWICU) under their respective Data Use Agreements (DUA).
Researchers with PhysioNet credentials can reconstruct the cohorts by running
the SQL check and extraction scripts against their own local database copies.

> **Version:** `1.1.5` — packaging-only update, 2026-11-16. **No estimate, table
> or figure changed**: every number is the one already reproducible from `1.1.4`.
> This release exists so that the deposit matches the paper as submitted, after a
> fresh reviewer-style audit of the repository:
>
> - the README carries the final title and the version the manuscript cites;
> - database credentials are no longer stored in the code. Thirteen scripts read
>   `NPSLE_PG_HOST/PORT/USER/PASSWORD` from the environment through the new
>   `analysis/db.py`, so a reproducer points the pipeline at their own copy
>   without editing source files;
> - the documented run order no longer contains a step that cannot execute.
>   `v11_algorithm_spec.py` (Table S31) used to read `scripts/_jce_body.html`
>   from the manuscript build layer, a file that is not part of this deposit; it
>   now rebuilds S31 from `analysis/fixtures/s27_legacy_table_body.html`, a
>   fixture that is shipped;
> - the legacy figure writers (`fig_revision.py`, `part1_epidemiology.py`,
>   `positive_analyses.py`, `v7_identification_models.py`) now emit **English**
>   panel labels, so no run of this code can produce a figure that differs from
>   the published one by language;
> - `fig_en.py` now **fails** when the SHAP panel cannot be rebuilt instead of
>   printing a warning and leaving a stale file, and `create_fig_composites.py`
>   refuses to assemble a composite whose panels are missing. Both close the same
>   hole: a partially-run pipeline could previously carry a non-English or
>   superseded panel into Supplementary Figure S2;
> - `make_first_stays.py` is now `select_one_stay_per_patient.py`, matching the
>   cohort name used throughout the paper. The intermediate file
>   `out/_first_stays.json` keeps its original name for provenance.
>
> **Release history.** `1.1.4` (2026-10-24) made the pipeline self-contained —
> eight defects found by the reproduction audit (missing helpers, a backup-CSV
> dependency, two absolute paths, one hard-coded era label, a silent figure-check
> exit). `1.1.3` completed the "one stay per patient" renaming in the archived
> code; `1.1.2` fixed the ROC/calibration panels and the composite rebuild;
> `1.1.1` corrected the eICU-CRD selection rule (offsets are measured from each
> ICU admission, so ascending `hospitaladmitoffset` is biased towards the *later*
> stay), the legacy arm's core universe in `extract_tier.py`, and a dead
> `try/except` around `RepeatedStratifiedGroupKFold`, a class scikit-learn does
> not have; `1.1.0` (2026-10-05) shipped the review-10 pass; `1.0.0`
> (2026-09-18) was an earlier, partially inconsistent pass. Two superseded
> scripts are kept under `legacy_pre_review10/` for transparency.

## What the study does

Neuropsychiatric SLE (NPSLE) cannot be attributed from administrative data
alone, so the study treats the *recorded* phenotype as the object of study
rather than as a proxy for the clinical entity. Three public ICU databases are
compared on:

1. **Prevalence and composition** of coded neuropsychiatric events in critically
   ill patients with SLE, and how much of the observed variation is definitional
   rather than clinical;
2. **Diagnostic-confidence tiering** — a transparent, investigator-proposed
   stratification of the coding terms (specific / compatible / non-specific),
   reported separately per database rather than pooled;
3. **Sepsis co-occurrence** — the association between sepsis and coded
   neuropsychiatric events, and the extent to which it is carried by
   non-specific codes that overlap by construction with septic encephalopathy;
4. **Quantitative bias framework** — the algebra needed to decide whether an
   observed association survives imperfect positive predictive value,
   definitional overlap, and residual confounding;
5. **Transportability of routine-data prediction models** — internal versus
   external discrimination and calibration across databases.

Two databases (MIMIC-IV, eICU-CRD) contribute to every pooled estimate;
NWICU is retained for descriptive corroboration only. INSPIRE was screened and
excluded because it contains no SLE diagnostic entries (its ICD-10 subset omits
the entire M30–M35 block).

## Repository layout

```
sql/feasibility.sql                 database inventory: SLE / NPSLE coding availability,
                                    term-level mappings, per-database schema differences
analysis/npsle_io.py                single unified loader: every analysis script builds its
                                    stay-level frame through load(); ROOT is resolved relative
                                    to the repository root, so outputs are written to out/
analysis/extract_cohort.py          builds the aligned three-database cohorts (core + broad
                                    phenotype, seven neuropsychiatric domains, covariates)
analysis/extract_tier.py            diagnostic-confidence tier assignment
analysis/db.py                      one place that defines the PostgreSQL connection: reads
                                    NPSLE_PG_HOST/PORT/USER/PASSWORD from the environment, so no
                                    credential is stored in the repository
analysis/select_one_stay_per_patient.py
                                    (formerly make_first_stays.py) prerequisite: writes
                                    out/_first_stays.json — ONE ICU stay per
                                    patient (the first ICU stay of the earliest admission; for
                                    eICU-CRD the earliest hospitaldischargeyear, then the smallest
                                    unitvisitnumber, then the largest hospitaladmitoffset). A
                                    one-stay-per-patient selection, deliberately not called a
                                    verified 'first stay', because eICU-CRD offsets are measured
                                    from each admission and are not comparable across admissions.
                                    Consumed by npsle_io.load() and every one-stay-per-patient
                                    sensitivity analysis.
analysis/describe_cohort.py         feature completeness / baseline / subtype–outcome summaries
analysis/part1_epidemiology.py      Part 1: baseline, prevalence, outcomes, pooling
analysis/positive_analyses.py       sepsis co-occurrence, tier stratification, severity and
                                    immunosuppression sensitivity
analysis/v4_analyses.py … v11_algorithm_spec.py
                                    the sequential analysis chain; each script adds or refines a
                                    specific table (t20–t46, the bias-framework JSONs, and the
                                    Table S31 algorithm specification). They are kept in
                                    manuscript order so the provenance of every number is auditable.
analysis/v7_identification_models.py Part 2: prediction models with patient-grouped internal
                                    validation — five repeats of five-fold
                                    StratifiedGroupKFold, each repeat with a prespecified random
                                    seed. scikit-learn has no RepeatedStratifiedGroupKFold class,
                                    so the repetition is implemented explicitly.
analysis/v6_measurement_model.py    three-layer quantitative bias framework (Layers 1–3)
analysis/v6_tier_multinomial.py     tier multinomial → tier coefficient ratio (ROR)
analysis/v8_tables.py / v9_tables.py  table generators for S30/S31/S32 and S22–S25/S29
analysis/fig_en.py, fig_revision.py, fig1_v9.py, fig_ladder.py,
            create_fig_composites.py, make_strobe.py   figures and participant flow
analysis/fixtures/s27_legacy_table_body.html
                                    the Table S27 body, shipped so that v11_algorithm_spec.py
                                    can rebuild Table S31 without the manuscript build layer
analysis/export_final_numbers.py     provenance tool: consolidates all result CSVs/JSONs into
                                    out/final_numbers.json (machine-readable, keyed by manuscript
                                    figure/table)
legacy_pre_review10/                 superseded scripts retained for transparency:
                                      - part2_model.py        (un-grouped RepeatedStratifiedKFold;
                                                               replaced by v7_identification_models.py)
                                      - qbf_bias_framework.py (replaced by v6_measurement_model.py)
```

## Pipeline

Run every script from the **repository root**; all paths are resolved relative to
the repository (`npsle_io.py` is the single source of `ROOT`/`OUT`), and outputs
are written to `out/` (created on demand).

**No data are distributed.** `out/` is generated, not shipped: the PhysioNet Data
Use Agreement forbids redistributing patient-level extracts, so a reproducer needs
their own credentialed copies of MIMIC-IV, eICU-CRD and NWICU and must run steps 1-2
before anything else. Every statistic in the manuscript can be regenerated from them;
`out/final_numbers.json` is the final consolidated export.

Reproduction is **value-level, not byte-level**: `extract_cohort.py` issues no
`ORDER BY`, so the row order of `cohort_*.csv` can differ between runs on the same
data, and a few pooled estimates differ in the last floating-point digit. Every
reported statistic is unchanged; do not read a differing md5 as a differing result.
A clean copy regenerates `out/final_numbers.json` with all of its keys and the same
values.

1. **Screen the databases** — `psql -f sql/feasibility.sql` against each local
   copy records which SLE and neuropsychiatric code families are actually
   present, and which are harmonisable across databases.
2. **Extract the cohorts** —
   `python analysis/extract_cohort.py mimiciv eicu nwicu`
   writes `out/cohort_<db>.csv` with column names aligned across databases.
2b. **Generate the one-stay-per-patient index (required prerequisite)** —
   `python analysis/select_one_stay_per_patient.py`
   reads the cohort CSVs and writes `out/_first_stays.json` — one ICU stay per
   subject, which defines the main-analysis cohort (MIMIC-IV 354,
   eICU-CRD 186, NWICU 43; total 583). This file is required before
   `extract_tier.py`, `v4_analyses.py` and `v7_identification_models.py`, which
   consume it through `npsle_io.load()`. Without it those scripts raise
   `FileNotFoundError`.
2c. **Exclusion-list audit (eICU-CRD)** — `python analysis/_r11_exclusion_audit.py`
   writes `out/_r11_exclusion_audit.json`, the ON/OFF core, Tier B and control
   counts that Table S31 reports; `v11_algorithm_spec.py` asserts against it.
2d. **Two count artefacts the manuscript quotes** —
   `python analysis/_r9_exclusion_stay_audit.py` writes `t45`/`t46` (the
   exclusion-list stay audit and its per-domain effect) and
   `python analysis/_r8_extra_counts.py` writes `t43` (eICU-CRD diagnostic
   timing) and `t44` (complete-case composition of the Tier A+B models). All four
   are read by `export_final_numbers.py`; without them the export stops with
   `FileNotFoundError`.
3. **Assign diagnostic-confidence tiers** — `python analysis/extract_tier.py`
   writes `out/tier_<db>.csv` (tier flags per ICU stay) and `out/t11_tier_terms.csv`.
4. **Describe the cohort** — `python analysis/describe_cohort.py`.
5. **Part 1 / signal analyses** —
   `python analysis/part1_epidemiology.py`,
   `python analysis/positive_analyses.py`, then the sequential chain
   `python analysis/v4_analyses.py` … `python analysis/v11_algorithm_spec.py`
   (each reads the cohort/tier CSVs through `npsle_io.load()` and writes its
   result table into `out/`). Three later steps are prerequisites of the
   provenance export and are easy to miss: `v8_addendum.py` (writes
   `t40`/`t41`/`t41m`/`t42`), `v9_revision.py` (writes `t27_v9` and the
   revision key) and `v8_cv_label_2x2.py` (writes `v8_cv_label_2x2.json`, read by
   `v8_tables.py`).
5b. **ICD-9-CM sepsis-definition sensitivity** — `python analysis/_icd9_sensitivity.py`
   writes `out/_icd9_sensitivity.json`, the primary-versus-prefix arms read by
   `export_final_numbers.py`.
6. **Part 2 models** —
   `python analysis/v7_identification_models.py` fits and cross-validates the
   routine-data prediction models with **patient-grouped** internal
   validation: five repeats of five-fold `StratifiedGroupKFold`, each repeat
   with a prespecified random seed and always grouped by patient.
7. **Quantitative bias framework** — `python analysis/v6_measurement_model.py`
   writes `out/v6_qbf.json` (Layers 1–3), consumed by `v9_tables.py`.
8. **Table assembly** — `python analysis/v8_tables.py` and
   `python analysis/v9_tables.py` build the supplementary tables; `v6_tier_multinomial.py`
   builds the tier coefficient ratio. `v11_algorithm_spec.py` (Table S31)
   rebuilds the legacy term inventory from
   `analysis/fixtures/s27_legacy_table_body.html`, which is shipped, so this step
   runs on a clean clone with no manuscript files present.
9. **Figures** — `python analysis/fig_en.py` writes every panel that enters the
   main figures and must run **last among the panel producers**, because
   `fig_revision.py`, `positive_analyses.py`, `part1_epidemiology.py` and
   `v7_identification_models.py` write the same file names earlier in the
   pipeline (their labels are English as of 1.1.5, but they are still superseded).
   Then `python analysis/fig1_v9.py` (Figure 1 layout),
   `python analysis/fig_ladder.py` (attribution ladder) and
   `python analysis/make_strobe.py` (participant flow). Finish with
   `python analysis/create_fig_composites.py`, which assembles the multi-panel
   figures, refuses to run if a panel is missing, and must be re-run after **any**
   panel is regenerated. `fig_en.py` also rebuilds the SHAP panel and now stops
   with an error if `shap` is not installed, so that a stale or non-English panel
   cannot reach Supplementary Figure S2.
   `analysis/fig_revision.py` and the retired `legacy_pre_review10/part2_model.py`
   are kept for provenance only and are not part of this chain.
10. **Provenance export** — `python analysis/export_final_numbers.py` consolidates
    every result file into `out/final_numbers.json`.

### Sepsis definition (per database)

Sepsis is derived from diagnosis records **without temporal restriction**:
ICD-based diagnoses were not timestamped for this analysis, whereas eICU-CRD
diagnosis offsets were available (`out/t21_eicu_timing.csv`) but were **not**
used to constrain feature eligibility. The primary definition is ICD-9-CM
`038.x` + `995.91`/`995.92` and ICD-10-CM `A40`/`A41`/`R65.2`; a prefix rule
(`995.9x`) is retained only as a sensitivity analysis. The extraction regex in
`extract_cohort.py` is `^(038|9959[12])` (primary) with `^(038|9959)` (legacy
sensitivity).

### SLE entry criterion (per database)

* MIMIC-IV and NWICU: ICD `M32.*` / `710.0*`.
* eICU-CRD: a free-text `diagnosisstring` keyword match on `'lupus'`, which may
  also capture cutaneous or discoid lupus; this is reported faithfully in the
  manuscript and its limitations.

## Database connection

No credential is stored in this repository. Every script that touches a database
imports `analysis/db.py`, which reads four environment variables:

| variable | default |
|---|---|
| `NPSLE_PG_HOST` | `localhost` |
| `NPSLE_PG_PORT` | `5432` |
| `NPSLE_PG_USER` | `postgres` |
| `NPSLE_PG_PASSWORD` | none, must be set |

```bash
export NPSLE_PG_PASSWORD='***'          # bash
$env:NPSLE_PG_PASSWORD = "***"          # PowerShell
python analysis/extract_cohort.py mimiciv eicu nwicu
```

Scripts that do not read a database (`v8_tables.py`, `v9_tables.py`,
`v11_algorithm_spec.py`, the figure scripts) run without these variables.

## Script → output map

Every script below writes at least one file that either the manuscript or a later
step consumes. Files land in `out/` unless stated otherwise; `fig/` holds panels.

| script | writes | consumed by |
|---|---|---|
| `extract_cohort.py` | `cohort_<db>.csv` | everything downstream |
| `select_one_stay_per_patient.py` | `_first_stays.json` | `npsle_io.load()`, all one-stay analyses |
| `extract_tier.py` | `tier_<db>.csv`, `t11_tier_terms.csv` | `npsle_io`, `v11_algorithm_spec.py` |
| `_r11_exclusion_audit.py` | `_r11_exclusion_audit.json` | `v11_algorithm_spec.py` (asserts) |
| `_r9_exclusion_stay_audit.py` | `t45`, `t46` | `export_final_numbers.py` |
| `_r8_extra_counts.py` | `t43`, `t44` | `export_final_numbers.py` |
| `describe_cohort.py` | `desc_*.csv` | descriptive reporting only |
| `part1_epidemiology.py` | `t1`–`t4`, `forest_*.png` | `export_final_numbers.py` (values); the PNGs are superseded by `fig_en.py` |
| `positive_analyses.py` | `t16`–`t19`, `forest_sepsis.png`, `tierC_reprod.png` | `export_final_numbers.py`; panels superseded by `fig_en.py` |
| `v4_analyses.py` | `t20`–`t24`, `t22a`–`t22d` | `export_final_numbers.py` |
| `v5_analyses.py` | `t25a`–`t25d`, `t26`–`t28` | `export_final_numbers.py` |
| `v6_measurement_model.py` | `v6_qbf.json` | `v9_tables.py` |
| `v6_tier_multinomial.py` | tier ROR (`S25`) | `v9_tables.py`, `export_final_numbers.py` |
| `v7_identification_models.py` | `t5`, `t5b`, `t6`–`t11`, model panels | `export_final_numbers.py`; panels superseded by `fig_en.py` |
| `v7_eicu_algorithm.py` | eICU algorithm evaluation | `extract_tier.py` design, `v11_algorithm_spec.py` |
| `v7_icd_attribution.py` | ICD-side attribution check | `v11_algorithm_spec.py` (Panel E) |
| `v7_gcs_check.py`, `v7_ladder.py`, `v7_promote_restricted.py`, `v7_rule_missingness_audit.py` | the four `v7` audit outputs | `export_final_numbers.py`, supplement notes |
| `v8_addendum.py` | `t40`, `t41`, `t41m`, `t42` | `export_final_numbers.py` |
| `v8_cv_label_2x2.py` | `v8_cv_label_2x2.json` | `v8_tables.py` |
| `v8_core_tier_reconcile.py`, `v8_eicu_paths.py`, `v8_eicu_rules.py`, `v8_rule_strings.py`, `v8_truth_*.py` | rule-string and truth-table audits | `v11_algorithm_spec.py`, supplement notes |
| `v9_revision.py` | `t27_v9`, revision keys | `export_final_numbers.py` |
| `v10_s12_rerun.py` | `t16_sepsis_assoc.csv` (clustered) | `v8_tables.py` (S12) |
| `_icd9_sensitivity.py` | `_icd9_sensitivity.json` | `export_final_numbers.py`, S19 note |
| `_r11_*`, `_r8_*`, `_icd9_*` | as above | as above |
| `v8_tables.py`, `v9_tables.py`, `v11_algorithm_spec.py` | `S28`–`S33` fragments, `v11_S31.json` | manuscript build |
| `fig_en.py` … `create_fig_composites.py` | `fig/*.png`, composites | manuscript figures |
| `export_final_numbers.py` | `final_numbers.json`, `FINAL_NUMBERS.md` | the traceability gates |

`result_or.py` and `revision_analyses.py` are shared model helpers used by the
`v4`–`v11` chain; `v6_exclusion_check.py` and `v6_verify.py` are the audit steps
that produced the numbers in the Table S31 note.

## Note on clustered standard errors

The primary association model uses patient-clustered robust standard errors
(`positive_analyses.py`, `cluster=True`). Two places deliberately differ, and the
difference is a finite-sample factor, not a different estimand:

- `fig_en.py` draws the Figure 2a panel with `cluster=False`. On the
  one-stay-per-patient cohort every subject contributes one stay, so the only
  effect of the cluster correction is the `G/(G-1) * (N-1)/(N-K)` factor. Refitting
  both ways gives 2.28 (1.25-4.15) vs 2.28 (1.25-4.16) in MIMIC-IV, 1.31
  (0.63-2.76) vs 1.31 (0.62-2.80) in eICU-CRD, and 1.81 (1.06-3.08) vs 1.82
  (1.07-3.09) pooled — identical to two decimals, which is the precision printed
  in Table 2.
- `_icd9_sensitivity.py` (the ICD-9-CM 995.9x sensitivity analysis) also uses
  model-based intervals, so that its two arms differ in the sepsis definition and
  nothing else. It reproduces the manuscript's numbers when run as shipped —
  primary 2.28 (1.25-4.15) in MIMIC-IV and 1.81 (1.06-3.08) pooled, prefix
  2.39 (1.32-4.33) and 1.85 (1.04-3.31) — and the primary arm agrees with the
  patient-clustered Table 2 to two decimals, so the estimator choice does not
  enter the comparison.

## Environment

- Python 3.13.14
- `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn (≥ 1.5)`, `xgboost`,
  `matplotlib`, `Pillow`, `psycopg2-binary`. `requirements.txt` gives the floor;
  `requirements-lock.txt` pins the exact versions that produced the submitted
  numbers.
- `shap` and `cloudpickle` are needed only to regenerate the SHAP panel of the
  model figure; without `shap`, `fig_en.py` stops with an error rather than
  leaving a stale panel for the composite step.
- A local PostgreSQL instance with the PhysioNet databases loaded
  (tested with PostgreSQL 18, MIMIC-IV v3.1, eICU-CRD v2.0, NWICU)

Scripts that read a database take the connection from `analysis/db.py`, which
reads `NPSLE_PG_HOST`, `NPSLE_PG_PORT`, `NPSLE_PG_USER` and `NPSLE_PG_PASSWORD`
from the environment (see **Database connection** above). No credential and no
patient-level data are stored in the repository.

```bash
python analysis/extract_cohort.py mimiciv eicu nwicu
```

## Data availability

Cohorts are reconstructed from MIMIC-IV, eICU-CRD and NWICU, which are
available to certified researchers through PhysioNet (https://physionet.org)
under their respective Data Use Agreements. Analysis code and extracted cohort
schemas are published in this repository; **individual patient data are not
published**, in accordance with those agreements.

## Scope and intended use

The tier definitions are investigator-proposed and were not validated against
chart review; they are reported to make the coding basis of every estimate
explicit, not as a clinical classification. The phenotype definitions, the
pooled estimates (two databases with differing exposure ascertainment) and the
prediction models should all be read as descriptive of administrative coding
behaviour rather than as clinical or mechanistic evidence. Glucocorticoid and
immunosuppressant exposure was taken over the whole admission or ICU stay, so it
cannot establish treatment precedence or exclude treatment-related confounding.

## License

MIT — see [LICENSE](LICENSE).

## Citation

Wang K, Liu Y, Li S, Gao Y, Li J. *Implementation dependence and cross-database
evaluation of administrative neuropsychiatric phenotyping in critically ill
patients with systemic lupus erythematosus.* (submitted).

Code: https://github.com/morrosun/npsle-coding-transportability
