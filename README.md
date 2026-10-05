# NPSLE Coding Transportability

**Reproducible analysis code** for the study:

*Transportability and structural limits of administrative neuropsychiatric
phenotyping in critically ill patients with systemic lupus erythematosus:
a three-database evaluation.*

This repository contains **analysis code only**. No patient-level data are
included; all cohorts are derived from PhysioNet-credentialed databases
(MIMIC-IV, eICU-CRD, NWICU) under their respective Data Use Agreements (DUA).
Researchers with PhysioNet credentials can reconstruct the cohorts by running
the SQL check and extraction scripts against their own local database copies.

> **Version:** `1.1.2` — synchronized with the submitted manuscript
> (review-11 snapshot, 2026-10-05). `1.1.1` corrected three things in the code
> itself, found on the eleventh external review: (i) the eICU-CRD
> one-ICU-stay-per-patient selection rule in `make_first_stays.py` sorted by
> `hospitaladmitoffset` ascending, which selects the *last* ICU stay of the
> *latest* admission because eICU offsets are measured from each admission;
> (ii) `extract_tier.py` intersected the legacy arm with the audited core flag
> instead of the legacy arm's own core universe; and (iii) 
> `v7_identification_models.py` carried a dead `try/except` around
> `RepeatedStratifiedGroupKFold`, a class that does not exist in scikit-learn.
> This release (`1.1.2`) fixes two further defects found while checking the
> *outputs* rather than the text: `fig_en.py` still reproduced the ROC,
> calibration and decision-curve panels from the retired `part2_model.py`
> (stay-level `RepeatedStratifiedKFold`), so the ROC legend read AUC 0.578 /
> 0.689 instead of the patient-grouped 0.514 / 0.686 reported in the
> manuscript; and the composite figures had not been rebuilt after the `1.1.1`
> reruns, so Figure 2 still carried pre-review estimates. It also removes the
> last machine-specific paths (`make_strobe.py` output file, the Arial font
> path in `create_fig_composites.py` / `fig1_v9.py`) and renames the remaining
> "first stay" labels to "one stay per patient".
> Earlier releases: `1.1.0` (2026-10-05) shipped the review-10 pass; `1.0.0`
> (2026-09-18) an earlier, partially inconsistent analysis pass. Two superseded
> scripts are retained under `legacy_pre_review10/` for transparency.

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
analysis/make_first_stays.py        prerequisite: writes out/_first_stays.json — ONE ICU stay per
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

1. **Screen the databases** — `psql -f sql/feasibility.sql` against each local
   copy records which SLE and neuropsychiatric code families are actually
   present, and which are harmonisable across databases.
2. **Extract the cohorts** —
   `python analysis/extract_cohort.py mimiciv eicu nwicu`
   writes `out/cohort_<db>.csv` with column names aligned across databases.
2b. **Generate the first-stay index (required prerequisite)** —
   `python analysis/make_first_stays.py`
   reads the cohort CSVs and writes `out/_first_stays.json` — the first ICU stay
   per subject, which defines the first-stay main-analysis cohort (MIMIC-IV 354,
   eICU-CRD 186, NWICU 43; total 583). This file is required before
   `extract_tier.py`, `v4_analyses.py` and `v7_identification_models.py`, which
   consume it through `npsle_io.load()`. Without it those scripts raise
   `FileNotFoundError`.
2c. **Exclusion-list audit (eICU-CRD)** — `python analysis/_r11_exclusion_audit.py`
   writes `out/_r11_exclusion_audit.json`, the ON/OFF core, Tier B and control
   counts that Table S31 reports; `v11_algorithm_spec.py` asserts against it.
3. **Assign diagnostic-confidence tiers** — `python analysis/extract_tier.py`
   writes `out/tier_<db>.csv` (tier flags per ICU stay) and `out/t11_tier_terms.csv`.
4. **Describe the cohort** — `python analysis/describe_cohort.py`.
5. **Part 1 / signal analyses** —
   `python analysis/part1_epidemiology.py`,
   `python analysis/positive_analyses.py`, then the sequential chain
   `python analysis/v4_analyses.py` … `python analysis/v11_algorithm_spec.py`
   (each reads the cohort/tier CSVs through `npsle_io.load()` and writes its
   result table into `out/`).
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
   builds the tier coefficient ratio.
9. **Figures** — run the panel producers first, in this order, because several of
   them write the same file name and the last writer wins:
   `python analysis/fig_en.py` (English panels, including the ROC / calibration /
   decision curves, which use the patient-grouped cross-validation of
   `v7_identification_models.py`), then `python analysis/fig1_v9.py` (Figure 1
   layout), `python analysis/fig_ladder.py` (attribution ladder) and
   `python analysis/make_strobe.py` (participant flow); finish with
   `python analysis/create_fig_composites.py`, which assembles the multi-panel
   figures and must be re-run after **any** panel is regenerated.
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

## Environment

- Python 3.13
- `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn (≥ 1.5)`, `xgboost`,
  `matplotlib`, `Pillow`, `psycopg2-binary` (see `requirements.txt`)
- `shap` and `cloudpickle` are required only to regenerate the SHAP panel of the
  model figure; they are listed as optional in `requirements.txt`.
- A local PostgreSQL instance with the PhysioNet databases loaded
  (tested with PostgreSQL 18, MIMIC-IV v3.1, eICU-CRD v2.0, NWICU)

The analysis scripts connect to the local PhysioNet databases through a
hard-coded `CONFIG` dict (`host`, `port`, `user`, `password`) near the top of
each script (the default points at the author's local credentialed instance at
`localhost:5432`). Point this at your own local copies by editing the `CONFIG`
dict (and the `dbname` per database) before running. No patient-level data are
read from or written to the repository.

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

Wang K, Liu Y, Li S, Gao Y, Li J. *Transportability and structural limits of
administrative neuropsychiatric phenotyping in critically ill patients with
systemic lupus erythematosus: a three-database evaluation.* (submitted).

Code: https://github.com/morrosun/npsle-coding-transportability
