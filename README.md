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
sql/feasibility.sql          database inventory: SLE / NPSLE coding availability,
                             term-level mappings, per-database schema differences
analysis/extract_cohort.py   builds the aligned three-database cohorts
analysis/extract_tier.py     diagnostic-confidence tier assignment
analysis/describe_cohort.py  feature completeness / baseline / subtype summaries
analysis/part1_epidemiology.py   Part 1: baseline, prevalence, outcomes, pooling
analysis/part2_model.py          Part 2: prediction models, external validation
analysis/positive_analyses.py    sepsis co-occurrence, tier stratification,
                                 Tier C reproducibility, severity sensitivity
analysis/qbf_bias_framework.py   quantitative bias framework (see below)
analysis/fig_*.py, make_strobe.py, create_fig_composites.py   figures
```

## Pipeline

Run every script from the **repository root**; all paths are resolved relative to
the repository, and outputs are written to `out/` (created on demand).

1. **Screen the databases** — `psql -f sql/feasibility.sql` against each local
   copy records which SLE and neuropsychiatric code families are actually
   present, and which are harmonisable across databases.
2. **Extract the cohorts** —
   `python analysis/extract_cohort.py [mimiciv] [eicu] [nwicu]`
   writes `out/cohort_<db>.csv` with column names aligned across databases,
   covering the core and broad phenotype definitions, the seven
   neuropsychiatric domains, and the harmonised covariates.
3. **Assign diagnostic-confidence tiers** — `python analysis/extract_tier.py`
   writes `out/tier_<db>.csv` (tier flags per ICU stay) and
   `out/t11_tier_terms.csv`, the raw term lists behind each tier.
4. **Describe the cohort** — `python analysis/describe_cohort.py` for feature
   completeness, baseline distributions and subtype–outcome cross-tabs.
5. **Part 1 / Part 2 analyses** — `python analysis/part1_epidemiology.py` and
   `python analysis/part2_model.py`.
6. **Signal analyses** — `python analysis/positive_analyses.py` produces the
   per-database sepsis odds ratios, the random-effects pooled estimate, the
   tier-stratified estimates, and the cross-database Tier C reproducibility
   test.
7. **Revision analyses** — the numbered `v4_analyses.py` … `v9_revision.py`
   scripts trace the analysis history in the order the questions were raised:
   immunosuppression and severity sensitivity, eICU diagnostic-timestamp
   sequencing, the first-stay primary analysis, dual outcome models, the
   inclusion/exclusion audit, the clustering-estimator grid, the attribution
   ladder, and mutually exclusive tier counts. They are retained because each
   one fixes or qualifies a specific earlier number.
8. **Quantitative bias framework** — `python analysis/qbf_bias_framework.py`
   writes `out/qbf_report.txt` and `out/qbf_results.json`.
9. **Figures** — `analysis/fig_revision.py`, `analysis/fig_en.py` (Latin-font
   regeneration of every manuscript figure), `analysis/fig1_v9.py`,
   `analysis/fig_ladder.py`, `analysis/make_strobe.py` (participant flow), then
   `analysis/create_fig_composites.py` to assemble the multi-panel figures.

## The quantitative bias framework

`analysis/qbf_bias_framework.py` takes only the published first-stay 2×2 tables
as input — no patient-level data are needed — and derives three quantities that
any multi-database computable-phenotype study can report:

| Layer | Question | Output |
|---|---|---|
| 1 | Does imperfect positive predictive value bias the odds ratio, or only cost precision? | Corrected OR range over a PPV grid, plus effective event counts and the identifiability bound implied by the observed coded prevalence |
| 2 | How much definitional overlap would be needed to explain the association away? | The overlap magnitude (`loop term`) at which the true OR reaches 1, for each database |
| 3 | How strong would unmeasured confounding have to be to do the same? | E-values for the pooled estimates |

It also tests *effect homogeneity against measurement equivalence*: a
random-effects I² of 0% is shown to be compatible with a large cross-database
difference in tier composition.

## Environment

- Python 3.13
- `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `xgboost`,
  `matplotlib`, `Pillow`, `psycopg2-binary` (see `requirements.txt`)
- A local PostgreSQL instance with the PhysioNet databases loaded
  (tested with PostgreSQL 18, MIMIC-IV v3.1, eICU-CRD v2.0, NWICU)

Database connection settings are read from environment variables; nothing is
hard-coded:

| Variable | Purpose | Default |
|---|---|---|
| `NPSLE_DB_HOST` | PostgreSQL host | `localhost` |
| `NPSLE_DB_PORT` | PostgreSQL port | `5432` |
| `NPSLE_DB_USER` | PostgreSQL user | `postgres` |
| `NPSLE_DB_PASSWORD` | PostgreSQL password | *(empty)* |
| `NPSLE_FONT` | optional font path for figure composition | auto-detected |

```bash
export NPSLE_DB_HOST=localhost NPSLE_DB_USER=postgres NPSLE_DB_PASSWORD='****'
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
behaviour rather than as clinical or mechanistic evidence.

## License

MIT — see [LICENSE](LICENSE).

## Citation

Wang K, Liu Y, Li S, Gao Y, Li J. *Transportability and structural limits of
administrative neuropsychiatric phenotyping in critically ill patients with
systemic lupus erythematosus: a three-database evaluation.* (submitted).

Code: https://github.com/morrosun/npsle-coding-transportability
