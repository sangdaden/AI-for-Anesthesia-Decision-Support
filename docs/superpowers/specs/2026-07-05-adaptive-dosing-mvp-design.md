# Adaptive Dosing MVP — Design Spec

Date: 2026-07-05
Status: Approved (design phase) — v2, pivoted after data prototyping
Author: Sang Phan

> Research/demo only. **Not** for clinical use.

## 0. Why this pivoted (recorded honestly)

The first design modeled within-case dose→BIS dynamics (a linear one-step PK/PD model).
Prototyping on the real M1 cohort showed this is **not identifiable** from observational
VitalDB data: clinicians already hold BIS in a narrow band, so there is little
dose-response variation to learn, and a patient's early propofol requirement does not
predict their later requirement within the same case (train→test correlation ≈ −0.17;
one-step propofol sensitivity had the wrong sign in ~40% of patients). That is a real
property of observational maintenance data, not a code defect.

A **covariate-based** framing IS identifiable and demonstrable: predicting each
patient's propofol maintenance requirement from their characteristics beats a
one-size-fits-all dose, leave-one-out cross-validated, with clinically correct
directions (heavier → more, older → less). This spec adopts that framing.

## 1. Purpose and Claim

Demonstrate that anesthesia dosing can be personalized per patient. Capability MVP,
not full M2.

**Claim.** A model personalizes the recommended propofol maintenance requirement from
patient characteristics (age, weight, BMI, sex), predicting each individual's dose need
**better than a one-size-fits-all population dose**, validated leave-one-out across
patients, with clinically sensible coefficient directions.

Prototype result on the 39-case demo cohort: leave-one-out MAE 3.45 (covariate model)
vs 3.91 (population mean) — a ~12% improvement — with corr(weight)=+0.54, corr(age)=
−0.38, corr(sex=M)=+0.41, corr(bmi)=+0.33.

## 2. Target: per-patient propofol maintenance requirement

For each case, the requirement is the mean propofol infusion rate during adequate
anesthesia:

```
requirement = mean( propofol_rate[t]  for t where 40 <= BIS[t] <= 60 and BIS[t] > 10 )
```

The `BIS[t] > 10` floor drops sensor-artifact zeros (a known M1 data issue). A case is
usable only if it has at least `min_in_band_rows` (default 10) such rows; otherwise it
is excluded and logged.

## 3. Model: covariate → requirement (linear)

A linear regression predicts requirement from patient covariates:

```
requirement ~ age + weight + bmi + sex_male + asa
```

Fit by least squares (`numpy.linalg.lstsq`), no new dependency. Linear is chosen for
interpretability — the fitted coefficients directly show each covariate's effect
direction and are reported in the demonstration.

## 4. Recommendation (the "controller")

The personalized recommended maintenance dose for a patient is simply the model's
prediction from their covariates:

```
recommend_dose(covariates) = model.predict(covariates)
```

The population baseline recommendation is the cohort-mean requirement (one dose for
everyone).

## 5. Demonstration (non-circular, cross-patient)

Leave-one-out cross-validation avoids leakage: for each patient, fit the model on all
**other** patients and predict theirs.

- **model MAE** = mean |requirement − LOO prediction|.
- **baseline MAE** = mean |requirement − mean-of-others| (one-size-fits-all).
- Personalization value = `(baseline_MAE − model_MAE) / baseline_MAE`.

**Three evidence outputs:**
1. LOO MAE improvement (model beats population baseline) — the core proof.
2. Fitted coefficients with directions (weight +, age −, …) — clinical plausibility.
3. Spread of recommended dose across patients (std of predictions) — different patients
   get different doses.

## 6. Architecture (spec structure + logic/UI separation)

```
src/adaptivedose/models/
├── __init__.py
├── requirement.py   # case_requirement(df, ...) -> float | None
└── personalize.py   # RequirementModel: fit/predict/coefficients; leave_one_out(X, y)
src/adaptivedose/adaptive/
├── __init__.py
└── demo.py          # build modeling table (covariates + requirement), run LOO, metrics
src/adaptivedose/dashboard/
└── charts.py        # ADD scatter_actual_vs_predicted (tested figure builder)
apps/dashboard/views/
└── adaptive.py      # "Adaptive demo" tab (render-only)
tests/models/{test_requirement.py, test_personalize.py}
tests/adaptive/test_demo.py
```

All computation lives in the tested library layer (`models/`, `adaptive/`, `charts.py`).
The Streamlit tab is render-only, matching the M1 dashboard pattern.

**Dashboard tab contents:** a cohort panel with the LOO improvement metric, a coefficient
table (covariate → effect direction), and an actual-vs-predicted scatter; plus a
per-patient selector showing that patient's covariates, personalized recommended dose vs
the population dose vs their actual requirement, and where their dose sits in the cohort
spread.

## 7. Data Flow

- Covariates come from the clinical cache via `adaptivedose.dashboard.data_access.load_clinical`
  (columns `age, sex, height, weight, bmi, asa`).
- Per-case requirement comes from `data_access.load_case` case frames (columns `bis,
  propofol_rate`).
- The cohort of cases comes from `data_access.build_cohort_table` (kept cases). No
  network at runtime.

## 8. Scope (YAGNI) and Honesty

- **Single drug (propofol), single target (BIS band).** Remifentanil, MAP safety, and
  time-varying dosing are out of scope.
- **Linear model, cross-sectional** (one requirement per patient) — not within-case
  closed-loop control.
- Stated limitations carried in the spec and surfaced in the dashboard tab:
  1. requirement is a per-case summary, not a moment-to-moment recommendation;
  2. covariate personalization explains part of the variance, not all (modest but real
     improvement over population dosing);
  3. observational data — the model learns clinician behavior conditioned on covariates,
     validated by cross-patient generalization, not by clinical trial.

## 9. Testing

- `requirement.case_requirement`: correct in-band mean; excludes BIS==0/<10 artifacts;
  returns None below `min_in_band_rows`.
- `personalize.RequirementModel`: `fit` recovers known coefficients on synthetic data;
  `predict`; `leave_one_out` produces one held-out prediction per row and, on synthetic
  data with a real covariate signal, beats the mean-of-others baseline.
- `adaptive.demo`: builds the modeling table from injected case frames + clinical frame;
  drops cases with no requirement; computes LOO model/baseline MAE and improvement;
  coefficient signs match the synthetic generator.
- `charts.scatter_actual_vs_predicted`: returns a figure with one scatter trace plus an
  identity reference line.
- Dashboard tab: render-only, exercised via `streamlit.testing.AppTest`.

## 10. Self-Review Notes

- **Identifiability:** the target and evaluation were validated by prototyping on real
  data before this spec was written (Section 0/1) — the demonstration will show a real,
  positive, honest effect.
- **Non-circular:** leave-one-out CV, no patient appears in its own training set.
- **Consistency:** column names (`bis, propofol_rate`; clinical `age, sex, height,
  weight, bmi, asa`) match M1 outputs and the existing dashboard data-access layer.
- **Scope:** single implementation plan; reuses M1 data access and dashboard pattern.
