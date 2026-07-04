# Adaptive Dosing MVP — Design Spec

Date: 2026-07-05
Status: Approved (design phase)
Author: Sang Phan

> Research/demo only. **Not** for clinical use.

## 1. Purpose and Claim

Demonstrate that personalized anesthesia dosing works per patient. This is a
capability MVP, not the full M2 milestone.

**Claim.** A personalized drug-response model learns each patient's individual
sensitivity, so that (a) the same state and BIS target produce **different**
recommended propofol doses across patients, and (b) a controller using each patient's
own parameters **tracks the BIS target better** than a controller using
population-average parameters.

This extends the research design
(`docs/superpowers/specs/2026-07-04-adaptive-dose-ai-research-design.md`): it builds
the M2 response model + controller in a minimal, honest form and proves the adaptation
value on the M1 VitalDB dataset.

## 2. Response Model (per-patient, linear one-step)

For each patient, fit a one-step-ahead autoregressive model by least squares on that
patient's own samples:

```
BIS[t+1] = alpha + beta * BIS[t] + gamma * propofol[t] + delta * remifentanil[t]
```

- `gamma` is the patient's **propofol sensitivity** (expected negative: more drug ->
  lower BIS). Variation in `gamma` across patients is the evidence of personalization.
- `beta` captures depth inertia.
- Interpretable, fast to fit, robust at the dataset's 10-second cadence.

The module reports the one-step BIS prediction MAE so model fit is quantified.

## 3. Controller (closed-form one-step inverse)

To move BIS toward target `BIS*` at the next step:

```
propofol* = clip( (BIS* - alpha - beta * BIS[t] - delta * remifentanil[t]) / gamma,
                  0, propofol_max )
```

If `gamma >= 0` (the model did not learn a sensible sensitivity), the patient is
flagged **non-identifiable** and excluded from the personalized-vs-population
comparison. No forcing.

## 4. Demonstration (avoiding circular self-proof)

This is the crux that keeps the result defensible.

- Split each case timeline into **train (first 60%)** and **test (last 40%)**.
- Fit `sim_model` (the patient's "true" dynamics for evaluation) on the **test
  window** — an independent simulator.
- Roll two controllers through `sim_model` over the test window:
  - **personalized** — parameters fit on the patient's **train** window.
  - **population** — parameters equal to the cohort mean of all patients' train fits.
- Personalized winning means the patient's early dynamics predict their later dynamics
  better than the population average does — i.e. personalization has real value, not a
  same-model artifact.

**Three evidence metrics:**
1. Spread of `gamma` (propofol sensitivity) across the cohort — inter-patient
   heterogeneity.
2. At a fixed reference state (`BIS = 70`, target `50`, remifentanil at cohort median),
   the recommended propofol per patient — spread of recommended dose.
3. Target-tracking error `mean|BIS - target|` over the test window: personalized vs
   population, aggregated across the cohort (and time-in-band [40, 60]).

## 5. Architecture (spec structure + logic/UI separation)

```
src/adaptivedose/models/
├── __init__.py
├── response.py     # fit_response(df) -> ResponseParams; predict_next; one-step MAE
└── controller.py   # recommend_dose(params, bis, target, remifentanil, propofol_max)
src/adaptivedose/adaptive/
├── __init__.py
└── demo.py         # train/test split, fit, simulate personalized vs population, metrics
apps/dashboard/views/
└── adaptive.py     # "Adaptive demo" tab (render-only)
tests/models/
├── test_response.py
└── test_controller.py
tests/adaptive/
└── test_demo.py
```

All computation lives in the tested library layer (`models/`, `adaptive/`). The
Streamlit tab is render-only, consistent with the M1 dashboard pattern. Chart builders
for the tab go in the existing `src/adaptivedose/dashboard/charts.py` (tested) where a
reusable figure is warranted; otherwise the tab uses existing builders.

**Dashboard tab contents:** pick a patient -> show that patient's `gamma` against the
cohort distribution, the recommended-dose curve, and a personalized-vs-population
BIS-tracking plot over the test window; plus a cohort-level panel with the three
evidence metrics from Section 4.

## 6. Data Flow

- Reads M1 outputs via the existing `adaptivedose.dashboard.data_access`
  (`build_cohort_table`, `load_case`) — no new disk readers, no network.
- `adaptive/demo.py` consumes per-case DataFrames (columns `bis, propofol_rate,
  remifentanil_rate, time_sec`) and the cohort list.

## 7. Scope (YAGNI) and Honesty

- **Single objective: BIS only.** MAP hemodynamic constraint is deferred to full M2.
- **Linear one-step model**, not full PK/PD; **one-step inverse**, not multi-step MPC.
- Three stated limitations carried in the spec and surfaced in the dashboard tab:
  1. linear surrogate response model (not pharmacological PK/PD),
  2. model-based evaluation (the simulator is a fitted model, so this demonstrates the
     value of personalization given the model, not clinical efficacy),
  3. single objective (BIS), no hemodynamic safety.

## 8. Testing

- `response.fit_response` / `predict_next`: recover known coefficients from synthetic
  data generated with a fixed `(alpha, beta, gamma, delta)`; MAE near zero on clean
  synthetic input.
- `controller.recommend_dose`: closed-form inverse correctness; clipping to
  `[0, propofol_max]`; non-identifiable (`gamma >= 0`) handling.
- `adaptive.demo`: train/test split boundaries; personalized beats population on
  synthetic data where a per-patient signal exists; non-identifiable patients excluded;
  metric computations correct on small synthetic cohorts.
- Dashboard tab: render-only, exercised via `streamlit.testing.AppTest`.

## 9. Self-Review Notes

- **Consistency:** signal column names (`bis, propofol_rate, remifentanil_rate,
  time_sec`) match M1 case Parquet output and the existing dashboard layer.
- **Non-circular evaluation:** train-fit controllers evaluated on an independent
  test-fit simulator (Section 4) — the design's key defensibility point.
- **Scope:** single implementation plan; single objective; reuses M1 data access and
  the dashboard pattern. No decomposition needed.
- **Honesty:** the three limitations are explicit so the MVP reads as a capability
  demonstration, not a clinical claim.
