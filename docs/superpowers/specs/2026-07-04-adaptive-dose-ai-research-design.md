# AdaptiveDose AI — Research Design Spec

Date: 2026-07-04
Status: Approved (design phase)
Author: Sang Phan

> Research/demo only. **Not** for clinical use.

## 1. Purpose and Research Question

AdaptiveDose AI is a research-grade Clinical Decision Support System that recommends
**personalized, adaptive anesthesia dosing** and updates the recommendation as a
patient's vital signs evolve during surgery.

**Research question.** Can a personalized drug-response model recommend anesthetic
doses (propofol / remifentanil) that maintain the target depth of anesthesia
(BIS 40–60) *without* inducing hypotension (MAP > 65 mmHg), better than a
population-level baseline?

**Why not naive dose prediction.** Predicting the dose a clinician actually gave is
confounded: the clinician chooses the dose *in response to* the observed effect, and
observational data contains no ground-truth "optimal dose." We break this confounding
by conditioning the recommendation on an **objective, measurable clinical target**
(BIS in range, MAP above threshold) rather than on the historical dose.

## 2. Scientific Framing (the core of the paper)

Three logical blocks:

1. **Response model** — `f_θ(patient, dose_history, vitals_history) → next (BIS, MAP)`.
   Learns each patient's individualized dose-response.
2. **Controller** — inverts the response model via Model Predictive Control (MPC):
   selects the dose that minimizes deviation from the BIS target while penalizing
   violations of the MAP safety constraint.
3. **Counterfactual evaluation** — uses the response model as a simulator to estimate
   "what would BIS/MAP have been had we followed the recommendation," compared against
   the clinician's actual trajectory (off-policy evaluation).

**Contribution claim.** Per-patient adaptation + dual safety constraint (depth AND
hemodynamics) + an honest counterfactual evaluation on observational data.

## 3. Scope (YAGNI)

### Core (required for the paper)
- VitalDB ETL + cohort selection
- Time-series feature engineering
- Response model (PK/PD baseline + ML)
- MPC controller
- Counterfactual off-policy evaluation + clinical/safety metrics
- SHAP explainability
- Reproducibility (config-driven, seeded, MLflow tracking)

### Optional (post-core, "stretch")
- Streamlit dashboard
- LLM explanation layer (explains predictions only; never prescribes)
- Drift detection (PSI / KL / Wasserstein)
- PostgreSQL (core uses Parquet + DuckDB instead)
- Docker Compose / full CI
- FHIR / HL7 integration

Rationale: for a publication the value is in a clean data pipeline and rigorous
evaluation, not in the dashboard or drift monitoring. Those are deferred.

## 4. Target Definition

Primary objective — **staged**:
- **Stage A:** maintain depth of anesthesia, BIS in [40, 60].
- **Stage B:** add hemodynamic safety, MAP > 65 mmHg, as a constraint / secondary
  objective. The final claim is the multi-objective version.

## 5. Data

**Source:** VitalDB (fully open access, no credentialing). Python `vitaldb` package.

**Cohort:** general-anesthesia surgical cases using **propofol TIVA with a valid BIS
signal** (required to define the depth target). Exclude cases lacking BIS or
drug-infusion channels.

**Signals:**
- Static: age, sex, height, weight, BMI, ASA score.
- Time-series: propofol infusion, remifentanil infusion, BIS, MAP / arterial BP,
  HR, SpO2, ETCO2.

**Pipeline:** raw → cleaning → missing-value handling → outlier removal → resampling
to a fixed cadence → feature engineering → train/val/test split (patient-level, no
leakage across splits).

## 6. Modeling Approach

Chosen: **PK/PD-informed response model + model-based controller**.

- **Response model:** start with a pharmacokinetic/pharmacodynamic-inspired baseline;
  extend with ML sequence models where they improve BIS/MAP prediction.
- **Controller:** MPC / inverse of the response model — minimum-deviation dose subject
  to the MAP constraint.
- Sequence models (LSTM/TCN/Transformer) and offline RL (bandit/CQL) are noted as
  future comparison arms, not part of the core.

## 7. Code Architecture

```
adaptive-dose-ai/
├── configs/            # YAML (Hydra/OmegaConf): cohort, features, model, eval
├── src/
│   ├── data/           # VitalDB download, ETL, cohort, cleaning, resample
│   ├── features/       # time-series feature engineering
│   ├── models/
│   │   ├── response/   # PK/PD + ML response model (dose → effect)
│   │   └── controller/ # MPC / inverse controller
│   ├── evaluation/     # regression + clinical + safety + off-policy metrics
│   ├── explainability/ # SHAP
│   └── utils/          # seed, logging, io
├── pipelines/          # etl.py, train.py, evaluate.py
├── notebooks/          # EDA
├── tests/
└── docs/
```

Each module has a single responsibility, communicates through explicit interfaces
(DataFrame / tensor + config), and is independently testable.

## 8. Evaluation

**Prediction (response model):** MAE, RMSE, R² for BIS and MAP forecasting.

**Clinical / control quality (counterfactual):**
- Time-in-target-range for BIS (fraction of time in [40, 60] under the recommended
  policy, per the simulator).
- Hypotension exposure (fraction of time MAP < 65).
- Comparison against the clinician's actual trajectory.

**Safety:**
- Overdose / underdose rate.
- Constraint-violation rate.

**Calibration / uncertainty:** where the response model produces confidence,
report calibration (e.g., ECE) — optional in core.

Off-policy evaluation is explicitly framed as simulator-based and its limitations
stated honestly (model-based estimates, observational data).

## 9. Reproducibility

- All runs config-driven (single source of truth in `configs/`).
- Fixed random seeds; deterministic where feasible.
- MLflow tracks parameters, metrics, artifacts, model versions.
- Dataset versioning via DVC (or documented Parquet snapshots) in the core.

## 10. Roadmap (staged, runnable early)

- **M1 — Data foundation:** VitalDB download → ETL → cohort → EDA.
  *Milestone: clean, versioned dataset.*
- **M2 — Response model:** PK/PD baseline + ML response model; BIS/MAP prediction eval.
- **M3 — Controller + Evaluation:** MPC controller + counterfactual off-policy eval +
  safety metrics. *Primary result of the paper.*
- **M4 — Explainability + reproducibility:** SHAP + MLflow + full config-isation.
- **M5 (optional):** dashboard, LLM, drift, Docker.

## 11. Testing

- Unit tests for data transforms, feature functions, response-model interfaces,
  controller logic, and metric computations.
- Target meaningful coverage on core `src/` modules (aspirational >80%, not gated on
  optional layers).
