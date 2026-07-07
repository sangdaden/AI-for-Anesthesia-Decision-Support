# AdaptiveDose AI — AI for Anesthesia Decision Support

A research-grade Clinical Decision Support System (CDSS) exploring **personalized,
adaptive anesthesia dosing** from the open [VitalDB](https://vitaldb.net) dataset.

> ⚠️ **Research and demonstration only. NOT for clinical use.** Nothing here is a
> medical device or clinical advice.

## What this project does

The core research question: *can a model recommend anesthetic doses personalized to each
patient, and can we demonstrate that personalization has real value?*

Three milestones have been built and validated on real data:

| Milestone | What it delivers | Status |
|-----------|------------------|--------|
| **M1 — Data foundation** | Reproducible VitalDB ETL → cohort selection → cleaning → resampling → patient-level split, DVC-versioned | ✅ Done |
| **M1 — Data explorer** | Streamlit dashboard: cohort overview, per-case signal viewer, signal-quality/EDA, multi-case compare | ✅ Done |
| **Adaptive MVP** | Covariate-based personalized propofol requirement, proven leave-one-out, with an "Adaptive demo" dashboard tab | ✅ Done |

### The adaptive result (honest capability demo)

A linear model predicts each patient's propofol **maintenance requirement** (mean
infusion rate while BIS is in the target band [40, 60]) from patient characteristics
(age, weight, BMI, sex, ASA).

On the demo cohort, **leave-one-out cross-validation**:

- Personalized model MAE **3.45** vs one-size-fits-all population dose MAE **3.91** →
  **~12% improvement**, out-of-sample.
- Clinically sensible directions: heavier → more propofol, older → less.
- A **noise-target control** confirms the improvement is real signal, not overfitting
  (a target independent of covariates does *not* beat the population baseline under LOO).

Honest scope: this personalizes by patient characteristics (cross-sectional), not
real-time closed-loop control, and it learns clinician behavior conditioned on
covariates — validated by cross-patient generalization, not a clinical trial.

## Architecture

Computation lives in a tested Python library (`src/adaptivedose/`); the Streamlit app
(`apps/dashboard/`) is render-only. No network access at dashboard runtime.

```
src/adaptivedose/
├── config.py              # typed YAML config (OmegaConf)
├── data/                  # VitalDB ETL pipeline
│   ├── vitaldb_client.py  #   cached clinical-info + track-index download
│   ├── cohort.py          #   track+clinical cohort selection
│   ├── loader.py          #   per-case tidy loader
│   ├── clean.py           #   physiologic clamping + imputation
│   ├── resample.py        #   trim to the BIS-monitored window
│   ├── builder.py         #   cohort loop -> per-case Parquet + manifest
│   └── split.py           #   subject-level leak-free train/val/test split
├── models/                # adaptive dosing
│   ├── requirement.py     #   per-case propofol maintenance requirement target
│   └── personalize.py     #   linear requirement model + leave-one-out CV
├── adaptive/
│   └── demo.py            #   assemble cohort table, run LOO demonstration, metrics
├── dashboard/             # tested dashboard logic (no Streamlit here)
│   ├── data_access.py     #   manifest-driven loaders + cohort table
│   ├── stats.py           #   cohort/signal statistics
│   └── charts.py          #   pure Plotly figure builders
└── utils/io.py            # parquet/dir helpers

apps/dashboard/            # Streamlit UI (render-only)
├── app.py                 #   entry, sidebar nav, missing-data guard
└── views/                 #   cohort, case, quality, compare, adaptive

pipelines/build_dataset.py # end-to-end M1 dataset build CLI
docs/superpowers/          # design specs and implementation plans
```

## Dataset

[VitalDB](https://vitaldb.net) is fully open (no credentialing). The cohort is
general-anesthesia surgical cases using **propofol TIVA with a valid BIS signal**
(BIS + propofol infusion + arterial MAP ≈ 1860 cases at full scale). Key tracks:
`BIS/BIS`, `Orchestra/PPF20_RATE`, `Orchestra/RFTN20_RATE`, `Solar8000/ART_MBP`,
`Solar8000/HR`, `Solar8000/PLETH_SPO2`, `Solar8000/ETCO2`.

## Setup

Requires Python ≥ 3.10 (developed on 3.14).

```bash
git clone https://github.com/sangdaden/AI-for-Anesthesia-Decision-Support.git
cd AI-for-Anesthesia-Decision-Support
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,dashboard]"
```

## Usage

**1. Build the dataset from live VitalDB** (downloads data; `--limit` for a quick run):

```bash
.venv/bin/python pipelines/build_dataset.py --config configs/data.yaml --limit 40
# omit --limit for the full ~1860-case cohort
```

Produces `data/processed/`: per-case Parquet, `manifest.parquet`, `splits.json`.

**2. Launch the dashboard** (runs from any working directory):

```bash
.venv/bin/streamlit run apps/dashboard/app.py
```

Tabs: **Cohort overview**, **Case viewer**, **Signal quality / EDA**,
**Compare cases**, **Adaptive demo**.

**3. Reproduce the adaptive result in one script:**

```bash
.venv/bin/python - <<'PY'
from adaptivedose.dashboard import data_access as da
from adaptivedose.adaptive import demo
table = demo.build_modeling_table(da.build_cohort_table(), da.load_clinical(), da.load_case)
res = demo.run_demo(table)
print(f"cases={len(table)} personalized_MAE={res['model_mae']:.2f} "
      f"population_MAE={res['baseline_mae']:.2f} improvement={res['improvement']*100:.0f}%")
PY
```

## Testing

```bash
.venv/bin/pytest -q      # 56 tests, network-free
```

All pipeline stages, statistics, models, and the leave-one-out demonstration are unit
tested. The Streamlit UI is exercised via `streamlit.testing.AppTest`.

## Configuration

Everything is driven by [`configs/data.yaml`](configs/data.yaml): cohort criteria,
track names, resample cadence, split fractions and seed. Edit it and re-run — no code
changes needed.

## Reproducibility

Config-driven, fixed seeds, subject-level splits (no patient leakage), and DVC-versioned
processed data. Design specs and step-by-step implementation plans live in
[`docs/superpowers/`](docs/superpowers/).

## Known data-quality notes (for future work)

Surfaced during M1 and visible in the dashboard's Signal-quality tab:

- **BIS artifact zeros** (sensor disconnects) pass the [0, 100] range check; future work
  should reject them via a physiologic floor or the `BIS/SQI` signal-quality index.
- **MAP missingness**: some cases have sparse invasive MAP; a `Solar8000/NIBP_MBP`
  fallback is documented but not yet wired in.

## Roadmap

- **Full M2:** real-time closed-loop control (PK/PD response model + MPC controller),
  hemodynamic safety constraint (MAP > 65), and counterfactual off-policy evaluation.
- Confidence intervals on the personalization improvement over a larger cohort.
- Optional layers from the research design: SHAP explainability, LLM explanation, drift
  monitoring.

## License / disclaimer

Research artifact. Not a medical device. Not for clinical decision-making.
```
