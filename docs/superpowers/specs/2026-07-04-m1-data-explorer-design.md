# M1 Data Explorer — Design Spec

Date: 2026-07-04
Status: Approved (design phase)
Author: Sang Phan

> Research/demo only. **Not** for clinical use.

## 1. Purpose

A Streamlit dashboard to explore the M1 VitalDB dataset (cohort, per-case signals,
data quality). It exists to support EDA and paper figures **now**, before any model
(M2) exists — so it visualizes the data foundation, not predictions. It reads only
the artifacts the M1 pipeline already produces (`data/processed/`), never the network.

## 2. Architecture Principle: logic separate from UI

Streamlit code is hard to unit-test, so all computation lives in a pure-Python library
layer that is tested; the Streamlit files only call those functions and render.

- `src/adaptivedose/dashboard/data_access.py` — load `manifest.parquet`, `splits.json`,
  `cases/*.parquet`, and cached clinical info; join manifest ↔ clinical for
  age/sex/ASA. Pure functions, unit-tested.
- `src/adaptivedose/dashboard/stats.py` — `cohort_summary`, `signal_distribution`,
  `time_in_target`, `missingness`. Pure functions, unit-tested.
- `apps/dashboard/app.py` + `apps/dashboard/views/*.py` — call the above and render.
  No business logic.

## 3. Views (in priority order)

1. **Cohort overview** — summary table plus distribution charts for age, sex, ASA,
   surgery type; case duration; kept/dropped ratio; train/val/test split sizes.
2. **Case viewer** — pick one case → multi-axis time-series: BIS (0–100, with target
   band [40, 60]), MAP (with 65 mmHg threshold line), HR, SpO2, propofol and
   remifentanil infusion rates.
3. **Signal quality / EDA** — per-signal distributions across the cohort, percent
   missing, artifact flag (BIS == 0), percent time in BIS target. Ties directly to the
   two data-quality issues found in M1 (BIS artifact zeros, MAP missingness).
4. **Multi-case compare** — pick ≥2 cases and overlay their BIS/MAP trajectories.

## 4. Tech and Data Flow

- **Streamlit** with sidebar navigation across the four views.
- **Plotly** for interactive multi-axis charts (added to the `dashboard` extra).
- Data source: read directly from `data/processed/` (pipeline M1 output) plus the
  cached clinical-info Parquet. No network calls while the dashboard runs.
- **Error handling:** if `data/processed/` is missing (pipeline not yet run), show a
  clear message instructing the user to run `pipelines/build_dataset.py`, rather than
  crashing. `data_access` raises a typed error; `app.py` catches and displays it.

## 5. File Structure and Testing

```
apps/dashboard/
├── app.py                      # entry: sidebar nav, wire the 4 views
└── views/
    ├── cohort.py
    ├── case.py
    ├── quality.py
    └── compare.py
src/adaptivedose/dashboard/
├── __init__.py
├── data_access.py              # load manifest/splits/cases/clinical (testable)
└── stats.py                    # cohort_summary/distribution/time_in_target/missingness
tests/dashboard/
├── __init__.py
├── test_data_access.py
└── test_stats.py
```

Tests cover `data_access` (correct load/join; typed error when data absent) and `stats`
(correct computation on small synthetic frames). The Streamlit UI itself is not
unit-tested — it is exercised manually via `streamlit run`.

## 6. Data Contracts (from M1)

- `manifest.parquet`: columns `caseid, n_samples, kept` (and `subjectid` after the
  pipeline merge).
- `splits.json`: `{"train": [...], "val": [...], "test": [...]}` of caseids.
- `cases/case_<id>.parquet`: columns `caseid, time_sec, bis, propofol_rate,
  remifentanil_rate, map, hr, spo2, etco2` (subset present per case).
- Clinical cache (`.vitaldb_cache/cases.parquet`): includes `caseid, age, sex, asa,
  optype`.

## 7. Run

Add `streamlit` and `plotly` to a `dashboard` optional-dependency group in
`pyproject.toml`, then:

```
.venv/bin/streamlit run apps/dashboard/app.py
```

## 8. Scope (YAGNI)

- No authentication, no multi-user state, no persistence beyond reading M1 outputs.
- No model/prediction/SHAP views — those wait for M2.
- No caching layer beyond Streamlit's built-in `@st.cache_data` on the loaders.

## 9. Self-Review Notes

- **Data contract consistency:** column names match the M1 implementation
  (`bis, propofol_rate, remifentanil_rate, map, hr, spo2, etco2`; manifest
  `caseid, n_samples, kept, subjectid`).
- **Testability:** every computed value has a home in `data_access` or `stats`; the
  Streamlit layer is render-only, so the untested surface is minimal.
- **No network at runtime:** dashboard reads local Parquet only; the clinical cache is
  populated by the already-run pipeline.
