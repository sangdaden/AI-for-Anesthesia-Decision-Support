# Adaptive Dosing MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Demonstrate personalized anesthesia dosing: a model predicts each patient's propofol maintenance requirement from covariates, beats a one-size-fits-all dose leave-one-out, and is shown in a dashboard "Adaptive demo" tab.

**Architecture:** Pure-Python tested library layer (`models/requirement.py`, `models/personalize.py`, `adaptive/demo.py`, a new chart builder) plus a render-only Streamlit tab. Reuses the M1 data-access layer; no network at runtime.

**Tech Stack:** numpy (least squares, no new dep), pandas, plotly, streamlit. Python 3.14.5, all installed.

---

## Verified Facts (prototyped on real M1 data, 2026-07-05)

- Per-case requirement = mean `propofol_rate` where `40 <= bis <= 60` and `bis > 10`.
- On the 39-case demo cohort: 38 cases usable; leave-one-out covariate model MAE **3.45**
  vs population-mean MAE **3.91** → **~12% improvement**.
- Covariate correlations with requirement: weight **+0.54**, sex_male **+0.41**, bmi
  **+0.33**, age **−0.38**, asa −0.09 — clinically sensible.
- Clinical covariates available via `data_access.load_clinical()`: `age, sex, height,
  weight, bmi, asa` (plus caseid).
- Case frames (`data_access.load_case`) have `bis, propofol_rate` columns.
- The within-case dynamics approach was rejected in prototyping (train→test requirement
  corr −0.17); this plan uses the covariate framing only.

---

## File Structure

```
src/adaptivedose/models/
├── __init__.py
├── requirement.py    # case_requirement(df, ...) -> float | None
└── personalize.py    # RequirementModel (fit/predict/coefficients) + leave_one_out
src/adaptivedose/adaptive/
├── __init__.py
└── demo.py           # build_modeling_table, run_demo -> metrics dict
src/adaptivedose/dashboard/
└── charts.py         # ADD scatter_actual_vs_predicted
apps/dashboard/views/
└── adaptive.py       # "Adaptive demo" tab (render-only)
apps/dashboard/app.py # wire the tab
tests/models/{__init__.py, test_requirement.py, test_personalize.py}
tests/adaptive/{__init__.py, test_demo.py}
tests/dashboard/test_charts.py  # ADD scatter test
```

Feature order is fixed everywhere as `FEATURES = ["age", "weight", "bmi", "sex_male", "asa"]`.

---

## Task 1: Per-case requirement target

**Files:**
- Create: `src/adaptivedose/models/__init__.py` (empty)
- Create: `src/adaptivedose/models/requirement.py`
- Create: `tests/models/__init__.py` (empty)
- Test: `tests/models/test_requirement.py`

- [ ] **Step 1: Create empty `src/adaptivedose/models/__init__.py` and `tests/models/__init__.py`**

- [ ] **Step 2: Write the failing test**

```python
# tests/models/test_requirement.py
import pandas as pd
from adaptivedose.models.requirement import case_requirement

def _case(bis, ppf):
    return pd.DataFrame({"bis": bis, "propofol_rate": ppf})

def test_requirement_is_mean_rate_in_band():
    # 3 rows in [40,60]; rates 4,6,5 -> mean 5.0
    df = _case([45, 50, 55, 30], [4.0, 6.0, 5.0, 9.0])
    assert case_requirement(df, min_in_band_rows=3) == 5.0

def test_requirement_excludes_artifact_zero_bis():
    # bis 0 is an artifact, must not count even though it's < 60
    df = _case([0, 45, 50, 55], [99.0, 4.0, 6.0, 5.0])
    assert case_requirement(df, min_in_band_rows=3) == 5.0

def test_requirement_none_when_too_few_in_band():
    df = _case([45, 50, 70, 80], [4.0, 6.0, 5.0, 5.0])  # only 2 in band
    assert case_requirement(df, min_in_band_rows=3) is None

def test_requirement_ignores_nan_rows():
    df = _case([45, 50, 55], [4.0, None, 6.0])  # middle rate NaN dropped -> mean 5.0
    assert case_requirement(df, min_in_band_rows=2) == 5.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/models/test_requirement.py -v`
Expected: FAIL — module not found.

- [ ] **Step 4: Write `requirement.py`**

```python
# src/adaptivedose/models/requirement.py
from __future__ import annotations
import pandas as pd

def case_requirement(
    df: pd.DataFrame,
    low: float = 40.0,
    high: float = 60.0,
    artifact_floor: float = 10.0,
    min_in_band_rows: int = 10,
) -> float | None:
    """Mean propofol infusion rate while anesthesia is adequate (BIS in band).

    Rows with BIS at or below `artifact_floor` are treated as sensor artifacts and
    excluded. Returns None if fewer than `min_in_band_rows` usable rows remain.
    """
    d = df.dropna(subset=["bis", "propofol_rate"])
    in_band = d[(d["bis"] >= low) & (d["bis"] <= high) & (d["bis"] > artifact_floor)]
    if len(in_band) < min_in_band_rows:
        return None
    return float(in_band["propofol_rate"].mean())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/models/test_requirement.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/adaptivedose/models/__init__.py src/adaptivedose/models/requirement.py tests/models/__init__.py tests/models/test_requirement.py
git commit -m "feat(models): per-case propofol maintenance requirement target"
```

---

## Task 2: Personalization model

**Files:**
- Create: `src/adaptivedose/models/personalize.py`
- Test: `tests/models/test_personalize.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_personalize.py
import numpy as np
from adaptivedose.models.personalize import RequirementModel, leave_one_out, FEATURES

def _synth(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    # true signal: first feature strong positive, second negative
    y = 20 + 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(scale=0.5, size=n)
    return X, y

def test_fit_recovers_known_coefficients():
    X, y = _synth()
    m = RequirementModel().fit(X, y)
    coef = m.coefficients(FEATURES)
    assert abs(coef["age"] - 3.0) < 0.3      # FEATURES[0] slot
    assert abs(coef["weight"] - (-2.0)) < 0.3  # FEATURES[1] slot
    assert abs(coef["intercept"] - 20.0) < 0.5

def test_predict_shape_and_values():
    X, y = _synth()
    m = RequirementModel().fit(X, y)
    preds = m.predict(X)
    assert preds.shape == (len(y),)
    assert np.mean(np.abs(preds - y)) < 1.0   # good fit on clean synthetic

def test_leave_one_out_beats_mean_baseline_when_signal_exists():
    X, y = _synth()
    preds = leave_one_out(X, y)
    assert preds.shape == (len(y),)
    n = len(y)
    baseline = np.array([y[np.arange(n) != i].mean() for i in range(n)])
    model_mae = np.mean(np.abs(y - preds))
    base_mae = np.mean(np.abs(y - baseline))
    assert model_mae < base_mae   # covariates help out-of-sample
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/models/test_personalize.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `personalize.py`**

```python
# src/adaptivedose/models/personalize.py
from __future__ import annotations
from typing import Dict, List
import numpy as np

FEATURES: List[str] = ["age", "weight", "bmi", "sex_male", "asa"]

class RequirementModel:
    """Ordinary least squares: covariates -> propofol maintenance requirement."""

    def __init__(self):
        self.coef_ = None  # coef_[0] is the intercept

    def fit(self, X, y) -> "RequirementModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        A = np.column_stack([np.ones(len(X)), X])
        self.coef_, *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        A = np.column_stack([np.ones(len(X)), X])
        return A @ self.coef_

    def coefficients(self, feature_names: List[str]) -> Dict[str, float]:
        names = ["intercept"] + list(feature_names)
        return {n: float(c) for n, c in zip(names, self.coef_)}

def leave_one_out(X, y) -> np.ndarray:
    """Return one out-of-sample prediction per row (fit on all others)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        preds[i] = RequirementModel().fit(X[mask], y[mask]).predict(X[i])[0]
    return preds
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/models/test_personalize.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/adaptivedose/models/personalize.py tests/models/test_personalize.py
git commit -m "feat(models): linear requirement model with leave-one-out CV"
```

---

## Task 3: Demonstration assembly

**Files:**
- Create: `src/adaptivedose/adaptive/__init__.py` (empty)
- Create: `src/adaptivedose/adaptive/demo.py`
- Create: `tests/adaptive/__init__.py` (empty)
- Test: `tests/adaptive/test_demo.py`

- [ ] **Step 1: Create empty `src/adaptivedose/adaptive/__init__.py` and `tests/adaptive/__init__.py`**

- [ ] **Step 2: Write the failing test**

```python
# tests/adaptive/test_demo.py
import numpy as np
import pandas as pd
from adaptivedose.adaptive import demo

def _clinical():
    # weight drives requirement; 10 patients
    return pd.DataFrame({
        "caseid": list(range(1, 11)),
        "age": [40, 50, 60, 45, 55, 65, 35, 70, 48, 52],
        "weight": [60, 70, 80, 65, 75, 85, 55, 90, 68, 72],
        "bmi": [22, 24, 26, 23, 25, 27, 21, 28, 23, 24],
        "sex": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
        "asa": [1, 2, 2, 1, 2, 3, 1, 3, 2, 2],
    })

def _case_frames():
    # requirement proportional to weight/10 so covariates carry signal;
    # each case: 12 in-band rows at a constant rate
    frames = {}
    for cid, w in zip(range(1, 11), [60, 70, 80, 65, 75, 85, 55, 90, 68, 72]):
        rate = w / 10.0
        frames[cid] = pd.DataFrame({"bis": [50.0] * 12, "propofol_rate": [rate] * 12})
    return frames

def test_build_modeling_table_joins_and_computes_requirement():
    frames = _case_frames()
    cohort = pd.DataFrame({"caseid": list(range(1, 11))})
    table = demo.build_modeling_table(cohort, _clinical(), lambda cid: frames[cid])
    assert len(table) == 10
    assert set(demo.FEATURES) <= set(table.columns)
    assert "requirement" in table.columns
    # case 3 has weight 80 -> requirement 8.0
    assert table[table["caseid"] == 3]["requirement"].iloc[0] == 8.0
    assert table[table["caseid"] == 1]["sex_male"].iloc[0] == 1.0

def test_build_modeling_table_drops_cases_without_requirement():
    frames = _case_frames()
    frames[5] = pd.DataFrame({"bis": [90.0] * 12, "propofol_rate": [7.0] * 12})  # none in band
    cohort = pd.DataFrame({"caseid": list(range(1, 11))})
    table = demo.build_modeling_table(cohort, _clinical(), lambda cid: frames[cid])
    assert 5 not in set(table["caseid"])

def test_run_demo_reports_improvement_and_coefficients():
    frames = _case_frames()
    cohort = pd.DataFrame({"caseid": list(range(1, 11))})
    table = demo.build_modeling_table(cohort, _clinical(), lambda cid: frames[cid])
    res = demo.run_demo(table)
    assert res["model_mae"] < res["baseline_mae"]     # personalization helps
    assert res["improvement"] > 0
    assert "weight" in res["coefficients"]
    assert res["coefficients"]["weight"] > 0          # heavier -> more propofol
    assert len(res["table"]) == len(table)
    assert "loo_pred" in res["table"].columns
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/adaptive/test_demo.py -v`
Expected: FAIL — module not found.

- [ ] **Step 4: Write `demo.py`**

```python
# src/adaptivedose/adaptive/demo.py
from __future__ import annotations
from typing import Callable, Dict
import numpy as np
import pandas as pd
from adaptivedose.models.requirement import case_requirement
from adaptivedose.models.personalize import RequirementModel, leave_one_out, FEATURES

def build_modeling_table(
    cohort: pd.DataFrame,
    clinical: pd.DataFrame,
    load_case_fn: Callable[[int], pd.DataFrame],
) -> pd.DataFrame:
    """Join per-case requirement with covariates; drop cases lacking either."""
    rows = []
    for cid in cohort["caseid"]:
        req = case_requirement(load_case_fn(int(cid)))
        if req is not None:
            rows.append({"caseid": int(cid), "requirement": req})
    reqs = pd.DataFrame(rows)
    if reqs.empty:
        return pd.DataFrame(columns=["caseid", "requirement", *FEATURES])
    table = reqs.merge(clinical, on="caseid", how="left")
    table["sex_male"] = (table["sex"] == "M").astype(float)
    return table.dropna(subset=[*FEATURES, "requirement"]).reset_index(drop=True)

def run_demo(table: pd.DataFrame) -> Dict:
    """Leave-one-out demonstration: personalized vs population dosing."""
    X = table[FEATURES].to_numpy(dtype=float)
    y = table["requirement"].to_numpy(dtype=float)
    n = len(y)
    preds = leave_one_out(X, y)
    baseline = np.array([y[np.arange(n) != i].mean() for i in range(n)])
    model_mae = float(np.mean(np.abs(y - preds)))
    baseline_mae = float(np.mean(np.abs(y - baseline)))
    improvement = (baseline_mae - model_mae) / baseline_mae if baseline_mae else 0.0
    coefficients = RequirementModel().fit(X, y).coefficients(FEATURES)
    out = table.copy()
    out["loo_pred"] = preds
    out["population_dose"] = float(y.mean())
    return {
        "table": out,
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "improvement": improvement,
        "coefficients": coefficients,
        "dose_spread_std": float(np.std(preds)),
    }

# re-export for convenience
__all__ = ["build_modeling_table", "run_demo", "FEATURES"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/adaptive/test_demo.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/adaptivedose/adaptive/__init__.py src/adaptivedose/adaptive/demo.py tests/adaptive/__init__.py tests/adaptive/test_demo.py
git commit -m "feat(adaptive): leave-one-out personalization demonstration"
```

---

## Task 4: Actual-vs-predicted scatter chart

**Files:**
- Modify: `src/adaptivedose/dashboard/charts.py`
- Test: `tests/dashboard/test_charts.py`

- [ ] **Step 1: Append the failing test to `tests/dashboard/test_charts.py`**

```python
def test_scatter_actual_vs_predicted_has_points_and_identity():
    import plotly.graph_objects as go
    fig = charts.scatter_actual_vs_predicted([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2                     # points + identity line
    assert fig.data[0].mode == "markers"
    assert fig.data[1].name == "identity"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dashboard/test_charts.py::test_scatter_actual_vs_predicted_has_points_and_identity -v`
Expected: FAIL — `scatter_actual_vs_predicted` not defined.

- [ ] **Step 3: Append `scatter_actual_vs_predicted` to `src/adaptivedose/dashboard/charts.py`**

```python
def scatter_actual_vs_predicted(actual, predicted,
                                title: str = "Requirement: actual vs predicted") -> go.Figure:
    actual = list(actual)
    predicted = list(predicted)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=predicted, mode="markers", name="patients"))
    lo = min(min(actual), min(predicted))
    hi = max(max(actual), max(predicted))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="identity"))
    fig.update_layout(title=title, xaxis_title="actual", yaxis_title="predicted",
                      margin=dict(l=40, r=20, t=40, b=40))
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/dashboard/test_charts.py -v`
Expected: all charts tests pass (3 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add src/adaptivedose/dashboard/charts.py tests/dashboard/test_charts.py
git commit -m "feat(dashboard): actual-vs-predicted scatter figure"
```

---

## Task 5: Adaptive demo tab

**Files:**
- Create: `apps/dashboard/views/adaptive.py`
- Modify: `apps/dashboard/app.py`

- [ ] **Step 1: Write `apps/dashboard/views/adaptive.py`**

```python
# apps/dashboard/views/adaptive.py
import pandas as pd
import streamlit as st
from adaptivedose.dashboard import data_access as da, charts
from adaptivedose.adaptive import demo

@st.cache_data
def _run(caseids: tuple):
    cohort = pd.DataFrame({"caseid": list(caseids)})
    clinical = da.load_clinical()
    table = demo.build_modeling_table(cohort, clinical, da.load_case)
    if len(table) < len(demo.FEATURES) + 2:
        return None
    return demo.run_demo(table)

def render_adaptive_demo(ctx):
    st.header("Adaptive demo — personalized propofol requirement")
    st.caption("Model-based capability demo on observational data — not for clinical use.")

    res = _run(tuple(sorted(int(c) for c in ctx["cohort"]["caseid"])))
    if res is None:
        st.info("Not enough usable cases. Rebuild the dataset with more cases: "
                "`.venv/bin/python pipelines/build_dataset.py --config configs/data.yaml --limit 60`.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Personalized MAE", f"{res['model_mae']:.2f}")
    c2.metric("Population MAE", f"{res['baseline_mae']:.2f}")
    c3.metric("Improvement", f"{res['improvement'] * 100:.0f}%")

    st.subheader("Covariate effects (direction of personalization)")
    coefs = {k: v for k, v in res["coefficients"].items() if k != "intercept"}
    st.dataframe(pd.DataFrame({"covariate": list(coefs), "coefficient": list(coefs.values())}),
                 width='stretch')

    st.subheader("Actual vs personalized prediction (leave-one-out)")
    st.plotly_chart(
        charts.scatter_actual_vs_predicted(res["table"]["requirement"], res["table"]["loo_pred"]),
        width='stretch',
    )

    st.subheader("Per patient")
    table = res["table"]
    cid = st.selectbox("Case", table["caseid"].tolist())
    row = table[table["caseid"] == cid].iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Personalized dose", f"{row['loo_pred']:.1f}")
    m2.metric("Population dose", f"{row['population_dose']:.1f}")
    m3.metric("Actual requirement", f"{row['requirement']:.1f}")
```

- [ ] **Step 2: Wire it into `apps/dashboard/app.py`**

Change the views import line to include `adaptive`:

```python
from apps.dashboard.views import cohort, case, quality, compare, adaptive
```

Add `"Adaptive demo"` to the sidebar radio list so it reads:

```python
    view = st.sidebar.radio(
        "View",
        ["Cohort overview", "Case viewer", "Signal quality / EDA", "Compare cases",
         "Adaptive demo"],
    )
```

Add a dispatch branch before the chain ends:

```python
    elif view == "Adaptive demo":
        adaptive.render_adaptive_demo(ctx)
```

- [ ] **Step 3: Verify the tab renders on real data via AppTest**

Run:
```bash
.venv/bin/python - <<'PY'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("apps/dashboard/app.py", default_timeout=60); at.run()
at.sidebar.radio[0].set_value("Adaptive demo").run()
assert not at.exception, at.exception
print("headers:", [h.value for h in at.header])
print("metrics:", [(m.label, m.value) for m in at.metric])
PY
```
Expected: no exception; prints the "Adaptive demo" header and metric values (Personalized MAE, Population MAE, Improvement). On the 39-case dataset the improvement should be positive (~12%).

- [ ] **Step 4: Confirm the demonstration numbers directly**

Run:
```bash
.venv/bin/python - <<'PY'
import pandas as pd
from adaptivedose.dashboard import data_access as da
from adaptivedose.adaptive import demo
cohort = da.build_cohort_table()
table = demo.build_modeling_table(cohort, da.load_clinical(), da.load_case)
res = demo.run_demo(table)
print(f"cases={len(table)} model_mae={res['model_mae']:.2f} "
      f"baseline_mae={res['baseline_mae']:.2f} improvement={res['improvement']*100:.0f}%")
print("coefficients:", {k: round(v,3) for k,v in res['coefficients'].items()})
assert res["improvement"] > 0, "expected personalization to beat population dose"
PY
```
Expected: prints ~`model_mae 3.4 baseline_mae 3.9 improvement 12%` and a positive `weight` coefficient; the assertion passes.

- [ ] **Step 5: Run the full unit suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (M1 + dashboard + models `requirement` 4, `personalize` 3, `adaptive` 3, charts scatter 1).

- [ ] **Step 6: Commit**

```bash
git add apps/dashboard/views/adaptive.py apps/dashboard/app.py
git commit -m "feat(dashboard): adaptive demo tab proving per-patient personalization"
```

---

## Definition of Done

- `pytest` green across new modules (`requirement`, `personalize`, `demo`, scatter chart)
  and all prior suites.
- The "Adaptive demo" tab renders on the real 39-case dataset and shows a positive
  leave-one-out improvement of personalized over population dosing, with a positive
  weight coefficient.
- All computation is in the tested library layer; the Streamlit tab is render-only.

## Self-Review Notes

- **Spec coverage:** §2 requirement target (Task 1) ✓; §3 linear model (Task 2) ✓;
  §4 recommendation = prediction (demo `loo_pred`/`population_dose`, Task 3) ✓;
  §5 LOO demonstration + three evidence outputs (Task 3 metrics + Task 5 tab) ✓;
  §6 architecture ✓; §7 data flow via data_access ✓; §8 scope/honesty (tab caption +
  limitations) — the tab shows a not-for-clinical-use caption; the three written
  limitations live in the spec ✓; §9 testing ✓.
- **Type consistency:** `FEATURES = ["age","weight","bmi","sex_male","asa"]` is defined
  once in `personalize.py` and imported by `demo.py`; the modeling table always carries
  these plus `caseid, requirement`, and `run_demo` adds `loo_pred, population_dose`.
- **Non-circular:** `leave_one_out` excludes each patient from its own fit; the baseline
  is mean-of-others. No leakage.
