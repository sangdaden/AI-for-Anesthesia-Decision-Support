# M1 Data Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Streamlit dashboard to explore the M1 VitalDB dataset — cohort overview, per-case signal viewer, data-quality/EDA, and multi-case comparison — reading only local M1 pipeline outputs.

**Architecture:** All computation lives in a pure-Python, unit-tested library layer (`src/adaptivedose/dashboard/`: `data_access`, `stats`, `charts`). The Streamlit files (`apps/dashboard/`) are render-only. The dashboard is **manifest-driven**: `manifest.parquet` is the source of truth for which cases belong to the dataset, never the `cases/` directory listing.

**Tech Stack:** Streamlit 1.58, Plotly 6.8, pandas, pyarrow. Both already installed in `.venv`. Python 3.14.5.

---

## Verified Facts (from the live workspace, 2026-07-04)

- `.venv` already has `streamlit==1.58.0`, `plotly==6.8.0` (install them into pyproject anyway for reproducibility).
- `data/processed/manifest.parquet` columns: `caseid, n_samples, kept`. (The pipeline's split step merges `subjectid` in-memory but the persisted manifest has only these three — do NOT assume `subjectid` in the manifest file.)
- `data/processed/splits.json`: `{"train": [...], "val": [...], "test": [...]}` of caseids.
- `data/processed/cases/case_<id>.parquet` columns: `caseid, time_sec, bis, propofol_rate, remifentanil_rate, map, hr, spo2, etco2`.
- `.vitaldb_cache/cases.parquet` (clinical) includes: `caseid, age, sex, asa, optype, subjectid` (74 cols total).
- **Known drift:** the `cases/` dir currently holds 325 files while the manifest lists 5 — a prior larger run's files persisted when a later `--limit 5` run overwrote the manifest. Task 1 fixes the pipeline to prevent this; the dashboard ignores stray files by being manifest-driven.
- `interval_sec` (sampling cadence) is 10 in `configs/data.yaml`; duration_minutes = `n_samples * interval_sec / 60`.

---

## File Structure

```
apps/dashboard/
├── app.py                          # Streamlit entry: sidebar nav + missing-data guard
└── views/
    ├── __init__.py
    ├── cohort.py                   # render_cohort_overview(ctx)
    ├── case.py                     # render_case_viewer(ctx)
    ├── quality.py                  # render_signal_quality(ctx)
    └── compare.py                  # render_compare(ctx)
src/adaptivedose/dashboard/
├── __init__.py
├── data_access.py                  # load manifest/splits/case/clinical; cohort table (tested)
├── stats.py                        # cohort_summary, distributions, missingness, targets (tested)
└── charts.py                       # pure Plotly figure builders (tested)
tests/dashboard/
├── __init__.py
├── test_data_access.py
├── test_stats.py
└── test_charts.py
```

**Responsibilities:** `data_access` is the only reader of disk. `stats` computes numbers from DataFrames. `charts` turns DataFrames/Series into Plotly figures. Views compose these and call `st.*`. Nothing in `apps/` computes anything testable.

---

## Task 1: Pipeline hygiene — clear stale case files on rebuild

**Files:**
- Modify: `src/adaptivedose/data/builder.py`
- Test: `tests/data/test_builder.py`

Prevents the manifest↔cases drift: `build_dataset` must clear old `case_*.parquet` before writing, so a rebuild never leaves orphans.

- [ ] **Step 1: Write the failing test** (append to `tests/data/test_builder.py`)

```python
def test_build_dataset_clears_stale_case_files(fake_load_case, tmp_path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True)
    stale = cases_dir / "case_999.parquet"
    stale.write_text("stale")  # orphan from a previous, larger run
    tracks = {"bis": "BIS/BIS", "propofol_rate": "Orchestra/PPF20_RATE",
              "map": "Solar8000/ART_MBP"}
    build_dataset([1], tracks, interval_sec=10, output_dir=tmp_path,
                  load_fn=fake_load_case, min_samples=5)
    assert not stale.exists()                       # orphan removed
    assert (cases_dir / "case_1.parquet").exists()  # fresh case written
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/data/test_builder.py::test_build_dataset_clears_stale_case_files -v`
Expected: FAIL — `case_999.parquet` still exists.

- [ ] **Step 3: Edit `build_dataset` in `src/adaptivedose/data/builder.py`**

Replace the line `out_dir = ensure_dir(Path(output_dir) / "cases")` with:

```python
    out_dir = ensure_dir(Path(output_dir) / "cases")
    for stale in out_dir.glob("case_*.parquet"):  # avoid manifest<->cases drift
        stale.unlink()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/data/test_builder.py -v`
Expected: all builder tests pass (including the new one).

- [ ] **Step 5: Commit**

```bash
git add src/adaptivedose/data/builder.py tests/data/test_builder.py
git commit -m "fix(data): clear stale case parquet on rebuild to prevent manifest drift"
```

---

## Task 2: Dependencies and package scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `src/adaptivedose/dashboard/__init__.py`
- Create: `apps/dashboard/views/__init__.py`
- Create: `tests/dashboard/__init__.py`

- [ ] **Step 1: Add a `dashboard` optional-dependency group to `pyproject.toml`**

Insert into the `[project.optional-dependencies]` table (which already has `dev`):

```toml
dashboard = ["streamlit>=1.58", "plotly>=6.8"]
```

- [ ] **Step 2: Create the empty package init files**

Create these three empty files:
- `src/adaptivedose/dashboard/__init__.py`
- `apps/dashboard/views/__init__.py`
- `tests/dashboard/__init__.py`

- [ ] **Step 3: Install the new extra**

Run: `.venv/bin/pip install -e ".[dev,dashboard]"`
Expected: succeeds; `streamlit` and `plotly` already present, so this is fast.

- [ ] **Step 4: Verify import**

Run: `.venv/bin/python -c "import streamlit, plotly; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/adaptivedose/dashboard/__init__.py apps/dashboard/views/__init__.py tests/dashboard/__init__.py
git commit -m "chore(dashboard): add streamlit/plotly extra and package scaffolding"
```

---

## Task 3: Data access layer

**Files:**
- Create: `src/adaptivedose/dashboard/data_access.py`
- Test: `tests/dashboard/test_data_access.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dashboard/test_data_access.py
import json
import pandas as pd
import pytest
from adaptivedose.dashboard import data_access as da

def _make_processed(tmp_path):
    proc = tmp_path / "processed"
    (proc / "cases").mkdir(parents=True)
    manifest = pd.DataFrame({"caseid": [1, 2, 3],
                             "n_samples": [100, 200, 5],
                             "kept": [True, True, False]})
    manifest.to_parquet(proc / "manifest.parquet", index=False)
    (proc / "splits.json").write_text(json.dumps(
        {"train": [1], "val": [2], "test": []}))
    for cid in (1, 2):
        pd.DataFrame({"caseid": [cid, cid], "time_sec": [0, 10],
                      "bis": [45.0, 46.0], "map": [70.0, 71.0]}
                     ).to_parquet(proc / "cases" / f"case_{cid}.parquet", index=False)
    return proc

def _make_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    pd.DataFrame({"caseid": [1, 2, 3], "age": [50, 60, 70],
                  "sex": ["M", "F", "M"], "asa": [2, 3, 2],
                  "optype": ["Colorectal", "Biliary", "Thyroid"]}
                 ).to_parquet(cache / "cases.parquet", index=False)
    return cache

def test_load_manifest_reads_parquet(tmp_path):
    proc = _make_processed(tmp_path)
    m = da.load_manifest(proc)
    assert list(m["caseid"]) == [1, 2, 3]

def test_load_manifest_missing_raises_typed_error(tmp_path):
    with pytest.raises(da.DataNotFoundError):
        da.load_manifest(tmp_path / "nope")

def test_load_case_reads_by_id(tmp_path):
    proc = _make_processed(tmp_path)
    df = da.load_case(1, proc)
    assert (df["caseid"] == 1).all()
    assert "bis" in df.columns

def test_load_splits_returns_dict(tmp_path):
    proc = _make_processed(tmp_path)
    s = da.load_splits(proc)
    assert s["train"] == [1] and s["test"] == []

def test_build_cohort_table_joins_clinical_and_split_and_keeps_only_kept(tmp_path):
    proc = _make_processed(tmp_path)
    cache = _make_cache(tmp_path)
    cohort = da.build_cohort_table(proc, cache)
    # case 3 is kept=False -> excluded
    assert set(cohort["caseid"]) == {1, 2}
    assert set(cohort.columns) >= {"caseid", "n_samples", "split", "age", "sex", "asa", "optype"}
    row1 = cohort[cohort["caseid"] == 1].iloc[0]
    assert row1["split"] == "train"
    assert row1["age"] == 50
    row2 = cohort[cohort["caseid"] == 2].iloc[0]
    assert row2["split"] == "val"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dashboard/test_data_access.py -v`
Expected: FAIL — module `adaptivedose.dashboard.data_access` not found.

- [ ] **Step 3: Write `data_access.py`**

```python
# src/adaptivedose/dashboard/data_access.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd

DEFAULT_PROCESSED = Path("data/processed")
DEFAULT_CACHE = Path(".vitaldb_cache")

class DataNotFoundError(RuntimeError):
    """Raised when an expected M1 pipeline output is missing."""

def _require(path: Path) -> Path:
    if not path.exists():
        raise DataNotFoundError(
            f"{path} not found. Run: .venv/bin/python pipelines/build_dataset.py "
            f"--config configs/data.yaml"
        )
    return path

def load_manifest(processed_dir: str | Path = DEFAULT_PROCESSED) -> pd.DataFrame:
    return pd.read_parquet(_require(Path(processed_dir) / "manifest.parquet"))

def load_splits(processed_dir: str | Path = DEFAULT_PROCESSED) -> Dict[str, List[int]]:
    return json.loads(_require(Path(processed_dir) / "splits.json").read_text())

def load_case(caseid: int, processed_dir: str | Path = DEFAULT_PROCESSED) -> pd.DataFrame:
    return pd.read_parquet(
        _require(Path(processed_dir) / "cases" / f"case_{caseid}.parquet")
    )

def load_clinical(cache_dir: str | Path = DEFAULT_CACHE) -> pd.DataFrame:
    return pd.read_parquet(_require(Path(cache_dir) / "cases.parquet"))

def split_label_map(splits: Dict[str, List[int]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for name, ids in splits.items():
        for cid in ids:
            out[int(cid)] = name
    return out

def build_cohort_table(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    cache_dir: str | Path = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Kept cases joined with clinical info and their split label.

    Manifest is the source of truth; stray case files on disk are ignored.
    """
    manifest = load_manifest(processed_dir)
    kept = manifest[manifest["kept"]].copy()
    labels = split_label_map(load_splits(processed_dir))
    kept["split"] = kept["caseid"].map(labels).fillna("unassigned")
    clinical = load_clinical(cache_dir)[["caseid", "age", "sex", "asa", "optype"]]
    return kept.merge(clinical, on="caseid", how="left")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/dashboard/test_data_access.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/adaptivedose/dashboard/data_access.py tests/dashboard/test_data_access.py
git commit -m "feat(dashboard): manifest-driven data-access layer"
```

---

## Task 4: Stats layer

**Files:**
- Create: `src/adaptivedose/dashboard/stats.py`
- Test: `tests/dashboard/test_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dashboard/test_stats.py
import numpy as np
import pandas as pd
from adaptivedose.dashboard import stats

def _cohort():
    return pd.DataFrame({
        "caseid": [1, 2, 3],
        "n_samples": [60, 120, 180],
        "kept": [True, True, True],
        "split": ["train", "train", "val"],
        "age": [50, 60, 70],
    })

def _case():
    return pd.DataFrame({
        "caseid": [1] * 5,
        "time_sec": [0, 10, 20, 30, 40],
        "bis": [0.0, 45.0, 50.0, 55.0, 65.0],   # one artifact zero, one out of target
        "map": [70.0, np.nan, 60.0, 62.0, 80.0],
    })

def test_cohort_summary_counts_and_splits():
    s = stats.cohort_summary(_cohort())
    assert s["n_cases"] == 3
    assert s["split_sizes"] == {"train": 2, "val": 1}

def test_duration_minutes():
    d = stats.duration_minutes(_cohort(), interval_sec=10)
    # 60 samples * 10s / 60 = 10 min
    assert d.iloc[0] == 10.0
    assert d.iloc[2] == 30.0

def test_time_in_target_bis():
    # bis in [40,60] -> values 45,50,55 of 5 -> 0.6
    assert abs(stats.time_in_target(_case()) - 0.6) < 1e-9

def test_artifact_rate_counts_zeros():
    # one bis==0 of 5 -> 0.2
    assert abs(stats.artifact_rate(_case()) - 0.2) < 1e-9

def test_missingness_reports_fraction_per_column():
    m = stats.missingness([_case()], ["bis", "map"])
    assert m["bis"] == 0.0
    assert abs(m["map"] - 0.2) < 1e-9   # one NaN of 5

def test_signal_distribution_concatenates_values():
    vals = stats.signal_distribution([_case(), _case()], "bis")
    assert len(vals) == 10   # two cases x 5 rows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dashboard/test_stats.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `stats.py`**

```python
# src/adaptivedose/dashboard/stats.py
from __future__ import annotations
from typing import Dict, List
import pandas as pd

def cohort_summary(cohort: pd.DataFrame) -> Dict:
    return {
        "n_cases": int(len(cohort)),
        "split_sizes": cohort["split"].value_counts().to_dict(),
    }

def duration_minutes(cohort: pd.DataFrame, interval_sec: int) -> pd.Series:
    return cohort["n_samples"] * interval_sec / 60.0

def time_in_target(case: pd.DataFrame, low: float = 40.0, high: float = 60.0,
                   col: str = "bis") -> float:
    if col not in case.columns or len(case) == 0:
        return 0.0
    return float(case[col].between(low, high).mean())

def artifact_rate(case: pd.DataFrame, col: str = "bis", value: float = 0.0) -> float:
    if col not in case.columns or len(case) == 0:
        return 0.0
    return float((case[col] == value).mean())

def missingness(case_frames: List[pd.DataFrame], cols: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in cols:
        present = [f[col] for f in case_frames if col in f.columns]
        if not present:
            out[col] = 1.0
            continue
        s = pd.concat(present, ignore_index=True)
        out[col] = float(s.isna().mean())
    return out

def signal_distribution(case_frames: List[pd.DataFrame], col: str) -> pd.Series:
    present = [f[col] for f in case_frames if col in f.columns]
    if not present:
        return pd.Series(dtype=float)
    return pd.concat(present, ignore_index=True).dropna()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/dashboard/test_stats.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/adaptivedose/dashboard/stats.py tests/dashboard/test_stats.py
git commit -m "feat(dashboard): cohort/signal statistics layer"
```

---

## Task 5: Chart builders (pure Plotly figures)

**Files:**
- Create: `src/adaptivedose/dashboard/charts.py`
- Test: `tests/dashboard/test_charts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dashboard/test_charts.py
import pandas as pd
import plotly.graph_objects as go
from adaptivedose.dashboard import charts

def _case(cid=1):
    return pd.DataFrame({
        "caseid": [cid] * 3,
        "time_sec": [0, 10, 20],
        "bis": [45.0, 50.0, 55.0],
        "map": [70.0, 72.0, 68.0],
        "propofol_rate": [4.0, 5.0, 5.0],
        "remifentanil_rate": [3.0, 3.0, 4.0],
    })

def test_case_timeseries_figure_has_traces_per_signal():
    fig = charts.case_timeseries_figure(_case())
    assert isinstance(fig, go.Figure)
    names = {t.name for t in fig.data}
    assert {"BIS", "MAP", "propofol_rate", "remifentanil_rate"} <= names

def test_distribution_figure_is_histogram():
    fig = charts.distribution_figure(pd.Series([1.0, 2.0, 2.0, 3.0]), "bis")
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "histogram"

def test_compare_bis_figure_one_trace_per_case():
    fig = charts.compare_bis_figure({1: _case(1), 2: _case(2)})
    assert len(fig.data) == 2
    assert {t.name for t in fig.data} == {"case 1", "case 2"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/dashboard/test_charts.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `charts.py`**

```python
# src/adaptivedose/dashboard/charts.py
from __future__ import annotations
from typing import Dict
import pandas as pd
import plotly.graph_objects as go

def case_timeseries_figure(df: pd.DataFrame) -> go.Figure:
    """Multi-axis time-series: BIS (with target band) + MAP + drug rates."""
    t = df["time_sec"] / 60.0  # minutes
    fig = go.Figure()
    if "bis" in df.columns:
        fig.add_trace(go.Scatter(x=t, y=df["bis"], name="BIS", yaxis="y1"))
    if "map" in df.columns:
        fig.add_trace(go.Scatter(x=t, y=df["map"], name="MAP", yaxis="y2"))
    for drug in ("propofol_rate", "remifentanil_rate"):
        if drug in df.columns:
            fig.add_trace(go.Scatter(x=t, y=df[drug], name=drug, yaxis="y3"))
    # BIS target band [40,60] and MAP 65 threshold as reference shapes
    fig.add_hrect(y0=40, y1=60, line_width=0, fillcolor="green", opacity=0.08,
                  yref="y1")
    fig.update_layout(
        xaxis=dict(title="time (min)"),
        yaxis=dict(title="BIS", range=[0, 100]),
        yaxis2=dict(title="MAP (mmHg)", overlaying="y", side="right"),
        yaxis3=dict(title="rate", overlaying="y", side="right", position=0.95,
                    showgrid=False),
        legend=dict(orientation="h"),
        margin=dict(l=40, r=60, t=30, b=40),
    )
    return fig

def distribution_figure(values: pd.Series, title: str) -> go.Figure:
    fig = go.Figure(data=[go.Histogram(x=values, nbinsx=40)])
    fig.update_layout(title=title, margin=dict(l=40, r=20, t=40, b=40))
    return fig

def compare_bis_figure(frames: Dict[int, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for cid, df in frames.items():
        fig.add_trace(go.Scatter(x=df["time_sec"] / 60.0, y=df["bis"],
                                 name=f"case {cid}"))
    fig.add_hrect(y0=40, y1=60, line_width=0, fillcolor="green", opacity=0.08)
    fig.update_layout(xaxis=dict(title="time (min)"),
                      yaxis=dict(title="BIS", range=[0, 100]),
                      legend=dict(orientation="h"),
                      margin=dict(l=40, r=20, t=30, b=40))
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/dashboard/test_charts.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/adaptivedose/dashboard/charts.py tests/dashboard/test_charts.py
git commit -m "feat(dashboard): pure Plotly figure builders"
```

---

## Task 6: Streamlit app shell + missing-data guard

**Files:**
- Create: `apps/dashboard/app.py`
- Create: `apps/dashboard/views/cohort.py` (stub for now)

The four view modules are filled in Tasks 7-10. This task wires navigation and the
missing-data guard. Views are render-only (no unit tests) — verified with `streamlit run`.

- [ ] **Step 1: Create a minimal `cohort.py` stub so the app imports**

```python
# apps/dashboard/views/cohort.py
import streamlit as st

def render_cohort_overview(ctx):
    st.header("Cohort overview")
    st.write(f"{ctx['summary']['n_cases']} kept cases")
```

- [ ] **Step 2: Write `apps/dashboard/app.py`**

```python
# apps/dashboard/app.py
"""AdaptiveDose AI — M1 Data Explorer (Streamlit).

Run: .venv/bin/streamlit run apps/dashboard/app.py
"""
import streamlit as st
from adaptivedose.dashboard import data_access as da
from adaptivedose.dashboard import stats
from apps.dashboard.views import cohort

st.set_page_config(page_title="AdaptiveDose M1 Explorer", layout="wide")

@st.cache_data
def _load_context():
    cohort_df = da.build_cohort_table()
    return {
        "cohort": cohort_df,
        "summary": stats.cohort_summary(cohort_df),
    }

def main():
    st.title("AdaptiveDose AI — M1 Data Explorer")
    try:
        ctx = _load_context()
    except da.DataNotFoundError as exc:
        st.error(str(exc))
        st.info("The dashboard reads the M1 pipeline outputs under `data/processed/`. "
                "Run the pipeline first, then reload.")
        return

    view = st.sidebar.radio(
        "View",
        ["Cohort overview", "Case viewer", "Signal quality / EDA", "Compare cases"],
    )
    if view == "Cohort overview":
        cohort.render_cohort_overview(ctx)
    else:
        st.info(f"'{view}' is added in a later task.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the app boots against real data**

Run: `.venv/bin/streamlit run apps/dashboard/app.py --server.headless true --server.port 8599 & sleep 8; curl -s -o /dev/null -w "%{http_code}" http://localhost:8599; kill %1`
Expected: prints `200` (Streamlit served the page without a Python error). If `data/processed/` is missing it still returns 200 and shows the guidance message — that is acceptable.

- [ ] **Step 4: Commit**

```bash
git add apps/dashboard/app.py apps/dashboard/views/cohort.py
git commit -m "feat(dashboard): streamlit shell with nav and missing-data guard"
```

---

## Task 7: Cohort overview view

**Files:**
- Modify: `apps/dashboard/views/cohort.py`

- [ ] **Step 1: Replace `cohort.py` with the full view**

```python
# apps/dashboard/views/cohort.py
import streamlit as st
from adaptivedose.dashboard import stats, charts

def render_cohort_overview(ctx):
    cohort = ctx["cohort"]
    summary = ctx["summary"]

    st.header("Cohort overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Kept cases", summary["n_cases"])
    c2.metric("Train / Val / Test",
              " / ".join(str(summary["split_sizes"].get(k, 0))
                         for k in ("train", "val", "test")))
    dur = stats.duration_minutes(cohort, interval_sec=10)
    c3.metric("Median duration (min)", f"{dur.median():.1f}")

    st.subheader("Distributions")
    a, b = st.columns(2)
    with a:
        if "age" in cohort:
            st.plotly_chart(charts.distribution_figure(cohort["age"].dropna(), "Age"),
                            use_container_width=True)
        if "asa" in cohort:
            st.bar_chart(cohort["asa"].value_counts().sort_index())
    with b:
        if "sex" in cohort:
            st.bar_chart(cohort["sex"].value_counts())
        st.plotly_chart(charts.distribution_figure(dur, "Case duration (min)"),
                        use_container_width=True)

    st.subheader("Cases")
    st.dataframe(cohort, use_container_width=True)
```

- [ ] **Step 2: Verify the view renders**

Run: `.venv/bin/streamlit run apps/dashboard/app.py --server.headless true --server.port 8599 & sleep 8; curl -s -o /dev/null -w "%{http_code}" http://localhost:8599; kill %1`
Expected: `200`, no exception in the terminal output.

- [ ] **Step 3: Commit**

```bash
git add apps/dashboard/views/cohort.py
git commit -m "feat(dashboard): cohort overview view"
```

---

## Task 8: Case viewer view

**Files:**
- Create: `apps/dashboard/views/case.py`
- Modify: `apps/dashboard/app.py` (wire the view)

- [ ] **Step 1: Write `apps/dashboard/views/case.py`**

```python
# apps/dashboard/views/case.py
import streamlit as st
from adaptivedose.dashboard import data_access as da
from adaptivedose.dashboard import stats, charts

def render_case_viewer(ctx):
    cohort = ctx["cohort"]
    st.header("Case viewer")
    caseid = st.selectbox("Case", sorted(cohort["caseid"].tolist()))
    df = da.load_case(int(caseid))

    m1, m2, m3 = st.columns(3)
    m1.metric("Samples", len(df))
    m2.metric("Time in BIS target", f"{stats.time_in_target(df) * 100:.0f}%")
    m3.metric("BIS artifact (==0)", f"{stats.artifact_rate(df) * 100:.1f}%")

    st.plotly_chart(charts.case_timeseries_figure(df), use_container_width=True)
```

- [ ] **Step 2: Wire it in `apps/dashboard/app.py`**

Change the import line `from apps.dashboard.views import cohort` to:

```python
from apps.dashboard.views import cohort, case
```

Replace the `else:` branch of the view dispatch with:

```python
    elif view == "Cohort overview":
        cohort.render_cohort_overview(ctx)
    elif view == "Case viewer":
        case.render_case_viewer(ctx)
    else:
        st.info(f"'{view}' is added in a later task.")
```

(and change the existing `if view == "Cohort overview":` to be the first branch of this
`if/elif` chain — i.e. remove the old standalone `if/else` and use the chain above.)

- [ ] **Step 3: Verify**

Run: `.venv/bin/streamlit run apps/dashboard/app.py --server.headless true --server.port 8599 & sleep 8; curl -s -o /dev/null -w "%{http_code}" http://localhost:8599; kill %1`
Expected: `200`, no exception.

- [ ] **Step 4: Commit**

```bash
git add apps/dashboard/views/case.py apps/dashboard/app.py
git commit -m "feat(dashboard): per-case time-series viewer"
```

---

## Task 9: Signal quality / EDA view

**Files:**
- Create: `apps/dashboard/views/quality.py`
- Modify: `apps/dashboard/app.py` (wire the view + load kept case frames once)

- [ ] **Step 1: Add a cached loader for all kept case frames in `app.py`**

Add this function next to `_load_context` in `apps/dashboard/app.py`:

```python
@st.cache_data
def _load_case_frames():
    cohort = da.build_cohort_table()
    return {int(cid): da.load_case(int(cid)) for cid in cohort["caseid"]}
```

- [ ] **Step 2: Write `apps/dashboard/views/quality.py`**

```python
# apps/dashboard/views/quality.py
import streamlit as st
from adaptivedose.dashboard import stats, charts

SIGNALS = ["bis", "map", "hr", "spo2", "propofol_rate", "remifentanil_rate", "etco2"]

def render_signal_quality(ctx, case_frames):
    st.header("Signal quality / EDA")

    frames = list(case_frames.values())
    miss = stats.missingness(frames, SIGNALS)
    st.subheader("Missingness (fraction NaN across cohort)")
    st.bar_chart({k: [v] for k, v in miss.items()})

    st.subheader("BIS artifacts and target")
    total_zero = sum(stats.artifact_rate(f) * len(f) for f in frames)
    total_rows = sum(len(f) for f in frames)
    in_target = sum(stats.time_in_target(f) * len(f) for f in frames)
    c1, c2 = st.columns(2)
    c1.metric("BIS == 0 (artifact) share",
              f"{(total_zero / total_rows * 100) if total_rows else 0:.1f}%")
    c2.metric("Time in BIS target [40,60]",
              f"{(in_target / total_rows * 100) if total_rows else 0:.1f}%")

    st.subheader("Signal distribution")
    sig = st.selectbox("Signal", SIGNALS)
    values = stats.signal_distribution(frames, sig)
    if len(values):
        st.plotly_chart(charts.distribution_figure(values, sig),
                        use_container_width=True)
    else:
        st.info(f"No data for {sig}.")
```

- [ ] **Step 3: Wire it in `app.py`**

Update the import to `from apps.dashboard.views import cohort, case, quality` and add to the
dispatch chain:

```python
    elif view == "Signal quality / EDA":
        quality.render_signal_quality(ctx, _load_case_frames())
```

- [ ] **Step 4: Verify**

Run: `.venv/bin/streamlit run apps/dashboard/app.py --server.headless true --server.port 8599 & sleep 8; curl -s -o /dev/null -w "%{http_code}" http://localhost:8599; kill %1`
Expected: `200`, no exception.

- [ ] **Step 5: Commit**

```bash
git add apps/dashboard/views/quality.py apps/dashboard/app.py
git commit -m "feat(dashboard): signal-quality / EDA view"
```

---

## Task 10: Compare-cases view

**Files:**
- Create: `apps/dashboard/views/compare.py`
- Modify: `apps/dashboard/app.py` (wire the view)

- [ ] **Step 1: Write `apps/dashboard/views/compare.py`**

```python
# apps/dashboard/views/compare.py
import streamlit as st
from adaptivedose.dashboard import data_access as da
from adaptivedose.dashboard import charts

def render_compare(ctx):
    cohort = ctx["cohort"]
    st.header("Compare cases")
    ids = st.multiselect("Cases (pick 2 or more)",
                         sorted(cohort["caseid"].tolist()))
    if len(ids) < 2:
        st.info("Select at least two cases to compare.")
        return
    frames = {int(cid): da.load_case(int(cid)) for cid in ids}
    st.plotly_chart(charts.compare_bis_figure(frames), use_container_width=True)
```

- [ ] **Step 2: Wire it in `app.py`**

Update the import to `from apps.dashboard.views import cohort, case, quality, compare` and
replace the final `else:` branch with:

```python
    elif view == "Compare cases":
        compare.render_compare(ctx)
```

- [ ] **Step 3: Verify**

Run: `.venv/bin/streamlit run apps/dashboard/app.py --server.headless true --server.port 8599 & sleep 8; curl -s -o /dev/null -w "%{http_code}" http://localhost:8599; kill %1`
Expected: `200`, no exception.

- [ ] **Step 4: Commit**

```bash
git add apps/dashboard/views/compare.py apps/dashboard/app.py
git commit -m "feat(dashboard): multi-case comparison view"
```

---

## Task 11: Regenerate a consistent demo dataset + full manual smoke

**Files:** none (data + manual verification)

The current `data/processed/` is inconsistent (manifest lists 5 cases, 325 stray case
files). Regenerate a clean, consistent dataset now that Task 1 clears stale files.

- [ ] **Step 1: Rebuild a demo-sized consistent dataset**

Run: `.venv/bin/python pipelines/build_dataset.py --config configs/data.yaml --limit 40`
Expected: prints `Selected 40 cases`, `Kept N/40 ...`, and a split dict. `data/processed/cases/`
now contains exactly the kept cases (stale files cleared), and `manifest.parquet` matches.

- [ ] **Step 2: Confirm manifest ↔ cases consistency**

Run: `.venv/bin/python -c "import pandas as pd, os; m=pd.read_parquet('data/processed/manifest.parquet'); kept=m[m['kept']]; files=set(os.listdir('data/processed/cases')); want={f'case_{c}.parquet' for c in kept['caseid']}; print('match:', want==files, 'kept:', len(kept), 'files:', len(files))"`
Expected: `match: True`.

- [ ] **Step 3: Run the full unit suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (M1 data tests + dashboard: data_access 5, stats 6, charts 3).

- [ ] **Step 4: Manual smoke of every view**

Run: `.venv/bin/streamlit run apps/dashboard/app.py`
Then in the browser click through all four views: Cohort overview (metrics + charts +
table), Case viewer (pick a case, see the multi-axis plot), Signal quality (missingness
bars, BIS artifact metric, distribution selector), Compare cases (pick 2+, overlaid BIS).
Confirm no exceptions appear in the terminal.

- [ ] **Step 5: Commit the DVC pointer update for the regenerated dataset**

```bash
.venv/bin/dvc add data/processed
git add data/processed.dvc
git commit -m "chore(data): regenerate consistent 40-case demo dataset"
```

---

## Definition of Done

- `pytest` green: M1 data stages + dashboard `data_access` (5), `stats` (6), `charts` (3).
- `streamlit run apps/dashboard/app.py` serves all four views without error against a
  consistent `data/processed/`.
- Dashboard is manifest-driven; stray case files never appear.
- Pipeline no longer leaves orphan case files (Task 1).

## Self-Review Notes

- **Spec coverage:** §2 logic/UI separation (data_access/stats/charts vs apps/) ✓;
  §3 all four views ✓ (Tasks 7-10); §4 Streamlit+Plotly, local-only, missing-data guard
  (Task 6) ✓; §5 file structure + tests ✓; §6 data contracts honored (manifest has no
  subjectid; clinical join columns) ✓; §7 run instructions ✓; §8 YAGNI (no auth, no model
  views) ✓.
- **Type consistency:** `build_cohort_table` returns `caseid, n_samples, kept, split,
  age, sex, asa, optype`; every view reads only those plus per-case signal columns.
  `ctx` dict keys (`cohort`, `summary`) are identical across `app.py` and all views.
  `_load_case_frames()` returns `{caseid: DataFrame}`, matching `compare_bis_figure` and
  `render_signal_quality` inputs.
- **Added scope (justified):** Task 1 fixes a pre-existing M1 pipeline bug (manifest↔cases
  drift) because the dashboard's correctness depends on that data source being consistent.
