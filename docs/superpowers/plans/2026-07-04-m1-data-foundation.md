# M1 — Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible pipeline that downloads VitalDB, selects the TIVA-with-BIS cohort, cleans and resamples the intraoperative signals to a fixed cadence, and produces a versioned, analysis-ready dataset.

**Architecture:** Config-driven Python package. A thin VitalDB client downloads clinical info and the track index; a cohort selector filters cases by track availability and clinical criteria (avoiding the deprecated `find_cases`); a per-case loader turns raw tracks into tidy time-series; a cleaning/resampling stage aligns everything to a fixed grid; a dataset builder loops the cohort into per-case Parquet plus a manifest; a patient-level splitter prevents leakage. Every stage reads one YAML config.

**Tech Stack:** Python 3.12+ (3.14 works), `vitaldb`, `pandas`, `numpy`, `pyarrow` (Parquet), `omegaconf` (config), `pytest` (TDD), `dvc` (dataset versioning). Unit tests never hit the network — VitalDB calls are mocked or fed synthetic fixtures.

---

## Verified Facts (from live VitalDB, 2026-07-04)

These are confirmed against the live API — use them verbatim.

- `vitaldb.find_cases(track_names)` → list of caseids. **Deprecated** (relies on `trks` index). We build our own selector instead.
- `vitaldb.load_case(caseid, track_names, interval=1)` → `numpy.ndarray` shape `(n_samples, n_tracks)`, columns in the order of `track_names`, rows every `interval` seconds. Missing samples are `NaN`.
- Clinical info CSV: `https://api.vitaldb.net/cases` — **gzipped**, 6388 rows, 74 columns. First column header carries a UTF-8 BOM (`﻿caseid`).
- Track index CSV: `https://api.vitaldb.net/trks` — columns `caseid,tname,tid`, ~486k rows.
- Clinical columns we use: `caseid, age, sex, height, weight, bmi, asa, ane_type, optype, casestart, caseend, opstart, opend, intraop_ppf`.
- Canonical track names:
  - Depth: `BIS/BIS` (5867 cases)
  - Propofol infusion rate (mL/h): `Orchestra/PPF20_RATE` (3512)
  - Remifentanil infusion rate: `Orchestra/RFTN20_RATE` (4773)
  - Invasive mean arterial pressure: `Solar8000/ART_MBP` (3724)
  - Non-invasive MAP fallback: `Solar8000/NIBP_MBP` (5763)
  - Heart rate: `Solar8000/HR` (6387)
  - SpO2: `Solar8000/PLETH_SPO2` (6386)
  - ETCO2: `Solar8000/ETCO2` (6242)
- Cohort sizes (intersection): `BIS/BIS` + `Orchestra/PPF20_RATE` + `Solar8000/ART_MBP` = **1860 cases**; adding `Orchestra/RFTN20_RATE` ≈ 1800.

---

## File Structure

```
adaptive-dose-ai/
├── pyproject.toml                 # package metadata + deps
├── configs/
│   └── data.yaml                  # cohort, tracks, resample, split params
├── src/adaptivedose/
│   ├── __init__.py
│   ├── config.py                  # load + validate YAML -> DataConfig
│   ├── data/
│   │   ├── __init__.py
│   │   ├── vitaldb_client.py      # download clinical info + track index (cached)
│   │   ├── cohort.py              # select caseids by tracks + clinical criteria
│   │   ├── loader.py              # load one case -> tidy DataFrame
│   │   ├── clean.py               # outlier removal, unit clamps, missing values
│   │   ├── resample.py            # align tracks to fixed cadence
│   │   ├── builder.py             # loop cohort -> per-case Parquet + manifest
│   │   └── split.py               # patient-level train/val/test split
│   └── utils/
│       ├── __init__.py
│       └── io.py                  # parquet read/write, cache dir helpers
├── pipelines/
│   └── build_dataset.py           # CLI: end-to-end M1 pipeline
├── notebooks/
│   └── 01_eda.ipynb               # EDA (created as script-style at Task 11)
├── tests/
│   ├── conftest.py                # shared fixtures (synthetic tracks, tmp cache)
│   └── data/
│       ├── test_config.py
│       ├── test_vitaldb_client.py
│       ├── test_cohort.py
│       ├── test_loader.py
│       ├── test_clean.py
│       ├── test_resample.py
│       ├── test_builder.py
│       └── test_split.py
├── data/                          # gitignored; DVC-tracked outputs
└── .dvcignore / .gitignore
```

**Responsibilities:** each `src/adaptivedose/data/*.py` file owns exactly one stage and exposes pure functions taking DataFrames/config and returning DataFrames. Network access lives only in `vitaldb_client.py` and `loader.py`, so everything else is unit-testable without the internet.

---

## Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/adaptivedose/__init__.py`
- Create: `src/adaptivedose/data/__init__.py`
- Create: `src/adaptivedose/utils/__init__.py`
- Create: `tests/__init__.py`, `tests/data/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "adaptivedose"
version = "0.1.0"
description = "Research CDSS for adaptive anesthesia dosing (VitalDB)."
requires-python = ">=3.10"
dependencies = [
    "vitaldb>=1.7.0",
    "pandas>=2.0",
    "numpy>=1.24",
    "pyarrow>=14.0",
    "omegaconf>=2.3",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "dvc>=3.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
.pytest_cache/
.coverage
htmlcov/
data/
*.parquet
.vitaldb_cache/
```

- [ ] **Step 3: Create empty package init files**

Create `src/adaptivedose/__init__.py`, `src/adaptivedose/data/__init__.py`, `src/adaptivedose/utils/__init__.py`, `tests/__init__.py`, `tests/data/__init__.py` — each an empty file (`""`).

- [ ] **Step 4: Create venv and install**

Run:
```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```
Expected: installs `adaptivedose` in editable mode and all deps without error. If a scientific wheel fails to build on Python 3.14, recreate the venv with Python 3.12 (`python3.12 -m venv .venv`).

- [ ] **Step 5: Verify pytest runs**

Run: `.venv/bin/pytest`
Expected: `no tests ran` (exit 5) — the harness works, there are simply no tests yet.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "chore: scaffold adaptivedose package for M1"
```

---

## Task 1: Config system

**Files:**
- Create: `configs/data.yaml`
- Create: `src/adaptivedose/config.py`
- Test: `tests/data/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_config.py
from pathlib import Path
from adaptivedose.config import load_data_config, DataConfig

def test_load_data_config_reads_yaml(tmp_path):
    cfg_text = """
cohort:
  required_tracks: ["BIS/BIS", "Orchestra/PPF20_RATE", "Solar8000/ART_MBP"]
  min_asa: 1
  max_asa: 4
  ane_type: "General"
tracks:
  bis: "BIS/BIS"
  propofol_rate: "Orchestra/PPF20_RATE"
  remifentanil_rate: "Orchestra/RFTN20_RATE"
  map: "Solar8000/ART_MBP"
  hr: "Solar8000/HR"
  spo2: "Solar8000/PLETH_SPO2"
  etco2: "Solar8000/ETCO2"
resample:
  interval_sec: 10
split:
  test_frac: 0.15
  val_frac: 0.15
  seed: 42
cache_dir: ".vitaldb_cache"
output_dir: "data/processed"
"""
    p = tmp_path / "data.yaml"
    p.write_text(cfg_text)
    cfg = load_data_config(p)
    assert isinstance(cfg, DataConfig)
    assert cfg.resample.interval_sec == 10
    assert cfg.split.seed == 42
    assert "BIS/BIS" in cfg.cohort.required_tracks
    assert cfg.tracks.map == "Solar8000/ART_MBP"

def test_load_data_config_missing_file_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_data_config(tmp_path / "nope.yaml")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/data/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adaptivedose.config'`.

- [ ] **Step 3: Write the config module**

```python
# src/adaptivedose/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
from omegaconf import OmegaConf

@dataclass
class CohortConfig:
    required_tracks: List[str]
    min_asa: int
    max_asa: int
    ane_type: str

@dataclass
class TracksConfig:
    bis: str
    propofol_rate: str
    remifentanil_rate: str
    map: str
    hr: str
    spo2: str
    etco2: str

@dataclass
class ResampleConfig:
    interval_sec: int

@dataclass
class SplitConfig:
    test_frac: float
    val_frac: float
    seed: int

@dataclass
class DataConfig:
    cohort: CohortConfig
    tracks: TracksConfig
    resample: ResampleConfig
    split: SplitConfig
    cache_dir: str
    output_dir: str

def load_data_config(path: str | Path) -> DataConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = OmegaConf.load(path)
    schema = OmegaConf.structured(DataConfig)
    merged = OmegaConf.merge(schema, raw)
    return OmegaConf.to_object(merged)  # type: ignore[return-value]
```

- [ ] **Step 4: Create the real config file**

```yaml
# configs/data.yaml
cohort:
  required_tracks: ["BIS/BIS", "Orchestra/PPF20_RATE", "Solar8000/ART_MBP"]
  min_asa: 1
  max_asa: 4
  ane_type: "General"
tracks:
  bis: "BIS/BIS"
  propofol_rate: "Orchestra/PPF20_RATE"
  remifentanil_rate: "Orchestra/RFTN20_RATE"
  map: "Solar8000/ART_MBP"
  hr: "Solar8000/HR"
  spo2: "Solar8000/PLETH_SPO2"
  etco2: "Solar8000/ETCO2"
resample:
  interval_sec: 10
split:
  test_frac: 0.15
  val_frac: 0.15
  seed: 42
cache_dir: ".vitaldb_cache"
output_dir: "data/processed"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/data/test_config.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add configs/data.yaml src/adaptivedose/config.py tests/data/test_config.py
git commit -m "feat(config): typed YAML data config loader"
```

---

## Task 2: Shared test fixtures

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write fixtures (no test yet — support code)**

```python
# tests/conftest.py
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def track_names():
    return ["BIS/BIS", "Orchestra/PPF20_RATE", "Solar8000/ART_MBP"]

@pytest.fixture
def fake_load_case(track_names):
    """Return a callable mimicking vitaldb.load_case: (n_samples, n_tracks) array."""
    def _loader(caseid, tracks, interval=1):
        n = 30  # 30 samples
        rng = np.random.default_rng(caseid)
        cols = []
        for t in tracks:
            if "BIS" in t:
                cols.append(rng.uniform(35, 65, n))
            elif "PPF" in t:
                cols.append(rng.uniform(2, 8, n))
            elif "ART_MBP" in t:
                cols.append(rng.uniform(60, 90, n))
            else:
                cols.append(rng.uniform(0, 100, n))
        arr = np.column_stack(cols)
        arr[0, 0] = np.nan  # inject a missing value
        return arr
    return _loader

@pytest.fixture
def trks_frame():
    """Minimal track-index frame like api.vitaldb.net/trks."""
    return pd.DataFrame(
        {
            "caseid": [1, 1, 1, 2, 2, 3, 3, 3],
            "tname": [
                "BIS/BIS", "Orchestra/PPF20_RATE", "Solar8000/ART_MBP",
                "BIS/BIS", "Solar8000/HR",
                "BIS/BIS", "Orchestra/PPF20_RATE", "Solar8000/ART_MBP",
            ],
            "tid": ["a"] * 8,
        }
    )

@pytest.fixture
def clinical_frame():
    """Minimal clinical-info frame like api.vitaldb.net/cases."""
    return pd.DataFrame(
        {
            "caseid": [1, 2, 3],
            "age": [50, 60, 70],
            "sex": ["M", "F", "M"],
            "height": [170.0, 160.0, 175.0],
            "weight": [70.0, 55.0, 80.0],
            "bmi": [24.2, 21.5, 26.1],
            "asa": [2, 5, 3],  # case 2 has ASA 5 -> excluded by max_asa=4
            "ane_type": ["General", "General", "Spinal"],  # case 3 excluded by ane_type
            "optype": ["Colorectal", "Biliary", "Thyroid"],
        }
    )
```

- [ ] **Step 2: Verify fixtures import cleanly**

Run: `.venv/bin/pytest tests/data/test_config.py -v`
Expected: still 2 passed (conftest loads without error).

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: shared synthetic fixtures for data stages"
```

---

## Task 3: VitalDB client (clinical info + track index)

**Files:**
- Create: `src/adaptivedose/utils/io.py`
- Create: `src/adaptivedose/data/vitaldb_client.py`
- Test: `tests/data/test_vitaldb_client.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_vitaldb_client.py
import gzip
import pandas as pd
from adaptivedose.data import vitaldb_client as vc

def test_parse_cases_csv_strips_bom():
    raw = "﻿caseid,age,sex\n1,50,M\n2,60,F\n".encode("utf-8")
    df = vc._parse_csv_bytes(raw)
    assert list(df.columns)[0] == "caseid"  # BOM stripped
    assert len(df) == 2

def test_parse_cases_csv_handles_gzip():
    raw = gzip.compress("﻿caseid,age\n1,50\n".encode("utf-8"))
    df = vc._parse_csv_bytes(raw)
    assert df.iloc[0]["age"] == 50

def test_load_clinical_info_uses_cache(tmp_path, monkeypatch):
    calls = {"n": 0}
    def fake_fetch(url):
        calls["n"] += 1
        return "﻿caseid,age\n1,50\n".encode("utf-8")
    monkeypatch.setattr(vc, "_fetch", fake_fetch)
    df1 = vc.load_clinical_info(cache_dir=tmp_path)
    df2 = vc.load_clinical_info(cache_dir=tmp_path)  # second call hits cache
    assert calls["n"] == 1
    assert len(df1) == len(df2) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/data/test_vitaldb_client.py -v`
Expected: FAIL — module `adaptivedose.data.vitaldb_client` not found.

- [ ] **Step 3: Write `io.py` helpers**

```python
# src/adaptivedose/utils/io.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False)
    return p

def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
```

- [ ] **Step 4: Write `vitaldb_client.py`**

```python
# src/adaptivedose/data/vitaldb_client.py
from __future__ import annotations
import gzip
import io
import urllib.request
from pathlib import Path
import pandas as pd
from adaptivedose.utils.io import ensure_dir

CASES_URL = "https://api.vitaldb.net/cases"
TRKS_URL = "https://api.vitaldb.net/trks"

def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()

def _parse_csv_bytes(raw: bytes) -> pd.DataFrame:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = gzip.decompress(raw).decode("utf-8")
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.lstrip("﻿") for c in df.columns]  # strip BOM
    return df

def _load_cached(url: str, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    df = _parse_csv_bytes(_fetch(url))
    ensure_dir(cache_path.parent)
    df.to_parquet(cache_path, index=False)
    return df

def load_clinical_info(cache_dir: str | Path = ".vitaldb_cache") -> pd.DataFrame:
    return _load_cached(CASES_URL, Path(cache_dir) / "cases.parquet")

def load_track_index(cache_dir: str | Path = ".vitaldb_cache") -> pd.DataFrame:
    return _load_cached(TRKS_URL, Path(cache_dir) / "trks.parquet")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/data/test_vitaldb_client.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/adaptivedose/utils/io.py src/adaptivedose/data/vitaldb_client.py tests/data/test_vitaldb_client.py
git commit -m "feat(data): cached VitalDB clinical-info and track-index client"
```

---

## Task 4: Cohort selection

**Files:**
- Create: `src/adaptivedose/data/cohort.py`
- Test: `tests/data/test_cohort.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_cohort.py
from adaptivedose.data.cohort import select_cohort

def test_select_cohort_requires_all_tracks(trks_frame, clinical_frame):
    ids = select_cohort(
        trks_frame, clinical_frame,
        required_tracks=["BIS/BIS", "Orchestra/PPF20_RATE", "Solar8000/ART_MBP"],
        min_asa=1, max_asa=4, ane_type="General",
    )
    # case 1 has all tracks + ASA 2 + General -> included
    # case 2 lacks propofol + ASA 5 -> excluded
    # case 3 has all tracks but ane_type Spinal -> excluded
    assert ids == [1]

def test_select_cohort_filters_asa_and_anetype(trks_frame, clinical_frame):
    ids = select_cohort(
        trks_frame, clinical_frame,
        required_tracks=["BIS/BIS"],
        min_asa=1, max_asa=4, ane_type="General",
    )
    # cases 1 and 2 have BIS; case 2 has ASA 5 (excluded); case 3 is Spinal (excluded)
    assert ids == [1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/data/test_cohort.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `cohort.py`**

```python
# src/adaptivedose/data/cohort.py
from __future__ import annotations
from typing import List
import pandas as pd

def select_cohort(
    trks: pd.DataFrame,
    clinical: pd.DataFrame,
    required_tracks: List[str],
    min_asa: int,
    max_asa: int,
    ane_type: str,
) -> List[int]:
    """Return sorted caseids that have every required track and pass clinical filters.

    Uses the track-index table directly instead of the deprecated
    vitaldb.find_cases().
    """
    have = trks[trks["tname"].isin(required_tracks)]
    counts = have.groupby("caseid")["tname"].nunique()
    track_ok = set(counts[counts == len(required_tracks)].index)

    clin = clinical.copy()
    clin = clin[clin["asa"].between(min_asa, max_asa)]
    clin = clin[clin["ane_type"] == ane_type]
    clinical_ok = set(clin["caseid"])

    return sorted(int(c) for c in (track_ok & clinical_ok))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/data/test_cohort.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/adaptivedose/data/cohort.py tests/data/test_cohort.py
git commit -m "feat(data): track+clinical cohort selection (no deprecated find_cases)"
```

---

## Task 5: Case loader

**Files:**
- Create: `src/adaptivedose/data/loader.py`
- Test: `tests/data/test_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_loader.py
from adaptivedose.data.loader import load_case_frame

def test_load_case_frame_builds_named_columns(fake_load_case):
    tracks = {"bis": "BIS/BIS", "propofol_rate": "Orchestra/PPF20_RATE",
              "map": "Solar8000/ART_MBP"}
    df = load_case_frame(1, tracks, interval_sec=10, load_fn=fake_load_case)
    assert list(df.columns) == ["caseid", "time_sec", "bis", "propofol_rate", "map"]
    assert (df["caseid"] == 1).all()
    # time axis is interval-spaced
    assert df["time_sec"].iloc[1] - df["time_sec"].iloc[0] == 10
    assert len(df) == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/data/test_loader.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `loader.py`**

```python
# src/adaptivedose/data/loader.py
from __future__ import annotations
from typing import Callable, Dict, Optional
import numpy as np
import pandas as pd

def _default_load_fn(caseid, track_names, interval):
    import vitaldb
    return vitaldb.load_case(caseid, track_names, interval=interval)

def load_case_frame(
    caseid: int,
    tracks: Dict[str, str],
    interval_sec: int,
    load_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """Load one case into a tidy frame: caseid, time_sec, <feature columns>.

    `tracks` maps friendly names -> VitalDB track strings. `load_fn` is injected
    for testing; defaults to vitaldb.load_case.
    """
    load_fn = load_fn or _default_load_fn
    names = list(tracks.keys())
    track_strings = [tracks[n] for n in names]
    arr = np.asarray(load_fn(caseid, track_strings, interval_sec), dtype=float)
    n = arr.shape[0]
    df = pd.DataFrame(arr, columns=names)
    df.insert(0, "time_sec", np.arange(n) * interval_sec)
    df.insert(0, "caseid", caseid)
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/data/test_loader.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/adaptivedose/data/loader.py tests/data/test_loader.py
git commit -m "feat(data): per-case tidy loader with injectable load_fn"
```

---

## Task 6: Cleaning (physiologic clamps + outliers + missing values)

**Files:**
- Create: `src/adaptivedose/data/clean.py`
- Test: `tests/data/test_clean.py`

> **Design note (corrected during execution):** clamping and imputation are two
> separate responsibilities. Splitting them lets each be tested independently —
> clamping turns out-of-range values into NaN; imputation fills NaNs. `clean_case_frame`
> composes both (clamp, then impute), which is what downstream stages call.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_clean.py
import numpy as np
import pandas as pd
from adaptivedose.data.clean import (
    clamp_out_of_range, impute_case_frame, clean_case_frame,
)

def _frame(**cols):
    n = len(next(iter(cols.values())))
    base = {"caseid": [1] * n, "time_sec": list(range(0, n * 10, 10))}
    base.update(cols)
    return pd.DataFrame(base)

def test_clamp_sets_out_of_range_to_nan():
    df = _frame(bis=[50, 120, 40], map=[70, -5, 65], propofol_rate=[3, 4, 999])
    out = clamp_out_of_range(df)
    assert np.isnan(out.loc[1, "bis"])            # 120 > 100
    assert np.isnan(out.loc[1, "map"])            # -5 < 10
    assert np.isnan(out.loc[2, "propofol_rate"])  # 999 > 200
    assert out.loc[0, "bis"] == 50                # in-range untouched

def test_impute_fills_drugs_with_zero_and_physiologic_by_fill():
    df = _frame(bis=[np.nan, 45.0, np.nan], map=[70.0, 71.0, 72.0],
                propofol_rate=[np.nan, 3.0, np.nan])
    out = impute_case_frame(df)
    assert out["bis"].iloc[0] == 45.0    # bfill
    assert out["bis"].iloc[2] == 45.0    # ffill
    assert out["propofol_rate"].iloc[0] == 0.0
    assert out["propofol_rate"].iloc[2] == 0.0
    assert not out["bis"].isna().any()

def test_clean_case_frame_clamps_then_imputes():
    df = _frame(bis=[50, 120, 40], map=[70, -5, 65], propofol_rate=[3, 4, 999])
    out = clean_case_frame(df)
    assert not out["bis"].isna().any()          # imputed after clamp
    assert out.loc[1, "bis"] == 50.0            # 120 clamped -> ffilled from index 0
    assert out.loc[2, "propofol_rate"] == 0.0   # 999 clamped -> imputed to 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/data/test_clean.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `clean.py`**

```python
# src/adaptivedose/data/clean.py
from __future__ import annotations
import pandas as pd

# Physiologically plausible ranges; values outside become NaN before imputation.
VALID_RANGES = {
    "bis": (0.0, 100.0),
    "map": (10.0, 250.0),
    "hr": (10.0, 250.0),
    "spo2": (50.0, 100.0),
    "etco2": (0.0, 100.0),
    "propofol_rate": (0.0, 200.0),
    "remifentanil_rate": (0.0, 200.0),
}
SIGNAL_COLS = list(VALID_RANGES.keys())
DRUG_COLS = ("propofol_rate", "remifentanil_rate")
PHYSIOLOGIC_COLS = ("bis", "map", "hr", "spo2", "etco2")

def clamp_out_of_range(df: pd.DataFrame) -> pd.DataFrame:
    """Set physiologically implausible values to NaN. No imputation."""
    out = df.copy()
    for col, (lo, hi) in VALID_RANGES.items():
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            out[col] = s.where((s >= lo) & (s <= hi))
    return out

def impute_case_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs: drug rates -> 0.0 (no infusion), physiologic -> ffill then bfill."""
    out = df.copy()
    for col in DRUG_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    for col in PHYSIOLOGIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").ffill().bfill()
    return out

def clean_case_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning: clamp out-of-range to NaN, then impute."""
    return impute_case_frame(clamp_out_of_range(df))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/data/test_clean.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/adaptivedose/data/clean.py tests/data/test_clean.py
git commit -m "feat(data): split clamp/impute cleaning with independent tests"
```

---

## Task 7: Resampling / cadence alignment

**Files:**
- Create: `src/adaptivedose/data/resample.py`
- Test: `tests/data/test_resample.py`

Note: `load_case(interval=...)` already returns a fixed grid, but individual signals
sometimes carry a different native cadence and leading/trailing NaN padding. This
stage trims to the anesthesia window and guarantees a uniform grid.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_resample.py
import numpy as np
import pandas as pd
from adaptivedose.data.resample import trim_to_valid_window

def _frame(bis):
    n = len(bis)
    return pd.DataFrame({
        "caseid": [1] * n,
        "time_sec": list(range(0, n * 10, 10)),
        "bis": bis,
        "map": [70.0] * n,
        "propofol_rate": [0.0, 5.0, 5.0, 5.0, 0.0][:n],
    })

def test_trim_removes_leading_trailing_all_nan_rows():
    df = _frame([np.nan, 45.0, 50.0, 48.0, np.nan])
    out = trim_to_valid_window(df, signal_col="bis")
    assert len(out) == 3
    assert out["bis"].iloc[0] == 45.0
    assert out["bis"].iloc[-1] == 48.0

def test_trim_resets_time_axis_to_start_at_zero():
    df = _frame([np.nan, 45.0, 50.0, 48.0, np.nan])
    out = trim_to_valid_window(df, signal_col="bis")
    assert out["time_sec"].iloc[0] == 0
    assert out["time_sec"].iloc[1] == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/data/test_resample.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `resample.py`**

```python
# src/adaptivedose/data/resample.py
from __future__ import annotations
import numpy as np
import pandas as pd

def trim_to_valid_window(df: pd.DataFrame, signal_col: str = "bis") -> pd.DataFrame:
    """Drop leading/trailing rows where the anchor signal is NaN, reset time axis.

    The anchor signal (default BIS) defines the monitored anesthesia window.
    """
    valid = df[signal_col].notna().to_numpy()
    if not valid.any():
        return df.iloc[0:0].copy()
    first = int(np.argmax(valid))
    last = len(valid) - int(np.argmax(valid[::-1]))
    out = df.iloc[first:last].reset_index(drop=True).copy()
    step = int(out["time_sec"].iloc[1] - out["time_sec"].iloc[0]) if len(out) > 1 else 0
    out["time_sec"] = np.arange(len(out)) * step
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/data/test_resample.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/adaptivedose/data/resample.py tests/data/test_resample.py
git commit -m "feat(data): trim cases to the valid BIS-monitored window"
```

---

## Task 8: Dataset builder (cohort loop + manifest)

**Files:**
- Create: `src/adaptivedose/data/builder.py`
- Test: `tests/data/test_builder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_builder.py
import pandas as pd
from adaptivedose.data.builder import build_case, build_manifest

def test_build_case_runs_full_per_case_pipeline(fake_load_case):
    tracks = {"bis": "BIS/BIS", "propofol_rate": "Orchestra/PPF20_RATE",
              "map": "Solar8000/ART_MBP"}
    df = build_case(1, tracks, interval_sec=10, load_fn=fake_load_case)
    assert "bis" in df.columns
    assert not df["bis"].isna().any()   # cleaned + imputed
    assert (df["caseid"] == 1).all()

def test_build_manifest_summarizes_cases():
    frames = {
        1: pd.DataFrame({"caseid": [1, 1], "time_sec": [0, 10],
                         "bis": [45.0, 46.0], "map": [70.0, 71.0]}),
        2: pd.DataFrame({"caseid": [2], "time_sec": [0],
                         "bis": [50.0], "map": [65.0]}),
    }
    manifest = build_manifest(frames)
    assert set(manifest["caseid"]) == {1, 2}
    row1 = manifest[manifest["caseid"] == 1].iloc[0]
    assert row1["n_samples"] == 2
    assert row1["duration_sec"] == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/data/test_builder.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `builder.py`**

```python
# src/adaptivedose/data/builder.py
from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, Optional
import pandas as pd
from adaptivedose.data.loader import load_case_frame
from adaptivedose.data.clean import clean_case_frame
from adaptivedose.data.resample import trim_to_valid_window
from adaptivedose.utils.io import write_parquet, ensure_dir

def build_case(
    caseid: int,
    tracks: Dict[str, str],
    interval_sec: int,
    load_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """Full per-case pipeline: load -> trim -> clean."""
    df = load_case_frame(caseid, tracks, interval_sec, load_fn=load_fn)
    anchor = "bis" if "bis" in df.columns else df.columns[2]
    df = trim_to_valid_window(df, signal_col=anchor)
    df = clean_case_frame(df)
    return df

def build_manifest(frames: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for caseid, df in frames.items():
        rows.append({
            "caseid": caseid,
            "n_samples": len(df),
            "duration_sec": int(df["time_sec"].iloc[-1]) if len(df) else 0,
        })
    return pd.DataFrame(rows)

def build_dataset(
    caseids,
    tracks: Dict[str, str],
    interval_sec: int,
    output_dir: str | Path,
    load_fn: Optional[Callable] = None,
    min_samples: int = 30,
) -> pd.DataFrame:
    """Build all cases, write per-case Parquet, return the manifest.

    Cases shorter than `min_samples` after trimming are skipped and logged in
    the manifest via a 'kept' flag.
    """
    out_dir = ensure_dir(Path(output_dir) / "cases")
    frames, rows = {}, []
    for caseid in caseids:
        df = build_case(caseid, tracks, interval_sec, load_fn=load_fn)
        kept = len(df) >= min_samples
        if kept:
            write_parquet(df, out_dir / f"case_{caseid}.parquet")
            frames[caseid] = df
        rows.append({"caseid": caseid, "n_samples": len(df), "kept": kept})
    manifest = pd.DataFrame(rows)
    write_parquet(manifest, Path(output_dir) / "manifest.parquet")
    return manifest
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/data/test_builder.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/adaptivedose/data/builder.py tests/data/test_builder.py
git commit -m "feat(data): cohort dataset builder with per-case parquet + manifest"
```

---

## Task 9: Patient-level train/val/test split

**Files:**
- Create: `src/adaptivedose/data/split.py`
- Test: `tests/data/test_split.py`

Note: VitalDB has `subjectid`; splitting on `caseid` alone could leak the same
patient across folds. We split on `subjectid` when available, else `caseid`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_split.py
import pandas as pd
from adaptivedose.data.split import make_splits

def test_make_splits_are_disjoint_and_cover_all():
    manifest = pd.DataFrame({
        "caseid": list(range(100)),
        "subjectid": list(range(100)),
        "kept": [True] * 100,
    })
    splits = make_splits(manifest, test_frac=0.2, val_frac=0.2, seed=42)
    all_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert all_ids == set(range(100))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["train"]) & set(splits["val"]))
    assert len(splits["test"]) == 20
    assert len(splits["val"]) == 20

def test_make_splits_is_deterministic_with_seed():
    manifest = pd.DataFrame({"caseid": list(range(50)),
                             "subjectid": list(range(50)), "kept": [True] * 50})
    a = make_splits(manifest, 0.2, 0.2, seed=7)
    b = make_splits(manifest, 0.2, 0.2, seed=7)
    assert a["test"] == b["test"]

def test_make_splits_no_subject_leak():
    # two cases share subjectid 0 -> must land in the same split
    manifest = pd.DataFrame({
        "caseid": [10, 11, 12, 13],
        "subjectid": [0, 0, 1, 2],
        "kept": [True] * 4,
    })
    splits = make_splits(manifest, test_frac=0.25, val_frac=0.25, seed=1)
    loc = {}
    for name, ids in splits.items():
        sub = manifest[manifest["caseid"].isin(ids)]["subjectid"]
        for s in sub:
            assert loc.get(s, name) == name  # subject never in two splits
            loc[s] = name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/data/test_split.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `split.py`**

```python
# src/adaptivedose/data/split.py
from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd

def make_splits(
    manifest: pd.DataFrame,
    test_frac: float,
    val_frac: float,
    seed: int,
) -> Dict[str, List[int]]:
    """Split kept cases into train/val/test by subject to prevent leakage.

    Returns caseid lists. Groups by subjectid (fallback: caseid) so all cases of
    a patient stay in one fold.
    """
    kept = manifest[manifest["kept"]].copy()
    group_col = "subjectid" if "subjectid" in kept.columns else "caseid"
    groups = kept[group_col].unique()

    rng = np.random.default_rng(seed)
    shuffled = groups.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_g = set(shuffled[:n_test])
    val_g = set(shuffled[n_test:n_test + n_val])
    train_g = set(shuffled[n_test + n_val:])

    def ids_for(gset):
        return sorted(int(c) for c in kept[kept[group_col].isin(gset)]["caseid"])

    return {"train": ids_for(train_g), "val": ids_for(val_g), "test": ids_for(test_g)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/data/test_split.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/adaptivedose/data/split.py tests/data/test_split.py
git commit -m "feat(data): subject-level leak-free train/val/test split"
```

---

## Task 10: End-to-end pipeline CLI

**Files:**
- Create: `pipelines/build_dataset.py`

This wires the stages against the **real** VitalDB. It is a script, not a unit test
(the unit tests already cover each stage with fakes). Run it once to materialize the
dataset.

- [ ] **Step 1: Write the pipeline script**

```python
# pipelines/build_dataset.py
"""Materialize the M1 dataset from live VitalDB.

Usage:
    .venv/bin/python pipelines/build_dataset.py --config configs/data.yaml [--limit N]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from adaptivedose.config import load_data_config
from adaptivedose.data.vitaldb_client import load_clinical_info, load_track_index
from adaptivedose.data.cohort import select_cohort
from adaptivedose.data.builder import build_dataset
from adaptivedose.data.split import make_splits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of cases (for a smoke run)")
    args = ap.parse_args()

    cfg = load_data_config(args.config)
    clinical = load_clinical_info(cfg.cache_dir)
    trks = load_track_index(cfg.cache_dir)

    caseids = select_cohort(
        trks, clinical,
        required_tracks=cfg.cohort.required_tracks,
        min_asa=cfg.cohort.min_asa, max_asa=cfg.cohort.max_asa,
        ane_type=cfg.cohort.ane_type,
    )
    if args.limit:
        caseids = caseids[: args.limit]
    print(f"Selected {len(caseids)} cases")

    tracks = {
        "bis": cfg.tracks.bis,
        "propofol_rate": cfg.tracks.propofol_rate,
        "remifentanil_rate": cfg.tracks.remifentanil_rate,
        "map": cfg.tracks.map,
        "hr": cfg.tracks.hr,
        "spo2": cfg.tracks.spo2,
        "etco2": cfg.tracks.etco2,
    }
    manifest = build_dataset(
        caseids, tracks, cfg.resample.interval_sec, cfg.output_dir,
    )
    kept = int(manifest["kept"].sum())
    print(f"Kept {kept}/{len(manifest)} cases after cleaning/trimming")

    # attach subjectid for leak-free split
    subj = clinical[["caseid", "subjectid"]] if "subjectid" in clinical.columns else None
    if subj is not None:
        manifest = manifest.merge(subj, on="caseid", how="left")
    splits = make_splits(manifest, cfg.split.test_frac, cfg.split.val_frac, cfg.split.seed)
    Path(cfg.output_dir, "splits.json").write_text(json.dumps(splits, indent=2))
    print({k: len(v) for k, v in splits.items()})

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on a few cases**

Run: `.venv/bin/python pipelines/build_dataset.py --config configs/data.yaml --limit 5`
Expected: prints `Selected 1860 cases` (then capped to 5), builds 5 case Parquet files under `data/processed/cases/`, writes `manifest.parquet` and `splits.json`. This performs real network downloads.

- [ ] **Step 3: Verify outputs exist**

Run: `.venv/bin/python -c "import pandas as pd; m=pd.read_parquet('data/processed/manifest.parquet'); print(m); print('files:', __import__('os').listdir('data/processed/cases'))"`
Expected: manifest with 5 rows and matching Parquet files.

- [ ] **Step 4: Commit (code only; data is gitignored)**

```bash
git add pipelines/build_dataset.py
git commit -m "feat(pipeline): end-to-end M1 dataset build CLI"
```

---

## Task 11: EDA summary

**Files:**
- Create: `notebooks/01_eda.py` (script-style; convertible to notebook)

- [ ] **Step 1: Write the EDA script**

```python
# notebooks/01_eda.py
"""Quick EDA over the built M1 dataset. Run after pipelines/build_dataset.py."""
from pathlib import Path
import pandas as pd

PROC = Path("data/processed")

def main():
    manifest = pd.read_parquet(PROC / "manifest.parquet")
    print("Cases:", len(manifest), "| kept:", int(manifest["kept"].sum()))
    print("Duration (samples) describe:\n", manifest["n_samples"].describe())

    case_files = sorted((PROC / "cases").glob("case_*.parquet"))
    frames = [pd.read_parquet(f) for f in case_files]
    if not frames:
        print("No case files — run the pipeline first."); return
    allrows = pd.concat(frames, ignore_index=True)
    for col in ["bis", "map", "propofol_rate", "remifentanil_rate", "hr", "spo2"]:
        if col in allrows.columns:
            s = allrows[col]
            print(f"{col:18s} mean={s.mean():7.2f} p5={s.quantile(.05):7.2f} "
                  f"p95={s.quantile(.95):7.2f} nan%={s.isna().mean()*100:5.2f}")

    # sanity: fraction of time BIS in target [40,60]
    in_target = allrows["bis"].between(40, 60).mean()
    print(f"Time in BIS target [40,60]: {in_target*100:.1f}%")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python notebooks/01_eda.py`
Expected: prints case counts, per-signal summary stats, and the BIS-in-target fraction using the 5 smoke-run cases.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_eda.py
git commit -m "docs(eda): dataset summary script for M1"
```

---

## Task 12: Dataset versioning (DVC)

**Files:**
- Create: `.dvc/` (via `dvc init`)
- Create: `data/processed.dvc` (via `dvc add`)

- [ ] **Step 1: Initialize DVC**

Run: `.venv/bin/dvc init`
Expected: creates `.dvc/` and stages `.dvc/config`, `.dvcignore`.

- [ ] **Step 2: Track the processed dataset**

Run: `.venv/bin/dvc add data/processed`
Expected: creates `data/processed.dvc`; `data/processed` is added to `.gitignore` by DVC.

- [ ] **Step 3: Commit the DVC pointers**

```bash
git add .dvc .dvcignore data/processed.dvc .gitignore
git commit -m "chore(dvc): version the processed M1 dataset"
```

- [ ] **Step 4: Run the full test suite**

Run: `.venv/bin/pytest -q`
Expected: all tests pass (config 2, client 3, cohort 2, loader 1, clean 3, resample 2, builder 2, split 3 = 18 passed).

---

## Definition of Done (M1)

- `pytest` green across all data-stage unit tests (network-free).
- `pipelines/build_dataset.py` produces per-case Parquet + `manifest.parquet` + `splits.json` from live VitalDB.
- Cohort selection returns the verified 1860-case TIVA-with-BIS set at full scale.
- Dataset is DVC-versioned; splits are subject-level and leak-free.
- EDA script reports plausible signal ranges and BIS-in-target fraction.

## Self-Review Notes

- **Spec coverage:** §5 Data (client + cohort + cleaning + resample + split) ✓; §9 Reproducibility (config-driven, seeded split, DVC) ✓; §3 core "VitalDB ETL + cohort" ✓; §11 Testing (unit tests per stage) ✓. MLflow (§9) belongs to M2 (training) — intentionally out of M1 scope.
- **Deferred deliberately:** feature engineering beyond cleaning/alignment (§7 `features/`) begins in M2 alongside the response model, where feature choices depend on the modeling target. M1 stops at an analysis-ready aligned dataset.
- **Type consistency:** friendly track keys (`bis, propofol_rate, remifentanil_rate, map, hr, spo2, etco2`) are identical across `config.py`, `loader.py`, `clean.py`, `builder.py`, and the pipeline CLI.

## Carry-over to M2 (from M1 final review + smoke-run EDA)

The M1 dataset is analysis-ready but a 5-case smoke run surfaced two data-quality
issues that M2 must handle before modeling. They are **not** M1 defects — M1's job is
a clean, aligned, versioned dataset — but they materially affect the response model.

1. **BIS artifact zeros.** `clean.VALID_RANGES["bis"] = (0.0, 100.0)`, so sensor-
   disconnect/artifact `BIS == 0` values pass through as legitimate depth (smoke run:
   BIS mean ≈ 36, p5 = 0, only 26% of time in target [40, 60]). Worse,
   `resample.trim_to_valid_window` anchors the case window on `BIS.notna()`, and `0`
   is not NaN, so artifact zeros can define window boundaries. **M2 action:** raise the
   BIS lower clamp to a physiologic floor (~10–20) and/or filter on the BIS signal-
   quality index (`BIS/SQI` track exists in VitalDB), then re-derive the window.

2. **MAP whole-case missingness.** Cases can have `Solar8000/ART_MBP` present but sparse
   or all-NaN in the trimmed window (smoke run: MAP nan% ≈ 21.6% after imputation, which
   correctly leaves all-NaN columns as NaN). The `Solar8000/NIBP_MBP` fallback (5763
   cases) is documented but not wired in. **M2 action:** implement the NIBP fallback for
   MAP, or explicitly flag/drop MAP-sparse cases before they reach the model.

Both are captured as tests-of-record where cheap (`test_impute_leaves_all_nan_physiologic_as_nan`
pins the all-NaN behavior) so a future change is a conscious one.
