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
            "asa": [2, 5, 3],
            "ane_type": ["General", "General", "Spinal"],
            "optype": ["Colorectal", "Biliary", "Thyroid"],
        }
    )
