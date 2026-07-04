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
        "bis": [0.0, 45.0, 50.0, 55.0, 65.0],
        "map": [70.0, np.nan, 60.0, 62.0, 80.0],
    })

def test_cohort_summary_counts_and_splits():
    s = stats.cohort_summary(_cohort())
    assert s["n_cases"] == 3
    assert s["split_sizes"] == {"train": 2, "val": 1}

def test_duration_minutes():
    d = stats.duration_minutes(_cohort(), interval_sec=10)
    assert d.iloc[0] == 10.0
    assert d.iloc[2] == 30.0

def test_time_in_target_bis():
    assert abs(stats.time_in_target(_case()) - 0.6) < 1e-9

def test_artifact_rate_counts_zeros():
    assert abs(stats.artifact_rate(_case()) - 0.2) < 1e-9

def test_missingness_reports_fraction_per_column():
    m = stats.missingness([_case()], ["bis", "map"])
    assert m["bis"] == 0.0
    assert abs(m["map"] - 0.2) < 1e-9

def test_signal_distribution_concatenates_values():
    vals = stats.signal_distribution([_case(), _case()], "bis")
    assert len(vals) == 10
