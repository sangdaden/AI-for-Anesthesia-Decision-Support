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

def test_missingness_absent_column_reports_full():
    m = stats.missingness([_case()], ["bis", "etco2"])  # etco2 not in _case()
    assert m["etco2"] == 1.0

def test_signal_distribution_absent_column_is_empty():
    vals = stats.signal_distribution([_case()], "etco2")
    assert len(vals) == 0

def test_pooled_fraction_weights_by_row_count():
    f_long = pd.DataFrame({"bis": [0.0, 0.0]})   # artifact_rate 1.0, 2 rows
    f_short = pd.DataFrame({"bis": [45.0]})       # artifact_rate 0.0, 1 row
    # pooled = (1.0*2 + 0.0*1) / 3
    assert abs(stats.pooled_fraction([f_long, f_short], stats.artifact_rate) - 2 / 3) < 1e-9

def test_pooled_fraction_empty_returns_zero():
    assert stats.pooled_fraction([], stats.artifact_rate) == 0.0
