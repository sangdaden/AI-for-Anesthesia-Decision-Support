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
