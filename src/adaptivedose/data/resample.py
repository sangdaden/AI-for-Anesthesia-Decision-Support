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
