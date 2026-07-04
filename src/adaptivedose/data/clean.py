from __future__ import annotations
import pandas as pd

VALID_RANGES = {
    "bis": (0.0, 100.0),
    "map": (10.0, 250.0),
    "hr": (10.0, 250.0),
    "spo2": (50.0, 100.0),
    "etco2": (0.0, 100.0),
    "propofol_rate": (0.0, 200.0),
    "remifentanil_rate": (0.0, 200.0),
}
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
