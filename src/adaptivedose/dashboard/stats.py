from __future__ import annotations
from typing import Callable, Dict, List
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

def pooled_fraction(
    case_frames: List[pd.DataFrame],
    per_case_fn: Callable[[pd.DataFrame], float],
) -> float:
    """Length-weighted mean of a per-case fraction across the cohort.

    Weighting by row count pools samples correctly rather than averaging
    per-case rates (which would over-weight short cases).
    """
    total = sum(len(f) for f in case_frames)
    if total == 0:
        return 0.0
    return sum(per_case_fn(f) * len(f) for f in case_frames) / total
