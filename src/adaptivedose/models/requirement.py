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
    excluded. With the default band [40, 60] the `>= low` bound already excludes such
    artifacts, so the floor is a safeguard that only bites when `low` is set below it.
    Returns None if fewer than `min_in_band_rows` usable rows remain.
    """
    d = df.dropna(subset=["bis", "propofol_rate"])
    in_band = d[(d["bis"] >= low) & (d["bis"] <= high) & (d["bis"] > artifact_floor)]
    if len(in_band) < min_in_band_rows:
        return None
    return float(in_band["propofol_rate"].mean())
