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
