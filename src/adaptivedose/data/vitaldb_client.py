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
