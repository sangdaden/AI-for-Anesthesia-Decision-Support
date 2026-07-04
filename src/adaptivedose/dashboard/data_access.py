from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd

DEFAULT_PROCESSED = Path("data/processed")
DEFAULT_CACHE = Path(".vitaldb_cache")

class DataNotFoundError(RuntimeError):
    """Raised when an expected M1 pipeline output is missing."""

def _require(path: Path) -> Path:
    if not path.exists():
        raise DataNotFoundError(
            f"{path} not found. Run: .venv/bin/python pipelines/build_dataset.py "
            f"--config configs/data.yaml"
        )
    return path

def load_manifest(processed_dir: str | Path = DEFAULT_PROCESSED) -> pd.DataFrame:
    return pd.read_parquet(_require(Path(processed_dir) / "manifest.parquet"))

def load_splits(processed_dir: str | Path = DEFAULT_PROCESSED) -> Dict[str, List[int]]:
    return json.loads(_require(Path(processed_dir) / "splits.json").read_text())

def load_case(caseid: int, processed_dir: str | Path = DEFAULT_PROCESSED) -> pd.DataFrame:
    return pd.read_parquet(
        _require(Path(processed_dir) / "cases" / f"case_{caseid}.parquet")
    )

def load_clinical(cache_dir: str | Path = DEFAULT_CACHE) -> pd.DataFrame:
    return pd.read_parquet(_require(Path(cache_dir) / "cases.parquet"))

def split_label_map(splits: Dict[str, List[int]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for name, ids in splits.items():
        for cid in ids:
            out[int(cid)] = name
    return out

def build_cohort_table(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    cache_dir: str | Path = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Kept cases joined with clinical info and their split label.

    Manifest is the source of truth; stray case files on disk are ignored.
    """
    manifest = load_manifest(processed_dir)
    kept = manifest[manifest["kept"]].copy()
    labels = split_label_map(load_splits(processed_dir))
    kept["split"] = kept["caseid"].map(labels).fillna("unassigned")
    clinical = load_clinical(cache_dir)[["caseid", "age", "sex", "asa", "optype"]]
    return kept.merge(clinical, on="caseid", how="left")
