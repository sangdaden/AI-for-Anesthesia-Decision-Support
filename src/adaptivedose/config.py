from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
from omegaconf import OmegaConf

@dataclass
class CohortConfig:
    required_tracks: List[str]
    min_asa: int
    max_asa: int
    ane_type: str

@dataclass
class TracksConfig:
    bis: str
    propofol_rate: str
    remifentanil_rate: str
    map: str
    hr: str
    spo2: str
    etco2: str

@dataclass
class ResampleConfig:
    interval_sec: int

@dataclass
class SplitConfig:
    test_frac: float
    val_frac: float
    seed: int

@dataclass
class DataConfig:
    cohort: CohortConfig
    tracks: TracksConfig
    resample: ResampleConfig
    split: SplitConfig
    cache_dir: str
    output_dir: str

def load_data_config(path: str | Path) -> DataConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = OmegaConf.load(path)
    schema = OmegaConf.structured(DataConfig)
    merged = OmegaConf.merge(schema, raw)
    return OmegaConf.to_object(merged)  # type: ignore[return-value]
