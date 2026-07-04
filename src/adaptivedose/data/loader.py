from __future__ import annotations
from typing import Callable, Dict, Optional
import numpy as np
import pandas as pd


def _default_load_fn(caseid, track_names, interval):
    import vitaldb
    return vitaldb.load_case(caseid, track_names, interval=interval)


def load_case_frame(
    caseid: int,
    tracks: Dict[str, str],
    interval_sec: int,
    load_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """Load one case into a tidy frame: caseid, time_sec, <feature columns>.

    `tracks` maps friendly names -> VitalDB track strings. `load_fn` is injected
    for testing; defaults to vitaldb.load_case.
    """
    load_fn = load_fn or _default_load_fn
    names = list(tracks.keys())
    track_strings = [tracks[n] for n in names]
    arr = np.asarray(load_fn(caseid, track_strings, interval_sec), dtype=float)
    n = arr.shape[0]
    df = pd.DataFrame(arr, columns=names)
    df.insert(0, "time_sec", np.arange(n) * interval_sec)
    df.insert(0, "caseid", caseid)
    return df
