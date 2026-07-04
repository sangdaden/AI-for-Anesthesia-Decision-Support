from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, Optional
import pandas as pd
from adaptivedose.data.loader import load_case_frame
from adaptivedose.data.clean import clean_case_frame
from adaptivedose.data.resample import trim_to_valid_window
from adaptivedose.utils.io import write_parquet, ensure_dir


def build_case(
    caseid: int,
    tracks: Dict[str, str],
    interval_sec: int,
    load_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """Full per-case pipeline: load -> trim -> clean."""
    df = load_case_frame(caseid, tracks, interval_sec, load_fn=load_fn)
    anchor = "bis" if "bis" in df.columns else df.columns[2]
    df = trim_to_valid_window(df, signal_col=anchor)
    df = clean_case_frame(df)
    return df


def build_manifest(frames: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for caseid, df in frames.items():
        rows.append({
            "caseid": caseid,
            "n_samples": len(df),
            "duration_sec": int(df["time_sec"].iloc[-1]) if len(df) else 0,
        })
    return pd.DataFrame(rows)


def build_dataset(
    caseids,
    tracks: Dict[str, str],
    interval_sec: int,
    output_dir: str | Path,
    load_fn: Optional[Callable] = None,
    min_samples: int = 30,
) -> pd.DataFrame:
    """Build all cases, write per-case Parquet, return the manifest.

    Cases shorter than `min_samples` after trimming are skipped and logged in
    the manifest via a 'kept' flag.
    """
    out_dir = ensure_dir(Path(output_dir) / "cases")
    frames, rows = {}, []
    for caseid in caseids:
        df = build_case(caseid, tracks, interval_sec, load_fn=load_fn)
        kept = len(df) >= min_samples
        if kept:
            write_parquet(df, out_dir / f"case_{caseid}.parquet")
            frames[caseid] = df
        rows.append({"caseid": caseid, "n_samples": len(df), "kept": kept})
    manifest = pd.DataFrame(rows)
    write_parquet(manifest, Path(output_dir) / "manifest.parquet")
    return manifest
