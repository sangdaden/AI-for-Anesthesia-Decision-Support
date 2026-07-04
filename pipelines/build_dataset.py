"""Materialize the M1 dataset from live VitalDB.

Usage:
    .venv/bin/python pipelines/build_dataset.py --config configs/data.yaml [--limit N]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from adaptivedose.config import load_data_config
from adaptivedose.data.vitaldb_client import load_clinical_info, load_track_index
from adaptivedose.data.cohort import select_cohort
from adaptivedose.data.builder import build_dataset
from adaptivedose.data.split import make_splits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of cases (for a smoke run)")
    args = ap.parse_args()

    cfg = load_data_config(args.config)
    clinical = load_clinical_info(cfg.cache_dir)
    trks = load_track_index(cfg.cache_dir)

    caseids = select_cohort(
        trks, clinical,
        required_tracks=cfg.cohort.required_tracks,
        min_asa=cfg.cohort.min_asa, max_asa=cfg.cohort.max_asa,
        ane_type=cfg.cohort.ane_type,
    )
    if args.limit:
        caseids = caseids[: args.limit]
    print(f"Selected {len(caseids)} cases")

    tracks = {
        "bis": cfg.tracks.bis,
        "propofol_rate": cfg.tracks.propofol_rate,
        "remifentanil_rate": cfg.tracks.remifentanil_rate,
        "map": cfg.tracks.map,
        "hr": cfg.tracks.hr,
        "spo2": cfg.tracks.spo2,
        "etco2": cfg.tracks.etco2,
    }
    manifest = build_dataset(
        caseids, tracks, cfg.resample.interval_sec, cfg.output_dir,
    )
    kept = int(manifest["kept"].sum())
    print(f"Kept {kept}/{len(manifest)} cases after cleaning/trimming")

    subj = clinical[["caseid", "subjectid"]] if "subjectid" in clinical.columns else None
    if subj is not None:
        manifest = manifest.merge(subj, on="caseid", how="left")
    splits = make_splits(manifest, cfg.split.test_frac, cfg.split.val_frac, cfg.split.seed)
    Path(cfg.output_dir, "splits.json").write_text(json.dumps(splits, indent=2))
    print({k: len(v) for k, v in splits.items()})

if __name__ == "__main__":
    main()
