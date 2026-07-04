import numpy as np
import pandas as pd
from adaptivedose.data.builder import build_case, build_manifest, build_dataset


def test_build_case_runs_full_per_case_pipeline(fake_load_case):
    tracks = {"bis": "BIS/BIS", "propofol_rate": "Orchestra/PPF20_RATE",
              "map": "Solar8000/ART_MBP"}
    df = build_case(1, tracks, interval_sec=10, load_fn=fake_load_case)
    assert "bis" in df.columns
    assert not df["bis"].isna().any()   # cleaned + imputed
    assert (df["caseid"] == 1).all()


def test_build_manifest_summarizes_cases():
    frames = {
        1: pd.DataFrame({"caseid": [1, 1], "time_sec": [0, 10],
                         "bis": [45.0, 46.0], "map": [70.0, 71.0]}),
        2: pd.DataFrame({"caseid": [2], "time_sec": [0],
                         "bis": [50.0], "map": [65.0]}),
    }
    manifest = build_manifest(frames)
    assert set(manifest["caseid"]) == {1, 2}
    row1 = manifest[manifest["caseid"] == 1].iloc[0]
    assert row1["n_samples"] == 2
    assert row1["duration_sec"] == 10


def test_build_dataset_writes_kept_cases_and_manifest(fake_load_case, tmp_path):
    tracks = {"bis": "BIS/BIS", "propofol_rate": "Orchestra/PPF20_RATE",
              "map": "Solar8000/ART_MBP"}
    manifest = build_dataset(
        [1, 2], tracks, interval_sec=10, output_dir=tmp_path,
        load_fn=fake_load_case, min_samples=5,
    )
    assert manifest["kept"].all()
    assert (tmp_path / "cases" / "case_1.parquet").exists()
    assert (tmp_path / "cases" / "case_2.parquet").exists()
    assert (tmp_path / "manifest.parquet").exists()


def test_build_dataset_skips_cases_below_min_samples(tmp_path):
    tracks = {"bis": "BIS/BIS"}

    def short_loader(caseid, track_strings, interval):
        return np.array([[45.0], [46.0], [47.0]])  # 3 samples < min

    manifest = build_dataset(
        [7], tracks, interval_sec=10, output_dir=tmp_path,
        load_fn=short_loader, min_samples=30,
    )
    assert not bool(manifest.iloc[0]["kept"])
    assert not (tmp_path / "cases" / "case_7.parquet").exists()
    assert (tmp_path / "manifest.parquet").exists()


def test_build_dataset_clears_stale_case_files(fake_load_case, tmp_path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True)
    stale = cases_dir / "case_999.parquet"
    stale.write_text("stale")  # orphan from a previous, larger run
    tracks = {"bis": "BIS/BIS", "propofol_rate": "Orchestra/PPF20_RATE",
              "map": "Solar8000/ART_MBP"}
    build_dataset([1], tracks, interval_sec=10, output_dir=tmp_path,
                  load_fn=fake_load_case, min_samples=5)
    assert not stale.exists()                       # orphan removed
    assert (cases_dir / "case_1.parquet").exists()  # fresh case written
