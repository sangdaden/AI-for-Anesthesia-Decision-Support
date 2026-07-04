import pandas as pd
from adaptivedose.data.builder import build_case, build_manifest


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
