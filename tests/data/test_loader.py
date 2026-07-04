from adaptivedose.data.loader import load_case_frame


def test_load_case_frame_builds_named_columns(fake_load_case):
    tracks = {"bis": "BIS/BIS", "propofol_rate": "Orchestra/PPF20_RATE",
              "map": "Solar8000/ART_MBP"}
    df = load_case_frame(1, tracks, interval_sec=10, load_fn=fake_load_case)
    assert list(df.columns) == ["caseid", "time_sec", "bis", "propofol_rate", "map"]
    assert (df["caseid"] == 1).all()
    assert df["time_sec"].iloc[1] - df["time_sec"].iloc[0] == 10
    assert len(df) == 30
