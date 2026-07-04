import json
import pandas as pd
import pytest
from adaptivedose.dashboard import data_access as da

def _make_processed(tmp_path):
    proc = tmp_path / "processed"
    (proc / "cases").mkdir(parents=True)
    manifest = pd.DataFrame({"caseid": [1, 2, 3],
                             "n_samples": [100, 200, 5],
                             "kept": [True, True, False]})
    manifest.to_parquet(proc / "manifest.parquet", index=False)
    (proc / "splits.json").write_text(json.dumps(
        {"train": [1], "val": [2], "test": []}))
    for cid in (1, 2):
        pd.DataFrame({"caseid": [cid, cid], "time_sec": [0, 10],
                      "bis": [45.0, 46.0], "map": [70.0, 71.0]}
                     ).to_parquet(proc / "cases" / f"case_{cid}.parquet", index=False)
    return proc

def _make_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    pd.DataFrame({"caseid": [1, 2, 3], "age": [50, 60, 70],
                  "sex": ["M", "F", "M"], "asa": [2, 3, 2],
                  "optype": ["Colorectal", "Biliary", "Thyroid"]}
                 ).to_parquet(cache / "cases.parquet", index=False)
    return cache

def test_load_manifest_reads_parquet(tmp_path):
    proc = _make_processed(tmp_path)
    m = da.load_manifest(proc)
    assert list(m["caseid"]) == [1, 2, 3]

def test_load_manifest_missing_raises_typed_error(tmp_path):
    with pytest.raises(da.DataNotFoundError):
        da.load_manifest(tmp_path / "nope")

def test_load_case_reads_by_id(tmp_path):
    proc = _make_processed(tmp_path)
    df = da.load_case(1, proc)
    assert (df["caseid"] == 1).all()
    assert "bis" in df.columns

def test_load_splits_returns_dict(tmp_path):
    proc = _make_processed(tmp_path)
    s = da.load_splits(proc)
    assert s["train"] == [1] and s["test"] == []

def test_build_cohort_table_joins_clinical_and_split_and_keeps_only_kept(tmp_path):
    proc = _make_processed(tmp_path)
    cache = _make_cache(tmp_path)
    cohort = da.build_cohort_table(proc, cache)
    assert set(cohort["caseid"]) == {1, 2}
    assert set(cohort.columns) >= {"caseid", "n_samples", "split", "age", "sex", "asa", "optype"}
    row1 = cohort[cohort["caseid"] == 1].iloc[0]
    assert row1["split"] == "train"
    assert row1["age"] == 50
    row2 = cohort[cohort["caseid"] == 2].iloc[0]
    assert row2["split"] == "val"
