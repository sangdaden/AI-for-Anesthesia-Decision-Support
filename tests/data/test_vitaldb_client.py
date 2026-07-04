import gzip
import pandas as pd
from adaptivedose.data import vitaldb_client as vc

def test_parse_cases_csv_strips_bom():
    raw = "﻿caseid,age,sex\n1,50,M\n2,60,F\n".encode("utf-8")
    df = vc._parse_csv_bytes(raw)
    assert list(df.columns)[0] == "caseid"  # BOM stripped
    assert len(df) == 2

def test_parse_cases_csv_handles_gzip():
    raw = gzip.compress("﻿caseid,age\n1,50\n".encode("utf-8"))
    df = vc._parse_csv_bytes(raw)
    assert df.iloc[0]["age"] == 50

def test_load_clinical_info_uses_cache(tmp_path, monkeypatch):
    calls = {"n": 0}
    def fake_fetch(url):
        calls["n"] += 1
        return "﻿caseid,age\n1,50\n".encode("utf-8")
    monkeypatch.setattr(vc, "_fetch", fake_fetch)
    df1 = vc.load_clinical_info(cache_dir=tmp_path)
    df2 = vc.load_clinical_info(cache_dir=tmp_path)  # second call hits cache
    assert calls["n"] == 1
    assert len(df1) == len(df2) == 1
