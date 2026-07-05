import numpy as np
import pandas as pd
from adaptivedose.adaptive import demo

def _clinical():
    return pd.DataFrame({
        "caseid": list(range(1, 11)),
        "age": [40, 50, 60, 45, 55, 65, 35, 70, 48, 52],
        "weight": [60, 70, 80, 65, 75, 85, 55, 90, 68, 72],
        "bmi": [22, 24, 26, 23, 25, 27, 21, 28, 23, 24],
        "sex": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
        "asa": [1, 2, 2, 1, 2, 3, 1, 3, 2, 2],
    })

def _case_frames():
    frames = {}
    for cid, w in zip(range(1, 11), [60, 70, 80, 65, 75, 85, 55, 90, 68, 72]):
        rate = w / 10.0
        frames[cid] = pd.DataFrame({"bis": [50.0] * 12, "propofol_rate": [rate] * 12})
    return frames

def test_build_modeling_table_joins_and_computes_requirement():
    frames = _case_frames()
    cohort = pd.DataFrame({"caseid": list(range(1, 11))})
    table = demo.build_modeling_table(cohort, _clinical(), lambda cid: frames[cid])
    assert len(table) == 10
    assert set(demo.FEATURES) <= set(table.columns)
    assert "requirement" in table.columns
    assert table[table["caseid"] == 3]["requirement"].iloc[0] == 8.0
    assert table[table["caseid"] == 1]["sex_male"].iloc[0] == 1.0

def test_build_modeling_table_drops_cases_without_requirement():
    frames = _case_frames()
    frames[5] = pd.DataFrame({"bis": [90.0] * 12, "propofol_rate": [7.0] * 12})
    cohort = pd.DataFrame({"caseid": list(range(1, 11))})
    table = demo.build_modeling_table(cohort, _clinical(), lambda cid: frames[cid])
    assert 5 not in set(table["caseid"])

def test_run_demo_reports_improvement_and_coefficients():
    frames = _case_frames()
    cohort = pd.DataFrame({"caseid": list(range(1, 11))})
    table = demo.build_modeling_table(cohort, _clinical(), lambda cid: frames[cid])
    res = demo.run_demo(table)
    assert res["model_mae"] < res["baseline_mae"]
    assert res["improvement"] > 0
    assert "weight" in res["coefficients"]
    assert res["coefficients"]["weight"] > 0
    assert len(res["table"]) == len(table)
    assert "loo_pred" in res["table"].columns

def test_noise_target_does_not_beat_population_baseline():
    # Control: a target independent of the covariates carries no signal. Leave-one-out
    # honestly penalizes the extra parameters, so the model must NOT beat the
    # population-mean baseline. Guards against silent leakage in run_demo's LOO.
    rng = np.random.default_rng(0)
    n = 40
    table = pd.DataFrame({f: rng.normal(size=n) for f in demo.FEATURES})
    table["caseid"] = range(n)
    table["requirement"] = rng.normal(size=n)  # pure noise, independent of covariates
    res = demo.run_demo(table)
    assert res["model_mae"] >= res["baseline_mae"]
