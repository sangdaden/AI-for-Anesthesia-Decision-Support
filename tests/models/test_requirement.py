import pandas as pd
from adaptivedose.models.requirement import case_requirement

def _case(bis, ppf):
    return pd.DataFrame({"bis": bis, "propofol_rate": ppf})

def test_requirement_is_mean_rate_in_band():
    df = _case([45, 50, 55, 30], [4.0, 6.0, 5.0, 9.0])
    assert case_requirement(df, min_in_band_rows=3) == 5.0

def test_requirement_excludes_artifact_zero_bis():
    df = _case([0, 45, 50, 55], [99.0, 4.0, 6.0, 5.0])
    assert case_requirement(df, min_in_band_rows=3) == 5.0

def test_requirement_none_when_too_few_in_band():
    df = _case([45, 50, 70, 80], [4.0, 6.0, 5.0, 5.0])
    assert case_requirement(df, min_in_band_rows=3) is None

def test_requirement_ignores_nan_rows():
    df = _case([45, 50, 55], [4.0, None, 6.0])
    assert case_requirement(df, min_in_band_rows=2) == 5.0
