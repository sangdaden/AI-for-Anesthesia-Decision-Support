import pandas as pd
from adaptivedose.models.requirement import case_requirement

def _case(bis, ppf):
    return pd.DataFrame({"bis": bis, "propofol_rate": ppf})

def test_requirement_is_mean_rate_in_band():
    df = _case([45, 50, 55, 30], [4.0, 6.0, 5.0, 9.0])
    assert case_requirement(df, min_in_band_rows=3) == 5.0

def test_requirement_excludes_out_of_band_rows():
    # BIS=0 is out of the [40,60] band, so it is excluded by the band bounds.
    df = _case([0, 45, 50, 55], [99.0, 4.0, 6.0, 5.0])
    assert case_requirement(df, min_in_band_rows=3) == 5.0

def test_artifact_floor_excludes_low_bis_when_band_extends_below_it():
    # With a band that reaches down to 0, only the floor keeps the BIS=0 artifact out.
    df = _case([0, 45, 50, 55], [99.0, 4.0, 6.0, 5.0])
    with_floor = case_requirement(df, low=0.0, artifact_floor=10.0, min_in_band_rows=3)
    without_floor = case_requirement(df, low=0.0, artifact_floor=-1.0, min_in_band_rows=3)
    assert with_floor == 5.0                     # 99.0 at BIS=0 excluded by floor
    assert without_floor == (99.0 + 4.0 + 6.0 + 5.0) / 4  # floor disabled -> zero counts

def test_requirement_none_when_too_few_in_band():
    df = _case([45, 50, 70, 80], [4.0, 6.0, 5.0, 5.0])
    assert case_requirement(df, min_in_band_rows=3) is None

def test_requirement_ignores_nan_rows():
    df = _case([45, 50, 55], [4.0, None, 6.0])
    assert case_requirement(df, min_in_band_rows=2) == 5.0
