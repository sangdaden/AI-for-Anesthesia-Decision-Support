import numpy as np
import pandas as pd
from adaptivedose.data.clean import (
    clamp_out_of_range, impute_case_frame, clean_case_frame,
)


def _frame(**cols):
    n = len(next(iter(cols.values())))
    base = {"caseid": [1] * n, "time_sec": list(range(0, n * 10, 10))}
    base.update(cols)
    return pd.DataFrame(base)


def test_clamp_sets_out_of_range_to_nan():
    df = _frame(bis=[50, 120, 40], map=[70, -5, 65], propofol_rate=[3, 4, 999])
    out = clamp_out_of_range(df)
    assert np.isnan(out.loc[1, "bis"])            # 120 > 100
    assert np.isnan(out.loc[1, "map"])            # -5 < 10
    assert np.isnan(out.loc[2, "propofol_rate"])  # 999 > 200
    assert out.loc[0, "bis"] == 50                # in-range untouched


def test_impute_fills_drugs_with_zero_and_physiologic_by_fill():
    df = _frame(bis=[np.nan, 45.0, np.nan], map=[70.0, 71.0, 72.0],
                propofol_rate=[np.nan, 3.0, np.nan])
    out = impute_case_frame(df)
    assert out["bis"].iloc[0] == 45.0    # bfill
    assert out["bis"].iloc[2] == 45.0    # ffill
    assert out["propofol_rate"].iloc[0] == 0.0
    assert out["propofol_rate"].iloc[2] == 0.0
    assert not out["bis"].isna().any()


def test_clean_case_frame_clamps_then_imputes():
    df = _frame(bis=[50, 120, 40], map=[70, -5, 65], propofol_rate=[3, 4, 999])
    out = clean_case_frame(df)
    assert not out["bis"].isna().any()          # imputed after clamp
    assert out.loc[1, "bis"] == 50.0            # 120 clamped -> ffilled from index 0
    assert out.loc[2, "propofol_rate"] == 0.0   # 999 clamped -> imputed to 0
