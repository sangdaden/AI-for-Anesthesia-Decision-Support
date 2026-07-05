from __future__ import annotations
from typing import Callable, Dict
import numpy as np
import pandas as pd
from adaptivedose.models.requirement import case_requirement
from adaptivedose.models.personalize import RequirementModel, leave_one_out, FEATURES

def build_modeling_table(
    cohort: pd.DataFrame,
    clinical: pd.DataFrame,
    load_case_fn: Callable[[int], pd.DataFrame],
) -> pd.DataFrame:
    """Join per-case requirement with covariates; drop cases lacking either."""
    rows = []
    for cid in cohort["caseid"]:
        req = case_requirement(load_case_fn(int(cid)))
        if req is not None:
            rows.append({"caseid": int(cid), "requirement": req})
    reqs = pd.DataFrame(rows)
    if reqs.empty:
        return pd.DataFrame(columns=["caseid", "requirement", *FEATURES])
    table = reqs.merge(clinical, on="caseid", how="left")
    table["sex_male"] = (table["sex"] == "M").astype(float)
    return table.dropna(subset=[*FEATURES, "requirement"]).reset_index(drop=True)

def run_demo(table: pd.DataFrame) -> Dict:
    """Leave-one-out demonstration: personalized vs population dosing."""
    X = table[FEATURES].to_numpy(dtype=float)
    y = table["requirement"].to_numpy(dtype=float)
    n = len(y)
    preds = leave_one_out(X, y)
    baseline = np.array([y[np.arange(n) != i].mean() for i in range(n)])
    model_mae = float(np.mean(np.abs(y - preds)))
    baseline_mae = float(np.mean(np.abs(y - baseline)))
    improvement = (baseline_mae - model_mae) / baseline_mae if baseline_mae else 0.0
    coefficients = RequirementModel().fit(X, y).coefficients(FEATURES)
    out = table.copy()
    out["loo_pred"] = preds
    out["population_dose"] = float(y.mean())
    return {
        "table": out,
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "improvement": improvement,
        "coefficients": coefficients,
        "dose_spread_std": float(np.std(preds)),
    }

__all__ = ["build_modeling_table", "run_demo", "FEATURES"]
