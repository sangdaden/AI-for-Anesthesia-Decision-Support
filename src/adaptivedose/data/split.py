from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd


def make_splits(
    manifest: pd.DataFrame,
    test_frac: float,
    val_frac: float,
    seed: int,
) -> Dict[str, List[int]]:
    """Split kept cases into train/val/test by subject to prevent leakage.

    Returns caseid lists. Groups by subjectid (fallback: caseid) so all cases of
    a patient stay in one fold.
    """
    kept = manifest[manifest["kept"]].copy()
    group_col = "subjectid" if "subjectid" in kept.columns else "caseid"
    groups = kept[group_col].unique()

    rng = np.random.default_rng(seed)
    shuffled = groups.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_g = set(shuffled[:n_test])
    val_g = set(shuffled[n_test:n_test + n_val])
    train_g = set(shuffled[n_test + n_val:])

    def ids_for(gset):
        return sorted(int(c) for c in kept[kept[group_col].isin(gset)]["caseid"])

    return {"train": ids_for(train_g), "val": ids_for(val_g), "test": ids_for(test_g)}
