from __future__ import annotations
from typing import List
import pandas as pd

def select_cohort(
    trks: pd.DataFrame,
    clinical: pd.DataFrame,
    required_tracks: List[str],
    min_asa: int,
    max_asa: int,
    ane_type: str,
) -> List[int]:
    """Return sorted caseids that have every required track and pass clinical filters.

    Uses the track-index table directly instead of the deprecated
    vitaldb.find_cases().
    """
    have = trks[trks["tname"].isin(required_tracks)]
    counts = have.groupby("caseid")["tname"].nunique()
    track_ok = set(counts[counts == len(required_tracks)].index)

    clin = clinical.copy()
    clin = clin[clin["asa"].between(min_asa, max_asa)]
    clin = clin[clin["ane_type"] == ane_type]
    clinical_ok = set(clin["caseid"])

    return sorted(int(c) for c in (track_ok & clinical_ok))
