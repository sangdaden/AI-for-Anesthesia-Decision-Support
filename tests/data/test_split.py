import pandas as pd
from adaptivedose.data.split import make_splits


def test_make_splits_are_disjoint_and_cover_all():
    manifest = pd.DataFrame({
        "caseid": list(range(100)),
        "subjectid": list(range(100)),
        "kept": [True] * 100,
    })
    splits = make_splits(manifest, test_frac=0.2, val_frac=0.2, seed=42)
    all_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert all_ids == set(range(100))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["train"]) & set(splits["val"]))
    assert len(splits["test"]) == 20
    assert len(splits["val"]) == 20


def test_make_splits_is_deterministic_with_seed():
    manifest = pd.DataFrame({"caseid": list(range(50)),
                             "subjectid": list(range(50)), "kept": [True] * 50})
    a = make_splits(manifest, 0.2, 0.2, seed=7)
    b = make_splits(manifest, 0.2, 0.2, seed=7)
    assert a["test"] == b["test"]


def test_make_splits_no_subject_leak():
    manifest = pd.DataFrame({
        "caseid": [10, 11, 12, 13],
        "subjectid": [0, 0, 1, 2],
        "kept": [True] * 4,
    })
    splits = make_splits(manifest, test_frac=0.25, val_frac=0.25, seed=1)
    loc = {}
    for name, ids in splits.items():
        sub = manifest[manifest["caseid"].isin(ids)]["subjectid"]
        for s in sub:
            assert loc.get(s, name) == name
            loc[s] = name
