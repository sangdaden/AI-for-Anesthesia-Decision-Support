from adaptivedose.data.cohort import select_cohort

def test_select_cohort_requires_all_tracks(trks_frame, clinical_frame):
    ids = select_cohort(
        trks_frame, clinical_frame,
        required_tracks=["BIS/BIS", "Orchestra/PPF20_RATE", "Solar8000/ART_MBP"],
        min_asa=1, max_asa=4, ane_type="General",
    )
    assert ids == [1]

def test_select_cohort_filters_asa_and_anetype(trks_frame, clinical_frame):
    ids = select_cohort(
        trks_frame, clinical_frame,
        required_tracks=["BIS/BIS"],
        min_asa=1, max_asa=4, ane_type="General",
    )
    assert ids == [1]
