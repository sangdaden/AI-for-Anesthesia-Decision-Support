"""AdaptiveDose AI — M1 Data Explorer (Streamlit).

Run: .venv/bin/streamlit run apps/dashboard/app.py
"""
import os
import sys
from pathlib import Path

# Make the dashboard runnable from any working directory. `streamlit run` puts the
# script's own directory on sys.path (not the repo root), and the app reads relative
# paths like `configs/data.yaml` and `data/processed`. Pin both to the repo root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import streamlit as st
from adaptivedose.config import load_data_config
from adaptivedose.dashboard import data_access as da
from adaptivedose.dashboard import stats
from apps.dashboard.views import cohort, case, quality, compare, adaptive

st.set_page_config(page_title="AdaptiveDose M1 Explorer", layout="wide")

@st.cache_data
def _load_context():
    cfg = load_data_config("configs/data.yaml")
    cohort_df = da.build_cohort_table()
    return {
        "cohort": cohort_df,
        "summary": stats.cohort_summary(cohort_df),
        "interval_sec": cfg.resample.interval_sec,
    }

@st.cache_data
def _load_case_frames():
    cohort = da.build_cohort_table()
    return {int(cid): da.load_case(int(cid)) for cid in cohort["caseid"]}

def main():
    st.title("AdaptiveDose AI — M1 Data Explorer")
    try:
        ctx = _load_context()
    except da.DataNotFoundError as exc:
        st.error(str(exc))
        st.info("The dashboard reads the M1 pipeline outputs under `data/processed/`. "
                "Run the pipeline first, then reload.")
        return

    view = st.sidebar.radio(
        "View",
        ["Cohort overview", "Case viewer", "Signal quality / EDA", "Compare cases",
         "Adaptive demo"],
    )
    if view == "Cohort overview":
        cohort.render_cohort_overview(ctx)
    elif view == "Case viewer":
        case.render_case_viewer(ctx)
    elif view == "Signal quality / EDA":
        quality.render_signal_quality(ctx, _load_case_frames())
    elif view == "Compare cases":
        compare.render_compare(ctx)
    elif view == "Adaptive demo":
        adaptive.render_adaptive_demo(ctx)

if __name__ == "__main__":
    main()
