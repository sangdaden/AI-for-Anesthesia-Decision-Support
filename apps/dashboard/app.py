"""AdaptiveDose AI — M1 Data Explorer (Streamlit).

Run: .venv/bin/streamlit run apps/dashboard/app.py
"""
import streamlit as st
from adaptivedose.dashboard import data_access as da
from adaptivedose.dashboard import stats
from apps.dashboard.views import cohort, case, quality

st.set_page_config(page_title="AdaptiveDose M1 Explorer", layout="wide")

@st.cache_data
def _load_context():
    cohort_df = da.build_cohort_table()
    return {
        "cohort": cohort_df,
        "summary": stats.cohort_summary(cohort_df),
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
        ["Cohort overview", "Case viewer", "Signal quality / EDA", "Compare cases"],
    )
    if view == "Cohort overview":
        cohort.render_cohort_overview(ctx)
    elif view == "Case viewer":
        case.render_case_viewer(ctx)
    elif view == "Signal quality / EDA":
        quality.render_signal_quality(ctx, _load_case_frames())
    else:
        st.info(f"'{view}' is added in a later task.")

if __name__ == "__main__":
    main()
