import streamlit as st
from adaptivedose.dashboard import data_access as da
from adaptivedose.dashboard import charts

def render_compare(ctx):
    cohort = ctx["cohort"]
    st.header("Compare cases")
    ids = st.multiselect("Cases (pick 2 or more)",
                         sorted(cohort["caseid"].tolist()))
    if len(ids) < 2:
        st.info("Select at least two cases to compare.")
        return
    frames = {int(cid): da.load_case(int(cid)) for cid in ids}
    st.plotly_chart(charts.compare_bis_figure(frames), width='stretch')
