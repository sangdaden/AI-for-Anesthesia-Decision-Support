import streamlit as st
from adaptivedose.dashboard import data_access as da
from adaptivedose.dashboard import stats, charts

def render_case_viewer(ctx):
    cohort = ctx["cohort"]
    st.header("Case viewer")
    caseid = st.selectbox("Case", sorted(cohort["caseid"].tolist()))
    df = da.load_case(int(caseid))

    m1, m2, m3 = st.columns(3)
    m1.metric("Samples", len(df))
    m2.metric("Time in BIS target", f"{stats.time_in_target(df) * 100:.0f}%")
    m3.metric("BIS artifact (==0)", f"{stats.artifact_rate(df) * 100:.1f}%")

    st.plotly_chart(charts.case_timeseries_figure(df), use_container_width=True)
