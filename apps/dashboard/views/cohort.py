import streamlit as st
from adaptivedose.dashboard import stats, charts

def render_cohort_overview(ctx):
    cohort = ctx["cohort"]
    summary = ctx["summary"]

    st.header("Cohort overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Kept cases", summary["n_cases"])
    c2.metric("Train / Val / Test",
              " / ".join(str(summary["split_sizes"].get(k, 0))
                         for k in ("train", "val", "test")))
    dur = stats.duration_minutes(cohort, interval_sec=10)
    c3.metric("Median duration (min)", f"{dur.median():.1f}")

    st.subheader("Distributions")
    a, b = st.columns(2)
    with a:
        if "age" in cohort:
            st.plotly_chart(charts.distribution_figure(cohort["age"].dropna(), "Age"),
                            use_container_width=True)
        if "asa" in cohort:
            st.bar_chart(cohort["asa"].value_counts().sort_index())
    with b:
        if "sex" in cohort:
            st.bar_chart(cohort["sex"].value_counts())
        st.plotly_chart(charts.distribution_figure(dur, "Case duration (min)"),
                        use_container_width=True)

    st.subheader("Cases")
    st.dataframe(cohort, use_container_width=True)
