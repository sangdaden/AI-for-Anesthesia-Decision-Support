import streamlit as st
from adaptivedose.dashboard import stats, charts

SIGNALS = ["bis", "map", "hr", "spo2", "propofol_rate", "remifentanil_rate", "etco2"]

def render_signal_quality(ctx, case_frames):
    st.header("Signal quality / EDA")

    frames = list(case_frames.values())
    miss = stats.missingness(frames, SIGNALS)
    st.subheader("Missingness (fraction NaN across cohort)")
    st.bar_chart({k: [v] for k, v in miss.items()})

    st.subheader("BIS artifacts and target")
    total_zero = sum(stats.artifact_rate(f) * len(f) for f in frames)
    total_rows = sum(len(f) for f in frames)
    in_target = sum(stats.time_in_target(f) * len(f) for f in frames)
    c1, c2 = st.columns(2)
    c1.metric("BIS == 0 (artifact) share",
              f"{(total_zero / total_rows * 100) if total_rows else 0:.1f}%")
    c2.metric("Time in BIS target [40,60]",
              f"{(in_target / total_rows * 100) if total_rows else 0:.1f}%")

    st.subheader("Signal distribution")
    sig = st.selectbox("Signal", SIGNALS)
    values = stats.signal_distribution(frames, sig)
    if len(values):
        st.plotly_chart(charts.distribution_figure(values, sig),
                        use_container_width=True)
    else:
        st.info(f"No data for {sig}.")
