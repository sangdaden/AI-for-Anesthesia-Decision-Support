import pandas as pd
import streamlit as st
from adaptivedose.dashboard import data_access as da, charts
from adaptivedose.adaptive import demo

@st.cache_data
def _run(caseids: tuple):
    cohort = pd.DataFrame({"caseid": list(caseids)})
    clinical = da.load_clinical()
    table = demo.build_modeling_table(cohort, clinical, da.load_case)
    if len(table) < len(demo.FEATURES) + 2:
        return None
    return demo.run_demo(table)

def render_adaptive_demo(ctx):
    st.header("Adaptive demo — personalized propofol requirement")
    st.caption("Model-based capability demo on observational data — not for clinical use.")

    res = _run(tuple(sorted(int(c) for c in ctx["cohort"]["caseid"])))
    if res is None:
        st.info("Not enough usable cases. Rebuild the dataset with more cases: "
                "`.venv/bin/python pipelines/build_dataset.py --config configs/data.yaml --limit 60`.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Personalized MAE", f"{res['model_mae']:.2f}")
    c2.metric("Population MAE", f"{res['baseline_mae']:.2f}")
    c3.metric("Improvement", f"{res['improvement'] * 100:.0f}%")

    st.subheader("Covariate effects (direction of personalization)")
    coefs = {k: v for k, v in res["coefficients"].items() if k != "intercept"}
    st.dataframe(pd.DataFrame({"covariate": list(coefs), "coefficient": list(coefs.values())}),
                 width='stretch')

    st.subheader("Actual vs personalized prediction (leave-one-out)")
    st.plotly_chart(
        charts.scatter_actual_vs_predicted(res["table"]["requirement"], res["table"]["loo_pred"]),
        width='stretch',
    )

    st.subheader("Per patient")
    table = res["table"]
    cid = st.selectbox("Case", table["caseid"].tolist())
    row = table[table["caseid"] == cid].iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Personalized dose", f"{row['loo_pred']:.1f}")
    m2.metric("Population dose", f"{row['population_dose']:.1f}")
    m3.metric("Actual requirement", f"{row['requirement']:.1f}")
