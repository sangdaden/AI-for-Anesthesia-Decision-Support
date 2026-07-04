import streamlit as st

def render_cohort_overview(ctx):
    st.header("Cohort overview")
    st.write(f"{ctx['summary']['n_cases']} kept cases")
