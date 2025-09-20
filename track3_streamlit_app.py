"""Streamlit dashboard for Track 3: early prediction of ICU patient outcomes."""
from __future__ import annotations

import streamlit as st

from data_utils import load_sample_data, parse_uploaded_csv
from ui_sections import render_patient_census, render_patient_detail


def main() -> None:
    st.set_page_config(page_title="Track 3: Early Prediction Dashboard", layout="wide")
    st.title("ICU Bounce-back Early Warning")
    st.caption(
        "Surface patients at risk of ICU readmission within 48 hours before deterioration is obvious."
    )

    st.sidebar.title("Data inputs")
    st.sidebar.write(
        "Upload a CSV with patient trajectories, or explore the built-in synthetic cohort."
    )

    uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        try:
            df = parse_uploaded_csv(uploaded)
            st.success("Custom dataset loaded.")
        except ValueError as err:
            st.error(str(err))
            st.stop()
    else:
        df = load_sample_data()
        st.info(
            "Using a synthetic cohort. Upload your own CSV with similar columns to evaluate your unit's data."
        )

    render_patient_census(df)
    st.write("---")

    render_patient_detail(df)


if __name__ == "__main__":
    main()
