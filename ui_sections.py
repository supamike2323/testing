"""UI rendering helpers for the Track 3 dashboard."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from explain_utils import explain_patient_risk, generate_recommendation


PATIENT_COLUMNS = [
    "readmit_7d_score",
    "readmit_30d_score",
    "hosp_los",
    "ed_len_hours",
    "prev90d_hosp_sum",
    "admit_year",
    "age",
    "disch_hour",
    "anchor_age",
    "comorbidity",
    "admit_hour",
    "prev90d_hosp",
    "admit_month",
    "admit_dow",
    "disch_dow",
    "disch_year",
    "icu_hx",
    "sepsistype",
    "race",
    "admission_type",
    "dod_nan",
]


def render_patient_census(snapshot: pd.DataFrame) -> None:
    """Display all available patient attributes in a single panel."""

    st.subheader("Patient Census")

    if snapshot.empty:
        st.info("No patient records available.")
        return

    missing = [col for col in PATIENT_COLUMNS if col not in snapshot.columns]
    if missing:
        st.error(
            "Missing required columns in dataset: " + ", ".join(sorted(missing))
        )
        return

    id_cols = [col for col in snapshot.columns if col.lower() == "patient_id"]
    census = snapshot[PATIENT_COLUMNS + id_cols].copy()

    # Remove any synthetic index columns that are just row numbers.
    drop_cols: list[str] = []
    for col in census.columns:
        if col in PATIENT_COLUMNS or col in id_cols:
            continue
        lowered = col.lower()
        if lowered.startswith("unnamed") or lowered in {"index", "level_0"}:
            drop_cols.append(col)
            continue
        series = census[col]
        if pd.api.types.is_integer_dtype(series):
            non_null = series.dropna().reset_index(drop=True)
            if non_null.empty:
                drop_cols.append(col)
            else:
                seq_zero = pd.Series(range(len(non_null)))
                seq_one = pd.Series(range(1, len(non_null) + 1))
                if non_null.equals(seq_zero) or non_null.equals(seq_one):
                    drop_cols.append(col)

    if drop_cols:
        census = census.drop(columns=drop_cols, errors="ignore")
    if id_cols:
        census = census.sort_values(id_cols[0])
    styled = census.style.background_gradient(
        subset=["readmit_7d_score", "readmit_30d_score"], cmap="Reds"
    )
    st.dataframe(styled, use_container_width=True)



def render_patient_detail(
    cohort: pd.DataFrame,
) -> None:
    """Render a detail panel with risk drivers and suggested actions."""

    st.subheader("Patient Detail")

    if cohort.empty:
        st.info("No patient records available.")
        return

    id_column = next((col for col in cohort.columns if col.lower() == "patient_id"), None)
    if id_column is None:
        st.error("The dataset must include a patient identifier column named `patient_id`.")
        return

    patient_ids = cohort[id_column].dropna().unique()
    if len(patient_ids) == 0:
        st.info("No patient IDs available for selection.")
        return

    selected = st.selectbox(
        "Select a patient",
        options=sorted(patient_ids),
        format_func=str,
    )

    patient_row = cohort[cohort[id_column] == selected].head(1)
    if patient_row.empty:
        st.info("No details available for the selected patient.")
        return

    latest_row = patient_row.iloc[0]

    st.markdown("**Patient profile**")
    profile_cols = [id_column] + PATIENT_COLUMNS
    profile = latest_row[profile_cols].to_frame(name="Value")
    profile.index.name = "Attribute"
    st.table(profile)

    contributions = explain_patient_risk(latest_row, cohort)
    if contributions.empty:
        st.info("Unable to compute drivers of elevated risk for this patient.")
    else:
        st.markdown("**Drivers of elevated risk**")
        display_cols = contributions[["Feature", "Value", "Contribution"]]
        st.dataframe(display_cols, use_container_width=True)

    recommendation = generate_recommendation(latest_row, contributions)
    st.markdown(f"**Suggested action:** {recommendation}")
