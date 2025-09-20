"""Explainability helpers for highlighting patient-level risk drivers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


NUMERIC_FEATURES: list[str] = [
    "hosp_los",
    "ed_len_hours",
    "prev90d_hosp_sum",
    "admit_year",
    "age",
    "disch_hour",
    "anchor_age",
    "admit_hour",
    "prev90d_hosp",
    "admit_month",
    "admit_dow",
    "disch_dow",
    "disch_year",
    "icu_hx",
]

CATEGORICAL_FEATURES: list[str] = [
    "comorbidity",
    "sepsistype",
    "race",
    "admission_type",
]

OPTIONAL_FLAGS: list[str] = ["dod_nan"]


CAT_WEIGHTS: dict[str, dict[str, float]] = {
    "comorbidity": {
        "ONC": 0.35,
        "CHF": 0.3,
        "CKD": 0.28,
        "COPD": 0.25,
        "DIABETES": 0.18,
        "NONE": -0.05,
    },
    "sepsistype": {"NONE": 0.0, "SUSPECTED": 0.25, "CONFIRMED": 0.4},
    "race": {
        "White": 0.1,
        "Black": 0.18,
        "Asian": 0.08,
        "Hispanic": 0.12,
        "Other": 0.1,
    },
    "admission_type": {
        "EMERGENCY": 0.32,
        "URGENT": 0.26,
        "ELECTIVE": -0.15,
        "DIRECT EMER.": 0.28,
        "TRANSFER": 0.2,
    },
}

FRIENDLY_NAMES: dict[str, str] = {
    "hosp_los": "extended length of stay",
    "ed_len_hours": "long ED dwell time",
    "prev90d_hosp_sum": "recent hospital days",
    "admit_year": "recent admission year",
    "age": "patient age",
    "disch_hour": "late discharge hour",
    "anchor_age": "anchor age",
    "comorbidity": "chronic condition profile",
    "admit_hour": "late admission hour",
    "prev90d_hosp": "recent admissions",
    "admit_month": "seasonal admission",
    "admit_dow": "admission day of week",
    "disch_dow": "discharge day of week",
    "disch_year": "discharge year",
    "icu_hx": "prior ICU history",
    "sepsistype": "sepsis status",
    "race": "race",
    "admission_type": "admission urgency",
    "dod_nan": "mortality flag",
}


@dataclass
class Contribution:
    feature: str
    value: object
    contribution: float
    percentile: float


def _numeric_contributions(patient: pd.Series, cohort: pd.DataFrame) -> list[Contribution]:
    contributions: list[Contribution] = []
    for feature in NUMERIC_FEATURES:
        if feature not in patient or feature not in cohort:
            continue
        values = cohort[feature].dropna()
        if values.empty:
            continue
        percentile = float((values <= patient[feature]).mean())
        centered = percentile - 0.5
        contribution = centered * 0.6
        contributions.append(
            Contribution(
                feature=feature,
                value=patient[feature],
                contribution=float(contribution),
                percentile=percentile,
            )
        )
    return contributions


def _categorical_contributions(patient: pd.Series) -> list[Contribution]:
    contributions: list[Contribution] = []
    for feature in CATEGORICAL_FEATURES:
        value = patient.get(feature)
        if value is None:
            continue
        weight_map = CAT_WEIGHTS.get(feature, {})
        weight = weight_map.get(value, 0.0)
        contributions.append(
            Contribution(
                feature=feature,
                value=value,
                contribution=float(weight),
                percentile=1.0,
            )
        )
    for flag in OPTIONAL_FLAGS:
        if patient.get(flag) is True:
            contributions.append(
                Contribution(
                    feature=flag,
                    value=True,
                    contribution=0.2,
                    percentile=1.0,
                )
            )
    return contributions


def explain_patient_risk(
    patient_row: pd.Series,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Return feature-level contributions for the selected patient."""

    if patient_row is None or patient_row.empty:
        return pd.DataFrame(columns=["Feature", "Value", "Contribution", "Percentile"])

    contributions = _numeric_contributions(patient_row, cohort)
    contributions.extend(_categorical_contributions(patient_row))

    if not contributions:
        return pd.DataFrame(columns=["Feature", "Value", "Contribution", "Percentile"])

    rows = [
        {
            "Feature": contrib.feature,
            "Value": contrib.value,
            "Contribution": contrib.contribution,
            "Percentile": contrib.percentile,
        }
        for contrib in contributions
    ]

    df = pd.DataFrame(rows)
    return df.sort_values("Contribution", ascending=False)


def _action_suggestions(features: Iterable[Contribution]) -> list[str]:
    suggestions: list[str] = []
    for contrib in features:
        feature = contrib.feature
        value = contrib.value
        if feature == "comorbidity":
            suggestions.append(
                "Coordinate chronic disease management and ensure follow-up appointments are scheduled."
            )
        elif feature == "sepsistype" and value in {"SUSPECTED", "CONFIRMED"}:
            suggestions.append(
                "Confirm sepsis bundle adherence and reassess infection markers before discharge."
            )
        elif feature in {"hosp_los", "ed_len_hours"}:
            suggestions.append(
                "Review discharge readiness checklist and verify care coordination for a safe transition."
            )
        elif feature in {"prev90d_hosp", "prev90d_hosp_sum"}:
            suggestions.append(
                "Engage case management to reduce short-interval readmissions and arrange early follow-up."
            )
        elif feature == "icu_hx":
            suggestions.append(
                "Discuss ICU history with the team and plan for close monitoring post-discharge."
            )
        elif feature == "admission_type" and value in {"EMERGENCY", "DIRECT EMER."}:
            suggestions.append(
                "Double-check stabilization status and ensure appropriate outpatient surveillance."
            )
        elif feature == "dod_nan":
            suggestions.append(
                "Evaluate for advanced care planning needs prior to disposition."
            )
    return suggestions


def generate_recommendation(
    patient_row: pd.Series,
    contributions: pd.DataFrame,
) -> str:
    """Craft a summary recommendation based on the top contributing drivers."""

    if patient_row is None or patient_row.empty:
        return "Review patient data to craft a discharge and follow-up plan."

    if contributions is None or contributions.empty:
        return "Ensure discharge readiness and confirm post-acute follow-up before release."

    ranked = contributions.sort_values("Contribution", ascending=False)
    top_positive = ranked[ranked["Contribution"] > 0].head(2)
    if top_positive.empty:
        top_positive = ranked.head(2)

    drivers = []
    driver_contribs: list[Contribution] = []
    for row in top_positive.itertuples():
        feature = row.Feature
        value = patient_row.get(feature, row.Value)
        friendly = FRIENDLY_NAMES.get(feature, feature.replace("_", " "))
        drivers.append(f"{friendly} ({value})")
        driver_contribs.append(
            Contribution(feature=feature, value=value, contribution=row.Contribution, percentile=row.Percentile)
        )

    reason_text = " and ".join(drivers) if drivers else "current clinical profile"
    suggestions = _action_suggestions(driver_contribs)
    action_text = suggestions[0] if suggestions else "Verify discharge readiness and ensure clear follow-up instructions."

    seven = patient_row.get("readmit_7d_score")
    thirty = patient_row.get("readmit_30d_score")
    score_text = ""
    if seven is not None and thirty is not None:
        score_text = f" (7-day risk {seven:.2f}, 30-day risk {thirty:.2f})"

    return f"This patient has elevated readmission risk due to {reason_text}{score_text}. {action_text}"
