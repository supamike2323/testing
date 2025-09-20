"""Risk scoring helpers for the Track 3 dashboard."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from model_utils import ALL_MODEL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


CATEGORY_WEIGHTS = {
    "sex": {"Female": 0.45, "Male": 0.55},
    "marital_status": {
        "Married": 0.4,
        "Single": 0.5,
        "Divorced": 0.55,
        "Widowed": 0.6,
    },
    "race": {
        "White": 0.45,
        "Black": 0.52,
        "Asian": 0.4,
        "Hispanic": 0.48,
        "Other": 0.5,
    },
    "health_insurance": {
        "Private": 0.4,
        "Medicare": 0.5,
        "Medicaid": 0.55,
        "Uninsured": 0.6,
    },
    "admission_type": {
        "Emergency": 0.6,
        "Urgent": 0.52,
        "Elective": 0.3,
        "Transfer": 0.48,
    },
    "sepsis_type": {"None": 0.35, "Suspected": 0.55, "Confirmed": 0.7},
    "elixhauser_condition": {
        "CHF": 0.62,
        "COPD": 0.58,
        "Diabetes": 0.48,
        "CKD": 0.6,
        "No chronic dx": 0.3,
    },
}


def _heuristic_risk(df: pd.DataFrame) -> np.ndarray:
    """Fallback risk score when no model is available."""

    if df.empty:
        return np.array([])

    numeric = df[NUMERIC_FEATURES].copy()
    numeric = (numeric - numeric.min()) / (numeric.max() - numeric.min() + 1e-6)

    numeric_score = (
        0.25 * numeric["hours_since_admission"]
        + 0.2 * numeric["age"]
        + 0.2 * numeric["los"]
        + 0.2 * numeric["previous_hospitalizations"]
        + 0.15 * numeric["icu_history"]
    )

    categorical_scores = np.zeros(len(df))
    for feature in CATEGORICAL_FEATURES:
        weights = CATEGORY_WEIGHTS.get(feature, {})
        categorical_scores += df[feature].map(lambda v: weights.get(v, 0.45)).to_numpy()

    categorical_scores = categorical_scores / max(len(CATEGORICAL_FEATURES), 1)
    risk = 0.55 * numeric_score + 0.45 * categorical_scores
    return np.clip(risk, 0, 1)


def assign_risk_band(score: float) -> str:
    """Map numeric risk to a categorical label with emoji indicator."""

    if score >= 0.7:
        return "🔴 High"
    if score >= 0.4:
        return "🟠 Medium"
    return "🟢 Low"


def compute_risk_trajectory(df: pd.DataFrame, model: Optional[object]) -> pd.DataFrame:
    """Return the full hourly trajectory with risk scores and bands."""

    trajectory = df.copy()

    if model is not None:
        trajectory["risk_score"] = model.predict_proba(
            trajectory[ALL_MODEL_FEATURES]
        )[:, 1]
    else:
        trajectory["risk_score"] = _heuristic_risk(trajectory)

    trajectory["risk_band"] = trajectory["risk_score"].apply(assign_risk_band)
    return trajectory


def latest_patient_snapshot(risk_trajectory: pd.DataFrame) -> pd.DataFrame:
    """Grab the most recent row for each patient."""

    latest = (
        risk_trajectory.sort_values("hours_since_admission")
        .groupby("patient_id", as_index=False)
        .tail(1)
    )
    return latest


def derive_alerts(risk_snapshot: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return patients whose risk exceeds the alerting threshold in the most recent hour."""

    if risk_snapshot.empty:
        return risk_snapshot.copy()

    recent_hour = risk_snapshot["hours_since_admission"].max()
    alerts = risk_snapshot[
        (risk_snapshot["risk_score"] >= threshold)
        & (risk_snapshot["hours_since_admission"] >= recent_hour)
    ].copy()
    return alerts.sort_values("risk_score", ascending=False)


def synthesize_interventions(risk_trajectory: pd.DataFrame) -> pd.DataFrame:
    """Create intervention markers based on notable risk changes and comorbidities."""

    events = []
    for patient_id, group in risk_trajectory.groupby("patient_id"):
        group = group.sort_values("hours_since_admission")
        delta = group["risk_score"].diff()

        spikes = group[(delta > 0.18) & (group["risk_score"] > 0.6)]
        for row in spikes.itertuples():
            events.append(
                {
                    "patient_id": patient_id,
                    "hours_since_admission": row.hours_since_admission,
                    "event": "Care team huddle initiated",
                }
            )

        sepsis_escalations = group[
            (group["sepsis_type"].isin(["Suspected", "Confirmed"]))
            & (group["risk_score"] > 0.65)
        ]
        for row in sepsis_escalations.itertuples():
            events.append(
                {
                    "patient_id": patient_id,
                    "hours_since_admission": row.hours_since_admission,
                    "event": "Sepsis bundle review",
                }
            )

        complex_history = group[
            (group["elixhauser_condition"].isin(["CHF", "CKD"]))
            & (group["risk_score"] > 0.6)
        ]
        for row in complex_history.itertuples():
            events.append(
                {
                    "patient_id": patient_id,
                    "hours_since_admission": row.hours_since_admission,
                    "event": "Case management consult",
                }
            )

    if not events:
        return pd.DataFrame(columns=["patient_id", "hours_since_admission", "event"])

    return pd.DataFrame.from_records(events).drop_duplicates()
