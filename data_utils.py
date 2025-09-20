"""Data loading utilities for the Track 3 dashboard."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd


RANDOM_SEED = 42


CANDIDATE_COLUMNS = {
    "patient_id",
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
}

OPTIONAL_COLUMNS = {"dod_nan"}

SCORE_COLUMNS = {"readmit_7d_score", "readmit_30d_score"}


def load_sample_data(num_patients: int = 200) -> pd.DataFrame:
    """Create a synthetic cohort using the candidate feature set and risk scores."""

    rng = np.random.default_rng(RANDOM_SEED)

    races = ["White", "Black", "Asian", "Hispanic", "Other"]
    admission_types = [
        "EMERGENCY",
        "URGENT",
        "ELECTIVE",
        "DIRECT EMER.",
        "TRANSFER",
    ]
    sepsis_types = ["NONE", "SUSPECTED", "CONFIRMED"]
    comorbidities = [
        "CHF",
        "COPD",
        "CKD",
        "DIABETES",
        "ONC",
        "NONE",
    ]

    admit_years = rng.integers(2017, 2024, size=num_patients)
    disch_years = admit_years

    df = pd.DataFrame(
        {
            "patient_id": np.arange(1, num_patients + 1),
            "hosp_los": rng.integers(2, 21, size=num_patients),
            "ed_len_hours": rng.gamma(shape=2.5, scale=1.5, size=num_patients),
            "prev90d_hosp_sum": rng.integers(0, 15, size=num_patients),
            "admit_year": admit_years,
            "age": rng.integers(18, 95, size=num_patients),
            "disch_hour": rng.integers(0, 24, size=num_patients),
            "anchor_age": rng.integers(18, 95, size=num_patients),
            "comorbidity": rng.choice(comorbidities, size=num_patients),
            "admit_hour": rng.integers(0, 24, size=num_patients),
            "prev90d_hosp": rng.integers(0, 5, size=num_patients),
            "admit_month": rng.integers(1, 13, size=num_patients),
            "admit_dow": rng.integers(0, 7, size=num_patients),
            "disch_dow": rng.integers(0, 7, size=num_patients),
            "disch_year": disch_years,
            "icu_hx": rng.integers(0, 6, size=num_patients),
            "sepsistype": rng.choice(sepsis_types, size=num_patients, p=[0.55, 0.3, 0.15]),
            "race": rng.choice(races, size=num_patients),
            "admission_type": rng.choice(admission_types, size=num_patients),
            "dod_nan": rng.uniform(0, 1, size=num_patients) < 0.08,
        }
    )

    chronic_weight = df["comorbidity"].map(
        {
            "CHF": 0.28,
            "COPD": 0.24,
            "CKD": 0.26,
            "DIABETES": 0.18,
            "ONC": 0.3,
            "NONE": -0.05,
        }
    )
    sepsis_weight = df["sepsistype"].map({"NONE": 0.0, "SUSPECTED": 0.22, "CONFIRMED": 0.38})
    admission_weight = df["admission_type"].map(
        {
            "EMERGENCY": 0.32,
            "URGENT": 0.25,
            "ELECTIVE": -0.12,
            "DIRECT EMER.": 0.28,
            "TRANSFER": 0.18,
        }
    )

    base_risk = 0.15
    normalized_age = (df["age"] - 18) / 77
    frequent_flyer = np.clip(df["prev90d_hosp_sum"] / 15, 0, 1)
    icu_factor = np.clip(df["icu_hx"] / 6, 0, 1)
    los_factor = np.clip((df["hosp_los"] - 2) / 19, 0, 1)

    seven_day_score = base_risk + 0.35 * normalized_age + 0.3 * chronic_weight + 0.25 * sepsis_weight
    seven_day_score += 0.2 * admission_weight + 0.25 * frequent_flyer + 0.2 * icu_factor
    seven_day_score += 0.18 * los_factor + 0.12 * (df["ed_len_hours"] / (df["ed_len_hours"].max() + 1e-6))

    thirty_day_score = seven_day_score * 0.8 + 0.25 * frequent_flyer + 0.15 * los_factor
    thirty_day_score += 0.1 * (df["admit_month"].isin([11, 12, 1]).astype(float))

    df["readmit_7d_score"] = np.clip(seven_day_score, 0, 1)
    df["readmit_30d_score"] = np.clip(thirty_day_score, 0, 1)

    return df


def parse_uploaded_csv(uploaded_file: io.BytesIO) -> pd.DataFrame:
    """Return a DataFrame from the uploaded CSV, raising friendly errors."""

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError("Unable to read the uploaded file as CSV") from exc

    required = CANDIDATE_COLUMNS | SCORE_COLUMNS
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Uploaded CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    for optional in OPTIONAL_COLUMNS:
        if optional not in df.columns:
            df[optional] = False

    return df
