"""Model training helpers for the Track 3 dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SKLEARN_AVAILABLE = False


RANDOM_SEED = 42


@dataclass
class ModelArtifacts:
    """Container for model outputs and diagnostics."""

    model: Optional[object]
    report: Optional[str]
    roc_auc: Optional[float]
    confusion: Optional[pd.DataFrame]
    warning: Optional[str] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    specificity: Optional[float] = None
    feature_names: Optional[list[str]] = None


NUMERIC_FEATURES = [
    "hours_since_admission",
    "age",
    "los",
    "previous_hospitalizations",
    "icu_history",
]

CATEGORICAL_FEATURES = [
    "sex",
    "marital_status",
    "race",
    "health_insurance",
    "admission_type",
    "sepsis_type",
    "elixhauser_condition",
]

ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _format_metric(value: float | None) -> Optional[float]:
    """Guard against NaNs while keeping None for missing metrics."""
    if value is None or np.isnan(value):
        return None
    return float(value)


def train_model(df: pd.DataFrame) -> ModelArtifacts:
    """Train a logistic regression model if scikit-learn is present."""

    if not SKLEARN_AVAILABLE:
        return ModelArtifacts(
            model=None,
            report=None,
            roc_auc=None,
            confusion=None,
            warning=(
                "Install scikit-learn (`pip install scikit-learn`) to enable on-the-fly model training. "
                "A heuristic risk score is being used instead."
            ),
        )

    missing_features = [col for col in ALL_MODEL_FEATURES if col not in df.columns]
    if missing_features:
        return ModelArtifacts(
            model=None,
            report=None,
            roc_auc=None,
            confusion=None,
            warning=(
                "Missing required columns for model training: "
                + ", ".join(sorted(missing_features))
            ),
        )

    X = df[ALL_MODEL_FEATURES]
    y = df["icu_readmit_48h"]

    if y.nunique() < 2:
        return ModelArtifacts(
            model=None,
            report=None,
            roc_auc=None,
            confusion=None,
            warning=(
                "Need at least two outcome classes to train the model. Upload more varied data or keep "
                "using the heuristic risk score."
            ),
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse=True),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    prec = _format_metric(precision_score(y_test, y_pred, zero_division=0))
    rec = _format_metric(recall_score(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    confusion_df = pd.DataFrame(
        cm,
        index=["Actual:No", "Actual:Yes"],
        columns=["Pred:No", "Pred:Yes"],
    )

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else None

    report = classification_report(y_test, y_pred, zero_division=0)
    roc_auc = _format_metric(roc_auc_score(y_test, y_prob))

    feature_names: Optional[list[str]] = None
    try:
        feature_names = list(model.named_steps["preprocessor"].get_feature_names_out())
    except Exception:  # pragma: no cover - defensive
        feature_names = None

    return ModelArtifacts(
        model=model,
        report=report,
        roc_auc=roc_auc,
        confusion=confusion_df,
        precision=prec,
        recall=rec,
        specificity=_format_metric(specificity),
        feature_names=feature_names,
    )
