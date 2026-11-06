"""Disease-progression risk model (machine learning on tabular vitals / labs).

Training data: the Efron et al. (2004) diabetes progression dataset bundled with
scikit-learn (442 patients; outcome = quantitative disease progression one year after
baseline). We use the routinely available features age, BMI, mean arterial pressure and
blood glucose, and label 'high progression' as the top tercile of the outcome.

It is a teaching / prototype model: it is NOT trained on Indian rural-clinic data and must be
re-trained and validated on local data before any clinical use.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["age", "bmi", "map", "glucose"]
LABELS = {"age": "Age (years)", "bmi": "BMI (kg/m2)", "map": "Mean arterial pressure (mmHg)", "glucose": "Blood glucose (mg/dL)"}


@dataclass
class ModelReport:
    name: str
    cv_auc: float
    cv_accuracy: float
    n_train: int
    threshold_outcome: float
    importances: dict


def _training_frame() -> tuple[pd.DataFrame, pd.Series, float]:
    d = load_diabetes(scaled=False, as_frame=True).frame
    X = pd.DataFrame({"age": d["age"], "bmi": d["bmi"], "map": d["bp"], "glucose": d["s6"]})
    thr = float(np.percentile(d["target"], 66.7))
    y = (d["target"] >= thr).astype(int)
    return X, y, thr


def _pipelines():
    return {
        "Logistic regression": Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=1000))]),
        "Gradient boosting": Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                                                          learning_rate=0.05, random_state=42))]),
    }


@lru_cache(maxsize=1)
def train() -> tuple[dict, list[ModelReport]]:
    X, y, thr = _training_frame()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models, reports = {}, []
    for name, pipe in _pipelines().items():
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        pipe.fit(X, y)
        imp = permutation_importance(pipe, X, y, n_repeats=10, random_state=0, scoring="roc_auc")
        reports.append(ModelReport(name, round(roc_auc_score(y, proba), 3), round(accuracy_score(y, proba >= 0.5), 3),
                                   len(X), round(thr, 1), {f: round(float(v), 3) for f, v in zip(FEATURES, imp.importances_mean)}))
        models[name] = pipe
    return models, reports


def best_model() -> tuple[str, Pipeline, ModelReport]:
    models, reports = train()
    rep = max(reports, key=lambda r: r.cv_auc)
    return rep.name, models[rep.name], rep


def predict(values: dict) -> dict:
    """values: any subset of FEATURES (missing values are imputed and reported)."""
    name, model, rep = best_model()
    X, _, _ = _training_frame()
    row = pd.DataFrame([{f: values.get(f) for f in FEATURES}], dtype=float)
    p = float(model.predict_proba(row)[0, 1])
    out_of_range = [f for f in FEATURES if values.get(f) is not None and not (X[f].min() <= values[f] <= X[f].max())]
    missing = [f for f in FEATURES if values.get(f) is None]
    band = "High" if p >= 0.6 else "Moderate" if p >= 0.35 else "Low"
    return {"model": name, "probability": round(p, 3), "band": band, "missing_imputed": missing,
            "outside_training_range": out_of_range, "cv_auc": rep.cv_auc}


def features_from_patient(patient: dict, vitals: pd.DataFrame, age: int | None) -> dict:
    v = vitals.ffill().iloc[-1].to_dict() if not vitals.empty else {}
    glucose = v.get("fasting_glucose_mg_dl", v.get("random_glucose_mg_dl"))
    return {"age": age, "bmi": _nan_none(v.get("bmi")), "map": _nan_none(v.get("map")), "glucose": _nan_none(glucose)}


def _nan_none(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
