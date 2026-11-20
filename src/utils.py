import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, recall_score,
                             precision_score, f1_score, brier_score_loss)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def save_json(file_path, data):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        raise CustomException(e, sys)


def lift_at_k(y_true, y_prob, k=0.10):
    """Response rate in the top k share of customers divided by the overall rate."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    n_top = int(round(len(y_true) * k))
    top = np.argsort(-y_prob)[:n_top]
    return float(y_true[top].mean() / y_true.mean())


def classification_metrics(y_true, y_prob, threshold=0.5):
    """All project metrics. PR-AUC is the main one."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "top_10pct_lift": lift_at_k(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "share_contacted": float(y_pred.mean()),
    }


def choose_threshold(y_true, y_prob):
    """Threshold with the highest F1 on validation data."""
    thresholds = np.round(np.arange(0.05, 0.61, 0.01), 2)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(f1s))])


def fairness_report(y_true, y_prob, groups, threshold):
    """Per-group contact rate, recall and precision (gender check)."""
    df = pd.DataFrame({"y": y_true, "p": y_prob, "g": groups})
    df["contacted"] = (df["p"] >= threshold).astype(int)
    report = {}
    for g, part in df.groupby("g"):
        report[str(g)] = {
            "customers": int(len(part)),
            "base_rate": float(part["y"].mean()),
            "share_contacted": float(part["contacted"].mean()),
            "recall": float(recall_score(part["y"], part["contacted"], zero_division=0)),
            "precision": float(precision_score(part["y"], part["contacted"], zero_division=0)),
        }
    rates = [v["share_contacted"] for v in report.values()]
    recalls = [v["recall"] for v in report.values()]
    report["contact_rate_ratio"] = float(min(rates) / max(rates))
    report["recall_gap"] = float(max(recalls) - min(recalls))
    return report


def evaluate_models(X_train, y_train, X_val, y_val, models, param, get_preprocessor,
                    n_iter=6, cv_splits=3, random_state=42):
    """Same idea as the tutorial: train every model, tune those with a search
    space, and score each on validation. Each model is a Pipeline
    (preprocessor -> model), so target encoding never sees validation data.
    Returns (report, fitted_models)."""
    try:
        report, fitted = {}, {}
        for name, (kind, model) in models.items():
            pipe = Pipeline([("preprocessor", get_preprocessor(kind)), ("model", model)])
            space = param.get(name, {})

            if space:
                search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="average_precision",
                                            cv=TimeSeriesSplit(n_splits=cv_splits),
                                            random_state=random_state, n_jobs=1)
                search.fit(X_train, y_train)
                pipe, best_params = search.best_estimator_, search.best_params_
            else:
                pipe.fit(X_train, y_train)
                best_params = {}

            train_score = average_precision_score(y_train, pipe.predict_proba(X_train)[:, 1])
            val_score = average_precision_score(y_val, pipe.predict_proba(X_val)[:, 1])
            report[name] = {"train_pr_auc": float(train_score), "val_pr_auc": float(val_score),
                            "best_params": best_params}
            fitted[name] = pipe
            logging.info(f"{name}: train PR-AUC {train_score:.4f}, validation PR-AUC {val_score:.4f}, {best_params}")
            print(f"{name}: validation PR-AUC {val_score:.4f}")
        return report, fitted
    except Exception as e:
        raise CustomException(e, sys)