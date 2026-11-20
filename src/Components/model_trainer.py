import os
import sys
from dataclasses import dataclass

from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import (save_object, save_json, evaluate_models, classification_metrics,
                       choose_threshold, fairness_report)
from src.Components.data_transformation import DataTransformation

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    report_file_path: str = os.path.join("artifacts", "model_report.json")
    min_val_pr_auc: float = 0.25
    random_state: int = 42
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "vehicle-insurance-cross-sell"


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def log_to_mlflow(self, model_report, report):
        """One parent run per training, with a child run for every model."""
        cfg = self.model_trainer_config
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.mlflow_experiment)

        with mlflow.start_run(run_name=f"pipeline - best {report['best_model']}"):
            for name, r in model_report.items():
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.log_params({k.replace("model__", ""): v for k, v in r["best_params"].items()})
                    mlflow.log_metrics({"train_pr_auc": r["train_pr_auc"], "val_pr_auc": r["val_pr_auc"]})

            mlflow.log_params({"best_model": report["best_model"], "threshold": report["threshold"]})
            mlflow.log_metrics({f"test_{k}": v for k, v in report["test_metrics"].items()})
            mlflow.log_metrics({f"calibration_{k}": v for k, v in report["calibration"].items()})
            mlflow.log_metrics({f"agent_{k}": v for k, v in report["agent_check"].items()})
            mlflow.log_metrics({"fairness_contact_rate_ratio": report["fairness_by_gender"]["contact_rate_ratio"],
                                "fairness_recall_gap": report["fairness_by_gender"]["recall_gap"]})
            mlflow.log_artifact(cfg.report_file_path)
            mlflow.log_artifact(cfg.trained_model_file_path)
        logging.info("Logged run to MLflow")

    def initiate_model_trainer(self, splits):
        try:
            cfg = self.model_trainer_config
            logging.info("Split training, validation and test input data")
            X_train, y_train = splits["train"]["X"], splits["train"]["y"]
            X_val, y_val = splits["validation"]["X"], splits["validation"]["y"]
            X_test, y_test = splits["test"]["X"], splits["test"]["y"]

            transformation = DataTransformation()
            cap = splits["premium_cap"]

            def get_preprocessor(kind, use_agent=True):
                return transformation.get_data_transformer_object(kind, use_agent=use_agent, premium_cap=cap)

            models = {
                "Baseline (Dummy)": ("tree", DummyClassifier(strategy="prior")),
                "Logistic Regression": ("linear", LogisticRegression(class_weight="balanced", max_iter=2000)),
                "Random Forest": ("tree", RandomForestClassifier(
                    n_estimators=150, max_depth=12, min_samples_leaf=200,
                    class_weight="balanced_subsample", n_jobs=-1, random_state=cfg.random_state)),
                "LightGBM": ("tree", LGBMClassifier(
                    class_weight="balanced", random_state=cfg.random_state, verbose=-1)),
            }
            params = {
                "Logistic Regression": {"model__C": [0.01, 0.1, 1.0, 10.0]},
                "LightGBM": {
                    "model__num_leaves": [15, 31, 63],
                    "model__learning_rate": [0.03, 0.05, 0.1],
                    "model__n_estimators": [200, 400],
                    "model__min_child_samples": [50, 200],
                },
            }

            model_report, fitted = evaluate_models(X_train, y_train, X_val, y_val,
                                                   models=models, param=params,
                                                   get_preprocessor=get_preprocessor,
                                                   random_state=cfg.random_state)

            candidates = {k: v["val_pr_auc"] for k, v in model_report.items() if not k.startswith("Baseline")}
            best_model_name = max(candidates, key=candidates.get)
            best_model_score = candidates[best_model_name]
            best_model = fitted[best_model_name]

            if best_model_score < cfg.min_val_pr_auc:
                raise Exception(f"No good model found: best validation PR-AUC {best_model_score:.4f}")
            logging.info(f"Best model: {best_model_name} (validation PR-AUC {best_model_score:.4f})")

            # agent-as-proxy check
            kind = models[best_model_name][0]
            no_agent = Pipeline([("preprocessor", get_preprocessor(kind, use_agent=False)),
                                 ("model", clone(best_model.named_steps["model"]))])
            no_agent.fit(X_train, y_train)
            agent_check = {
                "with_agent": best_model_score,
                "without_agent": float(average_precision_score(y_val, no_agent.predict_proba(X_val)[:, 1])),
            }

            # calibration and threshold on validation
            calibrated = CalibratedClassifierCV(FrozenEstimator(best_model), method="sigmoid")
            calibrated.fit(X_val, y_val)
            val_prob = calibrated.predict_proba(X_val)[:, 1]
            threshold = choose_threshold(y_val, val_prob)
            calibration = {
                "brier_before": float(brier_score_loss(y_val, best_model.predict_proba(X_val)[:, 1])),
                "brier_after": float(brier_score_loss(y_val, val_prob)),
            }

            # sealed test window, scored once
            test_prob = calibrated.predict_proba(X_test)[:, 1]
            test_metrics = classification_metrics(y_test, test_prob, threshold)
            fairness = fairness_report(y_test, test_prob, splits["test"]["gender"], threshold)
            logging.info(f"Test metrics: {test_metrics}")

            save_object(
                file_path=cfg.trained_model_file_path,
                obj={"model": calibrated, "base_model": best_model,
                     "threshold": threshold, "model_name": best_model_name},
            )
            report = {
                "models": model_report,
                "best_model": best_model_name,
                "agent_check": agent_check,
                "calibration": calibration,
                "threshold": threshold,
                "test_metrics": test_metrics,
                "fairness_by_gender": fairness,
            }
            save_json(cfg.report_file_path, report)

            if HAS_MLFLOW:
                self.log_to_mlflow(model_report, report)
            else:
                logging.warning("mlflow not installed: run not tracked")

            return test_metrics["pr_auc"], report
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.Components.data_ingestion import DataIngestion

    train_path, val_path, test_path, score_path = DataIngestion().initiate_data_ingestion()
    splits, _ = DataTransformation().initiate_data_transformation(train_path, val_path, test_path)
    test_pr_auc, report = ModelTrainer().initiate_model_trainer(splits)
    print("\nBest model:", report["best_model"])
    print("Test metrics:", {k: round(v, 4) for k, v in report["test_metrics"].items()})