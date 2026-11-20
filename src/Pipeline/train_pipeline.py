import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.Components.data_ingestion import DataIngestion
from src.Components.data_validation import DataValidation
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer
from src.Components.model_push import ModelPush
from src.Components.model_explainer import ModelExplainer
from src.Pipeline.predict_pipeline import PredictPipeline


class TrainPipeline:
    """Runs the whole project with one command:  python -m src.Pipeline.train_pipeline"""

    def run_pipeline(self):
        try:
            logging.info("===== Training pipeline started =====")

            train_path, val_path, test_path, score_path = DataIngestion().initiate_data_ingestion()
            DataValidation().initiate_data_validation(train_path, val_path, test_path, score_path)
            splits, _ = DataTransformation().initiate_data_transformation(train_path, val_path, test_path)
            test_pr_auc, report = ModelTrainer().initiate_model_trainer(splits)
            serving_dir = ModelPush().initiate_model_push(report)

            if serving_dir:
                ModelExplainer().initiate_model_explainer(val_path)
                PredictPipeline(model_path=os.path.join(serving_dir, "model.pkl")).score_file(
                    score_path, os.path.join("artifacts", "predictions.csv"))

            logging.info("===== Training pipeline finished =====")
            return report, serving_dir
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    report, serving_dir = TrainPipeline().run_pipeline()
    print("\nBest model:", report["best_model"])
    print("Threshold:", report["threshold"])
    print("Test metrics:", {k: round(v, 4) for k, v in report["test_metrics"].items()})
    print("Model pushed to:", serving_dir if serving_dir else "NOT pushed (failed quality gate)")
    print("SHAP charts: artifacts/shap/  |  Validation report: artifacts/validation_report.json")
    print("MLflow: run  mlflow ui --backend-store-uri sqlite:///mlflow.db  and open http://127.0.0.1:5000")