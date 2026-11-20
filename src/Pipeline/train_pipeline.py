import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.Components.data_ingestion import DataIngestion
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer
from src.Components.model_push import ModelPush
from src.Pipeline.predict_pipeline import PredictPipeline


class TrainPipeline:
    """Runs the whole project with one command, from the project root:
        python -m src.Pipeline.train_pipeline
    """

    def run_pipeline(self):
        try:
            logging.info("===== Training pipeline started =====")

            # 1) ingestion: chronological 70/15/15 split
            train_path, val_path, test_path, score_path = DataIngestion().initiate_data_ingestion()

            # 2) transformation: load splits, learn the premium cap
            splits, _ = DataTransformation().initiate_data_transformation(train_path, val_path, test_path)

            # 3) training: tune, choose, calibrate, threshold, sealed test
            test_pr_auc, report = ModelTrainer().initiate_model_trainer(splits)

            # 4) push: copy to final_model/ only if it passes the quality gate
            serving_dir = ModelPush().initiate_model_push(report)

            # 5) final scoring of the unlabelled customers -> ranked call list
            if serving_dir:
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
    print("Call list: artifacts/predictions.csv")