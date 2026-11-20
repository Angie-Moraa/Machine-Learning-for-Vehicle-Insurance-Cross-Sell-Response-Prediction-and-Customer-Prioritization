import os
import sys
import shutil
from dataclasses import dataclass
from datetime import datetime

from src.exception import CustomException
from src.logger import logging
from src.utils import save_json


@dataclass
class ModelPushConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")
    serving_dir: str = "final_model"     # the folder the API and Docker image load from
    min_test_pr_auc: float = 0.25        # quality gate: about 2x the 12.5% base rate


class ModelPush:
    def __init__(self):
        self.model_push_config = ModelPushConfig()

    def initiate_model_push(self, report):
        logging.info("Entered the model push component")
        try:
            cfg = self.model_push_config
            test_pr_auc = report["test_metrics"]["pr_auc"]

            if test_pr_auc < cfg.min_test_pr_auc:
                logging.warning(f"Model NOT pushed: test PR-AUC {test_pr_auc:.4f} < {cfg.min_test_pr_auc}")
                return None

            os.makedirs(cfg.serving_dir, exist_ok=True)
            shutil.copy(cfg.trained_model_path, os.path.join(cfg.serving_dir, "model.pkl"))
            save_json(os.path.join(cfg.serving_dir, "metadata.json"), {
                "pushed_at": datetime.now().isoformat(timespec="seconds"),
                "model_name": report["best_model"],
                "threshold": report["threshold"],
                "test_metrics": report["test_metrics"],
            })
            logging.info(f"Model pushed to {cfg.serving_dir}")
            return cfg.serving_dir
        except Exception as e:
            raise CustomException(e, sys)