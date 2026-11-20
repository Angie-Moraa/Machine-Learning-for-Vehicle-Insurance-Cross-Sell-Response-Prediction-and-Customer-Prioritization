import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts to files without opening windows
import matplotlib.pyplot as plt

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@dataclass
class ModelExplainerConfig:
    model_path: str = os.path.join("final_model", "model.pkl")
    output_dir: str = os.path.join("artifacts", "shap")
    sample_size: int = 5000
    random_state: int = 42


class ModelExplainer:
    def __init__(self):
        self.model_explainer_config = ModelExplainerConfig()

    def initiate_model_explainer(self, validation_path):
        logging.info("Entered the model explainer component")
        if not HAS_SHAP:
            logging.warning("shap not installed: explanations skipped")
            return None
        try:
            cfg = self.model_explainer_config
            os.makedirs(cfg.output_dir, exist_ok=True)

            bundle = load_object(cfg.model_path)
            base = bundle["base_model"]          # the uncalibrated pipeline: preprocessor -> model
            preprocessor = base.named_steps["preprocessor"]
            model = base.named_steps["model"]

            df = pd.read_csv(validation_path, dtype={"id": str})
            sample = df.drop(columns=["Response"]).sample(
                min(cfg.sample_size, len(df)), random_state=cfg.random_state)

            X_t = preprocessor.transform(sample)
            names = list(preprocessor.named_steps["columns"].get_feature_names_out())
            X_t = pd.DataFrame(np.asarray(X_t, dtype=float), columns=names)

            if hasattr(model, "feature_importances_"):          # LightGBM / Random Forest
                explainer = shap.TreeExplainer(model)
            else:                                                # Logistic Regression
                explainer = shap.LinearExplainer(model, X_t)
            shap_values = explainer.shap_values(X_t)
            if isinstance(shap_values, list):                    # older shap: one array per class
                shap_values = shap_values[1]
            if np.ndim(shap_values) == 3:                        # newer shap: (rows, features, classes)
                shap_values = shap_values[:, :, 1]

            # 1) beeswarm: direction and size of each feature's effect
            shap.summary_plot(shap_values, X_t, show=False)
            plt.title("SHAP values (validation sample)")
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.output_dir, "shap_summary.png"), dpi=150)
            plt.close()

            # 2) bar chart: average size of each feature's effect
            shap.summary_plot(shap_values, X_t, plot_type="bar", show=False)
            plt.title("Mean |SHAP value| by feature")
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.output_dir, "shap_importance.png"), dpi=150)
            plt.close()

            # 3) table
            importance = (pd.Series(np.abs(shap_values).mean(axis=0), index=names)
                          .sort_values(ascending=False).rename("mean_abs_shap"))
            importance.to_csv(os.path.join(cfg.output_dir, "shap_importance.csv"))
            logging.info(f"SHAP importance:\n{importance}")
            return importance
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    importance = ModelExplainer().initiate_model_explainer(os.path.join("artifacts", "validation.csv"))
    print(importance)
    print("Charts saved in artifacts/shap/")