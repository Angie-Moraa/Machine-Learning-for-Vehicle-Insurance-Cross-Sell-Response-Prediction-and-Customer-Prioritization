import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, KBinsDiscretizer, OneHotEncoder,
                                   StandardScaler, TargetEncoder)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.Components.data_ingestion import DATE_FORMAT

TARGET = "Response"
PREMIUM_FLOOR = 2630  # spike at the minimum premium (EDA 4.4.3)

NUM_RAW = ["age", "vehicle_age", "Annual_Premium"]
BINARY = ["Driving_License", "Previously_Insured", "premium_floor_flag"]
CAT_LOW = ["Gender", "Vehicle_Damage"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns -> model features (notebook section 2).
    Everything is measured at the policy start date. `id` is dropped."""
    policy_date = pd.to_datetime(df["Life_policy_start_date"], format=DATE_FORMAT)
    birth_date = pd.to_datetime(df["Customer_Date_of_Birth"], format=DATE_FORMAT)
    out = pd.DataFrame(index=df.index)
    out["age"] = (policy_date - birth_date).dt.days / 365.25
    out["vehicle_age"] = policy_date.dt.year - df["Vehicle_Year_of_Manufacture"].astype(int)
    out["Annual_Premium"] = df["Annual_Premium"].astype(float)
    out["premium_floor_flag"] = (df["Annual_Premium"] == PREMIUM_FLOOR).astype(int)
    out["Gender"] = df["Gender"]
    out["Vehicle_Damage"] = df["Vehicle_Damage"]
    out["Driving_License"] = df["Driving_License"].astype(int)
    out["Previously_Insured"] = df["Previously_Insured"].astype(int)
    out["damage_not_insured"] = ((df["Vehicle_Damage"] == "Yes")
                                 & (df["Previously_Insured"] == 0)).astype(int)
    out["Customer_Residence_Sub_County"] = df["Customer_Residence_Sub_County"].astype(str)
    out["Agent_name"] = df["Agent_name"].astype(str)
    return out


def log_cap(x, cap):
    """Cap premium at the 99th percentile of train, then log (Logistic Regression only)."""
    return np.log1p(np.clip(x, None, cap))


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    random_state: int = 42


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def make_target_encoder(self):
        # cross-fitted inside 5 folds, smoothed; unseen categories -> global response rate
        return TargetEncoder(target_type="binary", smooth="auto",
                             cv=KFold(n_splits=5, shuffle=True,
                                      random_state=self.data_transformation_config.random_state))

    def get_data_transformer_object(self, kind="tree", use_agent=True, premium_cap=73021.0):
        """kind='linear' for Logistic Regression, kind='tree' for Random Forest / LightGBM.
        The first step runs engineer_features, so the saved model takes raw columns."""
        try:
            high_card = (["Customer_Residence_Sub_County", "Agent_name"] if use_agent
                         else ["Customer_Residence_Sub_County"])

            if kind == "linear":
                transformers = [
                    ("age_bands", KBinsDiscretizer(n_bins=6, encode="onehot", strategy="quantile"), ["age"]),
                    ("premium", Pipeline([
                        ("log_cap", FunctionTransformer(log_cap, kw_args={"cap": premium_cap},
                                                        feature_names_out="one-to-one")),
                        ("scale", StandardScaler()),
                    ]), ["Annual_Premium"]),
                    ("vehicle_age", StandardScaler(), ["vehicle_age"]),
                    ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), CAT_LOW),
                    ("binary", "passthrough", BINARY + ["damage_not_insured"]),
                    ("target_enc", Pipeline([("enc", self.make_target_encoder()),
                                             ("scale", StandardScaler())]), high_card),
                ]
            elif kind == "tree":
                transformers = [
                    ("numeric", "passthrough", NUM_RAW),
                    ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), CAT_LOW),
                    ("binary", "passthrough", BINARY),
                    ("target_enc", self.make_target_encoder(), high_card),
                ]
            else:
                raise ValueError(f"kind must be 'linear' or 'tree', got {kind}")

            logging.info(f"Built {kind} preprocessor (use_agent={use_agent})")
            return Pipeline([
                ("features", FunctionTransformer(engineer_features)),
                ("columns", ColumnTransformer(transformers, remainder="drop",
                                              verbose_feature_names_out=False)),
            ])
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, validation_path, test_path):
        logging.info("Entered the data transformation method or component")
        try:
            splits = {}
            for name, path in [("train", train_path), ("validation", validation_path), ("test", test_path)]:
                df = pd.read_csv(path, dtype={"id": str})
                splits[name] = {
                    "X": df.drop(columns=[TARGET]),          # raw columns; features are made inside the pipeline
                    "y": df[TARGET].astype(int).values,
                    "gender": df["Gender"].values,           # kept aside for the fairness check
                }
                logging.info(f"{name}: X {splits[name]['X'].shape}, response rate {splits[name]['y'].mean():.4f}")

            # premium cap learnt from the training window only
            premium_cap = float(splits["train"]["X"]["Annual_Premium"].quantile(0.99))
            splits["premium_cap"] = premium_cap
            logging.info(f"Premium cap (99th percentile of train): {premium_cap:,.0f}")

            # save a fitted tree preprocessor, as in the tutorial (useful for inspection)
            preprocessor = self.get_data_transformer_object("tree", premium_cap=premium_cap)
            preprocessor.fit(splits["train"]["X"], splits["train"]["y"])
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)
            logging.info("Saved preprocessing object")

            return splits, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.Components.data_ingestion import DataIngestion

    train_path, val_path, test_path, score_path = DataIngestion().initiate_data_ingestion()
    transformation = DataTransformation()
    splits, preprocessor_path = transformation.initiate_data_transformation(train_path, val_path, test_path)

    for kind in ["linear", "tree"]:
        pre = transformation.get_data_transformer_object(kind, premium_cap=splits["premium_cap"])
        X_out = pre.fit_transform(splits["train"]["X"], splits["train"]["y"])
        names = list(pre.named_steps["columns"].get_feature_names_out())
        print(f"{kind}: {X_out.shape[1]} columns ->", names)

    print("Premium cap:", round(splits["premium_cap"]))
    print("Saved:", preprocessor_path)