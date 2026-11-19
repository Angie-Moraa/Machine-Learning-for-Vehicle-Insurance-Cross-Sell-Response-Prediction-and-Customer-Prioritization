import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging

DATE_FORMAT = "%A, %B %d, %Y"  # e.g. "Monday, March 7, 1977"


@dataclass
class DataIngestionConfig:
    # source files (paths are relative to the project root)
    source_train_path: str = os.path.join("Notebook", "data", "train_data_cross-sell.csv")
    source_score_path: str = os.path.join("Notebook", "data", "test_data_cross-sell.csv")
    # outputs
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    validation_data_path: str = os.path.join("artifacts", "validation.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    score_data_path: str = os.path.join("artifacts", "score.csv")
    # chronological split: 70% train, 15% validation, remaining 15% sealed test
    train_share: float = 0.70
    validation_share: float = 0.15


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            cfg = self.ingestion_config

            # id is read as text so it is never turned into scientific notation
            df = pd.read_csv(cfg.source_train_path, dtype={"id": str}, encoding="utf-8-sig")
            score_df = pd.read_csv(cfg.source_score_path, dtype={"id": str}, encoding="utf-8-sig")
            logging.info(f"Read labelled data {df.shape} and unlabelled data {score_df.shape}")

            os.makedirs(os.path.dirname(cfg.train_data_path), exist_ok=True)
            df.to_csv(cfg.raw_data_path, index=False, header=True)

            logging.info("Chronological train / validation / test split initiated")
            policy_date = pd.to_datetime(df["Life_policy_start_date"], format=DATE_FORMAT)
            cut_1 = policy_date.quantile(cfg.train_share)
            cut_2 = policy_date.quantile(cfg.train_share + cfg.validation_share)

            # sort by date so later steps (TimeSeriesSplit) see policies in time order
            order = policy_date.sort_values(kind="mergesort").index
            df, policy_date = df.loc[order], policy_date.loc[order]

            train_set = df[policy_date <= cut_1]
            validation_set = df[(policy_date > cut_1) & (policy_date <= cut_2)]
            test_set = df[policy_date > cut_2]

            for name, part in [("train", train_set), ("validation", validation_set), ("test", test_set)]:
                logging.info(f"{name}: {len(part)} rows, response rate {part['Response'].mean():.4f}")

            train_set.to_csv(cfg.train_data_path, index=False, header=True)
            validation_set.to_csv(cfg.validation_data_path, index=False, header=True)
            test_set.to_csv(cfg.test_data_path, index=False, header=True)
            score_df.to_csv(cfg.score_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                cfg.train_data_path,
                cfg.validation_data_path,
                cfg.test_data_path,
                cfg.score_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, val_path, test_path, score_path = obj.initiate_data_ingestion()
    print("Saved:", train_path, val_path, test_path, score_path)