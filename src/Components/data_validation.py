import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_json
from src.Components.data_ingestion import DATE_FORMAT

FEATURE_COLUMNS = [
    "Gender", "Customer_Date_of_Birth", "Driving_License", "Customer_Residence_Sub_County",
    "Previously_Insured", "Vehicle_Year_of_Manufacture", "Vehicle_Damage", "Annual_Premium",
    "Agent_name", "Life_policy_start_date",
]
TARGET = "Response"


@dataclass
class DataValidationConfig:
    report_path: str = os.path.join("artifacts", "validation_report.json")


class DataValidation:
    """Checks every split against the rules found in the EDA (section 3).
    Hard problems ('issues') stop the pipeline; softer ones ('warnings') are logged."""

    def __init__(self):
        self.data_validation_config = DataValidationConfig()

    @staticmethod
    def check_frame(df, labelled=True):
        issues, warnings = [], []

        expected = ["id"] + FEATURE_COLUMNS + ([TARGET] if labelled else [])
        missing_cols = [c for c in expected if c not in df.columns]
        if missing_cols:
            return {"rows": len(df), "issues": [f"missing columns {missing_cols}"], "warnings": []}

        # 3.1 missing values
        n_missing = int(df[expected].isna().sum().sum())
        if n_missing:
            warnings.append(f"{n_missing} missing values")

        # 3.2 duplicates (repeated ids are expected, fully duplicated rows are not)
        n_dup = int(df.drop(columns="id").duplicated().sum())
        if n_dup:
            warnings.append(f"{n_dup} duplicated rows (ignoring id)")

        # 3.7 dates must parse
        for col in ["Customer_Date_of_Birth", "Life_policy_start_date"]:
            bad = int(pd.to_datetime(df[col], format=DATE_FORMAT, errors="coerce").isna().sum())
            if bad:
                issues.append(f"{bad} unparseable dates in {col}")

        # 3.6 allowed categories
        for col, allowed in [("Gender", {"Male", "Female"}), ("Vehicle_Damage", {"Yes", "No"}),
                             ("Driving_License", {0, 1}), ("Previously_Insured", {0, 1})]:
            unexpected = set(df[col].dropna().unique()) - allowed
            if unexpected:
                issues.append(f"unexpected values in {col}: {sorted(map(str, unexpected))[:5]}")

        # 3.7 plausible values
        if (df["Annual_Premium"] <= 0).any():
            issues.append("Annual_Premium has values <= 0")
        policy = pd.to_datetime(df["Life_policy_start_date"], format=DATE_FORMAT, errors="coerce")
        birth = pd.to_datetime(df["Customer_Date_of_Birth"], format=DATE_FORMAT, errors="coerce")
        age = (policy - birth).dt.days / 365.25
        if ((age < 18) | (age > 100)).any():
            warnings.append(f"{int(((age < 18) | (age > 100)).sum())} ages outside 18-100")
        if (policy.dt.year < df["Vehicle_Year_of_Manufacture"]).any():
            warnings.append("vehicles manufactured after the policy start year")

        if labelled and not set(df[TARGET].unique()) <= {0, 1}:
            issues.append("Response is not binary")

        return {"rows": int(len(df)), "issues": issues, "warnings": warnings}

    def initiate_data_validation(self, train_path, validation_path, test_path, score_path):
        logging.info("Entered the data validation component")
        try:
            report = {}
            frames = {}
            for name, path, labelled in [("train", train_path, True), ("validation", validation_path, True),
                                         ("test", test_path, True), ("score", score_path, False)]:
                frames[name] = pd.read_csv(path, dtype={"id": str})
                report[name] = self.check_frame(frames[name], labelled)

            # 3.9.4 categories the model has never seen (target encoder falls back to the global rate)
            for col in ["Customer_Residence_Sub_County", "Agent_name"]:
                seen = set(frames["train"][col])
                for name in ["validation", "test", "score"]:
                    unseen = set(frames[name][col]) - seen
                    if unseen:
                        report[name]["warnings"].append(f"{len(unseen)} unseen {col} values")

            report["passed"] = all(not report[k]["issues"] for k in frames)
            save_json(self.data_validation_config.report_path, report)
            logging.info(f"Validation report: {report}")

            if not report["passed"]:
                raise Exception(f"Data validation failed, see {self.data_validation_config.report_path}")
            return self.data_validation_config.report_path
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.Components.data_ingestion import DataIngestion
    paths = DataIngestion().initiate_data_ingestion()
    print("Validation report:", DataValidation().initiate_data_validation(*paths))