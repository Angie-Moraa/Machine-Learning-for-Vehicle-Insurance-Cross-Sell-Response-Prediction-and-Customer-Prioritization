import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self, model_path=os.path.join("artifacts", "model.pkl")):
        self.model_path = model_path

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Raw customer columns in -> probability and contact decision out.
        Feature engineering and encoding happen inside the saved model."""
        try:
            bundle = load_object(file_path=self.model_path)
            model, threshold = bundle["model"], bundle["threshold"]

            probability = model.predict_proba(features)[:, 1]
            return pd.DataFrame({
                "probability": probability.round(4),
                "contact": (probability >= threshold).astype(int),
            }, index=features.index)
        except Exception as e:
            raise CustomException(e, sys)

    def score_file(self, input_path, output_path):
        """Scores every customer in a file and ranks them for outreach (rank 1 = call first)."""
        try:
            df = pd.read_csv(input_path, dtype={"id": str})
            result = pd.concat([df[["id"]], self.predict(df)], axis=1)
            result["priority_rank"] = result["probability"].rank(ascending=False, method="first").astype(int)
            result = result.sort_values("priority_rank")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result.to_csv(output_path, index=False)
            logging.info(f"Scored {len(result)} customers -> {output_path}")
            return result
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """One customer, in the same format as the source CSV files."""

    def __init__(self,
                 gender: str,
                 customer_date_of_birth: str,
                 driving_license: int,
                 customer_residence_sub_county: str,
                 previously_insured: int,
                 vehicle_year_of_manufacture: int,
                 vehicle_damage: str,
                 annual_premium: float,
                 agent_name: str,
                 life_policy_start_date: str):
        self.gender = gender
        self.customer_date_of_birth = customer_date_of_birth
        self.driving_license = driving_license
        self.customer_residence_sub_county = customer_residence_sub_county
        self.previously_insured = previously_insured
        self.vehicle_year_of_manufacture = vehicle_year_of_manufacture
        self.vehicle_damage = vehicle_damage
        self.annual_premium = annual_premium
        self.agent_name = agent_name
        self.life_policy_start_date = life_policy_start_date

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.gender],
                "Customer_Date_of_Birth": [self.customer_date_of_birth],
                "Driving_License": [self.driving_license],
                "Customer_Residence_Sub_County": [self.customer_residence_sub_county],
                "Previously_Insured": [self.previously_insured],
                "Vehicle_Year_of_Manufacture": [self.vehicle_year_of_manufacture],
                "Vehicle_Damage": [self.vehicle_damage],
                "Annual_Premium": [self.annual_premium],
                "Agent_name": [self.agent_name],
                "Life_policy_start_date": [self.life_policy_start_date],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = PredictPipeline()

    # 1) one customer: damaged vehicle, not insured (the high-response segment)
    customer = CustomData(
        gender="Male",
        customer_date_of_birth="Monday, March 7, 1977",
        driving_license=1,
        customer_residence_sub_county="EMBAKASI",
        previously_insured=0,
        vehicle_year_of_manufacture=2020,
        vehicle_damage="Yes",
        annual_premium=38294,
        agent_name="Jomo Kenyatta",
        life_policy_start_date="Thursday, December 21, 2023",
    )
    print("High-response customer:\n", pipeline.predict(customer.get_data_as_data_frame()))

    # 2) same customer, but already insured and no damage (should be close to 0)
    customer.previously_insured, customer.vehicle_damage = 1, "No"
    print("Already insured customer:\n", pipeline.predict(customer.get_data_as_data_frame()))

    # 3) whole unlabelled file -> ranked call list
    ranked = pipeline.score_file(os.path.join("artifacts", "score.csv"),
                                 os.path.join("artifacts", "predictions.csv"))
    print(f"\nRecommended for contact: {ranked['contact'].sum():,} of {len(ranked):,}")
    print(ranked.head(10))