"""FastAPI service for the vehicle insurance cross-sell model.

Run:   uvicorn app:app --reload
Docs:  http://127.0.0.1:8000/docs
Form:  http://127.0.0.1:8000/
"""
import os
import json
from datetime import date
from typing import List, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.Pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import load_object

MODEL_DIR = os.getenv("MODEL_DIR", "final_model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

app = FastAPI(
    title="Vehicle Insurance Cross-Sell API",
    description="Probability that a life-insurance policyholder takes up a vehicle-insurance offer.",
    version="1.0",
)
pipeline = PredictPipeline(model_path=MODEL_PATH)


def to_source_format(d: date) -> str:
    """2023-12-21 -> 'Thursday, December 21, 2023' (the format the model was trained on)."""
    return f"{d:%A, %B} {d.day}, {d.year}"


class Customer(BaseModel):
    """Input checked by Pydantic: wrong types or values are rejected with a clear error."""
    gender: Literal["Male", "Female"]
    customer_date_of_birth: date = Field(examples=["1977-03-07"])
    driving_license: Literal[0, 1]
    customer_residence_sub_county: str = Field(examples=["EMBAKASI"])
    previously_insured: Literal[0, 1]
    vehicle_year_of_manufacture: int = Field(ge=1950, le=2030, examples=[2020])
    vehicle_damage: Literal["Yes", "No"]
    annual_premium: float = Field(gt=0, examples=[38294])
    agent_name: str = Field(examples=["Jomo Kenyatta"])
    life_policy_start_date: date = Field(examples=["2023-12-21"])

    def to_frame(self) -> pd.DataFrame:
        values = self.model_dump()
        values["customer_date_of_birth"] = to_source_format(self.customer_date_of_birth)
        values["life_policy_start_date"] = to_source_format(self.life_policy_start_date)
        return CustomData(**values).get_data_as_data_frame()


class Prediction(BaseModel):
    probability: float
    contact: bool
    threshold: float


@app.get("/health")
def health():
    """Shows whether a model is loaded and which one."""
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(meta_path):
        raise HTTPException(status_code=503, detail=f"No model found in {MODEL_DIR}. Run the training pipeline.")
    with open(meta_path) as f:
        meta = json.load(f)
    return {"status": "ok", "model": meta["model_name"], "threshold": meta["threshold"],
            "pushed_at": meta["pushed_at"], "test_pr_auc": round(meta["test_metrics"]["pr_auc"], 4)}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    result = pipeline.predict(customer.to_frame()).iloc[0]
    threshold = load_object(MODEL_PATH)["threshold"]
    return Prediction(probability=float(result["probability"]),
                      contact=bool(result["contact"]), threshold=threshold)


@app.post("/predict_batch", response_model=List[Prediction])
def predict_batch(customers: List[Customer]):
    if not customers:
        return []
    frame = pd.concat([c.to_frame() for c in customers], ignore_index=True)
    results = pipeline.predict(frame)
    threshold = load_object(MODEL_PATH)["threshold"]
    return [Prediction(probability=float(p), contact=bool(c), threshold=threshold)
            for p, c in zip(results["probability"], results["contact"])]


@app.get("/")
def home():
    """Simple form, like the tutorial's home.html."""
    return FileResponse(os.path.join("templates", "index.html"))