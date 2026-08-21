"""FastAPI application for Census income inference."""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class Data(BaseModel):
    """A single Census record accepted by the inference endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(..., examples=[37])
    workclass: str = Field(..., examples=["Private"])
    fnlgt: int = Field(..., examples=[178356])
    education: str = Field(..., examples=["HS-grad"])
    education_num: int = Field(..., examples=[10], alias="education-num")
    marital_status: str = Field(
        ..., examples=["Married-civ-spouse"], alias="marital-status"
    )
    occupation: str = Field(..., examples=["Prof-specialty"])
    relationship: str = Field(..., examples=["Husband"])
    race: str = Field(..., examples=["White"])
    sex: str = Field(..., examples=["Male"])
    capital_gain: int = Field(..., examples=[0], alias="capital-gain")
    capital_loss: int = Field(..., examples=[0], alias="capital-loss")
    hours_per_week: int = Field(..., examples=[40], alias="hours-per-week")
    native_country: str = Field(
        ..., examples=["United-States"], alias="native-country"
    )


encoder = load_model(MODEL_DIR / "encoder.pkl")
model = load_model(MODEL_DIR / "model.pkl")

app = FastAPI(title="Census Income Classifier", version="1.0.0")


@app.get("/")
async def get_root():
    """Return an API greeting."""
    return {"message": "Welcome to the Census Income Classifier API"}


@app.post("/data/")
async def post_inference(data: Data):
    """Predict whether the supplied record's salary exceeds $50K."""
    values = data.model_dump()
    record = {key.replace("_", "-"): [value] for key, value in values.items()}
    frame = pd.DataFrame.from_dict(record)
    processed, _, _, _ = process_data(
        frame,
        categorical_features=CATEGORICAL_FEATURES,
        training=False,
        encoder=encoder,
    )
    prediction = inference(model, processed)
    return {"result": apply_label(prediction)}