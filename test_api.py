"""Tests for the FastAPI endpoints."""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

VALID_RECORD = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


def test_get_root_returns_greeting():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Census Income Classifier API"
    }


def test_post_data_returns_prediction():
    response = client.post("/data/", json=VALID_RECORD)
    assert response.status_code == 200
    assert response.json()["result"] in {">50K", "<=50K"}


def test_post_data_rejects_incomplete_record():
    response = client.post("/data/", json={"age": 37})
    assert response.status_code == 422