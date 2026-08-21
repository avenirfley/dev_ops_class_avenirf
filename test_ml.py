"""Unit tests for the Census machine learning pipeline."""

import numpy as np
import pandas as pd
import pytest

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)


def sample_data():
    """Return a small labeled dataset suitable for fast unit tests."""
    return pd.DataFrame(
        {
            "age": [25, 52, 31, 47, 28, 61, 36, 44],
            "workclass": ["Private", "Gov"] * 4,
            "salary": ["<=50K", ">50K", "<=50K", ">50K"] * 2,
        }
    )


def test_process_data_reuses_encoder_and_handles_unknown_category():
    data = sample_data()
    _, _, encoder, lb = process_data(
        data,
        categorical_features=["workclass"],
        label="salary",
        training=True,
    )
    unseen = pd.DataFrame(
        {"age": [40], "workclass": ["Self-employed"], "salary": [">50K"]}
    )
    X_unseen, y_unseen, _, _ = process_data(
        unseen,
        categorical_features=["workclass"],
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    assert X_unseen.shape == (1, 3)
    assert y_unseen.tolist() == [1]


def test_train_model_and_inference_return_binary_predictions():
    data = sample_data()
    X, y, _, _ = process_data(
        data,
        categorical_features=["workclass"],
        label="salary",
        training=True,
    )
    predictions = inference(train_model(X, y), X)
    assert predictions.shape == y.shape
    assert set(predictions).issubset({0, 1})


def test_compute_model_metrics_matches_expected_values():
    precision, recall, f1 = compute_model_metrics(
        np.array([1, 0, 1, 0]), np.array([1, 0, 0, 0])
    )
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(2 / 3)


def test_save_and_load_model_round_trip(tmp_path):
    artifact = {"model": "test", "version": 1}
    path = tmp_path / "artifact.pkl"
    save_model(artifact, path)
    assert load_model(path) == artifact


def test_performance_on_categorical_slice_returns_valid_metrics():
    data = sample_data()
    X, y, encoder, lb = process_data(
        data,
        categorical_features=["workclass"],
        label="salary",
        training=True,
    )
    model = train_model(X, y)
    metrics = performance_on_categorical_slice(
        data,
        "workclass",
        "Private",
        ["workclass"],
        "salary",
        encoder,
        lb,
        model,
    )
    assert len(metrics) == 3
    assert all(0.0 <= metric <= 1.0 for metric in metrics)