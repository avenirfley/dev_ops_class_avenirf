"""Training, persistence, inference, and evaluation helpers."""

import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


def train_model(X_train, y_train):
    """Train and return a deterministic random forest classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """Return precision, recall, and F1 for binary predictions."""
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=0)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inference and return predictions."""
    return model.predict(X)


def save_model(model, path):
    """Serialize a model or encoder to ``path``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file_handle:
        pickle.dump(model, file_handle)


def load_model(path):
    """Load and return a pickle artifact from ``path``."""
    with Path(path).open("rb") as file_handle:
        return pickle.load(file_handle)


def performance_on_categorical_slice(
    data,
    column_name,
    slice_value,
    categorical_features,
    label,
    encoder,
    lb,
    model,
):
    """Compute precision, recall, and F1 for one categorical data slice."""
    data_slice = data.loc[data[column_name] == slice_value]
    if data_slice.empty:
        raise ValueError(f"No rows found for {column_name}={slice_value!r}.")

    X_slice, y_slice, _, _ = process_data(
        data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X_slice)
    return compute_model_metrics(y_slice, preds)