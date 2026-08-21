"""Data preprocessing helpers for the Census income model."""

import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X,
    categorical_features=None,
    label=None,
    training=True,
    encoder=None,
    lb=None,
):
    """Encode categorical features and, when present, binarize the label."""
    categorical_features = categorical_features or []
    data = X.copy()

    if label is not None:
        y_values = data.pop(label)
    else:
        y_values = None

    X_categorical = data[categorical_features].values
    X_continuous = data.drop(columns=categorical_features).to_numpy()

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_categorical = encoder.fit_transform(X_categorical)
        lb = LabelBinarizer()
        if y_values is None:
            y = np.array([])
        else:
            y = lb.fit_transform(y_values).ravel()
    else:
        if encoder is None:
            raise ValueError("An encoder is required when training=False.")
        X_categorical = encoder.transform(X_categorical)
        if y_values is None:
            y = np.array([])
        else:
            if lb is None:
                raise ValueError("A label binarizer is required for labeled data.")
            y = lb.transform(y_values).ravel()

    X_processed = np.concatenate([X_continuous, X_categorical], axis=1)
    return X_processed, y, encoder, lb


def apply_label(prediction):
    """Convert a binary model prediction to the Census salary label."""
    value = np.asarray(prediction).ravel()[0]
    if value == 1:
        return ">50K"
    if value == 0:
        return "<=50K"
    raise ValueError(f"Unexpected binary prediction: {value}")