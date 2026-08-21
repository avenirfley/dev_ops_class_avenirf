"""Train and evaluate the Census income classifier."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "census.csv"
MODEL_DIR = PROJECT_ROOT / "model"
SLICE_OUTPUT_PATH = PROJECT_ROOT / "slice_output.txt"

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


def load_and_clean_data(path=DATA_PATH):
    """Load the Census CSV and normalize whitespace in names and values."""
    data = pd.read_csv(path, skipinitialspace=True)
    data.columns = data.columns.str.strip()
    text_columns = data.select_dtypes(include="object").columns
    data[text_columns] = data[text_columns].apply(lambda col: col.str.strip())
    return data


def write_slice_metrics(test, encoder, lb, model):
    """Evaluate every distinct categorical slice and replace the output file."""
    with SLICE_OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for column in CATEGORICAL_FEATURES:
            for slice_value in sorted(test[column].dropna().unique()):
                count = int((test[column] == slice_value).sum())
                precision, recall, f1 = performance_on_categorical_slice(
                    test,
                    column,
                    slice_value,
                    CATEGORICAL_FEATURES,
                    "salary",
                    encoder,
                    lb,
                    model,
                )
                print(f"{column}: {slice_value}, Count: {count:,}", file=output)
                print(
                    "Precision: "
                    f"{precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}",
                    file=output,
                )


def main():
    """Train the model, persist artifacts, and write evaluation results."""
    data = load_and_clean_data()
    train, test = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data["salary"],
    )

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CATEGORICAL_FEATURES,
        label="salary",
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CATEGORICAL_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    save_model(model, MODEL_DIR / "model.pkl")
    save_model(encoder, MODEL_DIR / "encoder.pkl")
    save_model(lb, MODEL_DIR / "label_binarizer.pkl")

    predictions = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, predictions)
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    write_slice_metrics(test, encoder, lb, model)
    print(f"Slice metrics written to {SLICE_OUTPUT_PATH}")
    return precision, recall, f1


if __name__ == "__main__":
    main()