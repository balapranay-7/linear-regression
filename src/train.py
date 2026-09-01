from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "california_housing.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "linear_regression.joblib"


def load_data() -> pd.DataFrame:
    """Load the raw housing dataset."""
    return pd.read_csv(DATA_PATH)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LinearRegression:
    """Train the Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: LinearRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate the trained model."""
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def main() -> None:
    print("Loading dataset...")

    df = load_data()

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    print(f"Dataset shape: {df.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    print("Training model...")

    model = train_model(X_train, y_train)

    print("Evaluating model...")

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print(f"MAE : {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"R²  : {metrics['r2']:.6f}")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()