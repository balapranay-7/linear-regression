from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "linear_regression.joblib"


FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


class HousePriceModel:
    """Service responsible for loading the model and making predictions."""

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)

    def predict(self, house: dict) -> float:
        """Predict house value from input features."""

        missing_features = [
            feature
            for feature in FEATURES
            if feature not in house
        ]

        if missing_features:
            raise ValueError(
                f"Missing features: {missing_features}"
            )

        input_data = pd.DataFrame(
            [[house[feature] for feature in FEATURES]],
            columns=FEATURES,
        )

        prediction = self.model.predict(input_data)[0]

        return float(prediction)