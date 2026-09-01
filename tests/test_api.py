from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction():
    house = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }

    response = client.post("/predict", json=house)

    assert response.status_code == 200

    result = response.json()

    assert "predicted_house_value" in result
    assert "estimated_price_usd" in result

    assert result["predicted_house_value"] > 0
    assert result["estimated_price_usd"] > 0


def test_invalid_input():
    house = {
        "MedInc": -5,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }

    response = client.post("/predict", json=house)

    assert response.status_code == 422