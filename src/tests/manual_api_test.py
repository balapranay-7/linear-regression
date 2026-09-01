from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def main():
    print("Testing /health...")

    response = client.get("/health")

    print("Status:", response.status_code)
    print("Response:", response.json())

    assert response.status_code == 200

    print("\nTesting /predict...")

    house = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    response = client.post("/predict", json=house)

    print("Status:", response.status_code)
    print("Response:", response.json())

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_house_value"] > 0
    assert result["estimated_price_usd"] > 0

    print("\nALL MANUAL API TESTS PASSED")


if __name__ == "__main__":
    main()
    