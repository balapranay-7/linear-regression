# House Price Prediction — Linear Regression

End-to-end machine learning project that predicts California house values using Linear Regression and serves predictions through a FastAPI REST API.

## Status

✅ MVP Completed

## Model Performance

| Metric | Score |
|---|---:|
| MAE | 0.5332 |
| RMSE | 0.7456 |
| R² | 0.5758 |

Dataset: California Housing  
Training samples: 16,512  
Test samples: 4,128

## Tech Stack

- Python
- NumPy / Pandas
- scikit-learn
- Joblib
- FastAPI
- Pytest
- Docker
- GitHub Actions

## Project Structure

```text
data/          Dataset
models/        Trained model
notebooks/     EDA
src/           Training, prediction & API
tests/         Automated tests
configs/       Configuration
Dockerfile     Container configuration
README.md      Documentation


Run Locally
python src/train.py
pytest
uvicorn src.api:app --reload
