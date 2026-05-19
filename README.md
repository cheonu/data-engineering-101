# Titanic Survival Predictor

End-to-end ML pipeline that trains a logistic regression model on the Titanic dataset and serves predictions via a FastAPI REST API. Built as a data engineering learning project covering data ingestion, transformation, model serialization, and containerized deployment.

## Project Structure

```
├── fetch_titanic.py      # Downloads Titanic dataset from Kaggle
├── transform.py          # Feature engineering & model training
├── save_model.py         # Serializes model + preprocessor to joblib artifact
├── api/main.py           # FastAPI prediction service
├── artifacts/            # Saved model artifacts (DVC-tracked)
├── models/               # Raw dataset (downloaded from Kaggle)
├── dockerfile            # Container image definition
└── requirements.txt      # Python dependencies
```

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Fetch the dataset

Requires a [Kaggle API token](https://www.kaggle.com/docs/api) configured.

```bash
python fetch_titanic.py
```

### 3. Train the model

```bash
python transform.py
```

This fits a `LogisticRegression` with a sklearn `ColumnTransformer` pipeline (imputation, scaling, one-hot encoding) and prints evaluation metrics.

### 4. Save the artifact

```bash
python save_model.py
```

Bundles the preprocessor, model, threshold, and feature names into `artifacts/pipeline_v1.joblib`.

### 5. Run the API locally

```bash
uvicorn api.main:app --reload
```

The service starts at `http://localhost:8000`.

## Docker

```bash
docker build -t titanic-predictor .
docker run -p 8000:8000 titanic-predictor
```

## API Endpoints

### `POST /predict`

Predict survival for a single passenger.

```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 29,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 211.3,
  "Embarked": "S"
}
```

Response:

```json
{
  "prediction": "survived",
  "probability": 0.9274,
  "threshold_used": 0.5,
  "status": "success"
}
```

### `GET /health`

Returns service health and model version.

### `POST /check-drift`

Accepts a list of passenger inputs (minimum 10) for distribution drift analysis. Currently a placeholder for Evidently AI integration.

## Tech Stack

- Python 3.12
- scikit-learn 1.8.0
- FastAPI + Uvicorn
- Pydantic v2
- DVC (artifact versioning)
- Docker

## Data Versioning

Model artifacts are tracked with [DVC](https://dvc.org/). To pull the latest artifact without retraining:

```bash
dvc pull
```
