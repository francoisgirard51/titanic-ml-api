# Titanic ML API – Predicting Passenger Survival

[![Docker Pulls](https://img.shields.io/docker/pulls/francoisgirard51/titanic-ml-api)](https://hub.docker.com/r/francoisgirard51/titanic-ml-api)
![CI](https://github.com/francoisgirard51/titanic-ml-api/actions/workflows/test.yml/badge.svg)
[![HF Space](https://img.shields.io/badge/🤗%20View%20on%20Hugging%20Face-blue)](https://huggingface.co/spaces/FrancoisGirard51/titanic-survival-predictor)


A Dockerized FastAPI microservice that predicts Titanic passenger survival using a custom-built scikit-learn pipeline. Designed for learning, deployment, and demonstration purposes.

---

## ✅ Key Features
- Train: Custom scikit-learn pipeline trained on [Kaggle's Titanic dataset](https://www.kaggle.com/competitions/titanic)
- Evaluate: Metrics include accuracy, precision, recall, and ROC AUC
- Serve: FastAPI REST endpoint for real-time predictions
- Documented: Interactive API docs auto-generated at /docs
- Dockerized: Easy build/run anywhere
- Tested: Unit and integration tests via pytest + httpx

---

## 📦 Project Structure

```
titanic-ml-api/
├── data/                 # Raw data: train.csv, test.csv
├── models/               # Serialized model: titanic_pipeline.joblib
├── notebooks/            # EDA notebook
├── src/                  # Application code
│   ├── api.py            # FastAPI app
│   ├── data.py           # Data loading utils
│   ├── features.py       # Feature engineering pipeline
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── __init__.py
├── tests/                # Tests
│   └── test_api.py
├── requirements.txt      # Python dependencies
├── Dockerfile            # Containerization
├── .gitignore            # Ignored files
├── .dockerignore         # Docker exclusions
└── README.md
```

---

## 🛠️ Prerequisites
- Python 3.10+
- Docker 20+
- (Optional) virtualenv or venv for local setup

---

## 💻 Local Setup

1. Clone repo
`git clone https://github.com/francoisgirard51/titanic-ml-api.git`
`cd titanic-ml-api`

2. Create & activate virtualenv
`python3 -m venv .venv`
`source .venv/bin/activate`

3. Install dependencies
`pip install --upgrade pip`
`pip install -r requirements.txt`

4. (Optional) Train & evaluate
`python src/train.py`
`python src/evaluate.py`

5. Launch the API
`uvicorn src.api:app --reload --host 127.0.0.1 --port 8000`

6. Open your browser at http://127.0.0.1:8000/docs

---

## ⚙️ Makefile
Run `make build`, `make run`, `make predict` etc. for simplified commands.

---

## 🐳 Docker Usage

1. Build image
`docker build -t titanic-api .`

2. Run container
`docker run -p 8000:8000 francoisgirard51/titanic-ml-api:latest`

3. Push to registry (after login)
`docker push francoisgirard51/titanic-ml-api:latest`

---

## 🧪 Testing
`pytest tests/`

---

## 🎯 Prediction Endpoint

Sample curl usage:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
      }'

```

Response:

```
{
  "prediction": 0,
  "probability": 0.0878
}
```

## 📝 License

- **Code**: Licensed under the [MIT License](https://opensource.org/licenses/MIT).
- **Data**: Sourced from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic), made available under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
