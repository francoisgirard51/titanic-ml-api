# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

@pytest.mark.parametrize("payload,expected_type", [
    ({"Pclass":3,"Sex":"male","Age":22,"SibSp":1,"Parch":0,"Fare":7.25,"Embarked":"S"}, dict),
])
def test_predict_smoke(payload, expected_type):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    # should contain both keys with correct types
    assert "prediction" in body and isinstance(body["prediction"], int)
    assert "probability" in body and isinstance(body["probability"], float)
