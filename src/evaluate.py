# src/evaluate.py
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import joblib

def load_and_split(path: str, test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    # our train.csv *does* have the Survived column
    y = df["Survived"]
    X = df.drop("Survived", axis=1)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def main():
    # 1. Load & split
    train_path = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
    X_train, X_test, y_train, y_test = load_and_split(train_path)

    # 2. Load your saved pipeline
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "titanic_pipeline.joblib",
    )
    pipeline = joblib.load(model_path)

    # 3. Predict on hold-out
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 4. Compute metrics
    print("Evaluation on 20% hold-out of train.csv\n" + "-"*40)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
    print(f"ROC AUC  : {roc_auc_score(y_test, y_proba):.3f}")

if __name__ == "__main__":
    main()
