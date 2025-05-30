import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import joblib

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    """load CSV and return pandas DataFrame."""
    return pd.read_csv(path)

def main():
    # 1. Load data
    df = load_data(os.path.join(os.path.dirname(__file__), "..", "data", "train.csv"))
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # 2. select columns and preprocess
    numeric_features = ["Age", "Fare", "SibSp", "Parch"]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_features = ["Pclass", "Sex", "Embarked"]
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    # 3. build du pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, n_jobs=-1)),
    ])

    # 4. train
    pipeline.fit(X, y)

    # 5. save model
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "titanic_pipeline.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model trained and saved here {model_path}")

if __name__ == "__main__":
    main()
