import gradio as gr
import joblib
import pandas as pd

model = joblib.load("models/titanic_pipeline.joblib")

def predict(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    df = pd.DataFrame([{
        "Pclass": int(Pclass),
        "Sex": Sex,
        "Age": float(Age),
        "SibSp": int(SibSp),
        "Parch": int(Parch),
        "Fare": float(Fare),
        "Embarked": Embarked
    }])
    y_proba = model.predict_proba(df)[0][1]
    y_pred = int(model.predict(df)[0])
    return {"Prediction": y_pred, "Probability": round(y_proba, 4)}

gr.Interface(
    fn=predict,
    inputs=["dropdown", "radio", "number", "number", "number", "number", "radio"],
    outputs="json",
    examples=[
        [3, "male", 22, 1, 0, 7.25, "S"],
        [1, "female", 38, 1, 0, 71.2833, "C"]
    ]
).launch()
