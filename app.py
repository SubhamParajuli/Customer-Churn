from flask import Flask, request, jsonify
import joblib
import pandas as pd

app=Flask(__name__)

model=joblib.load("models/full_pipeline.pkl")

@app.route('/')
def home():
    return "Customer Churn API Running"

@app.route("/predict", methods=['POST'])
def predict():
    data = request.json

    df = pd.DataFrame([data])

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return jsonify({
        "prediction":
        "Churn" if pred == 1 else "No Churn",

        "probability":
        round(float(prob), 4)
    })

if __name__ == "__main__":
    app.run(debug=True)