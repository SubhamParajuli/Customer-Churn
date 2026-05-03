# Customer Churn Prediction

This project predicts whether a telecom customer is likely to churn using machine learning. It includes an end-to-end training pipeline, saved model artifacts, a Flask prediction API, and a Streamlit dashboard for entering customer details through a web interface.

## Features

- Loads customer churn data from Excel
- Cleans duplicates, missing values, and unused columns
- Encodes categorical features and scales numeric features
- Trains and compares Random Forest and XGBoost classifiers
- Saves the best full prediction pipeline with preprocessing included
- Provides a Flask API for churn predictions
- Provides a Streamlit dashboard that calls the Flask API

## Project Structure

```text
Customer Churn/
|-- app.py                    # Flask API for predictions
|-- dashboard.py              # Streamlit dashboard
|-- main.py                   # Training and sample prediction pipeline
|-- requirements.txt          # Python dependencies
|-- data/
|   `-- customer_churn.xlsx   # Source dataset
|-- logs/                     # Application and training logs
|-- models/
|   |-- best_model.pkl
|   |-- full_pipeline.pkl
|   `-- preprocessor.pkl
|-- notebook/
|   `-- churn_predict.ipynb   # Notebook exploration
|-- reports/
|   `-- model_results.csv     # Model comparison results
`-- src/
    |-- data_loader.py
    |-- logger.py
    |-- model_trainer.py
    |-- predict_pipeline.py
    `-- preprocess.py
```

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- XGBoost
- Flask
- Streamlit
- joblib
- matplotlib, seaborn
- openpyxl

## Setup

Create and activate a virtual environment:

```bash
python -m venv myenv
myenv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run the main training pipeline:

```bash
python main.py
```

This will:

- Load `data/customer_churn.xlsx`
- Preprocess the dataset
- Train Random Forest and XGBoost models
- Save model comparison results to `reports/model_results.csv`
- Save the full pipeline to `models/full_pipeline.pkl`
- Run a sample prediction

## Model Results

Current saved results:

| Model | Accuracy | F1 Score | ROC AUC |
| --- | ---: | ---: | ---: |
| XGBClassifier | 0.7915 | 0.5802 | 0.7119 |
| RandomForestClassifier | 0.7986 | 0.5703 | 0.7042 |

The pipeline selects the best model based on ROC AUC and F1 score. In the current report, XGBoost is ranked first.

## Run the Flask API

Start the API server:

```bash
python app.py
```

The API will run at:

```text
http://127.0.0.1:5000
```

Health check:

```text
GET /
```

Prediction endpoint:

```text
POST /predict
```

Example JSON payload:

```json
{
  "Gender": "Male",
  "Senior Citizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "Tenure Months": 12,
  "Phone Service": "Yes",
  "Multiple Lines": "No",
  "Internet Service": "Fiber optic",
  "Online Security": "No",
  "Online Backup": "Yes",
  "Device Protection": "No",
  "Tech Support": "No",
  "Streaming TV": "Yes",
  "Streaming Movies": "Yes",
  "Contract": "Month-to-month",
  "Paperless Billing": "Yes",
  "Payment Method": "Electronic check",
  "Monthly Charges": 79.5,
  "Total Charges": 954.0,
  "CLTV": 3200
}
```

Example response:

```json
{
  "prediction": "Churn",
  "probability": 0.7345
}
```

## Run the Streamlit Dashboard

Start the Flask API first:

```bash
python app.py
```

In a second terminal, start the Streamlit app:

```bash
streamlit run dashboard.py
```

The dashboard collects customer details, sends them to the Flask API, and displays the churn prediction with probability.

## Input Features

The model expects these fields:

- `Gender`
- `Senior Citizen`
- `Partner`
- `Dependents`
- `Tenure Months`
- `Phone Service`
- `Multiple Lines`
- `Internet Service`
- `Online Security`
- `Online Backup`
- `Device Protection`
- `Tech Support`
- `Streaming TV`
- `Streaming Movies`
- `Contract`
- `Paperless Billing`
- `Payment Method`
- `Monthly Charges`
- `Total Charges`
- `CLTV`

## Outputs

The prediction pipeline returns:

- `prediction`: `Churn` or `No Churn`
- `probability`: probability that the customer will churn

## Notes

- `models/full_pipeline.pkl` is required by the Flask API.
- The Streamlit dashboard expects the Flask API to be running at `http://127.0.0.1:5000`.
- Training logs are written to the `logs/` folder.
- Generated reports are stored in the `reports/` folder.


