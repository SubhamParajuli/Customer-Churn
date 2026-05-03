# main.py

import os
import joblib

from sklearn.pipeline import Pipeline

from src.logger import logger
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model_trainer import ModelTrainer
from src.predict_pipeline import PredictionPipeline


def train_pipeline():
    """
    End-to-End Training Pipeline

    Steps:
    1. Load Dataset
    2. Preprocess Data
    3. Train Multiple Models
    4. Select Best Model
    5. Save Full Pipeline
    6. Plot Evaluation Metrics
    """

    try:
        logger.info("=" * 70)
        logger.info("TRAINING PIPELINE STARTED")
        logger.info("=" * 70)

        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

        data_path = "data/customer_churn.xlsx"

        logger.info(f"Loading Dataset: {data_path}")

        df = load_data(data_path)

        logger.info(
            f"Dataset Loaded Successfully | Shape = {df.shape}"
        )


        (
            preprocessor,
            X_train,
            X_test,
            y_train,
            y_test
        ) = preprocess_data(df)

        logger.info(
            "Preprocessing Completed Successfully"
        )

        trainer = ModelTrainer()

        results_df = trainer.train_all_models(
            X_train,
            y_train,
            X_test,
            y_test
        )

        logger.info(
            "All Models Trained Successfully"
        )


        print("\nMODEL COMPARISON\n")

        print(
            results_df[
                [
                    "Model_Name",
                    "Accuracy",
                    "F1_Score",
                    "ROC_AUC"
                ]
            ]
        )

        results_df.to_csv(
            "reports/model_results.csv",
            index=False
        )

        logger.info(
            "Model results saved to reports/"
        )

        best_model = trainer.best_model

        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", best_model)
        ])

        joblib.dump(
            full_pipeline,
            "models/full_pipeline.pkl"
        )

        logger.info(
            f"Best Model Saved: {trainer.best_name}"
        )

        logger.info(
            "Full Pipeline Saved Successfully"
        )


        trainer.plot_confusion_matrix(
            X_test,
            y_test
        )

        logger.info(
            "Confusion Matrix Generated"
        )

        logger.info("=" * 70)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("=" * 70)

    except Exception as e:

        logger.error(
            f"Training Pipeline Failed: {str(e)}"
        )

        raise



def prediction_pipeline():
    """
    Load Saved Pipeline
    Predict New Customer Churn
    """

    try:
        logger.info("=" * 70)
        logger.info("PREDICTION PIPELINE STARTED")
        logger.info("=" * 70)

        pipeline = PredictionPipeline(
            model_path="models/full_pipeline.pkl"
        )


        sample_customer = {

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


        result = pipeline.predict_full(
            sample_customer
        )

        print("\nPREDICTION RESULT\n")
        print(result)

        logger.info(
            f"Prediction Output: {result}"
        )

        logger.info("=" * 70)
        logger.info("PREDICTION PIPELINE COMPLETED")
        logger.info("=" * 70)

    except Exception as e:

        logger.error(
            f"Prediction Pipeline Failed: {str(e)}"
        )

        raise


if __name__ == "__main__":

    train_pipeline()
    prediction_pipeline()