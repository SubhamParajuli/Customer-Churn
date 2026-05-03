# src/predict_pipeline.py

import os
import joblib
import pandas as pd

from src.logger import logger


class PredictionPipeline:
    """
    End-to-End Prediction Pipeline

    Steps:
    1. Load trained model
    2. Accept user input
    3. Convert to DataFrame
    4. Predict churn
    5. Return label + probability
    """

    def __init__(self, model_path="models/best_model.pkl"):

        try:
            logger.info("Initializing Prediction Pipeline")

            self.model_path = model_path

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model not found at {model_path}"
                )

            self.model = joblib.load(model_path)

            logger.info(
                f"Model Loaded Successfully: {model_path}"
            )

        except Exception as e:
            logger.error(
                f"Prediction Pipeline Init Failed: {str(e)}"
            )
            raise

    def prepare_input(self, data: dict):

        try:
            logger.info("Preparing input data")

            df = pd.DataFrame([data])

            logger.info(
                f"Input Shape: {df.shape}"
            )

            return df

        except Exception as e:
            logger.error(
                f"Input Preparation Failed: {str(e)}"
            )
            raise


    def predict(self, data: dict):

        try:
            logger.info("Prediction Started")

            input_df = self.prepare_input(data)

            pred = self.model.predict(input_df)[0]

            result = "Churn" if pred == 1 else "No Churn"

            logger.info(
                f"Prediction Completed: {result}"
            )

            return result

        except Exception as e:
            logger.error(
                f"Prediction Failed: {str(e)}"
            )
            raise


    def predict_proba(self, data: dict):

        try:
            logger.info(
                "Probability Prediction Started"
            )

            input_df = self.prepare_input(data)

            proba = self.model.predict_proba(
                input_df
            )[0][1]

            logger.info(
                f"Churn Probability: {proba:.4f}"
            )

            return round(float(proba), 4)

        except Exception as e:
            logger.error(
                f"Probability Prediction Failed: {str(e)}"
            )
            raise


    def predict_full(self, data: dict):

        try:
            pred = self.predict(data)

            prob = self.predict_proba(data)

            response = {
                "Prediction": pred,
                "Churn_Probability": prob
            }

            logger.info(
                f"Final Response Generated"
            )

            return response

        except Exception as e:
            logger.error(
                f"Full Prediction Failed: {str(e)}"
            )
        