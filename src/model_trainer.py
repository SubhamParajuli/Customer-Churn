import os
import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt
from src.logger import logger

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



class ModelTrainer:

    def __init__(self, save_path="models"):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        logger.info("Initializing ModelTrainer")

        self.search_spaces = {
            "RandomForestClassifier": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5]
                }
            },

            "XGBClassifier": {
                "model": XGBClassifier(
                    random_state=42,
                    eval_metric="logloss"
                ),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5],
                    "learning_rate": [0.01, 0.1]
                }
            }
        }

    def calculate_metrics(self, y_test, y_pred):

        logger.info("Calculating evaluation metrics")

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1_Score": f1_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_pred),
            "Report": classification_report(y_test, y_pred)
        }

        return metrics


    def train_single_model(self, name, config, X_train, y_train):

        logger.info(f"Training Started: {name}")

        search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=5,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            random_state=42
        )

        search.fit(X_train, y_train)

        logger.info(f"Training Completed: {name}")
        logger.info(f"Best Params ({name}) = {search.best_params_}")

        return search.best_estimator_, search.best_params_


    def train_all_models(self, X_train, y_train, X_test, y_test):

        logger.info("Training all models started")

        results = []

        for name, config in self.search_spaces.items():

            try:
                model, params = self.train_single_model(
                    name,
                    config,
                    X_train,
                    y_train
                )

                y_pred = model.predict(X_test)

                metrics = self.calculate_metrics(
                    y_test,
                    y_pred
                )

                logger.info(
                    f"{name} | "
                    f"Accuracy={metrics['Accuracy']:.4f} | "
                    f"F1={metrics['F1_Score']:.4f} | "
                    f"ROC_AUC={metrics['ROC_AUC']:.4f}"
                )

                results.append({
                    "Model_Name": name,
                    "Accuracy": metrics["Accuracy"],
                    "F1_Score": metrics["F1_Score"],
                    "ROC_AUC": metrics["ROC_AUC"],
                    "Best_Params": params,
                    "Model_Object": model
                })

            except Exception as e:
                logger.error(f"Error while training {name}: {str(e)}")

        df = pd.DataFrame(results)

        df = df.sort_values(
            by=["ROC_AUC", "F1_Score"],
            ascending=False
        ).reset_index(drop=True)

        self.results_df = df
        self.best_model = df.loc[0, "Model_Object"]
        self.best_name = df.loc[0, "Model_Name"]

        logger.info(f"Best Model Selected: {self.best_name}")

        return df

    def save_best_model(self, filename="best_model.pkl"):

        path = os.path.join(self.save_path, filename)

        joblib.dump(self.best_model, path)

        logger.info(f"Best model saved at {path}")


    def plot_confusion_matrix(self, X_test, y_test):

        logger.info("Generating confusion matrix")

        y_pred = self.best_model.predict(X_test)

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred
        )

        plt.title(f"Confusion Matrix - {self.best_name}")
        plt.show()


    def feature_importance(self, X_train):

        logger.info("Generating feature importance")

        if hasattr(self.best_model, "feature_importances_"):

            imp = pd.DataFrame({
                "Feature": X_train.columns,
                "Importance":
                self.best_model.feature_importances_
            })

            imp = imp.sort_values(
                by="Importance",
                ascending=False
            )

            print(imp.head(10))

            logger.info("Feature importance displayed")

        else:
            logger.warning(
                f"{self.best_name} does not support feature importance"
            )