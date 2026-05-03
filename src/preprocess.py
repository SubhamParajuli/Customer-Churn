import pandas as pd

from src.logger import logger

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


def preprocess_data(data):
    """
    Preprocess customer churn dataset
    Returns:
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test
    """

    try:
        logger.info("Preprocessing Started")

        df = data.copy()

        before_rows = df.shape[0]
        df = df.drop_duplicates()

        after_rows = df.shape[0]

        logger.info(
            f"Duplicates Removed: {before_rows - after_rows}"
        )


        drop_cols = [
            "CustomerID",
            "Count",
            "Country",
            "State",
            "City",
            "Zip Code",
            "Lat Long",
            "Latitude",
            "Longitude",
            "Churn Label",
            "Churn Score",
            "Churn Reason"
        ]

        df = df.drop(
            columns=drop_cols,
            errors="ignore"
        )

        logger.info("Unnecessary columns dropped")


        df["Total Charges"] = pd.to_numeric(
            df["Total Charges"],
            errors="coerce"
        )

        missing_before = df.isnull().sum().sum()

        df = df.dropna()

        logger.info(
            f"Missing values removed: {missing_before}"
        )

        y = df["Churn Value"]

        X = df.drop(
            columns=["Churn Value"], axis=1
        )

        logger.info("Feature and target separated")

        numeric_features = [
            "Tenure Months",
            "Monthly Charges",
            "Total Charges",
            "CLTV"
        ]

        binary_features = [
            "Gender",
            "Senior Citizen",
            "Partner",
            "Dependents",
            "Phone Service",
            "Paperless Billing"
        ]

        nominal_features = [
            "Multiple Lines",
            "Internet Service",
            "Online Security",
            "Online Backup",
            "Device Protection",
            "Tech Support",
            "Streaming TV",
            "Streaming Movies",
            "Contract",
            "Payment Method"
        ]

        preprocessor = ColumnTransformer(
            transformers=[

                (
                    "num",
                    StandardScaler(),
                    numeric_features
                ),

                (
                    "bin",
                    OrdinalEncoder(),
                    binary_features
                ),

                (
                    "nom",
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore"
                    ),
                    nominal_features
                )
            ]
        )

        logger.info(
            "ColumnTransformer created successfully"
        )

        X = preprocessor.fit_transform(X)

        logger.info(
            "Feature transformation completed"
        )

        logger.info(
            "Splitting train/test data (70/30)"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y
        )

        logger.info(
            f"Train Shape: {X_train.shape}"
        )

        logger.info(
            f"Test Shape: {X_test.shape}"
        )

        logger.info("Preprocessing Completed")

        return (
            preprocessor,
            X_train,
            X_test,
            y_train,
            y_test
        )

    except Exception as e:

        logger.error(
            f"Preprocessing Failed: {str(e)}"
        )

        raise