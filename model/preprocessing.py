"""
preprocessing.py
----------------
Reusable preprocessing and feature engineering pipeline
for the IBM Telco Customer Churn dataset.

All transformations are reproducible and target-leakage free.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


# ──────────────────────────────────────────────
# 1.  Raw data loading
# ──────────────────────────────────────────────

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERICAL_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

ENGINEERED_COLS = [
    "AvgMonthlyCharge",
    "TotalServices",
    "HasMultipleServices",
    "TenureGroup",
    "ContractRisk",
]


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the IBM Telco CSV without modifying raw content."""
    df = pd.read_csv(path)
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Run basic validation checks and return a report dict.
    Does NOT raise errors — returns a report so the caller can decide.
    """
    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_customerID": int(df["customerID"].isna().sum()),
        "totalCharges_whitespace": int(
            (df["TotalCharges"].astype(str).str.strip() == "").sum()
        ),
        "churn_distribution": df["Churn"].value_counts().to_dict(),
    }
    return report


# ──────────────────────────────────────────────
# 2.  Cleaning
# ──────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe.
    Returns a copy — raw input is never modified.
    """
    df = df.copy()

    # Fix TotalCharges: whitespace → NaN → numeric
    df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # The 11 whitespace records all have tenure == 0 and TotalCharges == NaN.
    # Impute TotalCharges as 0 for zero-tenure customers (they haven't been billed yet).
    df.loc[df["tenure"] == 0, "TotalCharges"] = df.loc[
        df["tenure"] == 0, "TotalCharges"
    ].fillna(0.0)

    # Any remaining NaN in TotalCharges → median imputation (safe fallback)
    median_tc = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)

    # Encode target: Yes → 1, No → 0
    df["Churn"] = (df["Churn"].str.strip() == "Yes").astype(int)

    # SeniorCitizen is already 0/1 int — keep it; cast to str for uniform handling
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Remove duplicates
    df = df.drop_duplicates(subset=["customerID"])

    return df


# ──────────────────────────────────────────────
# 3.  Feature engineering
# ──────────────────────────────────────────────

# Service columns used to count subscribed add-ons
SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Positive values that indicate the service is active
ACTIVE_VALUES = {"Yes", "Fiber optic", "DSL"}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the cleaned dataframe.
    All features are derived from existing columns only.
    """
    df = df.copy()

    # AvgMonthlyCharge: protects against tenure=0 division
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)

    # TotalServices: how many services is the customer subscribed to
    df["TotalServices"] = df[SERVICE_COLS].apply(
        lambda row: sum(str(v) in ACTIVE_VALUES for v in row), axis=1
    )

    # HasMultipleServices: boolean flag (2+ services)
    df["HasMultipleServices"] = (df["TotalServices"] >= 2).astype(int)

    # TenureGroup: ordinal buckets
    def tenure_bucket(t):
        if t <= 12:
            return 0   # New
        elif t <= 24:
            return 1   # Growing
        elif t <= 48:
            return 2   # Established
        else:
            return 3   # Loyal

    df["TenureGroup"] = df["tenure"].apply(tenure_bucket)

    # ContractRisk: month-to-month is highest churn risk
    contract_risk_map = {
        "Month-to-month": 2,
        "One year": 1,
        "Two year": 0,
    }
    df["ContractRisk"] = df["Contract"].map(contract_risk_map).fillna(1).astype(int)

    return df


# ──────────────────────────────────────────────
# 4.  Sklearn preprocessing pipeline
# ──────────────────────────────────────────────

# After engineer_features we treat engineered numerics together with originals
ALL_NUMERICAL_COLS = NUMERICAL_COLS + ENGINEERED_COLS

# SeniorCitizen is already stored as str "0"/"1" so it fits in categorical
ALL_CATEGORICAL_COLS = CATEGORICAL_COLS


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Returns a fitted-ready ColumnTransformer.

    Numerical:  median imputation → standard scaling
    Categorical: most-frequent imputation → ordinal encoding
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, ALL_NUMERICAL_COLS),
            ("cat", categorical_pipeline, ALL_CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Return ordered feature names matching the preprocessor output columns.
    """
    num_names = ALL_NUMERICAL_COLS
    cat_names = ALL_CATEGORICAL_COLS
    return num_names + cat_names


# ──────────────────────────────────────────────
# 5.  Full data preparation helper
# ──────────────────────────────────────────────

def prepare_dataset(path: str = DATA_PATH):
    """
    End-to-end: load → validate → clean → feature engineer.

    Returns
    -------
    df_clean : pd.DataFrame   (cleaned + engineered, includes 'customerID' and 'Churn')
    X : pd.DataFrame          (feature matrix, no customerID, no Churn)
    y : pd.Series             (binary target)
    feature_names : list[str] (ordered column names used in X)
    validation_report : dict
    """
    df_raw = load_raw_data(path)
    validation_report = validate_data(df_raw)
    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean)

    feature_cols = ALL_NUMERICAL_COLS + ALL_CATEGORICAL_COLS
    X = df_feat[feature_cols].copy()
    y = df_feat["Churn"].copy()

    return df_feat, X, y, feature_cols, validation_report
