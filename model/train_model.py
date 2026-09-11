# -*- coding: utf-8 -*-
"""
train_model.py
--------------
Train Logistic Regression and XGBoost models on the IBM Telco
Customer Churn dataset.

Run this script once to produce `model/churn_model.pkl`.

Usage:
    python model/train_model.py
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.preprocessing import (
    build_preprocessing_pipeline,
    get_feature_names,
    prepare_dataset,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

RANDOM_STATE = 42
TEST_SIZE = 0.20
MODEL_DIR = os.path.join(os.path.dirname(__file__))

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
)


# ──────────────────────────────────────────────
# 1.  Data preparation
# ──────────────────────────────────────────────

def load_and_split():
    print("[1/5] Loading and preparing dataset...")
    df_feat, X, y, feature_cols, val_report = prepare_dataset(DATA_PATH)

    print(f"      Rows: {val_report['rows']}  |  Duplicates: {val_report['duplicates']}")
    print(f"      TotalCharges whitespace fixed: {val_report['totalCharges_whitespace']}")
    print(f"      Churn distribution: {val_report['churn_distribution']}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"      Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_cols, df_feat


# ──────────────────────────────────────────────
# 2.  Model training
# ──────────────────────────────────────────────

def build_lr_pipeline():
    preprocessor = build_preprocessing_pipeline()
    lr = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        C=0.1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", lr)])


def build_xgb_pipeline():
    preprocessor = build_preprocessing_pipeline()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=4607 / 2436,   # handle class imbalance
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", xgb)])


# ──────────────────────────────────────────────
# 3.  Evaluation
# ──────────────────────────────────────────────

def evaluate_model(name: str, pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True
        ),
    }
    print(f"\n  {name}")
    print(f"    Accuracy : {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall   : {metrics['recall']:.4f}")
    print(f"    F1-score : {metrics['f1']:.4f}")
    print(f"    ROC-AUC  : {metrics['roc_auc']:.4f}")
    return metrics


# ──────────────────────────────────────────────
# 4.  SHAP values (computed here, saved alongside model)
# ──────────────────────────────────────────────

def compute_shap_values(xgb_pipeline, X_train, feature_names):
    """
    Compute SHAP values on the *preprocessed* training data.
    Returns a dict with global feature importance.
    """
    import shap

    print("[4/5] Computing SHAP values...")

    # Transform training data through the preprocessor step only
    preprocessor = xgb_pipeline.named_steps["preprocessor"]
    X_train_transformed = preprocessor.transform(X_train)

    classifier = xgb_pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_train_transformed)

    # Global importance: mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True,
    )

    global_importance = [
        {"feature": f, "importance": round(float(v), 5)}
        for f, v in importance_df
    ]

    return {
        "global_importance": global_importance,
        "explainer": explainer,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }


# ──────────────────────────────────────────────
# 5.  Save artifacts
# ──────────────────────────────────────────────

def save_artifacts(xgb_pipeline, shap_info, lr_metrics, xgb_metrics, df_feat):
    print("[5/5] Saving artifacts...")

    # Save the full XGBoost pipeline
    joblib.dump(xgb_pipeline, os.path.join(MODEL_DIR, "churn_model.pkl"))

    # Save SHAP explainer + preprocessor separately for the dashboard
    joblib.dump(
        {
            "explainer": shap_info["explainer"],
            "preprocessor": shap_info["preprocessor"],
            "feature_names": shap_info["feature_names"],
        },
        os.path.join(MODEL_DIR, "shap_explainer.pkl"),
    )

    # Save metrics as JSON
    metrics_output = {
        "logistic_regression": lr_metrics,
        "xgboost": xgb_metrics,
        "best_model": "xgboost" if xgb_metrics["roc_auc"] >= lr_metrics["roc_auc"] else "logistic_regression",
    }
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics_output, f, indent=2)

    # Save processed predictions for dashboard (all customers)
    print("      Computing predictions for all customers...")

    # Load a fresh pipeline to get predictions on whole dataset
    df_feat = df_feat.copy()
    from model.preprocessing import ALL_CATEGORICAL_COLS, ALL_NUMERICAL_COLS

    feature_cols = ALL_NUMERICAL_COLS + ALL_CATEGORICAL_COLS
    X_all = df_feat[feature_cols]
    y_prob_all = xgb_pipeline.predict_proba(X_all)[:, 1]
    y_pred_all = xgb_pipeline.predict(X_all)

    df_feat["churn_probability"] = np.round(y_prob_all, 4)
    df_feat["prediction"] = y_pred_all

    def risk_level(p):
        if p < 0.30:
            return "LOW"
        elif p < 0.60:
            return "MEDIUM"
        elif p < 0.80:
            return "HIGH"
        else:
            return "CRITICAL"

    df_feat["risk_level"] = df_feat["churn_probability"].apply(risk_level)

    # Save predictions CSV
    predictions_path = os.path.join(MODEL_DIR, "predictions.csv")
    df_feat[
        [
            "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
            "AvgMonthlyCharge", "TotalServices", "HasMultipleServices",
            "TenureGroup", "ContractRisk",
            "churn_probability", "risk_level", "prediction",
        ]
    ].to_csv(predictions_path, index=False)
    print(f"      Predictions saved: {predictions_path}")

    print("\n  Artifacts saved:")
    print(f"    model/churn_model.pkl")
    print(f"    model/shap_explainer.pkl")
    print(f"    model/model_metrics.json")
    print(f"    model/predictions.csv")

    return metrics_output


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Customer Churn Prediction — Model Training")
    print("=" * 55)

    X_train, X_test, y_train, y_test, feature_cols, df_feat = load_and_split()

    # Train Logistic Regression
    print("\n[2/5] Training models...")
    lr_pipeline = build_lr_pipeline()
    lr_pipeline.fit(X_train, y_train)

    # Train XGBoost
    xgb_pipeline = build_xgb_pipeline()
    xgb_pipeline.fit(X_train, y_train)

    # Evaluate
    print("\n[3/5] Evaluating models...")
    lr_metrics = evaluate_model("Logistic Regression", lr_pipeline, X_test, y_test)
    xgb_metrics = evaluate_model("XGBoost", xgb_pipeline, X_test, y_test)

    best = "XGBoost" if xgb_metrics["roc_auc"] >= lr_metrics["roc_auc"] else "Logistic Regression"
    best_auc = max(xgb_metrics['roc_auc'], lr_metrics['roc_auc'])
    print(f"\n  [OK] Best model: {best} (ROC-AUC = {best_auc:.4f})")

    # SHAP (only for XGBoost — TreeExplainer)
    shap_info = compute_shap_values(xgb_pipeline, X_train, feature_cols)

    # Save everything
    metrics_output = save_artifacts(xgb_pipeline, shap_info, lr_metrics, xgb_metrics, df_feat)

    print("\n" + "=" * 55)
    print("  Training complete! Run the dashboard with:")
    print("  streamlit run app.py")
    print("=" * 55)



if __name__ == "__main__":
    main()
