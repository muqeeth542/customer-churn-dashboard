# 📉 Customer Churn Prediction Dashboard

A **student-level machine learning portfolio project** that predicts which customers are likely to leave a telecom subscription service — and explains *why*.

Built with Python, Scikit-learn, XGBoost, SHAP, and Streamlit.

---

## 🎯 Business Problem

Customer churn — when a subscriber cancels their service — is one of the most expensive problems for subscription businesses. Acquiring a new customer typically costs 5-7× more than retaining an existing one.

This dashboard helps a business answer:

> **"Which of our customers are most likely to leave, and what can we do about it?"**

By identifying high-risk customers *before* they cancel, a retention team can prioritise outreach and offer targeted incentives.

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Name** | IBM Telco Customer Churn |
| **Source** | IBM Business Analytics Community |
| **Rows** | 7,043 customers |
| **Columns** | 21 features |
| **Target** | `Churn` (Yes / No) |
| **License** | Apache 2.0 |
| **Nature** | Real-world telecom snapshot |

**⚠️ Limitation:** The dataset is a static cross-sectional snapshot. It does not contain true cancellation timestamps, longitudinal customer journeys, or causal treatment effects. All predictions are pattern-based estimates from this single snapshot.

---

## 🏗️ Architecture

```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
           │
           ▼
  model/preprocessing.py
  ┌─────────────────────────────────────────┐
  │  1. Load raw CSV                        │
  │  2. Fix TotalCharges (11 whitespace)    │
  │  3. Encode target: Yes→1, No→0          │
  │  4. Feature engineering                 │
  │  5. Sklearn ColumnTransformer pipeline  │
  └─────────────────────────────────────────┘
           │
           ▼
  model/train_model.py
  ┌─────────────────────────────────────────┐
  │  Logistic Regression (baseline)         │
  │  XGBoost (primary model)                │
  │  Stratified 80/20 train-test split      │
  │  Evaluation: Accuracy, P, R, F1, AUC    │
  │  SHAP TreeExplainer                     │
  └─────────────────────────────────────────┘
           │
           ▼
  model/ artifacts
  ┌────────────────────┐
  │ churn_model.pkl    │  Full XGBoost pipeline
  │ shap_explainer.pkl │  SHAP explainer + preprocessor
  │ model_metrics.json │  Evaluation metrics
  │ predictions.csv    │  Pre-computed predictions
  └────────────────────┘
           │
           ▼
  app.py — Streamlit Dashboard
  ┌─────────────────────────────────────────┐
  │  Page 1: Overview Dashboard             │
  │  Page 2: Customer Analysis + SHAP       │
  │  Page 3: Live Prediction Form           │
  │  Page 4: Model Performance Metrics      │
  └─────────────────────────────────────────┘
```

---

## ⚙️ Features

### Data Preprocessing
- **TotalCharges** converted from string to float (`11` whitespace records for `tenure=0` fixed → imputed as `0.0`)
- Duplicate removal
- Target encoding: `Yes → 1`, `No → 0`
- Categorical columns: OrdinalEncoder (handles unknown values)
- Numerical columns: StandardScaler with median imputation

### Engineered Features

| Feature | Formula / Logic | Interpretation |
|---|---|---|
| `AvgMonthlyCharge` | `TotalCharges / (tenure + 1)` | Average spend per month (guards division by zero) |
| `TotalServices` | Count of active services | Customers with more services tend to stay |
| `HasMultipleServices` | `TotalServices >= 2` | Binary flag for engagement |
| `TenureGroup` | 0 (0–12mo), 1 (13–24mo), 2 (25–48mo), 3 (49+mo) | Loyalty tier |
| `ContractRisk` | Month-to-month=2, One year=1, Two year=0 | Contract churn risk proxy |

### Models

| Model | Role | Handles Imbalance |
|---|---|---|
| Logistic Regression | Interpretable baseline | `class_weight="balanced"` |
| XGBoost | Primary model | `scale_pos_weight` tuned to class ratio |

### SHAP Explainability
- **Global**: Mean absolute SHAP values across training set — shows top churn drivers
- **Individual**: Per-customer SHAP waterfall — shows exactly why a customer scored high
- ⚠️ SHAP explains the model's decision. It does **not** prove causality.

### Risk Tiers

| Level | Probability | Recommended Action |
|---|---|---|
| LOW | < 30% | Standard engagement |
| MEDIUM | 30–60% | Monitor, soft outreach |
| HIGH | 60–80% | Proactive retention call |
| CRITICAL | > 80% | Urgent intervention |

---

## 🚀 Quick Start

### 1. Clone / navigate to the project
```bash
cd customer-churn-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (one-time)
```bash
python model/train_model.py
```
This generates `model/churn_model.pkl`, `model/shap_explainer.pkl`, `model/model_metrics.json`, and `model/predictions.csv`.

### 4. Launch the dashboard
```bash
streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

---

## 📱 Dashboard Pages

### Page 1: Dashboard
- KPI ribbon: Total Customers, Churned, Churn Rate, High-Risk, Critical Risk, Avg Monthly $
- Charts: Churn vs Stayed donut, Churn by Contract, by Internet Service, by Payment Method
- Churn Probability distribution histogram
- Customer Risk Distribution bar chart
- Sortable High-Risk Customers table

### Page 2: Customer Analysis
- Search customers by ID, filter by risk level and contract type
- Full customer profile panel
- Live churn probability gauge
- SHAP bar chart: individual feature contributions
- Plain-English "Top churn risk drivers" and "Top protective factors"

### Page 3: Live Prediction
- Input form with all 19 original IBM Telco features
- Real-time XGBoost prediction on submission
- Probability, risk level, and verdict displayed
- SHAP explanation for the entered customer

### Page 4: Model Performance
- Side-by-side Logistic Regression vs XGBoost metric cards
- ROC-AUC comparison bar chart
- Confusion matrix heatmap
- Feature importance chart
- Classification report table

---

## 📏 Evaluation Methodology

All metrics are computed on a **held-out 20% stratified test split**.

- Train/test split is stratified on the `Churn` label to preserve the 26%/74% imbalance ratio.
- `random_state=42` ensures full reproducibility.
- No metric is fabricated — all numbers come from `sklearn.metrics`.

---

## ⚠️ Limitations

1. **Static dataset**: The IBM Telco CSV is a snapshot. There are no timestamps, no true event dates, and no longitudinal behavior.
2. **No causal inference**: SHAP explains model behavior — it does not identify what *causes* customers to churn.
3. **No survival analysis**: This project does not predict *when* a customer will churn.
4. **No A/B or uplift**: Intervention effectiveness is not modelled.
5. **Dataset imbalance**: ~26% churn rate; class imbalance handled via `scale_pos_weight`, not oversampling.

---

## 🔮 Possible Future Improvements

- Survival analysis with Kaplan-Meier curves
- Uplift modeling on a randomized control trial dataset
- Real-time database integration
- Customer retention recommendation engine
- Automated retraining pipeline

---

## 📁 Project Structure

```
customer-churn-dashboard/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← IBM Telco dataset
│
├── model/
│   ├── preprocessing.py      ← Cleaning, feature engineering, sklearn pipeline
│   ├── train_model.py        ← Model training, evaluation, SHAP, artifact saving
│   ├── churn_model.pkl       ← Trained XGBoost pipeline (generated)
│   ├── shap_explainer.pkl    ← SHAP explainer artifact (generated)
│   ├── model_metrics.json    ← LR + XGBoost metrics (generated)
│   └── predictions.csv       ← All-customer predictions (generated)
│
├── notebooks/
│   └── analysis.ipynb        ← Exploratory data analysis notebook
│
├── app.py                    ← Streamlit dashboard (4 pages)
├── requirements.txt
└── README.md
```

---

## 📦 Dependencies

```
pandas        Data loading and manipulation
numpy         Numerical operations
scikit-learn  Preprocessing pipelines and Logistic Regression
xgboost       Primary churn classification model
shap          Model explainability (TreeExplainer)
streamlit     Interactive web dashboard
matplotlib    Plotting support
seaborn       Statistical visualizations
joblib        Model artifact serialization
plotly        Interactive charts in Streamlit
```

---

*Built as a portfolio project demonstrating end-to-end ML — from raw data to an explainable, interactive dashboard.*
