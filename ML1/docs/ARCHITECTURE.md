"""
ARCHITECTURE DIAGRAM

Data Pipeline with Multi-Dataset Support
=========================================

INPUT DATA
==========
transactions.csv          client_activity.csv
(fraud labels)            (fraud labels)
    |                           |
    v                           v
  ┌─────────────────────────────────────────┐
  │         src/preprocess.py                 │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │ Transactions │  │ Client Activity  │ │
│  │  Features    │  │  Features        │ │
│  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────┘
    |                           |
    v                           v
  processed_transactions.parquet  processed_client_activity.parquet


TRAINING
========
    processed_transactions.parquet          processed_client_activity.parquet
              |                                        |
              v                                        v
    ┌──────────────────────────┐      ┌──────────────────────────┐
    │  src/train.py            │      │  src/train.py            │
    │ (--dataset transactions) │      │ (--dataset client_activity)
    │                          │      │                          │
    │ ┌────────────────────┐   │      │ ┌────────────────────┐   │
    │ │ Train/Val/Test    │   │      │ │ Train/Val/Test    │   │
    │ │ Time-based Split  │   │      │ │ Time-based Split  │   │
    │ └────────────────────┘   │      │ └────────────────────┘   │
    │          |                │      │          |                │
    │ ┌────────v────────────┐   │      │ ┌────────v────────────┐   │
    │ │ Ensemble/LightGBM   │   │      │ │ Ensemble/LightGBM   │   │
    │ │ - LightGBM          │   │      │ │ - LightGBM          │   │
    │ │ - XGBoost (opt)     │   │      │ │ - XGBoost (opt)     │   │
    │ │ - CatBoost (opt)    │   │      │ │ - CatBoost (opt)    │   │
    │ └────────────────────┘   │      │ └────────────────────┘   │
    └──────────────────────────┘      └──────────────────────────┘
              |                                        |
              v                                        v
    ┌──────────────────────────┐      ┌──────────────────────────┐
    │  models/transactions/    │      │  models/client_activity/ │
    │  ├─ lightgbm_model.txt   │      │  ├─ lightgbm_model.txt   │
    │  ├─ xgboost_model.json   │      │  ├─ xgboost_model.json   │
    │  ├─ catboost_model.cbm   │      │  ├─ catboost_model.cbm   │
    │  ├─ model_meta.pkl       │      │  ├─ model_meta.pkl       │
    │  ├─ shap_background.pkl  │      │  ├─ shap_background.pkl  │
    │  └─ feature_importance   │      │  └─ feature_importance   │
    └──────────────────────────┘      └──────────────────────────┘


INFERENCE
=========
  src/infer_service.py
        ┌──────────────────────────────┐
        │  Load Both Models on Startup  │
        │  - transactions model        │
        │  - client_activity model     │
        └──────────────────────────────┘
                    |
        ┌───────────┼───────────┐
        |           |           |
        v           v           v
    [/predict/   [/predict/   [/predict/
     transaction] client_    combined]
                 activity]
        |           |           |
        └───────────┼───────────┘
                    v
        ┌──────────────────────────────┐
        │   Response with:              │
        │   - Fraud probability         │
        │   - Threshold                 │
        │   - Action (allow/challenge/  │
        │     block)                    │
        │   - SHAP explanations         │
        └──────────────────────────────┘


DATA FLOW EXAMPLE
================

Input (Transaction):
  {
    "amount": 50000.0,
    "timestamp": "2025-11-28T14:30:00",
    "src_account_id": "user_123",
    "beneficiary_id": "benef_456"
  }
        |
        v
    /predict/transaction
        |
        v
    Load models/transactions/ model
        |
        v
    compute_realtime_features() -> create feature vector
        |
        v
    model.predict() -> fraud probability
        |
        v
    SHAP explanation -> top features contributing to prediction
        |
        v
    Output:
  {
    "probability": 0.35,
    "threshold": 0.45,
    "action": "allow",
    "explanations": [
      {"feature": "amount", "shap_value": 0.15},
      {"feature": "hour", "shap_value": -0.05},
      {"feature": "is_new_beneficiary", "shap_value": 0.12}
    ]
  }


FEATURE SETS
============

TRANSACTIONS Features:
├─ Temporal: hour, dow, is_weekend, is_night, is_business_hours
├─ Amount: amount, log_amount, amount_squared, amount_sqrt
├─ Beneficiary: beneficiary_unique_users, beneficiary_tx_count, is_new_beneficiary
├─ User Aggregates: amount_count, amount_mean, amount_std, amount_sum, etc.
├─ Rolling Windows: amount_rolling_mean, amount_rolling_std, amount_rolling_count
└─ Interactions: amount_per_user_mean, amount_per_user_std, hour_diff_from_mean

CLIENT_ACTIVITY Features:
├─ Temporal: hour, dow, is_weekend, is_night, is_business_hours
├─ Login Stats: logins_last_7_days, logins_last_30_days, login_frequency_*
├─ Device Info: monthly_os_changes, monthly_phone_model_changes
├─ Intervals: avg_login_interval_30d, std_login_interval_30d
├─ Statistical: ewm_login_interval_7d, burstiness_login_interval, fano_factor
└─ Transformations: log_logins_7d, log_logins_30d, etc.


MODEL SELECTION
===============

Threshold Strategies:
  1. precision  -> max recall @ desired precision
  2. f1         -> max F1-score
  3. recall     -> max recall @ min precision
  4. balanced   -> balance precision/recall

Example:
  python src/train.py --threshold_strategy recall --min_recall 0.5
  -> prioritize catching frauds (recall) over false positives
  

DEPLOYMENT
==========

1. Preprocess Data:
  ./scripts/run_preprocess.sh  # or: python src/preprocess.py
   
2. Train Models:
  ./scripts/run_train.sh both --ensemble  # or: python src/train.py --dataset both --ensemble
   
3. Start API Server:
  ./scripts/run_service.sh  # or: python src/infer_service.py
   
4. Make Predictions:
   - Via HTTP (curl, requests, etc.)
   - Via Python requests library
   - Via integrated test script (test_api.py)

Server runs on http://localhost:8000
API docs available at http://localhost:8000/docs
"""

# This file serves as visual documentation of the system architecture.
# For actual code, see: preprocess.py, train.py, infer_service.py
