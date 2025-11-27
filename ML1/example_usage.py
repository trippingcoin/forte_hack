#!/usr/bin/env python3
"""
example_usage.py
Примеры использования переделанного кода с поддержкой двух датасетов
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ===== EXAMPLE 1: Preprocessing both datasets =====
print("="*70)
print("EXAMPLE 1: Preprocessing Both Datasets")
print("="*70)

print("""
# Run preprocessing for both datasets:
python preprocess.py \\
    --transactions_input data/transactions.csv \\
    --client_activity_input data/client_activity.csv \\
    --transactions_output data/processed_transactions.parquet \\
    --client_activity_output data/processed_client_activity.parquet

# This creates two processed files:
# - data/processed_transactions.parquet (for transaction fraud detection)
# - data/processed_client_activity.parquet (for client activity analysis)
""")

# ===== EXAMPLE 2: Training models for both datasets =====
print("\n" + "="*70)
print("EXAMPLE 2: Training Models for Both Datasets")
print("="*70)

print("""
# Option A: Train ensemble models for both datasets
python train.py \\
    --processed_transactions data/processed_transactions.parquet \\
    --processed_client_activity data/processed_client_activity.parquet \\
    --dataset both \\
    --ensemble \\
    --threshold_strategy precision \\
    --desired_precision 0.7

# Option B: Train only transaction model (single LightGBM)
python train.py \\
    --processed_transactions data/processed_transactions.parquet \\
    --dataset transactions

# Option C: Train only client activity model with recall optimization
python train.py \\
    --processed_client_activity data/processed_client_activity.parquet \\
    --dataset client_activity \\
    --threshold_strategy recall \\
    --min_recall 0.5

# This creates models in:
# - models/transactions/ (contains lightgbm_model.txt, model_meta.pkl, etc.)
# - models/client_activity/ (contains lightgbm_model.txt, model_meta.pkl, etc.)
""")

# ===== EXAMPLE 3: Using the inference service =====
print("\n" + "="*70)
print("EXAMPLE 3: Using the Inference Service")
print("="*70)

print("""
# Start the service
python infer_service.py

# Service runs on http://localhost:8000
# Check available models:
curl http://localhost:8000/health
""")

# ===== EXAMPLE 4: Making predictions =====
print("\n" + "="*70)
print("EXAMPLE 4: Making Predictions via API")
print("="*70)

print("""
# 1. Transaction Fraud Prediction
curl -X POST http://localhost:8000/predict/transaction \\
  -H "Content-Type: application/json" \\
  -d '{
    "transaction": {
      "amount": 50000.0,
      "timestamp": "2025-11-28T14:30:00",
      "src_account_id": "user_123",
      "beneficiary_id": "benef_456"
    }
  }'

Response Example:
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

# 2. Client Activity Analysis
curl -X POST http://localhost:8000/predict/client_activity \\
  -H "Content-Type: application/json" \\
  -d '{
    "activity": {
      "timestamp": "2025-11-28T14:30:00",
      "src_account_id": "user_123",
      "logins_last_7_days": 3,
      "logins_last_30_days": 15,
      "login_frequency_7d": 0.43,
      "login_frequency_30d": 0.5,
      "avg_login_interval_30d": 172800.0
    }
  }'

# 3. Combined Prediction (Both datasets at once)
curl -X POST http://localhost:8000/predict/combined \\
  -H "Content-Type: application/json" \\
  -d '{
    "transaction": {
      "amount": 50000.0,
      "timestamp": "2025-11-28T14:30:00",
      "src_account_id": "user_123",
      "beneficiary_id": "benef_456"
    },
    "activity": {
      "timestamp": "2025-11-28T14:30:00",
      "src_account_id": "user_123",
      "logins_last_7_days": 3,
      "logins_last_30_days": 15,
      "login_frequency_7d": 0.43
    }
  }'

Response Example:
{
  "transactions": {
    "probability": 0.35,
    "threshold": 0.45,
    "action": "allow",
    "explanations": [...]
  },
  "client_activity": {
    "probability": 0.22,
    "threshold": 0.50,
    "action": "allow",
    "explanations": [...]
  }
}
""")

# ===== EXAMPLE 5: Python API Usage =====
print("\n" + "="*70)
print("EXAMPLE 5: Python API Usage")
print("="*70)

print("""
import requests
import json

BASE_URL = "http://localhost:8000"

# Transaction prediction
payload = {
    "transaction": {
        "amount": 50000.0,
        "timestamp": "2025-11-28T14:30:00",
        "src_account_id": "user_123",
        "beneficiary_id": "benef_456"
    }
}

response = requests.post(
    f"{BASE_URL}/predict/transaction",
    json=payload
)

result = response.json()
print(f"Fraud probability: {result['probability']:.2%}")
print(f"Recommended action: {result['action']}")

for explanation in result['explanations']:
    print(f"  - {explanation['feature']}: {explanation['shap_value']:.3f}")
""")

# ===== EXAMPLE 6: Pipeline Script =====
print("\n" + "="*70)
print("EXAMPLE 6: Complete Pipeline with Script")
print("="*70)

print("""
# Run the complete pipeline (preprocess + train)
./run_pipeline.sh both --ensemble

# Or just preprocess and train without ensemble:
./run_pipeline.sh both

# Or train only on transactions:
./run_pipeline.sh transactions
""")

# ===== EXAMPLE 7: Understanding the model structure =====
print("\n" + "="*70)
print("EXAMPLE 7: Model Directory Structure")
print("="*70)

print("""
After training, your models directory will look like:

models/
├── transactions/
│   ├── lightgbm_model.txt          # Trained LightGBM model
│   ├── xgboost_model.json          # Trained XGBoost (if ensemble)
│   ├── catboost_model.cbm          # Trained CatBoost (if ensemble)
│   ├── model_meta.pkl              # Model metadata (features, threshold, etc.)
│   ├── shap_background.pkl         # Background data for SHAP explanations
│   └── feature_importance.csv      # Feature importance rankings
│
└── client_activity/
    ├── lightgbm_model.txt
    ├── xgboost_model.json
    ├── catboost_model.cbm
    ├── model_meta.pkl
    ├── shap_background.pkl
    └── feature_importance.csv

Each model independently learns to detect fraud patterns specific to its dataset type.
""")

# ===== EXAMPLE 8: Feature engineering differences =====
print("\n" + "="*70)
print("EXAMPLE 8: Feature Engineering Differences")
print("="*70)

print("""
TRANSACTIONS dataset features:
- Transaction amount (raw, log, squared, sqrt)
- Time features (hour, day of week, is_weekend, is_night, is_business_hours)
- Beneficiary features (unique users, tx count, is_new)
- User aggregates (transaction statistics: count, mean, std, sum, min, max)
- Rolling window features (5-transaction rolling mean/std)
- Feature interactions (amount per user, hour deviation from mean)

CLIENT_ACTIVITY dataset features:
- Time features (hour, day of week, is_weekend, is_night, is_business_hours)
- Login statistics (logins_last_7d, logins_last_30d, frequency, intervals)
- Device information (OS changes, phone model changes, categorical features)
- Statistical features (burstiness, Fano factor, Z-scores)
- Log transformations of numerical features

Each dataset has its own optimal feature set based on the data structure.
""")

# ===== EXAMPLE 9: Threshold strategies =====
print("\n" + "="*70)
print("EXAMPLE 9: Threshold Selection Strategies")
print("="*70)

print("""
Different threshold strategies for different business needs:

1. 'precision' (default):
   - Maximizes recall while maintaining desired precision
   - Use when: You want to catch frauds but can tolerate false positives
   - python train.py --threshold_strategy precision --desired_precision 0.7

2. 'f1':
   - Maximizes the F1-score (balance between precision and recall)
   - Use when: You want overall balance
   - python train.py --threshold_strategy f1

3. 'recall':
   - Maximizes recall with a minimum threshold on precision
   - Use when: You must catch most frauds (high recall is critical)
   - python train.py --threshold_strategy recall --min_recall 0.5

4. 'balanced':
   - Balances precision and recall with minimum recall constraint
   - Use when: You want balance but with guaranteed minimum recall
   - python train.py --threshold_strategy balanced --min_recall 0.4
""")

print("\n" + "="*70)
print("For more details, see: README_MULTI_DATASET.md")
print("="*70)
