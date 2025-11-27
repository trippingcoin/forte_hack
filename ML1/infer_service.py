"""
infer_service.py
- FastAPI service
- Загружает модель + metadata + shap background
- Принимает JSON с транзакцией, вычисляет базовые realtime фичи,
  запускает prediction, и возвращает probability + top-3 SHAP explanations

Note: Для продакшн: вынеси compute_realtime_features в отдельный модуль,
подключай Redis/feature store.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np
import shap
import uvicorn
from typing import Dict, Any, List

MODEL_DIR = "models"

# Пример ожидаемого input JSON:
# {
#   "transaction": {
#       "transaction_id": "...",
#       "src_account_id": "...",
#       "amount": 100.5,
#       "timestamp": "2025-11-26T12:34:56",
#       "channel": "mobile_app",
#       ...
#   }
# }

class TransactionInput(BaseModel):
    transaction: Dict[str, Any]

app = FastAPI(title="Fraud Detection Inference")

# Load model & meta
model = None
meta = None
shap_background = None

@app.on_event("startup")
def load_model():
    global model, meta, shap_background, explainer
    model_path = MODEL_DIR + "/lightgbm_model.txt"
    meta_path = MODEL_DIR + "/model_meta.pkl"
    bg_path = MODEL_DIR + "/shap_background.pkl"
    model = lgb.Booster(model_file=model_path)
    meta = joblib.load(meta_path)
    shap_background = joblib.load(bg_path)
    # TreeExplainer is efficient for tree models
    explainer = shap.TreeExplainer(model)
    # store explainer
    app.state.explainer = explainer
    print("Model and metadata loaded. Features:", len(meta['features']))

def compute_realtime_features(tx: Dict[str,Any], features: List[str]):
    """
    Простая реализация realtime фичей по одной транзакции.
    В реальном мире берём агрегаты из Redis/feature store.
    Здесь: маппим поля и заполняем отсутствующие фичи нулями.
    """
    # Convert to DataFrame single row with required features
    row = {}
    # Basic mapping examples (подстрой под свой feature list)
    row['amount'] = float(tx.get('amount', 0.0))
    row['log_amount'] = np.log1p(row['amount'])
    ts = pd.to_datetime(tx.get('timestamp'))
    row['timestamp'] = ts
    row['hour'] = ts.hour
    row['dow'] = ts.dayofweek
    row['is_weekend'] = int(row['dow'] in [5,6])
    # Example: dummy for channel one-hot columns
    # If your features contain ch_mobile_app, ch_web, etc. create them
    # We'll attempt to set any 'ch_*' features to 1 if matching, 0 otherwise
    channel = tx.get('channel','unknown')
    for f in features:
        if f.startswith('ch_'):
            ch_name = f.replace('ch_','')
            row[f] = 1 if channel == ch_name else 0
    # Fill other numeric features with zeros if not present
    for f in features:
        if f not in row:
            row[f] = tx.get(f, 0)
    df = pd.DataFrame([row])[features]
    # ensure numeric dtype
    df = df.fillna(0)
    return df

@app.post("/predict")
def predict(inp: TransactionInput):
    tx = inp.transaction
    features = meta['features']
    X = compute_realtime_features(tx, features)
    proba = float(model.predict(X, num_iteration=meta.get('best_iteration', None))[0])
    # Compute SHAP values (local explanation)
    explainer = app.state.explainer
    # Use background to compute shap_values faster/more stable
    try:
        shap_values = explainer.shap_values(X)[0]  # for binary classification returns list in some versions
    except Exception:
        # fallback
        shap_values = explainer(X.values)
    # Build top-K explanation
    feat_imp = list(zip(features, shap_values))
    feat_imp_sorted = sorted(feat_imp, key=lambda x: abs(x[1]), reverse=True)[:5]
    explanations = [{"feature": f, "shap_value": float(s)} for f,s in feat_imp_sorted]
    action = "allow"
    thr = float(meta.get('threshold', 0.5))
    if proba >= thr:
        action = "block"
    elif proba >= 0.5:
        action = "challenge"  # business rule example
    return {
        "probability": proba,
        "threshold": thr,
        "action": action,
        "explanations": explanations
    }

if __name__ == "__main__":
    uvicorn.run("infer_service:app", host="0.0.0.0", port=8000, reload=False)