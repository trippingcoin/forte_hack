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
from typing import Dict, Any, List, Optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

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
models = {}
meta = None
shap_background = None

@app.on_event("startup")
def load_model():
    global models, meta, shap_background, explainer
    meta_path = MODEL_DIR + "/model_meta.pkl"
    bg_path = MODEL_DIR + "/shap_background.pkl"
    meta = joblib.load(meta_path)
    shap_background = joblib.load(bg_path)
    
    model_type = meta.get('model_type', 'lightgbm')
    
    if model_type == 'ensemble':
        # Load ensemble models
        model_names = meta.get('models', ['lightgbm'])
        for name in model_names:
            if name == 'lightgbm':
                model_path = MODEL_DIR + "/lightgbm_model.txt"
                models['lightgbm'] = lgb.Booster(model_file=model_path)
            elif name == 'xgboost' and XGBOOST_AVAILABLE:
                model_path = MODEL_DIR + "/xgboost_model.json"
                models['xgboost'] = xgb.Booster()
                models['xgboost'].load_model(model_path)
            elif name == 'catboost' and CATBOOST_AVAILABLE:
                model_path = MODEL_DIR + "/catboost_model.cbm"
                models['catboost'] = CatBoostClassifier()
                models['catboost'].load_model(model_path)
        # Use LightGBM for SHAP (fastest)
        explainer = shap.TreeExplainer(models['lightgbm'])
        print(f"Ensemble models loaded: {list(models.keys())}. Features:", len(meta['features']))
    else:
        # Single LightGBM model
        model_path = MODEL_DIR + "/lightgbm_model.txt"
        models['lightgbm'] = lgb.Booster(model_file=model_path)
        explainer = shap.TreeExplainer(models['lightgbm'])
        print("LightGBM model loaded. Features:", len(meta['features']))
    
    app.state.explainer = explainer

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
    
    # Predict with ensemble or single model
    predictions = []
    for name, model in models.items():
        if name == 'lightgbm':
            pred = model.predict(X, num_iteration=meta.get('best_iteration', None))[0]
        elif name == 'xgboost':
            dmatrix = xgb.DMatrix(X)
            pred = model.predict(dmatrix)[0]
        elif name == 'catboost':
            pred = model.predict_proba(X)[0, 1]
        predictions.append(pred)
    
    # Average predictions for ensemble, or use single prediction
    proba = float(np.mean(predictions))
    
    # Compute SHAP values (local explanation) - use LightGBM for speed
    explainer = app.state.explainer
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