"""
infer_service.py
- FastAPI service
- Загружает модели для transactions и client_activity
- Принимает JSON с данными, вычисляет фичи
- Возвращает predictions с объяснениями для обоих датасетов

Поддерживает два типа предсказаний:
1. /predict/transaction - для транзакционных данных
2. /predict/client_activity - для активности клиента
3. /predict/combined - для обоих типов данных сразу
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np
import shap
import uvicorn
from typing import Dict, Any, List, Optional
import os
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

class TransactionInput(BaseModel):
    transaction: Dict[str, Any]

class ClientActivityInput(BaseModel):
    activity: Dict[str, Any]

class CombinedInput(BaseModel):
    transaction: Optional[Dict[str, Any]] = None
    activity: Optional[Dict[str, Any]] = None

app = FastAPI(title="Fraud Detection Inference - Multi-Dataset")

# Load models & metadata for both datasets
models = {}
meta = {}
shap_background = {}
explainers = {}

@app.on_event("startup")
def load_models():
    """Загружает модели для обоих датасетов"""
    global models, meta, shap_background, explainers
    
    datasets = ['transactions', 'client_activity']
    
    for dataset in datasets:
        dataset_model_dir = os.path.join(MODEL_DIR, dataset)
        
        # Проверяем наличие моделей для этого датасета
        if not os.path.exists(dataset_model_dir):
            print(f"Warning: Model directory for {dataset} not found at {dataset_model_dir}")
            continue
        
        print(f"\n=== Loading {dataset} model ===")
        
        try:
            meta_path = os.path.join(dataset_model_dir, "model_meta.pkl")
            bg_path = os.path.join(dataset_model_dir, "shap_background.pkl")
            
            meta[dataset] = joblib.load(meta_path)
            shap_background[dataset] = joblib.load(bg_path)
            
            models[dataset] = {}
            model_type = meta[dataset].get('model_type', 'lightgbm')
            
            if model_type == 'ensemble':
                # Load ensemble models
                model_names = meta[dataset].get('models', ['lightgbm'])
                for name in model_names:
                    if name == 'lightgbm':
                        model_path = os.path.join(dataset_model_dir, "lightgbm_model.txt")
                        models[dataset]['lightgbm'] = lgb.Booster(model_file=model_path)
                    elif name == 'xgboost' and XGBOOST_AVAILABLE:
                        model_path = os.path.join(dataset_model_dir, "xgboost_model.json")
                        models[dataset]['xgboost'] = xgb.Booster()
                        models[dataset]['xgboost'].load_model(model_path)
                    elif name == 'catboost' and CATBOOST_AVAILABLE:
                        model_path = os.path.join(dataset_model_dir, "catboost_model.cbm")
                        models[dataset]['catboost'] = CatBoostClassifier()
                        models[dataset]['catboost'].load_model(model_path)
                
                explainers[dataset] = shap.TreeExplainer(models[dataset]['lightgbm'])
                print(f"Ensemble models loaded for {dataset}: {list(models[dataset].keys())}. Features: {len(meta[dataset]['features'])}")
            else:
                # Single LightGBM model
                model_path = os.path.join(dataset_model_dir, "lightgbm_model.txt")
                models[dataset]['lightgbm'] = lgb.Booster(model_file=model_path)
                explainers[dataset] = shap.TreeExplainer(models[dataset]['lightgbm'])
                print(f"LightGBM model loaded for {dataset}. Features: {len(meta[dataset]['features'])}")
        
        except Exception as e:
            print(f"Error loading model for {dataset}: {e}")

def compute_realtime_features(data: Dict[str, Any], features: List[str], dataset_type: str):
    """
    Вычисляет realtime фичи.
    dataset_type: 'transactions' или 'client_activity'
    """
    row = {}
    
    # Mapping для разных типов данных
    if dataset_type == 'transactions':
        row['amount'] = float(data.get('amount', 0.0))
        row['log_amount'] = np.log1p(row['amount'])
        ts = pd.to_datetime(data.get('timestamp'))
        row['timestamp'] = ts
        row['hour'] = ts.hour
        row['dow'] = ts.dayofweek
        row['is_weekend'] = int(row['dow'] in [5,6])
    
    elif dataset_type == 'client_activity':
        ts = pd.to_datetime(data.get('timestamp'))
        row['timestamp'] = ts
        row['hour'] = ts.hour
        row['dow'] = ts.dayofweek
        row['is_weekend'] = int(row['dow'] in [5,6])
        
        # Копируем все доступные числовые признаки
        for key, val in data.items():
            if key not in ['timestamp', 'src_account_id']:
                try:
                    row[key] = float(val)
                except (ValueError, TypeError):
                    pass
    
    # Заполняем отсутствующие фичи нулями
    for f in features:
        if f not in row:
            row[f] = data.get(f, 0)
    
    df = pd.DataFrame([row])[features]
    df = df.fillna(0)
    return df

def predict_for_dataset(data: Dict[str, Any], dataset_type: str):
    """Делает предсказание для конкретного датасета"""
    
    if dataset_type not in models or dataset_type not in meta:
        raise HTTPException(
            status_code=400, 
            detail=f"Model for {dataset_type} not available"
        )
    
    features = meta[dataset_type]['features']
    X = compute_realtime_features(data, features, dataset_type)
    
    # Predict with ensemble or single model
    predictions = []
    for name, model in models[dataset_type].items():
        if name == 'lightgbm':
            pred = model.predict(X, num_iteration=meta[dataset_type].get('best_iteration', None))[0]
        elif name == 'xgboost':
            dmatrix = xgb.DMatrix(X)
            pred = model.predict(dmatrix)[0]
        elif name == 'catboost':
            pred = model.predict_proba(X)[0, 1]
        predictions.append(pred)
    
    # Average predictions for ensemble
    proba = float(np.mean(predictions))
    
    # Compute SHAP values
    explainer = explainers[dataset_type]
    try:
        shap_values = explainer.shap_values(X)[0]
    except Exception:
        shap_values = explainer(X.values)
    
    # Build top-K explanation
    feat_imp = list(zip(features, shap_values))
    feat_imp_sorted = sorted(feat_imp, key=lambda x: abs(x[1]), reverse=True)[:5]
    explanations = [{"feature": f, "shap_value": float(s)} for f,s in feat_imp_sorted]
    
    action = "allow"
    thr = float(meta[dataset_type].get('threshold', 0.5))
    if proba >= thr:
        action = "block"
    elif proba >= 0.5:
        action = "challenge"
    
    return {
        "probability": proba,
        "threshold": thr,
        "action": action,
        "explanations": explanations
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    available_models = list(models.keys())
    return {
        "status": "ok",
        "available_models": available_models
    }

@app.post("/predict/transaction")
def predict_transaction(inp: TransactionInput):
    """Предсказание для транзакций"""
    return predict_for_dataset(inp.transaction, 'transactions')

@app.post("/predict/client_activity")
def predict_client_activity(inp: ClientActivityInput):
    """Предсказание для активности клиента"""
    return predict_for_dataset(inp.activity, 'client_activity')

@app.post("/predict/combined")
def predict_combined(inp: CombinedInput):
    """Предсказание для обоих типов данных"""
    results = {}
    
    if inp.transaction:
        try:
            results['transactions'] = predict_for_dataset(inp.transaction, 'transactions')
        except HTTPException as e:
            results['transactions'] = {"error": str(e.detail)}
    
    if inp.activity:
        try:
            results['client_activity'] = predict_for_dataset(inp.activity, 'client_activity')
        except HTTPException as e:
            results['client_activity'] = {"error": str(e.detail)}
    
    return results

if __name__ == "__main__":
    uvicorn.run("infer_service:app", host="0.0.0.0", port=8000, reload=False)