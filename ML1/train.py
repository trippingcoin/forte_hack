"""
train.py
- Читает processed.parquet
- Делает time-based split (train / val / test)
- Обучает LightGBM
- Подбирает threshold по desired_precision (пример)
- Сохраняет модель и метаданные
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, confusion_matrix
import joblib
import argparse
import os
import shap

MODEL_DIR = "models"

def time_split(df, time_col='timestamp', train_days=90, val_days=14, test_days=14):
    df = df.sort_values(time_col)
    max_date = df[time_col].max()
    test_start = max_date - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)
    train_end = val_start
    train = df[df[time_col] < val_start].copy()
    val = df[(df[time_col] >= val_start) & (df[time_col] < test_start)].copy()
    test = df[df[time_col] >= test_start].copy()
    return train, val, test

def select_features(df):
    # Примеры фич — отредактируй под свой фрейм
    ignore = ['transaction_id','src_account_id','dst_account_id','beneficiary_id','timestamp','confirmed_fraud']
    features = [c for c in df.columns if c not in ignore]
    return features

def choose_threshold(y_true, y_scores, desired_precision=0.7):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    # precision_recall_curve returns arrays where thresholds length = len(precisions)-1
    valid = precisions[:-1] >= desired_precision
    if valid.any():
        # выберем threshold с максимальным recall при precision>=desired_precision
        idx = np.argmax(recalls[:-1][valid])
        # Map idx in valid to original thresholds:
        valid_indices = np.where(valid)[0]
        chosen_idx = valid_indices[idx]
        thr = thresholds[chosen_idx]
    else:
        thr = 0.5
    return float(thr)

def main(processed_path, model_dir, desired_precision):
    df = pd.read_parquet(processed_path)
    # ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_train, df_val, df_test = time_split(df, time_col='timestamp')
    print("Splits:", len(df_train), len(df_val), len(df_test))
    features = select_features(df)
    X_train, y_train = df_train[features], df_train['confirmed_fraud']
    X_val, y_val = df_val[features], df_val['confirmed_fraud']
    X_test, y_test = df_test[features], df_test['confirmed_fraud']
    # LightGBM dataset
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    # scale_pos_weight helps with imbalance
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = max(1.0, neg/pos) if pos>0 else 1.0
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'learning_rate': 0.05,
        'num_leaves': 64,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,
        'seed': 42
    }
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50)
        ]
    )
    # Predict on val/test
    val_scores = model.predict(X_val, num_iteration=model.best_iteration)
    test_scores = model.predict(X_test, num_iteration=model.best_iteration)
    print("Val ROC-AUC:", roc_auc_score(y_val, val_scores))
    print("Test ROC-AUC:", roc_auc_score(y_test, test_scores))
    # choose threshold
    threshold = choose_threshold(y_val, val_scores, desired_precision=desired_precision)
    print("Chosen threshold:", threshold)
    # Evaluate at threshold
    preds_test = (test_scores >= threshold).astype(int)
    print("Test precision:", precision_score(y_test, preds_test, zero_division=0))
    print("Test recall:", recall_score(y_test, preds_test, zero_division=0))
    print("Test confusion matrix:\n", confusion_matrix(y_test, preds_test))
    # Save model & metadata
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "lightgbm_model.txt")
    params_path = os.path.join(model_dir, "model_meta.pkl")
    model.save_model(model_path)
    joblib.dump({
        "features": features,
        "threshold": threshold,
        "best_iteration": model.best_iteration
    }, params_path)
    # Save a small background dataset for SHAP (sample from train)
    background = X_train.sample(n=min(1000, len(X_train)), random_state=42)
    joblib.dump(background, os.path.join(model_dir, "shap_background.pkl"))
    print("Saved model and metadata to", model_dir)
    # Optionally compute global feature importance (gain)
    fi = pd.DataFrame({
        'feature': model.feature_name(),
        'gain': model.feature_importance(importance_type='gain')
    }).sort_values('gain', ascending=False)
    fi.to_csv(os.path.join(model_dir, "feature_importance.csv"), index=False)
    # Save a SHAP explainer snapshot? we'll compute TreeExplainer on load (fast for tree models)
    # End

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=str, default="data/processed.parquet")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--desired_precision", type=float, default=0.7)
    args = parser.parse_args()
    main(args.processed, args.model_dir, args.desired_precision)