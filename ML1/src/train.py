"""
train.py
- Читает processed.parquet
- Делает time-based split (train / val / test)
- Обучает LightGBM (или Ensemble: LightGBM + XGBoost + CatBoost)
- Подбирает threshold по desired_precision (пример)
- Сохраняет модель и метаданные
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")

from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, confusion_matrix, f1_score, average_precision_score, accuracy_score
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
    # Исключаем категориальные и нечисловые столбцы
    features = [c for c in df.columns if c not in ignore 
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32', 'int16', 'int8', 'uint8', 'uint16', 'uint32', 'uint64']]
    return features

def choose_threshold(y_true, y_scores, desired_precision=0.7, strategy='precision', min_recall=0.4):
    """
    Выбор порога по разным стратегиям:
    - 'precision': максимальный recall при precision >= desired_precision
    - 'f1': максимальный F1-score
    - 'balanced': баланс между precision и recall с учетом min_recall
    - 'recall': оптимизация по recall (минимальный recall >= min_recall, максимальный precision)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    
    if strategy == 'precision':
        valid = precisions[:-1] >= desired_precision
        if valid.any():
            # выберем threshold с максимальным recall при precision>=desired_precision
            idx = np.argmax(recalls[:-1][valid])
            valid_indices = np.where(valid)[0]
            chosen_idx = valid_indices[idx]
            thr = thresholds[chosen_idx]
        else:
            # Если не нашли нужный precision, выберем порог с максимальным F1
            best_idx = np.argmax(f1_scores)
            thr = thresholds[best_idx]
            print(f"Warning: Could not achieve precision >= {desired_precision}")
            print(f"Best available: precision={precisions[best_idx]:.3f}, recall={recalls[best_idx]:.3f}, F1={f1_scores[best_idx]:.3f}")
    elif strategy == 'f1':
        # Выбираем порог с максимальным F1
        best_idx = np.argmax(f1_scores)
        thr = thresholds[best_idx]
        print(f"F1-optimized threshold: precision={precisions[best_idx]:.3f}, recall={recalls[best_idx]:.3f}, F1={f1_scores[best_idx]:.3f}")
    elif strategy == 'recall':
        # Оптимизация по recall: минимальный recall >= min_recall, максимальный precision
        valid = recalls[:-1] >= min_recall
        if valid.any():
            # Из валидных выбираем с максимальным precision
            valid_indices = np.where(valid)[0]
            best_precision_idx = np.argmax(precisions[:-1][valid_indices])
            chosen_idx = valid_indices[best_precision_idx]
            thr = thresholds[chosen_idx]
            print(f"Recall-optimized threshold: precision={precisions[chosen_idx]:.3f}, recall={recalls[chosen_idx]:.3f}, F1={f1_scores[chosen_idx]:.3f}")
        else:
            # Если не нашли, выбираем порог с максимальным recall
            best_recall_idx = np.argmax(recalls[:-1])
            thr = thresholds[best_recall_idx]
            print(f"Recall-optimized (max recall): precision={precisions[best_recall_idx]:.3f}, recall={recalls[best_recall_idx]:.3f}, F1={f1_scores[best_recall_idx]:.3f}")
    else:  # 'balanced'
        # Выбираем порог с оптимальным балансом precision/recall
        # Ищем порог где recall >= min_recall и precision >= 0.4 (разумный компромисс)
        valid = (recalls[:-1] >= min_recall) & (precisions[:-1] >= 0.4)
        if valid.any():
            # Из валидных выбираем с максимальным F1
            valid_indices = np.where(valid)[0]
            best_f1_idx = np.argmax(f1_scores[valid_indices])
            chosen_idx = valid_indices[best_f1_idx]
            thr = thresholds[chosen_idx]
            print(f"Balanced threshold: precision={precisions[chosen_idx]:.3f}, recall={recalls[chosen_idx]:.3f}, F1={f1_scores[chosen_idx]:.3f}")
        else:
            # Если не нашли, выбираем порог с максимальным F1 при recall >= 0.3
            valid_recall = recalls[:-1] >= max(0.3, min_recall * 0.75)
            if valid_recall.any():
                valid_indices = np.where(valid_recall)[0]
                best_f1_idx = np.argmax(f1_scores[valid_indices])
                chosen_idx = valid_indices[best_f1_idx]
                thr = thresholds[chosen_idx]
                print(f"Balanced threshold (relaxed): precision={precisions[chosen_idx]:.3f}, recall={recalls[chosen_idx]:.3f}, F1={f1_scores[chosen_idx]:.3f}")
            else:
                # Последний вариант - максимальный F1
                best_idx = np.argmax(f1_scores)
                thr = thresholds[best_idx]
                print(f"Balanced threshold (F1 max): precision={precisions[best_idx]:.3f}, recall={recalls[best_idx]:.3f}, F1={f1_scores[best_idx]:.3f}")
    
    return float(thr)

def train_ensemble(X_train, y_train, X_val, y_val, scale_pos_weight):
    """Обучает ensemble из LightGBM, XGBoost и CatBoost"""
    models = {}
    predictions = {}
    
    # LightGBM
    print("\n=== Training LightGBM ===")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'learning_rate': 0.03,  # Уменьшил для лучшей сходимости
        'num_leaves': 31,
        'max_depth': 7,  # Немного уменьшил для снижения переобучения
        'min_data_in_leaf': 3,  # Уменьшил для большей чувствительности
        'feature_fraction': 0.85,  # Немного уменьшил для регуляризации
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'lambda_l1': 0.01,  # Увеличил регуляризацию
        'lambda_l2': 0.01,
        'scale_pos_weight': scale_pos_weight * 1.2,  # Увеличил для лучшего recall
        'seed': 42,
        'force_row_wise': True
    }
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
    )
    models['lightgbm'] = lgb_model
    predictions['lightgbm'] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    print(f"LightGBM Val AUC: {roc_auc_score(y_val, predictions['lightgbm']):.4f}")
    
    # XGBoost - улучшенные параметры для fraud detection
    if XGBOOST_AVAILABLE:
        print("\n=== Training XGBoost ===")
        xgb_train = xgb.DMatrix(X_train, label=y_train)
        xgb_val = xgb.DMatrix(X_val, label=y_val)
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.02,  # Еще меньше LR для лучшей сходимости
            'max_depth': 7,
            'min_child_weight': 2,  # Еще меньше для большей чувствительности к мошенничеству
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'colsample_bylevel': 0.85,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'gamma': 0.05,  # Уменьшил для более агрессивного роста
            'max_delta_step': 2,  # Увеличил для лучшей работы с imbalance
            'scale_pos_weight': scale_pos_weight * 1.2,  # Увеличил для лучшего recall
            'tree_method': 'hist',
            'grow_policy': 'lossguide',
            'max_leaves': 31,
            'seed': 42,
            'verbosity': 0
        }
        xgb_model = xgb.train(
            xgb_params,
            xgb_train,
            num_boost_round=2000,
            evals=[(xgb_train, 'train'), (xgb_val, 'val')],
            early_stopping_rounds=100,
            verbose_eval=50
        )
        models['xgboost'] = xgb_model
        predictions['xgboost'] = xgb_model.predict(xgb_val)
        print(f"XGBoost Val AUC: {roc_auc_score(y_val, predictions['xgboost']):.4f}")
    else:
        print("\n=== Skipping XGBoost (not installed) ===")
    
    # CatBoost - улучшенные параметры для fraud detection
    if CATBOOST_AVAILABLE:
        print("\n=== Training CatBoost ===")
        cat_model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.02,  # Еще меньше LR для лучшей сходимости
            depth=7,
            min_data_in_leaf=2,  # Еще меньше для большей чувствительности
            l2_leaf_reg=0.01,
            # Параметры для борьбы с переобучением
            colsample_bylevel=0.85,
            max_leaves=31,
            # Параметры для несбалансированных данных (используем только class_weights)
            class_weights=[1.0, scale_pos_weight * 1.2],  # Увеличил вес для лучшего recall
            # Дополнительные параметры
            bootstrap_type='Bayesian',  # Bayesian bootstrap для лучшей стабильности
            bagging_temperature=0.8,  # Температура для Bayesian bootstrap (subsample не нужен для Bayesian)
            random_strength=0.5,  # Случайность в выборе split
            # Регуляризация
            leaf_estimation_method='Newton',  # Метод оценки листьев
            leaf_estimation_iterations=10,  # Итерации для оценки листьев
            random_seed=42,
            verbose=50,
            early_stopping_rounds=100,
            eval_metric='AUC',
            use_best_model=True,
            task_type='CPU'  # Явно указываем CPU
        )
        cat_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=50
        )
        models['catboost'] = cat_model
        predictions['catboost'] = cat_model.predict_proba(X_val)[:, 1]
        print(f"CatBoost Val AUC: {roc_auc_score(y_val, predictions['catboost']):.4f}")
    else:
        print("\n=== Skipping CatBoost (not installed) ===")
    
    # Ensemble prediction (weighted average)
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    print(f"\n=== Ensemble Val AUC: {roc_auc_score(y_val, ensemble_pred):.4f} ===")
    
    return models, ensemble_pred

def main(processed_path, model_dir, desired_precision, threshold_strategy='balanced', use_ensemble=False, min_recall=0.4):
    df = pd.read_parquet(processed_path)
    # ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Check if target column exists
    if 'confirmed_fraud' not in df.columns:
        print(f"Warning: 'confirmed_fraud' column not found in {processed_path}")
        print("Available columns:", list(df.columns))
        print("Skipping training for this dataset - no target variable found")
        return
    
    df_train, df_val, df_test = time_split(df, time_col='timestamp')
    print("Splits:", len(df_train), len(df_val), len(df_test))
    print(f"Fraud rates - Train: {df_train['confirmed_fraud'].mean():.3f}, Val: {df_val['confirmed_fraud'].mean():.3f}, Test: {df_test['confirmed_fraud'].mean():.3f}")
    features = select_features(df)
    X_train, y_train = df_train[features], df_train['confirmed_fraud']
    X_val, y_val = df_val[features], df_val['confirmed_fraud']
    X_test, y_test = df_test[features], df_test['confirmed_fraud']
    # LightGBM dataset
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    # scale_pos_weight helps with imbalance - увеличиваем для лучшего recall
    pos = y_train.sum()
    neg = len(y_train) - pos
    base_weight = max(1.0, neg/pos) if pos>0 else 1.0
    scale_pos_weight = base_weight * 1.5  # Увеличиваем вес для лучшего recall
    print(f"Class imbalance: {neg}:{pos}, scale_pos_weight: {scale_pos_weight:.2f}")
    
    if use_ensemble:
        # Train ensemble
        models, val_scores = train_ensemble(X_train, y_train, X_val, y_val, scale_pos_weight)
        
        # Predict on test with ensemble
        test_predictions = []
        for name, model in models.items():
            if name == 'lightgbm':
                test_pred = model.predict(X_test, num_iteration=model.best_iteration)
            elif name == 'xgboost':
                test_dmatrix = xgb.DMatrix(X_test)
                test_pred = model.predict(test_dmatrix)
            elif name == 'catboost':
                test_pred = model.predict_proba(X_test)[:, 1]
            test_predictions.append(test_pred)
        test_scores = np.mean(test_predictions, axis=0)
        
        # For train predictions (for threshold selection)
        train_predictions = []
        for name, model in models.items():
            if name == 'lightgbm':
                train_pred = model.predict(X_train, num_iteration=model.best_iteration)
            elif name == 'xgboost':
                train_dmatrix = xgb.DMatrix(X_train)
                train_pred = model.predict(train_dmatrix)
            elif name == 'catboost':
                train_pred = model.predict_proba(X_train)[:, 1]
            train_predictions.append(train_pred)
        train_scores = np.mean(train_predictions, axis=0)
    else:
        # Single LightGBM model (original)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 8,
            'min_data_in_leaf': 5,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.005,
            'lambda_l2': 0.005,
            'min_gain_to_split': 0.0,
            'scale_pos_weight': scale_pos_weight,
            'seed': 42,
            'force_row_wise': True
        }
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=2000,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(50)
            ]
        )
        models = {'lightgbm': model}
        val_scores = model.predict(X_val, num_iteration=model.best_iteration)
        test_scores = model.predict(X_test, num_iteration=model.best_iteration)
        train_scores = model.predict(X_train, num_iteration=model.best_iteration)
    
    # Отладочная информация
    print(f"\nScore ranges: Val [{val_scores.min():.4f}, {val_scores.max():.4f}], Test [{test_scores.min():.4f}, {test_scores.max():.4f}]")
    print(f"Fraud rate: Val {y_val.mean():.3f}, Test {y_test.mean():.3f}")
    
    print("\n=== Validation Metrics ===")
    print("Val ROC-AUC:", roc_auc_score(y_val, val_scores))
    print("Val PR-AUC:", average_precision_score(y_val, val_scores))
    
    print("\n=== Test Metrics ===")
    print("Test ROC-AUC:", roc_auc_score(y_test, test_scores))
    print("Test PR-AUC:", average_precision_score(y_test, test_scores))
    
    # choose threshold - используем train+val для большей стабильности при distribution shift
    use_combined_for_threshold = True
    if use_combined_for_threshold:
        y_combined = pd.concat([y_train, y_val])
        scores_combined = np.concatenate([train_scores, val_scores])
        threshold = choose_threshold(y_combined, scores_combined, desired_precision=desired_precision, strategy=threshold_strategy, min_recall=min_recall)
        print(f"\nChosen threshold ({threshold_strategy}, using train+val):", threshold)
    else:
        threshold = choose_threshold(y_val, val_scores, desired_precision=desired_precision, strategy=threshold_strategy, min_recall=min_recall)
        print(f"\nChosen threshold ({threshold_strategy}):", threshold)
    
    # Evaluate at threshold
    preds_val = (val_scores >= threshold).astype(int)
    preds_test = (test_scores >= threshold).astype(int)
    
    print("\n=== Validation at Threshold ===")
    print("Val precision:", precision_score(y_val, preds_val, zero_division=0))
    print("Val recall:", recall_score(y_val, preds_val, zero_division=0))
    print("Val F1:", f1_score(y_val, preds_val, zero_division=0))
    
    print("\n=== Test at Threshold ===")
    print("Test precision:", precision_score(y_test, preds_test, zero_division=0))
    print("Test recall:", recall_score(y_test, preds_test, zero_division=0))
    print("Test F1:", f1_score(y_test, preds_test, zero_division=0))
    test_accuracy = accuracy_score(y_test, preds_test)
    print(f"Test accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    print("Note: Accuracy can be misleading for imbalanced data (fraud rate: {:.1f}%)".format(y_test.mean()*100))
    print("Test confusion matrix:\n", confusion_matrix(y_test, preds_test))
    
    # Анализ различий между валидацией и тестом
    val_recall = recall_score(y_val, preds_val, zero_division=0)
    test_recall = recall_score(y_test, preds_test, zero_division=0)
    if abs(val_recall - test_recall) > 0.2:
        print(f"\n  Warning: Large recall gap between Val ({val_recall:.3f}) and Test ({test_recall:.3f})")
        print("This suggests overfitting or distribution shift. Consider:")
        print("  - Using combined train+val for threshold selection")
        print("  - More regularization")
        print("  - Checking data distribution differences")
    # Save model & metadata
    os.makedirs(model_dir, exist_ok=True)
    params_path = os.path.join(model_dir, "model_meta.pkl")
    
    if use_ensemble:
        # Save all models in ensemble
        for name, model in models.items():
            if name == 'lightgbm':
                model_path = os.path.join(model_dir, "lightgbm_model.txt")
                model.save_model(model_path)
            elif name == 'xgboost':
                model_path = os.path.join(model_dir, "xgboost_model.json")
                model.save_model(model_path)
            elif name == 'catboost':
                model_path = os.path.join(model_dir, "catboost_model.cbm")
                model.save_model(model_path)
        
        joblib.dump({
            "features": features,
            "threshold": threshold,
            "model_type": "ensemble",
            "models": list(models.keys())
        }, params_path)
    else:
        # Save single LightGBM model
        model = models['lightgbm']
        model_path = os.path.join(model_dir, "lightgbm_model.txt")
        model.save_model(model_path)
        joblib.dump({
            "features": features,
            "threshold": threshold,
            "best_iteration": model.best_iteration,
            "model_type": "lightgbm"
        }, params_path)
    # Save a small background dataset for SHAP (sample from train)
    background = X_train.sample(n=min(1000, len(X_train)), random_state=42)
    joblib.dump(background, os.path.join(model_dir, "shap_background.pkl"))
    print("Saved model and metadata to", model_dir)
    # Compute feature importance
    if use_ensemble:
        # Average feature importance across all models
        all_importances = []
        for name, model in models.items():
            if name == 'lightgbm':
                fi_dict = dict(zip(model.feature_name(), model.feature_importance(importance_type='gain')))
            elif name == 'xgboost':
                # XGBoost feature importance
                try:
                    xgb_scores = model.get_score(importance_type='gain')
                    fi_dict = {f: xgb_scores.get(f'f{i}', 0) for i, f in enumerate(features)}
                except:
                    fi_dict = {f: 0 for f in features}
            elif name == 'catboost':
                fi_dict = dict(zip(features, model.get_feature_importance()))
            all_importances.append(fi_dict)
        
        # Average importances
        avg_importance = {}
        for feat in features:
            avg_importance[feat] = np.mean([imp.get(feat, 0) for imp in all_importances])
        
        fi = pd.DataFrame({
            'feature': list(avg_importance.keys()),
            'gain': list(avg_importance.values())
        }).sort_values('gain', ascending=False)
    else:
        model = models['lightgbm']
        fi = pd.DataFrame({
            'feature': model.feature_name(),
            'gain': model.feature_importance(importance_type='gain')
        }).sort_values('gain', ascending=False)
    
    fi.to_csv(os.path.join(model_dir, "feature_importance.csv"), index=False)
    # Save a SHAP explainer snapshot? we'll compute TreeExplainer on load (fast for tree models)
    # End

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_transactions", type=str, default="data/processed_transactions.parquet")
    parser.add_argument("--processed_client_activity", type=str, default="data/processed_client_activity.parquet")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--dataset", type=str, default="both", 
                        choices=['transactions', 'client_activity', 'both'],
                        help="Which dataset to train on: transactions, client_activity, or both")
    parser.add_argument("--desired_precision", type=float, default=0.7)
    parser.add_argument("--threshold_strategy", type=str, default="balanced", 
                        choices=['precision', 'f1', 'balanced', 'recall'],
                        help="Strategy for threshold selection: precision (max recall at desired precision), f1 (max F1), balanced, recall (optimize for recall)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use ensemble of LightGBM + XGBoost + CatBoost instead of single LightGBM")
    parser.add_argument("--min_recall", type=float, default=0.4,
                        help="Minimum recall target for balanced/recall strategies (default: 0.4)")
    args = parser.parse_args()
    
    # Train on selected datasets
    if args.dataset in ['transactions', 'both']:
        print("\n" + "="*80)
        print("TRAINING ON TRANSACTIONS DATA")
        print("="*80)
        main(args.processed_transactions, os.path.join(args.model_dir, 'transactions'), 
             args.desired_precision, args.threshold_strategy, args.ensemble, args.min_recall)
    
    if args.dataset in ['client_activity', 'both']:
        print("\n" + "="*80)
        print("TRAINING ON CLIENT ACTIVITY DATA")
        print("="*80)
        main(args.processed_client_activity, os.path.join(args.model_dir, 'client_activity'), 
             args.desired_precision, args.threshold_strategy, args.ensemble, args.min_recall)