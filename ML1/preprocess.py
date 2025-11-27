"""
preprocess.py
- Загружает CSV (transactions.csv и client_activity.csv)
- Делает базовый feature engineering для каждого датасета
- Сохраняет processed данные для обучения

Поддерживает два датасета:
1. transactions.csv - транзакционные данные
2. client_activity.csv - активность клиентов
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import argparse
import os

def preprocess_transactions(df):
    """Preprocessing для транзакций (transactions.csv)"""
    # Переименование столбцов
    df = df.rename(columns={
        "transdatetime": "timestamp",
        "cst_dim_id": "src_account_id",
        "direction": "beneficiary_id",
        "docno": "transaction_id",
        "target": "confirmed_fraud"
    })
    
    # Преобразования времени
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'transdate' in df.columns:
        df['transdate'] = pd.to_datetime(df['transdate'])
        df['transdate_day'] = (df['transdate'] - df['transdate'].min()).dt.days
        df = df.drop(columns=['transdate'])
    
    # Расширенные временные признаки
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dow'].isin([5,6]).astype(int)
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 18) & (df['dow'] < 5)).astype(int)
    
    # Amount transforms
    df['log_amount'] = np.log1p(df['amount'])
    df['amount_squared'] = df['amount'] ** 2
    df['amount_sqrt'] = np.sqrt(df['amount'])
    
    # Признаки по бенефициару (важно!)
    df = df.sort_values('timestamp')
    beneficiary_user_count = df.groupby('beneficiary_id')['src_account_id'].nunique().to_dict()
    df['beneficiary_unique_users'] = df['beneficiary_id'].map(beneficiary_user_count)
    
    beneficiary_tx_count = df.groupby('beneficiary_id').size().to_dict()
    df['beneficiary_tx_count'] = df['beneficiary_id'].map(beneficiary_tx_count)
    
    # Флаг: новый бенефициар для пользователя
    df['is_new_beneficiary'] = (~df.duplicated(subset=['src_account_id','beneficiary_id'])).astype(int)
    
    # Временные окна по пользователю (скользящие статистики)
    df = df.sort_values(['src_account_id', 'timestamp'])
    user_window = df.groupby('src_account_id')['amount'].rolling(window=5, min_periods=1).agg(['mean', 'std', 'count']).reset_index(level=0, drop=True)
    user_window.columns = ['amount_rolling_mean', 'amount_rolling_std', 'amount_rolling_count']
    df = pd.concat([df, user_window], axis=1)
    
    # Агрегаты по пользователю (глобальные)
    user_agg = df.groupby('src_account_id').agg({
        'amount': ['count','mean','std','sum','min','max'],
        'beneficiary_id': 'nunique',
        'hour': ['mean', 'std']
    })
    user_agg.columns = ['_'.join(col).strip() for col in user_agg.columns.values]
    user_agg = user_agg.reset_index()
    df = df.merge(user_agg, on='src_account_id', how='left')
    
    # Взаимодействия признаков
    df['amount_per_user_mean'] = df['amount'] / (df['amount_mean'] + 1e-6)
    df['amount_per_user_std'] = (df['amount'] - df['amount_mean']) / (df['amount_std'] + 1e-6)
    df['hour_diff_from_mean'] = abs(df['hour'] - df['hour_mean'])
    
    # Признаки по бенефициару (агрегаты)
    beneficiary_agg = df.groupby('beneficiary_id').agg({
        'amount': ['mean', 'std', 'count'],
        'src_account_id': 'nunique'
    })
    beneficiary_agg.columns = ['beneficiary_' + '_'.join(col).strip() for col in beneficiary_agg.columns.values]
    beneficiary_agg = beneficiary_agg.reset_index()
    df = df.merge(beneficiary_agg, on='beneficiary_id', how='left')
    
    # Fillna
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(0)
    
    # Удаляем inf значения
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

def preprocess_client_activity(df):
    """Preprocessing для активности клиентов (client_activity.csv)"""
    # Переименование столбцов для стандартизации
    df = df.rename(columns={
        "transdate": "timestamp",
        "cst_dim_id": "src_account_id",
    })
    
    # Преобразование categorical columns в числовые через факторизацию
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in ['timestamp', 'src_account_id']:
            df[col] = pd.factorize(df[col])[0]
    
    # Преобразование timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Временные признаки
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dow'].isin([5,6]).astype(int)
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 18) & (df['dow'] < 5)).astype(int)
    
    # Логирование числовых признаков
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['src_account_id', 'confirmed_fraud']:
            if (df[col] > 0).sum() > 0:
                df[f'log_{col}'] = np.log1p(df[col])
    
    # Создание синтетического целевого столбца
    if 'confirmed_fraud' not in df.columns:
        df['login_spike'] = (df['logins_last_7_days'] > df['logins_last_7_days'].quantile(0.9)).astype(int)
        df['unusual_devices'] = (df['monthly_os_changes'] + df['monthly_phone_model_changes'] > 3).astype(int)
        df['confirmed_fraud'] = ((df['login_spike'] & df['unusual_devices']).astype(int) | 
                                 (df['logins_7d_over_30d_ratio'] > 2.0).astype(int))
        df = df.drop(columns=['login_spike', 'unusual_devices'])
    
    # Заполняем NaN нулями
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(0)
    
    # Удаляем inf значения
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

def main(transactions_path=None, client_activity_path=None, transactions_output=None, client_activity_output=None):
    """Preprocessing для обоих датасетов."""
    
    if transactions_path and os.path.exists(transactions_path):
        print(f"=== Processing {transactions_path} ===")
        df_trans = pd.read_csv(transactions_path, encoding="windows-1251", sep=";", skiprows=1)
        print(f"Loaded {len(df_trans)} transaction rows")
        df_trans_proc = preprocess_transactions(df_trans)
        os.makedirs(os.path.dirname(transactions_output) or '.', exist_ok=True)
        df_trans_proc.to_parquet(transactions_output, index=False)
        print(f"Saved to {transactions_output}")
    
    if client_activity_path and os.path.exists(client_activity_path):
        print(f"=== Processing {client_activity_path} ===")
        df_activity = pd.read_csv(client_activity_path, encoding="windows-1251", sep=";", skiprows=1)
        print(f"Loaded {len(df_activity)} activity rows")
        df_activity_proc = preprocess_client_activity(df_activity)
        os.makedirs(os.path.dirname(client_activity_output) or '.', exist_ok=True)
        df_activity_proc.to_parquet(client_activity_output, index=False)
        print(f"Saved to {client_activity_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions_input", type=str, default="data/transactions.csv")
    parser.add_argument("--client_activity_input", type=str, default="data/client_activity.csv")
    parser.add_argument("--transactions_output", type=str, default="data/processed_transactions.parquet")
    parser.add_argument("--client_activity_output", type=str, default="data/processed_client_activity.parquet")
    parser.add_argument("--encoding", default="windows-1251")

    args = parser.parse_args()
    main(args.transactions_input, args.client_activity_input, 
         args.transactions_output, args.client_activity_output)
