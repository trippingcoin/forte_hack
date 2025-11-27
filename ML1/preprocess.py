"""
preprocess.py
- Загружает CSV
- Делает базовый feature engineering (временные признаки, агрегаты за простые окна)
- Сохраняет processed CSV для обучения

Настрой: INPUT_PATH, OUTPUT_PATH, names of columns according to your CSV.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import argparse
import os

def basic_feature_engineering(df):
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

def main(input_path, output_path):
    df = pd.read_csv(input_path, encoding="windows-1251", sep=";", skiprows=1)

    df = df.rename(columns={
    "transdatetime": "timestamp",
    "cst_dim_id": "src_account_id",
    "direction": "beneficiary_id",
    "docno": "transaction_id",
    "target": "confirmed_fraud"
    })

    print("Loaded", len(df), "rows")
    df_proc = basic_feature_engineering(df)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df_proc.to_parquet(output_path, index=False)
    print("Saved processed data to", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/transactions.csv")
    parser.add_argument("--output", type=str, default="data/processed.parquet")
    parser.add_argument("--encoding", default="windows-1251")

    args = parser.parse_args()
    main(args.input, args.output)