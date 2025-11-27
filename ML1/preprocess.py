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
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dow'].isin([5,6]).astype(int)
    # Amount transforms
    df['log_amount'] = np.log1p(df['amount'])
    # Flag: новый бенефициар (пример — если первый раз в данных)
    df['is_new_beneficiary'] = (~df.duplicated(subset=['src_account_id','beneficiary_id'])).astype(int)
    # Простые агрегаты по пользователю (за последние X дней) — here: precomputed approx via groupby
    # NOTE: for production use streaming/incremental Redis or feature store
    user_agg = df.groupby('src_account_id').agg({
        'amount': ['count','mean','std','sum']
    })
    user_agg.columns = ['_'.join(col).strip() for col in user_agg.columns.values]
    user_agg = user_agg.reset_index().rename(columns={'index':'src_account_id'})
    df = df.merge(user_agg, on='src_account_id', how='left')
    # Fillna
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(0)
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