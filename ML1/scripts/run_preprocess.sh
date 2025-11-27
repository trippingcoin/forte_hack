#!/bin/bash
# Wrapper to run preprocessing
set -e
python src/preprocess.py \
  --transactions_input data/transactions.csv \
  --client_activity_input data/client_activity.csv \
  --transactions_output data/processed_transactions.parquet \
  --client_activity_output data/processed_client_activity.parquet
