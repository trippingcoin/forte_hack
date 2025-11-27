#!/bin/bash
# Wrapper to run training
set -e
# Usage: ./scripts/run_train.sh [dataset] [--ensemble]
DATASET=${1:-both}
ENSEMBLE=${2:-}
python src/train.py \
  --processed_transactions data/processed_transactions.parquet \
  --processed_client_activity data/processed_client_activity.parquet \
  --dataset "$DATASET" $ENSEMBLE
