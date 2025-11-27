#!/bin/bash
# run_pipeline.sh
# Скрипт для запуска полного pipeline обработки и обучения на обоих датасетах

set -e  # Exit on first error

echo "==============================================="
echo "Fraud Detection Pipeline - Multi-Dataset"
echo "==============================================="

# Проверяем наличие файлов данных
if [ ! -f "data/transactions.csv" ]; then
    echo "Error: data/transactions.csv not found!"
    exit 1
fi

if [ ! -f "data/client_activity.csv" ]; then
    echo "Error: data/client_activity.csv not found!"
    exit 1
fi

echo ""
echo "Step 1: Preprocessing both datasets..."
echo "-------------------------------------------"
python src/preprocess.py \
    --transactions_input data/transactions.csv \
    --client_activity_input data/client_activity.csv \
    --transactions_output data/processed_transactions.parquet \
    --client_activity_output data/processed_client_activity.parquet

echo ""
echo "Step 2: Training models for both datasets..."
echo "-------------------------------------------"
# Parse command line arguments for training
DATASET=${1:-"both"}
ENSEMBLE=${2:-"--ensemble"}

python src/train.py \
    --processed_transactions data/processed_transactions.parquet \
    --processed_client_activity data/processed_client_activity.parquet \
    --dataset "$DATASET" \
    $ENSEMBLE

echo ""
echo "==============================================="
echo "Pipeline completed successfully!"
echo "==============================================="
echo ""
echo "Models saved to: models/"
echo ""
echo "To start the inference service, run:"
echo "  python src/infer_service.py"
echo ""
echo "Then test with:"
echo "  curl -X POST http://localhost:8000/predict/transaction \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"transaction\": {\"amount\": 1000, \"timestamp\": \"2025-11-28T12:00:00\"}}'"
