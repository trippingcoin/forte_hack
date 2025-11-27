# Quick Reference - Multi-Dataset Fraud Detection

## Commands Quick Reference

### Preprocessing
```bash
# Both datasets
python preprocess.py

# With custom paths
python preprocess.py \
  --transactions_input data/transactions.csv \
  --client_activity_input data/client_activity.csv \
  --transactions_output data/processed_transactions.parquet \
  --client_activity_output data/processed_client_activity.parquet
```

### Training

```bash
# Both datasets with ensemble
python train.py --dataset both --ensemble

# Both datasets without ensemble (faster, less memory)
python train.py --dataset both

# Only transactions
python train.py --dataset transactions

# Only client activity
python train.py --dataset client_activity

# With threshold optimization
python train.py --dataset both --threshold_strategy recall --min_recall 0.5
```

**Threshold Strategies:**
- `precision` - max recall at desired precision (default)
- `f1` - maximize F1-score
- `recall` - maximize recall with min precision
- `balanced` - balance precision/recall

### Inference Service
```bash
# Start API server
python infer_service.py

# Server runs at: http://localhost:8000
# API docs at: http://localhost:8000/docs
# Health check: http://localhost:8000/health
```

### Testing
```bash
# Full test suite
python test_api.py

# Single endpoint test
curl -X POST http://localhost:8000/predict/transaction \
  -H "Content-Type: application/json" \
  -d '{"transaction": {"amount": 1000, "timestamp": "2025-11-28T12:00:00"}}'
```

### Automation
```bash
# Full pipeline
./run_pipeline.sh both --ensemble

# Just preprocessing
python preprocess.py

# Just training
python train.py --dataset both
```

---

## API Endpoints

### GET /health
```bash
curl http://localhost:8000/health
```
Response: `{"status": "ok", "available_models": ["transactions", "client_activity"]}`

### POST /predict/transaction
```bash
curl -X POST http://localhost:8000/predict/transaction \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "amount": 5000.0,
      "timestamp": "2025-11-28T14:30:00",
      "src_account_id": "user_123",
      "beneficiary_id": "benef_456"
    }
  }'
```

### POST /predict/client_activity
```bash
curl -X POST http://localhost:8000/predict/client_activity \
  -H "Content-Type: application/json" \
  -d '{
    "activity": {
      "timestamp": "2025-11-28T14:30:00",
      "src_account_id": "user_123",
      "logins_last_7_days": 5,
      "logins_last_30_days": 20,
      "login_frequency_7d": 0.71
    }
  }'
```

### POST /predict/combined
```bash
curl -X POST http://localhost:8000/predict/combined \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {...},
    "activity": {...}
  }'
```

---

## Response Format

```json
{
  "probability": 0.35,
  "threshold": 0.45,
  "action": "allow",
  "explanations": [
    {
      "feature": "amount",
      "shap_value": 0.15
    },
    {
      "feature": "hour",
      "shap_value": -0.05
    }
  ]
}
```

**Fields:**
- `probability` - fraud probability (0-1)
- `threshold` - decision threshold
- `action` - "allow", "challenge", or "block"
- `explanations` - top-5 feature contributions (SHAP values)

---

## File Structure After Training

```
models/
├── transactions/
│   ├── lightgbm_model.txt
│   ├── xgboost_model.json (if --ensemble)
│   ├── catboost_model.cbm (if --ensemble)
│   ├── model_meta.pkl
│   ├── shap_background.pkl
│   └── feature_importance.csv
│
└── client_activity/
    ├── lightgbm_model.txt
    ├── xgboost_model.json (if --ensemble)
    ├── catboost_model.cbm (if --ensemble)
    ├── model_meta.pkl
    ├── shap_background.pkl
    └── feature_importance.csv
```

---

## Python API Usage

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Predict transaction fraud
payload = {
    "transaction": {
        "amount": 50000.0,
        "timestamp": "2025-11-28T14:30:00",
        "src_account_id": "user_123",
        "beneficiary_id": "benef_456"
    }
}

response = requests.post(
    f"{BASE_URL}/predict/transaction",
    json=payload
)

result = response.json()
print(f"Fraud probability: {result['probability']:.2%}")
print(f"Action: {result['action']}")
for exp in result['explanations']:
    print(f"  - {exp['feature']}: {exp['shap_value']:.3f}")
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No such file` on CSV | Check paths: `data/transactions.csv`, `data/client_activity.csv` |
| `KeyError` on column names | CSV encoding is "windows-1251" with semicolon separator |
| Models not found | Run `python train.py --dataset both` first |
| API returns 400 error | Check model directory exists: `models/transactions/` or `models/client_activity/` |
| Connection refused | Start service: `python infer_service.py` |
| Out of memory | Remove `--ensemble` flag to use single LightGBM |
| Slow preprocessing | Normal for large datasets, increase patience |

---

## Common Workflows

### Workflow 1: Single Dataset (Transactions Only)
```bash
python preprocess.py
python train.py --dataset transactions
python infer_service.py
curl -X POST http://localhost:8000/predict/transaction -H "Content-Type: application/json" -d '{"transaction": {"amount": 1000, "timestamp": "2025-11-28T12:00:00"}}'
```

### Workflow 2: Both Datasets with Ensemble
```bash
python preprocess.py
python train.py --dataset both --ensemble
python infer_service.py
curl -X POST http://localhost:8000/predict/combined -H "Content-Type: application/json" -d '{"transaction": {...}, "activity": {...}}'
```

### Workflow 3: Optimize for Recall
```bash
python preprocess.py
python train.py --dataset both --threshold_strategy recall --min_recall 0.5
python infer_service.py
```

### Workflow 4: Quick Testing
```bash
python preprocess.py
python train.py  # defaults to both datasets
python infer_service.py &  # run in background
python test_api.py
```

---

## Directory Layout

```
ML1/
├── preprocess.py              # Preprocessing script
├── train.py                   # Training script
├── infer_service.py          # API service
├── test_api.py               # Test suite
├── run_pipeline.sh           # Automation script
├── requirements.txt          # Dependencies
│
├── data/
│   ├── transactions.csv
│   ├── client_activity.csv
│   ├── processed_transactions.parquet (after preprocess)
│   └── processed_client_activity.parquet (after preprocess)
│
├── models/                    # Created after training
│   ├── transactions/
│   └── client_activity/
│
└── docs/
    ├── README_MULTI_DATASET.md
    ├── ARCHITECTURE.md
    ├── CHANGES.md
    ├── CHECKLIST.md
    ├── SUMMARY.md
    └── QUICK_REFERENCE.md (this file)
```

---

## Performance Tips

1. **Use ensemble for better accuracy**: `--ensemble` flag
2. **Optimize for your use case**: Choose threshold strategy carefully
3. **Monitor feature importance**: Check `feature_importance.csv`
4. **Parallel processing**: Python handles parallel feature engineering
5. **Memory management**: Single LightGBM uses less memory than ensemble

---

## Important Notes

- ✓ Each dataset has **independent features** and **independent model**
- ✓ Models are saved in **separate directories**
- ✓ API **automatically detects** which model to use based on endpoint
- ✓ SHAP explanations are **specific to each model**
- ✓ Thresholds are **independently optimized** for each dataset
- ✓ Feature importance is **independently computed** for each model

---

## Help & Documentation

- **Full docs**: `README_MULTI_DATASET.md`
- **Architecture**: `ARCHITECTURE.md`
- **Changes**: `CHANGES.md`
- **Examples**: `example_usage.py`
- **Testing**: `test_api.py`

---

Last Updated: 2025-11-28
Status: Ready to Use ✅
