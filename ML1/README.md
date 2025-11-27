# forte_hack

Lightweight fraud-detection pipeline (multi-dataset).

Quick start

1. Activate your virtualenv:

```bash
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Preprocess, train and run tests (wrappers):

```bash
# Preprocess data
./scripts/run_preprocess.sh

# Train models (both datasets, ensemble)
./scripts/run_train.sh both --ensemble

# Start inference service
./scripts/run_service.sh

# Run API tests (ensure service is running)
./scripts/test_api.sh
```

Layout

- `src/` — main Python modules: `preprocess.py`, `train.py`, `infer_service.py`, `example_usage.py`
- `scripts/` — convenience shell wrappers: `run_preprocess.sh`, `run_train.sh`, `run_service.sh`, `run_pipeline.sh`, `test_api.sh`
- `data/` — raw and processed CSV / parquet files
- `models/` — trained model artifacts (per dataset)
- `docs/` — documentation and how-tos
- `tests/` — integration test scripts for the API

See `docs/README_MULTI_DATASET.md` for detailed usage and API examples.