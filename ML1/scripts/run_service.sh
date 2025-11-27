#!/bin/bash
# Wrapper to start inference service
set -e
python src/infer_service.py
# Alternatively: uvicorn src.infer_service:app --host 0.0.0.0 --port 8000
