# Fraud Detection - Multi-Dataset Support

Обновленная версия кода поддерживает два датасета одновременно:
1. **transactions.csv** - транзакционные данные
2. **client_activity.csv** - активность клиентов

## Структура

### 1. Preprocessing (`preprocess.py`)

Теперь обрабатывает оба датасета отдельно с оптимизированным feature engineering для каждого:

```bash
# Обработать оба датасета
python preprocess.py \
    --transactions_input data/transactions.csv \
    --client_activity_input data/client_activity.csv \
    --transactions_output data/processed_transactions.parquet \
    --client_activity_output data/processed_client_activity.parquet

# Или обработать только один датасет
python preprocess.py \
    --transactions_input data/transactions.csv \
    --transactions_output data/processed_transactions.parquet
```

**Выход:**
- `data/processed_transactions.parquet` - обработанные транзакции
- `data/processed_client_activity.parquet` - обработанная активность клиентов

### 2. Training (`train.py`)

Обучает модели для одного или обоих датасетов:

```bash
# Обучить модели для обоих датасетов (по умолчанию)
python train.py \
    --processed_transactions data/processed_transactions.parquet \
    --processed_client_activity data/processed_client_activity.parquet \
    --dataset both \
    --ensemble

# Обучить только модель для транзакций
python train.py \
    --processed_transactions data/processed_transactions.parquet \
    --dataset transactions

# Обучить только модель для активности клиентов
python train.py \
    --processed_client_activity data/processed_client_activity.parquet \
    --dataset client_activity

# Обучить с настройками порога
python train.py \
    --processed_transactions data/processed_transactions.parquet \
    --processed_client_activity data/processed_client_activity.parquet \
    --dataset both \
    --threshold_strategy recall \
    --min_recall 0.5
```

**Параметры:**
- `--dataset {transactions|client_activity|both}` - какие датасеты обучать
- `--ensemble` - использовать ensemble (LightGBM + XGBoost + CatBoost)
- `--threshold_strategy` - стратегия выбора порога (precision, f1, balanced, recall)
- `--min_recall` - минимальный recall для стратегий (по умолчанию 0.4)

**Выход:**
```
models/
├── transactions/
│   ├── lightgbm_model.txt
 # Fraud Detection - Multi-Dataset Support

 Обновлённая версия кода поддерживает два датасета одновременно:
 1. **transactions.csv** - транзакционные данные
 2. **client_activity.csv** - активность клиентов

 ## Структура

 ### 1. Preprocessing (`src/preprocess.py`)

 Теперь обрабатывает оба датасета отдельно с оптимизированным feature engineering для каждого.

 Примеры:

 ```bash
 # Обработать оба датасета (wrapper)
 ./scripts/run_preprocess.sh

 # Или вызвать напрямую
 python src/preprocess.py \
     --transactions_input data/transactions.csv \
     --client_activity_input data/client_activity.csv \
     --transactions_output data/processed_transactions.parquet \
     --client_activity_output data/processed_client_activity.parquet
 ```

 **Выход:**
 - `data/processed_transactions.parquet` - обработанные транзакции
 - `data/processed_client_activity.parquet` - обработанная активность клиентов

 ### 2. Training (`src/train.py`)

 Обучает модели для одного или обоих датасетов. Используйте wrapper:

 ```bash
 # Обучить модели для обоих датасетов (ensemble)
 ./scripts/run_train.sh both --ensemble

 # Или напрямую
 python src/train.py \
     --processed_transactions data/processed_transactions.parquet \
     --processed_client_activity data/processed_client_activity.parquet \
     --dataset both \
     --ensemble
 ```

 Параметры:
 - `--dataset {transactions|client_activity|both}`
 - `--ensemble` - использовать ensemble (LightGBM + XGBoost + CatBoost)
 - `--threshold_strategy` - стратегия выбора порога (precision, f1, balanced, recall)

 **Выход:** структура в `models/` с поддиректориями для каждого датасета.

 ### 3. Inference Service (`src/infer_service.py`)

 FastAPI сервис с поддержкой предсказаний для обоих датасетов.

 Запуск:

 ```bash
 # Wrapper
 ./scripts/run_service.sh

 # Or direct
 python src/infer_service.py
 ```

 Сервис по умолчанию доступен на `http://localhost:8000`.

 #### API Endpoints (пример)

 Health check:

 ```bash
 curl http://localhost:8000/health
 ```

 Transaction prediction:

 ```bash
 curl -X POST http://localhost:8000/predict/transaction \
   -H "Content-Type: application/json" \
   -d '{
     "transaction": {
       "amount": 5000.0,
       "timestamp": "2025-11-28T12:30:00",
       "src_account_id": "123456",
       "beneficiary_id": "789012"
     }
   }'
 ```

 Client activity prediction:

 ```bash
 curl -X POST http://localhost:8000/predict/client_activity \
   -H "Content-Type: application/json" \
   -d '{
     "activity": {
       "timestamp": "2025-11-28T12:30:00",
       "src_account_id": "123456",
       "logins_last_7_days": 5,
       "logins_last_30_days": 20
     }
   }'
 ```

 Combined prediction:

 ```bash
 curl -X POST http://localhost:8000/predict/combined \
   -H "Content-Type: application/json" \
   -d '{
     "transaction": {"amount": 5000.0, "timestamp": "2025-11-28T12:30:00", "src_account_id": "123456"},
     "activity": {"timestamp": "2025-11-28T12:30:00", "src_account_id": "123456", "logins_last_7_days": 5}
   }'
 ```

 ## Быстрый старт

 ```bash
 # 1. Activate env
 source venv/bin/activate

 # 2. Install deps
 pip install -r requirements.txt

 # 3. Preprocess
 ./scripts/run_preprocess.sh

 # 4. Train
 ./scripts/run_train.sh both --ensemble

 # 5. Start service
 ./scripts/run_service.sh

 # 6. Run tests (in separate terminal)
 ./scripts/test_api.sh
 ```

 ## Примечания

 - Модели обучаются независимо с временным разбиением (train/val/test).
 - По умолчанию `train.py` использует стратегию выбора порога `balanced`.
 - Документация по feature engineering и структуре моделей находится в этом каталоге.
