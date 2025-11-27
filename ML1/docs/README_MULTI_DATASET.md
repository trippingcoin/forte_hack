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
│   ├── xgboost_model.json (если ensemble)
│   ├── catboost_model.cbm (если ensemble)
│   ├── model_meta.pkl
│   ├── shap_background.pkl
│   └── feature_importance.csv
├── client_activity/
│   ├── lightgbm_model.txt
│   ├── xgboost_model.json (если ensemble)
│   ├── catboost_model.cbm (если ensemble)
│   ├── model_meta.pkl
│   ├── shap_background.pkl
│   └── feature_importance.csv
```

### 3. Inference Service (`infer_service.py`)

FastAPI сервис с поддержкой предсказаний для обоих датасетов:

```bash
# Запустить сервис (автоматически загружает обе модели)
python infer_service.py
```

Сервис будет запущен на `http://localhost:8000`

#### API Endpoints

**1. Health Check**
```bash
curl http://localhost:8000/health
```

**2. Предсказание для транзакций**
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

**3. Предсказание для активности клиента**
```bash
curl -X POST http://localhost:8000/predict/client_activity \
  -H "Content-Type: application/json" \
  -d '{
    "activity": {
      "timestamp": "2025-11-28T12:30:00",
      "src_account_id": "123456",
      "logins_last_7_days": 5,
      "logins_last_30_days": 20,
      "login_frequency_7d": 0.71,
      "avg_login_interval_30d": 100000.0
    }
  }'
```

**4. Комбинированное предсказание для обоих типов данных**
```bash
curl -X POST http://localhost:8000/predict/combined \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "amount": 5000.0,
      "timestamp": "2025-11-28T12:30:00",
      "src_account_id": "123456",
      "beneficiary_id": "789012"
    },
    "activity": {
      "timestamp": "2025-11-28T12:30:00",
      "src_account_id": "123456",
      "logins_last_7_days": 5,
      "logins_last_30_days": 20
    }
  }'
```

#### Response Format

```json
{
  "probability": 0.15,
  "threshold": 0.45,
  "action": "allow",
  "explanations": [
    {
      "feature": "amount",
      "shap_value": 0.023
    },
    {
      "feature": "hour",
      "shap_value": -0.012
    }
  ]
}
```

## Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Обработать данные
python preprocess.py

# 3. Обучить модели
python train.py --dataset both --ensemble

# 4. Запустить сервис
python infer_service.py

# 5. В другом терминале: сделать тестовое предсказание
curl -X POST http://localhost:8000/predict/transaction \
  -H "Content-Type: application/json" \
  -d '{"transaction": {"amount": 1000, "timestamp": "2025-11-28T12:00:00"}}'
```

## Отличия в Feature Engineering

### Для transactions.csv:
- Временные признаки (hour, dow, is_weekend, etc.)
- Amount трансформации (log, squared, sqrt)
- Признаки по бенефициару (unique users, tx count)
- Скользящие статистики по пользователю
- Взаимодействия признаков

### Для client_activity.csv:
- Временные признаки
- Log трансформации числовых признаков
- Стандартизация активности логинов
- Признаки девайса и ОС

## Структура моделей

Каждый датасет имеет свои:
- Обученную модель (LightGBM / Ensemble)
- Набор фич (features)
- Порог классификации (threshold)
- SHAP background для объяснений
- Feature importance

## Примечания

- Модели обучаются независимо с временным разбиением (train/val/test)
- Supports both single LightGBM и ensemble моделей
- SHAP объяснения показывают top-5 самых важных признаков для каждого предсказания
- Сервис автоматически выбирает подходящую модель в зависимости от типа запроса
