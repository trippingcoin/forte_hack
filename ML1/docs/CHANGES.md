# Изменения в коде для поддержки двух датасетов

## Краткое резюме

Код переделан так, чтобы **одновременно работать с двумя датасетами**:
- ✅ `transactions.csv` - транзакционные данные (fraud detection)
- ✅ `client_activity.csv` - активность клиентов (behavior analysis)

Каждый датасет имеет **свою модель**, обученную на специфических для него признаках.

---

## Файл: preprocess.py

### Что было:
- Обрабатывал только один датасет
- Функция: `basic_feature_engineering(df)`
- Выход: один `processed.parquet`

### Что стало:
- ✅ Обрабатывает **оба датасета одновременно**
- ✅ Функции: `preprocess_transactions()` и `preprocess_client_activity()`
- ✅ Выход:
  - `processed_transactions.parquet` (с признаками для транзакций)
  - `processed_client_activity.parquet` (с признаками для активности)

### Использование:
```bash
# Обработать оба датасета
python preprocess.py

# Или с явными параметрами
python preprocess.py \
    --transactions_input data/transactions.csv \
    --client_activity_input data/client_activity.csv \
    --transactions_output data/processed_transactions.parquet \
    --client_activity_output data/processed_client_activity.parquet
```

---

## Файл: train.py

### Что было:
- Обучал модель только на одном датасете
- Аргумент: `--processed` (один файл)
- Выход: модель в одной директории `models/`

### Что стало:
- ✅ Обучает **две модели независимо**
- ✅ Новый аргумент: `--dataset {transactions|client_activity|both}`
- ✅ Выход:
  - `models/transactions/` (модель для транзакций)
  - `models/client_activity/` (модель для активности)
- ✅ Каждая модель имеет свои:
  - Обученные веса
  - Набор признаков
  - Порог классификации
  - Feature importance

### Использование:
```bash
# Обучить обе модели (по умолчанию)
python train.py --dataset both --ensemble

# Обучить только модель для транзакций
python train.py --dataset transactions

# Обучить только модель для активности клиентов
python train.py --dataset client_activity
```

---

## Файл: infer_service.py

### Что было:
- Один API endpoint: `/predict`
- Загружал одну модель
- Работал только с транзакциями

### Что стало:
- ✅ Три API endpoint'а:
  1. `/predict/transaction` - предсказание для транзакций
  2. `/predict/client_activity` - предсказание для активности
  3. `/predict/combined` - предсказания для обоих датасетов одновременно

- ✅ Загружает **две модели** при старте
- ✅ Каждая модель имеет:
  - Свой explainer (для SHAP объяснений)
  - Свой набор признаков
  - Свой порог классификации

### Использование:
```bash
# Запустить сервис (автоматически загружает обе модели)
python infer_service.py

# Тестовый запрос для транзакций
curl -X POST http://localhost:8000/predict/transaction \
  -H "Content-Type: application/json" \
  -d '{"transaction": {"amount": 5000, "timestamp": "2025-11-28T12:00:00"}}'

# Тестовый запрос для активности
curl -X POST http://localhost:8000/predict/client_activity \
  -H "Content-Type: application/json" \
  -d '{"activity": {"timestamp": "2025-11-28T12:00:00", "logins_last_7_days": 5}}'

# Комбинированный запрос
curl -X POST http://localhost:8000/predict/combined \
  -H "Content-Type: application/json" \
  -d '{"transaction": {...}, "activity": {...}}'
```

---

## Структура моделей

```
models/
├── transactions/
│   ├── lightgbm_model.txt
│   ├── xgboost_model.json (если --ensemble)
│   ├── catboost_model.cbm (если --ensemble)
│   ├── model_meta.pkl
│   ├── shap_background.pkl
│   └── feature_importance.csv
│
└── client_activity/
    ├── lightgbm_model.txt
    ├── xgboost_model.json (если --ensemble)
    ├── catboost_model.cbm (если --ensemble)
    ├── model_meta.pkl
    ├── shap_background.pkl
    └── feature_importance.csv
```

---

## Быстрый старт (новый workflow)

```bash
# 1. Обработать оба датасета
python preprocess.py

# 2. Обучить модели для обоих датасетов (с ensemble)
python train.py --dataset both --ensemble

# 3. Запустить API сервис
python infer_service.py

# 4. В другом терминале: тестировать API
python test_api.py
```

Или использовать скрипт:
```bash
./run_pipeline.sh both --ensemble
python infer_service.py
python test_api.py
```

---

## Ключевые преимущества

1. **Независимые модели**: Каждый датасет обучается независимо с оптимизированными признаками
2. **Гибкость**: Можно обучать любой набор датасетов или добавлять новые
3. **API для обоих**: Единый сервис поддерживает предсказания для обоих типов данных
4. **SHAP объяснения**: Каждое предсказание получает объяснения специфичные для своего датасета
5. **Масштабируемость**: Легко добавить новые датасеты - просто нужны функции preprocessing и параметры в train.py/infer_service.py

---

## Файлы для помощи

- **README_MULTI_DATASET.md** - полная документация с примерами
- **example_usage.py** - примеры использования
- **test_api.py** - скрипт для тестирования API
- **run_pipeline.sh** - bash скрипт для запуска pipeline

---

## Примечания

- Каждая модель сохраняет свой набор признаков в `model_meta.pkl`
- SHAP background (для быстрых объяснений) сохраняется отдельно для каждой модели
- Feature importance вычисляется независимо для каждой модели
- Пороги классификации выбираются независимо (могут быть разными для разных датасетов)
