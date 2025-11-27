# 📊 Fraud Detection - Multi-Dataset Implementation Complete

## 🎉 Summary

Ваш код успешно переделан для работы с **двумя датасетами одновременно**:
- ✅ `transactions.csv` - транзакционные данные
- ✅ `client_activity.csv` - активность клиентов

Каждый датасет получает **свою оптимизированную модель** с собственным набором признаков.

---

## 📁 Структура Проекта

```
ML1/
├── Data Processing
│   ├── preprocess.py              ← Обновлено: работает с обоими датасетами
│   ├── data/
│   │   ├── transactions.csv
│   │   ├── client_activity.csv
│   │   ├── processed_transactions.parquet      ← Новое
│   │   └── processed_client_activity.parquet   ← Новое
│
├── Model Training
│   ├── train.py                   ← Обновлено: обучает две модели
│   ├── models/
│   │   ├── transactions/          ← Новая директория
│   │   │   ├── lightgbm_model.txt
│   │   │   ├── model_meta.pkl
│   │   │   ├── shap_background.pkl
│   │   │   └── feature_importance.csv
│   │   └── client_activity/       ← Новая директория
│   │       ├── lightgbm_model.txt
│   │       ├── model_meta.pkl
│   │       ├── shap_background.pkl
│   │       └── feature_importance.csv
│
├── API Inference
│   ├── infer_service.py           ← Обновлено: API для обоих датасетов
│   ├── test_api.py                ← Новое: тестирование API
│
├── Scripts
│   ├── run_pipeline.sh            ← Новое: автоматизация всего pipeline
│   └── example_usage.py           ← Новое: примеры использования
│
├── Documentation
│   ├── README_MULTI_DATASET.md    ← Новое: полная документация
│   ├── ARCHITECTURE.md            ← Новое: диаграммы системы
│   ├── CHANGES.md                 ← Новое: детали изменений
│   ├── CHECKLIST.md               ← Новое: чек-лист проверки
│   └── README.md                  ← Старое: оригинальная документация
│
└── Configuration
    ├── requirements.txt
    └── .gitignore
```

---

## 🔄 Data Pipeline

### Шаг 1: Preprocessing (preprocess.py)

```bash
python preprocess.py \
    --transactions_input data/transactions.csv \
    --client_activity_input data/client_activity.csv \
    --transactions_output data/processed_transactions.parquet \
    --client_activity_output data/processed_client_activity.parquet
```

**Выход:**
- `processed_transactions.parquet` (30+ признаков для транзакций)
- `processed_client_activity.parquet` (20+ признаков для активности)

### Шаг 2: Training (train.py)

```bash
# Обучить обе модели с ensemble
python train.py --dataset both --ensemble

# Или только одну
python train.py --dataset transactions
python train.py --dataset client_activity
```

**Выход:**
- `models/transactions/` (модель для транзакций)
- `models/client_activity/` (модель для активности)

### Шаг 3: Inference (infer_service.py)

```bash
python infer_service.py  # Runs on http://localhost:8000
```

**API Endpoints:**
- `GET /health` - статус сервиса
- `POST /predict/transaction` - предсказание для транзакций
- `POST /predict/client_activity` - предсказание для активности
- `POST /predict/combined` - оба предсказания одновременно

---

## 🚀 Quick Start

```bash
# 1️⃣ Обработать данные
python preprocess.py

# 2️⃣ Обучить модели
python train.py --dataset both --ensemble

# 3️⃣ Запустить API сервис
python infer_service.py

# 4️⃣ В другом терминале: тестировать API
python test_api.py
```

Или используйте скрипт:
```bash
./run_pipeline.sh both --ensemble
python infer_service.py
python test_api.py
```

---

## 📡 API Usage Examples

### 1. Transaction Prediction
```bash
curl -X POST http://localhost:8000/predict/transaction \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "amount": 50000.0,
      "timestamp": "2025-11-28T14:30:00",
      "src_account_id": "user_123",
      "beneficiary_id": "benef_456"
    }
  }'
```

### 2. Client Activity Prediction
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

### 3. Combined Prediction (Both)
```bash
curl -X POST http://localhost:8000/predict/combined \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {...},
    "activity": {...}
  }'
```

### Response Format
```json
{
  "probability": 0.35,
  "threshold": 0.45,
  "action": "allow",
  "explanations": [
    {"feature": "amount", "shap_value": 0.15},
    {"feature": "hour", "shap_value": -0.05},
    {"feature": "is_new_beneficiary", "shap_value": 0.12}
  ]
}
```

---

## 🎯 Key Features

### ✨ Двойные Модели
- Независимое обучение для каждого датасета
- Специфичные признаки для каждого типа данных
- Отдельные пороги классификации

### 🔍 SHAP Объяснения
- Интерпретируемость предсказаний
- Top-5 самых важных признаков
- Визуализация влияния признаков

### 🎛️ Гибкие Параметры
- Стратегии выбора порога: precision, f1, recall, balanced
- Выбор между single LightGBM и ensemble (LightGBM + XGBoost + CatBoost)
- Настройка желаемой precision/recall

### 📊 Comprehensive Metrics
- ROC-AUC, PR-AUC, F1, Precision, Recall
- Confusion matrix
- Feature importance
- Time-based train/val/test split

---

## 📚 Documentation Files

| Файл | Назначение |
|------|-----------|
| **README_MULTI_DATASET.md** | Полная документация с примерами |
| **ARCHITECTURE.md** | Диаграммы и flow'ы системы |
| **CHANGES.md** | Детали изменений в каждом файле |
| **CHECKLIST.md** | Чек-лист проверки и troubleshooting |
| **example_usage.py** | Примеры кода и использования |
| **test_api.py** | Скрипт для тестирования API |
| **run_pipeline.sh** | Bash скрипт для автоматизации |

---

## 🔧 Modified Files

### preprocess.py
- **Было**: 1 функция `basic_feature_engineering()` для одного датасета
- **Стало**: 2 функции `preprocess_transactions()` и `preprocess_client_activity()`
- **Параметры**: добавлены `--transactions_input`, `--client_activity_input`, и т.д.

### train.py
- **Было**: обучение одной модели в `models/`
- **Стало**: обучение двух моделей в `models/transactions/` и `models/client_activity/`
- **Параметры**: добавлен `--dataset {transactions|client_activity|both}`

### infer_service.py
- **Было**: 1 endpoint `/predict`
- **Стало**: 4 endpoint'а (`/health`, `/predict/transaction`, `/predict/client_activity`, `/predict/combined`)
- **Функциональность**: поддержка двух моделей с независимыми SHAP explainers

---

## ✅ Validation

✓ Python синтаксис проверен  
✓ Все параметры документированы  
✓ Примеры использования предоставлены  
✓ API документация готова  
✓ Скрипты для автоматизации созданы  

---

## 🎓 What You Can Do Now

1. **Обработать оба датасета** с оптимизированным feature engineering
2. **Обучить две независимые модели** для разных типов предсказаний
3. **Запустить единый API сервис** поддерживающий оба типа запросов
4. **Получать объяснения** для каждого предсказания через SHAP
5. **Выбирать стратегии обучения** (precision, recall, balanced, f1)
6. **Использовать ensemble** модели для лучшей точности

---

## 📞 Support & Troubleshooting

Если возникли вопросы:

1. **Прочитать** CHANGES.md для понимания что изменилось
2. **Посмотреть** example_usage.py для примеров кода
3. **Проверить** README_MULTI_DATASET.md для полной документации
4. **Запустить** test_api.py для проверки API
5. **Смотреть** ARCHITECTURE.md для понимания flow'ов

---

## 🎉 Ready to Go!

Ваш код полностью готов для работы с двумя датасетами.

**Начните с:**
```bash
python preprocess.py
python train.py --dataset both --ensemble
python infer_service.py
```

Успехов! 🚀
