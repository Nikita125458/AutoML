# AutoML
Данный проект представляет собой сравнительное исследование эффективности автоматического машинного обучения (AutoML с использованием AutoKeras) и ручного проектирования нейронных сетей. Эксперименты проводятся на трех различных датасетах и включают анализ производительности и воспроизводимости для обоих подходов.
## Структура проекта
## Структура проекта

**AutoML/** - Корневая папка проекта

**data/** - Исходные данные  
│  
└── **raw/** - Сырые датасеты  
  ├── `bank_marketing.csv`  
  ├── `housing.csv`  
  └── `customer_churn.csv`

**models/** - Обученные модели 
│  
├── **autokeras/** - Модели AutoKeras  
│  ├── `bank/`  
│  ├── `housing/`  
│  └── `churn/`  
│  
└── **neural/** - Ручные нейросети  
  ├── `bank/`  
  ├── `housing/`  
  └── `churn/`

**results/** - Результаты экспериментов
│  
├── **data_analysis/** - Анализ данных  
├── **autokeras/** - Результаты AutoKeras  
├── **neural/** - Результаты нейросетей  
├── **series_a/** - Серия A (AutoKeras анализ)  
├── **series_b/** - Серия B (Сравнение)  
├── **summary/** - Сводки  
└── **plots/** - Графики

**Код проекта:**  
├── `main.py` - Главный скрипт  
├── `data_analysis.py` - Анализ данных  
├── `data_preprocessing.py` - Предобработка  
├── `neural_experiment_simple.py` - Ручные нейросети  
├── `autokeras_experiment_simple.py` - AutoKeras  
└── `visualize_results.py` - Визуализация

**Конфигурационные файлы:**  
├── `requirements.txt` - Зависимости Python  
└──  `README.md` - Документация  

**Виртуальное окружение:**  
└── `venv_autokeras/` - Папка виртуального окружения

## Установка и запуск

1. **Клонирование репозитория:**
```bash
git clone <repository-url>
cd project_root
```
## 2. **Создание виртуального окружения:**
```bash
python -m venv venv_autokeras
```
# Активация окружения:

Windows:
```bash
venv_autokeras\Scripts\activate
```
Linux/Mac:
```bash
source venv_autokeras/bin/activate
```
# 4. Установка зависимостей:
```bash
pip install -r requirements.txt
```
# 5. Запуск экспериментов:

```bash
python main.py
```
Эксперименты
Series A: AutoKeras
- Анализ зависимости качества от количества trials (7, 15, 30, 60)
- Анализ воспроизводимости с разными random seeds (42, 43, 44)
Series B: Сравнение с ручными архитектурами
- AutoKeras (60 trials) vs две ручные архитектуры нейросетей (каждая архитектура обучается ~ 1 час)

Датасеты
- Bank Marketing - классификация подписки на депозит
- California Housing - регрессия стоимости жилья
- Telecom Churn - классификация оттока клиентов

Результаты
- Результаты сохраняются в папке results/, включая:
- JSON файлы с метриками
- Сохраненные модели
- Графики анализа
- Сводные таблицы
