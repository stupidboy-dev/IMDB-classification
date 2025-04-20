```markdown
# Классификация тональности текста с использованием BERT

Проект по обучению модели для классификации тональности текста на основе BERT с использованием датасета IMDB Reviews (50k) и кастомных примеров.

## Основные особенности
- Использование предобученной BERT-модели от TensorFlow Hub
- Поддержка GPU-ускорения обучения
- Обработка сложных лингвистических конструкций в тексте
- Расширенный набор кастомных примеров с нюансированными отзывами
- Оценка качества с помощью метрик: Accuracy, ROC-AUC, F1-score

## Требования
- Python 3.8+
- TensorFlow 2.15+
- TensorFlow Hub
- TensorFlow Text
- scikit-learn
- pandas
- numpy

## Установка
```bash
pip install tensorflow tensorflow_hub tensorflow_text scikit-learn pandas numpy
```

## Структура проекта
```
├── IMDB Dataset.csv           # Основной датасет отзывов
├── model.keras                # Сохраненная обученная модель
├── sentiment_analysis.ipynb   # Основной файл с кодом
└── README.md                  # Документация
```

## Использование
1. Загрузите датасет с [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. Запустите ноутбук `sentiment_analysis.ipynb`
3. Для предсказания используйте функцию:
```python
def predict(text: list) -> str:
    """
    Возвращает предсказанную тональность текста
    Аргументы:
        text: list - список строк для классификации
    Возвращает:
        'positive' или 'negative'
    """
```

## Основные компоненты кода
1. **Предобработка данных**
   - Обработка пропущенных значений
   - Балансировка классов
   - Разделение на train/test (80/20)

2. **Архитектура модели**
   - BERT-энкодер (uncased L-12 H-768 A-12)
   - Слои нормализации и дропаута
   - Полносвязные слои с ReLU и сигмоидной активацией

3. **Обучение**
   - Оптимизатор: SGD
   - Функция потерь: Binary Crossentropy
   - Метрики: Accuracy
   - 10 эпох обучения на GPU

4. **Оценка качества**
   - Classification Report
   - ROC-AUC Score
   - Тестирование на кастомных примерах

## Примеры предсказаний
```python
predict(["The movie blends innovative storytelling with technical brilliance"])  # -> 'positive'
predict({"Their 'premium' service delivers subpar quality with arrogant attitude"}) # -> 'negative'
```

## Результаты
Модель демонстрирует:
- Accuracy: ~0.88 на тестовом наборе
- ROC-AUC: ~0.88
- Точность классификации сложных примеров: 85-90%

## Особенности реализации
- Поддержка смешанных конструкций ("Хотя... но...")
- Распознавание профессиональной лексики
- Учет контекстуальных противопоставлений
- Обработка иронических высказываний
- Анализ длинных составных предложений

