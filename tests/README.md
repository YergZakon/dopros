# Система тестирования ДОПРОС MVP 2.0

Полная система тестирования и валидации для проекта мультимодального анализа эмоций.

## 🧪 Структура тестов

### Основные тестовые модули

- **`test_compatibility.py`** - Проверка окружения и совместимости
- **`test_pipeline.py`** - Интеграционные тесты пайплайна
- **`test_models.py`** - Тесты моделей машинного обучения

### Вспомогательные файлы

- **`fixtures/conftest.py`** - Фикстуры для тестов
- **`fixtures/test_data_generator.py`** - Генератор тестовых данных
- **`pytest.ini`** - Конфигурация pytest
- **`run_tests.py`** - Скрипт запуска тестов

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Активируем виртуальное окружение
source venv/bin/activate

# Устанавливаем зависимости для тестирования
pip install pytest pytest-cov pytest-mock pytest-asyncio psutil
```

### 2. Запуск всех тестов

```bash
# Простой запуск
pytest tests/ -v

# С анализом покрытия кода
pytest tests/ -v --cov=core --cov=models --cov-report=html

# Использование скрипта
python run_tests.py --type all --coverage
```

### 3. Запуск отдельных типов тестов

```bash
# Только проверка совместимости
python run_tests.py --type compatibility

# Только тесты моделей
python run_tests.py --type models

# Только интеграционные тесты
python run_tests.py --type integration
```

## 📋 Типы тестов

### 🔧 Тесты совместимости (`test_compatibility.py`)

Проверяют окружение и зависимости:

- ✅ Версия Python
- ✅ Наличие конфигурационных файлов
- ✅ Импорт основных библиотек
- ✅ Доступность GPU/CUDA
- ✅ Установка FFmpeg
- ✅ API ключи
- ✅ Загрузка моделей YOLO
- ✅ Структура директорий

**Запуск:**
```bash
pytest tests/test_compatibility.py -v
python run_tests.py --setup-only
```

### 🔄 Интеграционные тесты (`test_pipeline.py`)

Тестируют полный пайплайн обработки:

- ✅ Инициализация компонентов
- ✅ Обработка видео и аудио
- ✅ Fallback механизмы
- ✅ Восстановление после ошибок
- ✅ Целостность данных
- ✅ Производительность

**Запуск:**
```bash
pytest tests/test_pipeline.py -v
python run_tests.py --type pipeline
```

### 🤖 Тесты моделей (`test_models.py`)

Тестируют ML модели:

- ✅ Анализатор речи
- ✅ YOLO детекция лиц
- ✅ Анализ эмоций (DeepFace, FER)
- ✅ Fallback между моделями
- ✅ Производительность моделей

**Запуск:**
```bash
pytest tests/test_models.py -v
python run_tests.py --type models
```

## 🏷️ Маркеры тестов

### Основные маркеры

- `@pytest.mark.slow` - Медленные тесты (> 10 сек)
- `@pytest.mark.gpu` - Требуют GPU
- `@pytest.mark.integration` - Интеграционные тесты
- `@pytest.mark.unit` - Юнит-тесты
- `@pytest.mark.performance` - Тесты производительности

### Использование маркеров

```bash
# Исключить медленные тесты
pytest -m "not slow"

# Только GPU тесты
pytest -m "gpu"

# Только быстрые юнит-тесты
pytest -m "unit and not slow"
```

## 📊 Анализ покрытия кода

### Генерация отчета

```bash
# HTML отчет
pytest --cov=core --cov=models --cov-report=html

# Терминальный отчет
pytest --cov=core --cov=models --cov-report=term-missing

# Только покрытие без тестов
coverage report -m
```

### Просмотр результатов

```bash
# Открыть HTML отчет
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## 🛠️ Создание тестовых данных

### Автоматическая генерация

```python
from tests.fixtures.test_data_generator import TestDataGenerator

generator = TestDataGenerator()

# Создать тестовое видео
generator.create_test_video("test.mp4", duration=5, fps=2)

# Создать тестовое аудио
generator.create_test_audio("test.wav", duration=5, audio_type="speech")

# Создать датасет эмоций
emotion_data = generator.create_emotion_dataset(100)
```

### Использование фикстур

```python
def test_with_fixtures(test_video_file, test_audio_file, test_config):
    """Тест с использованием готовых фикстур"""
    pipeline = MasterPipeline(test_config)
    result = pipeline.process_video(test_video_file)
    assert result is not None
```

## ⚡ Производительность

### Бенчмарки

```bash
# Тесты производительности
pytest -m "performance" -v

# С профилированием
pytest --profile-svg
```

### Ожидаемые показатели

- **Обработка видео**: < 2 сек/кадр
- **Анализ аудио**: < 0.5 сек/сек аудио
- **Детекция эмоций**: < 0.5 сек/лицо
- **Память**: < 500 MB для видео 1080p

## 🐛 Отладка тестов

### Подробный вывод

```bash
# Максимальная детализация
pytest -vvv -s --tb=long

# Остановка на первой ошибке
pytest -x

# Запуск последних упавших тестов
pytest --lf
```

### Логирование

```bash
# Включить логирование
pytest --log-cli-level=DEBUG

# Сохранить логи в файл
pytest --log-file=test.log
```

## 🔧 Конфигурация

### Основные настройки (`pytest.ini`)

```ini
[tool:pytest]
testpaths = tests
markers = 
    slow: медленные тесты
    gpu: тесты для GPU
    integration: интеграционные тесты
addopts = --verbose --strict-markers --tb=short
```

### Переменные окружения

```bash
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=""  # Отключить GPU для тестов
export TF_CPP_MIN_LOG_LEVEL=3   # Минимальное логирование TensorFlow
```

## 📈 Continuous Integration

### GitHub Actions

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: python run_tests.py --type all --coverage
```

## 📝 Добавление новых тестов

### Структура теста

```python
import pytest
from unittest.mock import Mock, patch

class TestNewFeature:
    """Тесты новой функции"""
    
    def test_basic_functionality(self):
        """Базовый тест функциональности"""
        # Arrange
        input_data = "test"
        
        # Act
        result = new_function(input_data)
        
        # Assert
        assert result is not None
    
    @pytest.mark.slow
    def test_performance(self):
        """Тест производительности"""
        import time
        start = time.time()
        
        # Выполняем операцию
        heavy_operation()
        
        duration = time.time() - start
        assert duration < 10.0  # Не более 10 секунд
```

### Рекомендации

1. **Один тест = одна проверка**
2. **Используйте описательные имена**
3. **Добавляйте маркеры для категоризации**
4. **Мокируйте внешние зависимости**
5. **Тестируйте граничные случаи**
6. **Проверяйте обработку ошибок**

## 🆘 Решение проблем

### Частые ошибки

**1. Import Error**
```bash
# Проверьте PYTHONPATH
export PYTHONPATH=.
pytest tests/
```

**2. CUDA ошибки**
```bash
# Отключите GPU для тестов
export CUDA_VISIBLE_DEVICES=""
pytest tests/
```

**3. Медленные тесты**
```bash
# Исключите медленные тесты
pytest -m "not slow"
```

**4. Недостаток памяти**
```bash
# Запускайте тесты последовательно
pytest -n 1
```

### Полезные команды

```bash
# Список всех тестов
pytest --collect-only

# Список маркеров
pytest --markers

# Справка по опциям
pytest --help

# Версия pytest
pytest --version
```

## 📚 Дополнительные ресурсы

- [Документация pytest](https://docs.pytest.org/)
- [Руководство по тестированию ML](https://madewithml.com/courses/mlops/testing/)
- [Pytest-cov документация](https://pytest-cov.readthedocs.io/)
- [Моки в Python](https://docs.python.org/3/library/unittest.mock.html)