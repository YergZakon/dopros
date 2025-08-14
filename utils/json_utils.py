"""
Утилиты для работы с JSON и сериализацией данных
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Union
from dataclasses import is_dataclass, asdict

class NumpyJSONEncoder(json.JSONEncoder):
    """
    JSON энкодер для numpy типов данных
    """
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif is_dataclass(obj):
            return asdict(obj)
        
        return super().default(obj)

def safe_json_serialize(data: Any) -> str:
    """
    Безопасная сериализация данных в JSON
    
    Args:
        data: Данные для сериализации
        
    Returns:
        JSON строка
    """
    try:
        return json.dumps(data, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2)
    except Exception as e:
        # Если все еще есть проблемы, попробуем очистить данные
        cleaned_data = clean_for_json(data)
        return json.dumps(cleaned_data, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2)

def clean_for_json(obj: Any) -> Any:
    """
    Рекурсивная очистка объекта для JSON сериализации
    
    Args:
        obj: Объект для очистки
        
    Returns:
        Очищенный объект
    """
    if isinstance(obj, dict):
        return {key: clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_for_json(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.astype(float).tolist()
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif is_dataclass(obj):
        return clean_for_json(asdict(obj))
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj

def save_json_safe(data: Any, file_path: Union[str, Path], **kwargs) -> bool:
    """
    Безопасное сохранение данных в JSON файл
    
    Args:
        data: Данные для сохранения
        file_path: Путь к файлу
        **kwargs: Дополнительные параметры для json.dump
        
    Returns:
        True если успешно, False иначе
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2, **kwargs)
        
        return True
    except Exception as e:
        print(f"Ошибка сохранения JSON: {e}")
        try:
            # Попытка с очисткой данных
            cleaned_data = clean_for_json(data)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2, **kwargs)
            return True
        except Exception as e2:
            print(f"Ошибка сохранения очищенного JSON: {e2}")
            return False

def load_json_safe(file_path: Union[str, Path]) -> Union[Dict, List, None]:
    """
    Безопасная загрузка JSON файла
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Загруженные данные или None при ошибке
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки JSON: {e}")
        return None