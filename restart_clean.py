#!/usr/bin/env python3
"""
Скрипт для полной очистки кэша и перезапуска системы
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def clear_python_cache():
    """Очистка Python кэша"""
    print("🧹 Очистка Python кэша...")
    
    # Удаление __pycache__ директорий
    for root, dirs, files in os.walk('.'):
        for d in dirs:
            if d == '__pycache__':
                cache_path = os.path.join(root, d)
                try:
                    shutil.rmtree(cache_path)
                    print(f"   ✅ Удален {cache_path}")
                except:
                    pass
    
    # Удаление .pyc файлов
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.pyc'):
                pyc_path = os.path.join(root, f)
                try:
                    os.remove(pyc_path)
                    print(f"   ✅ Удален {pyc_path}")
                except:
                    pass

def clear_streamlit_cache():
    """Очистка Streamlit кэша"""
    print("🧹 Очистка Streamlit кэша...")
    
    streamlit_cache = Path.home() / '.streamlit'
    if streamlit_cache.exists():
        try:
            shutil.rmtree(streamlit_cache)
            print("   ✅ Streamlit кэш очищен")
        except:
            print("   ⚠️ Не удалось очистить Streamlit кэш")

def clear_system_modules():
    """Очистка системных модулей из памяти"""
    print("🧹 Очистка системных модулей...")
    
    modules_to_clear = []
    for module_name in list(sys.modules.keys()):
        if any(module_name.startswith(prefix) for prefix in ['core', 'models', 'utils', 'integrations']):
            modules_to_clear.append(module_name)
    
    for module_name in modules_to_clear:
        try:
            del sys.modules[module_name]
            print(f"   ✅ Модуль {module_name} выгружен")
        except:
            pass

def set_environment():
    """Установка переменных окружения"""
    print("🔧 Настройка окружения...")
    
    # Принудительное отключение GPU для DeepFace
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false' 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("   ✅ DeepFace переключен на CPU-only режим")

if __name__ == "__main__":
    print("🚀 Начинаю полную очистку системы...")
    
    clear_python_cache()
    clear_streamlit_cache() 
    clear_system_modules()
    set_environment()
    
    print("\n✅ Очистка завершена!")
    print("🎯 Теперь запустите: streamlit run main.py")
    print("🔥 Все оптимизации будут применены с чистого листа")