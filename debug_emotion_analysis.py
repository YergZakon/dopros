#!/usr/bin/env python3
"""
Диагностика проблем анализа эмоций
"""

import cv2
import numpy as np
import os
from pathlib import Path
import yaml

# Принудительно отключаем GPU для тестирования
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_fer_emotion_analysis():
    """Тест FER анализа эмоций"""
    print("🎭 Тестируем FER анализ эмоций...")
    
    try:
        from fer import FER
        detector = FER(mtcnn=True)
        
        # Загружаем тестовый кадр
        if os.path.exists('test_face_detection.jpg'):
            image = cv2.imread('test_face_detection.jpg')
            result = detector.detect_emotions(image)
            
            print(f"  📊 FER результат: {len(result)} лиц обнаружено")
            
            if result:
                for i, face_data in enumerate(result):
                    emotions = face_data['emotions']
                    dominant = max(emotions, key=emotions.get)
                    confidence = emotions[dominant]
                    
                    print(f"    🎯 Лицо {i+1}: {dominant} ({confidence:.2%})")
                    print(f"    📋 Все эмоции: {emotions}")
                
                return True
            else:
                print("  ❌ FER не обнаружил эмоции")
                return False
        else:
            print("  ❌ Тестовый кадр не найден")
            return False
            
    except Exception as e:
        print(f"  ❌ Ошибка FER: {e}")
        return False

def test_deepface_emotion_analysis():
    """Тест DeepFace анализа эмоций"""
    print("\n🧠 Тестируем DeepFace анализ эмоций...")
    
    try:
        from deepface import DeepFace
        
        if os.path.exists('test_face_detection.jpg'):
            result = DeepFace.analyze(
                img_path='test_face_detection.jpg',
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            print(f"  📊 DeepFace результат получен")
            
            if isinstance(result, list):
                result = result[0] if result else None
            
            if result and 'emotion' in result:
                emotions = result['emotion']
                dominant = max(emotions, key=emotions.get)
                confidence = emotions[dominant]
                
                print(f"    🎯 Доминирующая эмоция: {dominant} ({confidence:.1f}%)")
                print(f"    📋 Все эмоции: {emotions}")
                
                return True
            else:
                print("  ❌ DeepFace не вернул эмоции")
                return False
        else:
            print("  ❌ Тестовый кадр не найден")
            return False
            
    except Exception as e:
        print(f"  ❌ Ошибка DeepFace: {e}")
        return False

def test_yolo_model_loading():
    """Тест загрузки YOLO модели"""
    print("\n🎯 Тестируем загрузку YOLO модели...")
    
    try:
        from ultralytics import YOLO
        
        # Проверяем стандартную модель
        if os.path.exists('yolo11n.pt'):
            model = YOLO('yolo11n.pt')
            print(f"  ✅ YOLO модель yolo11n.pt загружена")
            print(f"  📋 Задача: {model.task}")
            print(f"  📋 Классы: {len(model.names)} ({list(model.names.values())[:5]}...)")
            return True
        else:
            print("  ❌ Модель yolo11n.pt не найдена")
            return False
            
    except Exception as e:
        print(f"  ❌ Ошибка YOLO: {e}")
        return False

def check_config_settings():
    """Проверяем настройки конфигурации"""
    print("\n⚙️ Проверяем настройки конфигурации...")
    
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Проверяем настройки анализа эмоций
        models = config.get('processing', {}).get('models', {})
        
        print("  📋 Настройки моделей:")
        
        # YOLO настройки
        yolo_config = models.get('yolo', {}).get('models', {})
        print(f"    🎯 YOLO лица: {yolo_config.get('face', 'не указано')}")
        print(f"    🎯 YOLO эмоции: {yolo_config.get('emotion', 'не указано')}")
        
        # DeepFace настройки
        deepface_config = models.get('deepface', {})
        print(f"    🧠 DeepFace включен: {deepface_config.get('enabled', False)}")
        print(f"    🧠 DeepFace backend: {deepface_config.get('backend', 'не указано')}")
        
        # FER настройки
        fer_config = models.get('fer', {})
        print(f"    🎭 FER включен: {fer_config.get('enabled', False)}")
        print(f"    🎭 FER MTCNN: {fer_config.get('mtcnn', False)}")
        
        # Пороги анализа
        analysis_config = config.get('analysis', {})
        print(f"  📊 Порог изменения эмоций: {analysis_config.get('emotion_change_threshold', 'не указано')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка чтения конфига: {e}")
        return False

def check_required_packages():
    """Проверяем наличие необходимых пакетов"""
    print("\n📦 Проверяем установленные пакеты...")
    
    packages = {
        'cv2': 'OpenCV',
        'fer': 'FER', 
        'deepface': 'DeepFace',
        'ultralytics': 'YOLO',
        'tensorflow': 'TensorFlow',
        'torch': 'PyTorch'
    }
    
    results = {}
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
            results[package] = True
        except ImportError:
            print(f"  ❌ {name}")
            results[package] = False
    
    return all(results.values())

if __name__ == "__main__":
    print("🔧 Запуск диагностики анализа эмоций...")
    
    # Проверки
    packages_ok = check_required_packages()
    config_ok = check_config_settings()
    yolo_ok = test_yolo_model_loading()
    fer_ok = test_fer_emotion_analysis()
    deepface_ok = test_deepface_emotion_analysis()
    
    print(f"\n🎯 Результаты диагностики:")
    print(f"  📦 Пакеты: {'✅' if packages_ok else '❌'}")
    print(f"  ⚙️ Конфигурация: {'✅' if config_ok else '❌'}")
    print(f"  🎯 YOLO: {'✅' if yolo_ok else '❌'}")
    print(f"  🎭 FER: {'✅' if fer_ok else '❌'}")
    print(f"  🧠 DeepFace: {'✅' if deepface_ok else '❌'}")
    
    working_methods = sum([fer_ok, deepface_ok, yolo_ok])
    
    if working_methods >= 1:
        print(f"🎉 Есть {working_methods} рабочих метода анализа эмоций!")
        if not fer_ok or not deepface_ok:
            print("⚠️ Рекомендуется исправить неработающие методы для лучшей точности")
    else:
        print("❌ Ни один метод анализа эмоций не работает!")
        print("🔧 Требуется срочное исправление настроек")