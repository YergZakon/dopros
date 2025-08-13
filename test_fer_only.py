#!/usr/bin/env python3
"""
Простой тест FER без TensorFlow конфликтов
"""

import cv2
import os

def test_fer_simple():
    """Простой тест FER"""
    print("🎭 Тестируем FER...")
    
    try:
        from fer import FER
        
        # Создаем детектор
        detector = FER()
        print("  ✅ FER детектор создан")
        
        # Проверяем на тестовом кадре
        if os.path.exists('test_face_detection.jpg'):
            image = cv2.imread('test_face_detection.jpg')
            print(f"  📷 Кадр загружен: {image.shape}")
            
            result = detector.detect_emotions(image)
            print(f"  📊 Результат: {len(result)} лиц")
            
            if result:
                for i, face in enumerate(result):
                    emotions = face['emotions']
                    dominant = max(emotions, key=emotions.get)
                    conf = emotions[dominant]
                    print(f"    🎯 Лицо {i+1}: {dominant} ({conf:.2%})")
                
                return True
            else:
                print("  ❌ FER не нашел лица")
                return False
        else:
            print("  ❌ Тестовый кадр не найден")
            return False
            
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return False

if __name__ == "__main__":
    test_fer_simple()