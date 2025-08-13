"""
Fixtures для тестов
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
import json


@pytest.fixture(scope="session")
def test_config():
    """Базовая конфигурация для тестов"""
    return {
        'processing': {
            'chunk_size': 30,
            'overlap': 0.1,
            'enable_gpu': False,
            'cpu_fallback': True
        },
        'video': {
            'fps': 1,
            'max_faces': 5,
            'face_confidence': 0.5,
            'max_resolution': (640, 480)
        },
        'audio': {
            'sample_rate': 16000,
            'segment_duration': 3.0,
            'overlap': 0.5,
            'min_segment_duration': 0.5
        },
        'models': {
            'yolo_model': 'yolo11n.pt',
            'confidence_threshold': 0.5,
            'device': 'cpu',
            'primary_emotion_model': 'basic',
            'enable_fallbacks': True
        },
        'logging': {
            'level': 'WARNING',  # Уменьшаем логирование в тестах
            'file': None  # Не пишем логи в файл
        }
    }


@pytest.fixture(scope="session")
def test_image():
    """Создание тестового изображения с лицом"""
    # Создаем изображение 640x480
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Добавляем фон
    image[:] = [50, 50, 50]
    
    # Рисуем простое "лицо"
    center_x, center_y = 320, 240
    face_size = 100
    
    # Лицо (овал)
    cv2.ellipse(image, (center_x, center_y), (face_size, face_size + 20), 
                0, 0, 360, (200, 180, 160), -1)
    
    # Глаза
    cv2.circle(image, (center_x - 30, center_y - 20), 8, (0, 0, 0), -1)
    cv2.circle(image, (center_x + 30, center_y - 20), 8, (0, 0, 0), -1)
    
    # Нос
    cv2.circle(image, (center_x, center_y), 4, (150, 130, 110), -1)
    
    # Рот (улыбка)
    cv2.ellipse(image, (center_x, center_y + 30), (25, 15), 
                0, 0, 180, (100, 50, 50), 2)
    
    return image


@pytest.fixture(scope="session")
def test_video_file(test_image):
    """Создание тестового видео файла"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_file.close()
    
    try:
        # Создаем видео с помощью OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_file.name, fourcc, 2.0, (640, 480))
        
        # Добавляем 10 кадров (5 секунд при 2 FPS)
        for i in range(10):
            # Немного изменяем изображение для каждого кадра
            frame = test_image.copy()
            
            # Добавляем номер кадра
            cv2.putText(frame, f'Frame {i}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Немного двигаем "лицо"
            if i > 0:
                shift_x = int(10 * np.sin(i * 0.5))
                shift_y = int(5 * np.cos(i * 0.5))
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                frame = cv2.warpAffine(frame, M, (640, 480))
            
            out.write(frame)
        
        out.release()
        
        yield temp_file.name
        
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@pytest.fixture(scope="session") 
def test_audio_file():
    """Создание тестового аудио файла"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.close()
    
    try:
        # Создаем простой аудио сигнал
        sample_rate = 16000
        duration = 5.0  # 5 секунд
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Смешиваем несколько частот для более реалистичного звука
        frequencies = [440, 523, 659]  # До-мажорное трезвучие
        audio = np.zeros_like(t)
        
        for freq in frequencies:
            audio += np.sin(2 * np.pi * freq * t) / len(frequencies)
        
        # Добавляем огибающую (затухание)
        envelope = np.exp(-t / 2)
        audio *= envelope
        
        # Нормализуем
        audio = audio / np.max(np.abs(audio)) * 0.7
        
        # Сохраняем как WAV
        import wave
        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # моно
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Конвертируем в 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        
        yield temp_file.name
        
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@pytest.fixture(scope="session")
def reference_results():
    """Эталонные результаты для тестов"""
    return {
        'video_analysis': {
            'total_faces_detected': 10,  # По одному лицу на кадр
            'dominant_emotion': 'счастье',  # Из-за улыбки
            'emotion_distribution': {
                'счастье': 0.6,
                'нейтральность': 0.3,
                'другие': 0.1
            },
            'average_confidence': 0.75
        },
        'audio_analysis': {
            'dominant_emotion': 'спокойствие',  # Простой тон
            'emotion_distribution': {
                'спокойствие': 0.5,
                'нейтральность': 0.4,
                'другие': 0.1
            },
            'average_confidence': 0.65,
            'segments_analyzed': 2  # 5 секунд / 3 сек сегмент с перекрытием
        },
        'multimodal_analysis': {
            'correlation_score': 0.8,
            'consistency_score': 0.75,
            'total_duration': 5.0
        }
    }


@pytest.fixture
def temp_directory():
    """Временная директория для тестов"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Очистка
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_openai_response():
    """Мокированный ответ от OpenAI"""
    return {
        'choices': [{
            'message': {
                'content': json.dumps({
                    'summary': 'Тестовый анализ показал нейтральные эмоции',
                    'key_findings': [
                        'Стабильное эмоциональное состояние',
                        'Отсутствие признаков стресса',
                        'Нормальные речевые паттерны'
                    ],
                    'recommendations': [
                        'Продолжить наблюдение',
                        'Рассмотреть дополнительные тесты'
                    ],
                    'confidence': 0.8
                })
            }
        }],
        'usage': {
            'prompt_tokens': 100,
            'completion_tokens': 50,
            'total_tokens': 150
        }
    }


@pytest.fixture
def sample_emotion_data():
    """Образцы данных эмоций для тестов"""
    return [
        {
            'timestamp': 0.0,
            'emotion': 'нейтральность',
            'confidence': 0.8,
            'source': 'video',
            'bbox': [100, 100, 150, 150]
        },
        {
            'timestamp': 1.0,
            'emotion': 'счастье',
            'confidence': 0.9,
            'source': 'video',
            'bbox': [105, 98, 148, 152]
        },
        {
            'timestamp': 2.0,
            'emotion': 'удивление',
            'confidence': 0.7,
            'source': 'video',
            'bbox': [102, 101, 149, 151]
        },
        {
            'timestamp': 0.5,
            'emotion': 'спокойствие',
            'confidence': 0.75,
            'source': 'audio'
        },
        {
            'timestamp': 1.5,
            'emotion': 'радость',
            'confidence': 0.85,
            'source': 'audio'
        }
    ]


@pytest.fixture(scope="session")
def performance_benchmarks():
    """Бенчмарки производительности"""
    return {
        'video_processing': {
            'max_time_per_frame': 2.0,  # секунд
            'max_memory_usage': 500,    # MB
        },
        'audio_processing': {
            'max_time_per_second': 0.5,  # секунд обработки на секунду аудио
            'max_memory_usage': 200,     # MB
        },
        'emotion_analysis': {
            'max_time_per_face': 0.5,   # секунд
            'max_time_per_audio_segment': 1.0,  # секунд
        }
    }


# Маркеры для pytest
pytest.mark.slow = pytest.mark.slow
pytest.mark.gpu = pytest.mark.skipif(
    not hasattr(pytest, '_gpu_available'), 
    reason="GPU не доступен"
)
pytest.mark.integration = pytest.mark.integration