"""
Генератор тестовых данных
"""

import numpy as np
import cv2
import json
from pathlib import Path
import tempfile
import wave


class TestDataGenerator:
    """Генератор различных типов тестовых данных"""
    
    @staticmethod
    def create_test_image(width=640, height=480, num_faces=1):
        """Создание тестового изображения с лицами"""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:] = [30, 30, 30]  # Темный фон
        
        face_positions = []
        
        for i in range(num_faces):
            # Позиция лица
            if num_faces == 1:
                center_x, center_y = width // 2, height // 2
            else:
                center_x = width // (num_faces + 1) * (i + 1)
                center_y = height // 2
            
            face_size = min(width, height) // (num_faces + 2)
            
            # Рисуем лицо
            TestDataGenerator._draw_face(image, center_x, center_y, face_size)
            
            face_positions.append({
                'bbox': [
                    center_x - face_size,
                    center_y - face_size,
                    face_size * 2,
                    face_size * 2
                ],
                'center': (center_x, center_y),
                'size': face_size
            })
        
        return image, face_positions
    
    @staticmethod
    def _draw_face(image, center_x, center_y, size):
        """Рисование простого лица"""
        # Лицо (овал)
        cv2.ellipse(image, (center_x, center_y), (size, size + 20), 
                    0, 0, 360, (200, 180, 160), -1)
        
        # Глаза
        eye_size = size // 8
        cv2.circle(image, (center_x - size//3, center_y - size//4), 
                   eye_size, (0, 0, 0), -1)
        cv2.circle(image, (center_x + size//3, center_y - size//4), 
                   eye_size, (0, 0, 0), -1)
        
        # Нос
        nose_size = size // 12
        cv2.circle(image, (center_x, center_y), nose_size, (150, 130, 110), -1)
        
        # Рот
        mouth_width = size // 2
        mouth_height = size // 6
        cv2.ellipse(image, (center_x, center_y + size//3), 
                    (mouth_width, mouth_height), 0, 0, 180, (100, 50, 50), 2)
    
    @staticmethod
    def create_test_video(filename, duration=5, fps=2, width=640, height=480, num_faces=1):
        """Создание тестового видео"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        total_frames = int(duration * fps)
        
        for frame_num in range(total_frames):
            # Создаем кадр
            image, face_positions = TestDataGenerator.create_test_image(
                width, height, num_faces
            )
            
            # Добавляем номер кадра
            cv2.putText(image, f'Frame {frame_num}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Добавляем время
            time_sec = frame_num / fps
            cv2.putText(image, f'Time: {time_sec:.1f}s', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Небольшое движение лиц для реалистичности
            if frame_num > 0:
                shift_x = int(5 * np.sin(frame_num * 0.3))
                shift_y = int(3 * np.cos(frame_num * 0.3))
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                image = cv2.warpAffine(image, M, (width, height))
            
            out.write(image)
        
        out.release()
        return total_frames
    
    @staticmethod
    def create_test_audio(filename, duration=5, sample_rate=16000, audio_type='speech'):
        """Создание тестового аудио"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        if audio_type == 'speech':
            # Имитация речи с несколькими формантами
            formants = [800, 1200, 2400]  # Типичные форманты
            audio = np.zeros_like(t)
            
            for formant in formants:
                # Добавляем формант с модуляцией
                modulation = 1 + 0.1 * np.sin(2 * np.pi * 5 * t)  # 5 Hz модуляция
                audio += np.sin(2 * np.pi * formant * t) * modulation / len(formants)
            
            # Добавляем огибающую для реалистичности
            envelope = np.where(t % 1 < 0.6, 1, 0.1)  # Пауза каждую секунду
            audio *= envelope
            
        elif audio_type == 'music':
            # Простая мелодия
            notes = [440, 494, 523, 587, 659, 698, 784, 880]  # До мажор
            audio = np.zeros_like(t)
            
            note_duration = duration / len(notes)
            for i, freq in enumerate(notes):
                start_idx = int(i * note_duration * sample_rate)
                end_idx = int((i + 1) * note_duration * sample_rate)
                if end_idx > len(audio):
                    end_idx = len(audio)
                
                note_t = t[start_idx:end_idx] - t[start_idx]
                note_audio = np.sin(2 * np.pi * freq * note_t)
                
                # Огибающая ноты
                note_envelope = np.exp(-note_t * 2)
                audio[start_idx:end_idx] = note_audio * note_envelope
        
        elif audio_type == 'noise':
            # Белый шум
            audio = np.random.normal(0, 0.1, len(t))
        
        else:  # 'tone'
            # Простой тон
            frequency = 440  # Ля первой октавы
            audio = np.sin(2 * np.pi * frequency * t)
        
        # Нормализация
        audio = audio / np.max(np.abs(audio)) * 0.7
        
        # Сохранение как WAV
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        
        return len(audio)
    
    @staticmethod
    def create_emotion_dataset(num_samples=50):
        """Создание датасета эмоций для тестов"""
        emotions = [
            'нейтральность', 'счастье', 'грусть', 'злость',
            'страх', 'удивление', 'отвращение', 'спокойствие'
        ]
        
        dataset = []
        
        for i in range(num_samples):
            # Случайные данные
            emotion = np.random.choice(emotions)
            confidence = np.random.uniform(0.5, 0.95)
            timestamp = i * 0.5  # Каждые 0.5 секунд
            
            # Видео данные
            video_data = {
                'timestamp': timestamp,
                'emotion': emotion,
                'confidence': confidence,
                'source': 'video',
                'bbox': [
                    np.random.randint(50, 200),
                    np.random.randint(50, 200),
                    np.random.randint(100, 200),
                    np.random.randint(100, 200)
                ],
                'face_id': np.random.randint(0, 3)
            }
            
            # Аудио данные (со смещением)
            audio_emotion = emotion if np.random.random() > 0.3 else np.random.choice(emotions)
            audio_data = {
                'timestamp': timestamp + 0.25,  # Небольшое смещение
                'emotion': audio_emotion,
                'confidence': np.random.uniform(0.4, 0.9),
                'source': 'audio',
                'segment_duration': 3.0
            }
            
            dataset.extend([video_data, audio_data])
        
        return sorted(dataset, key=lambda x: x['timestamp'])
    
    @staticmethod
    def create_performance_data():
        """Создание данных для тестов производительности"""
        return {
            'video_processing_times': np.random.lognormal(0, 0.5, 100).tolist(),
            'audio_processing_times': np.random.lognormal(-1, 0.3, 100).tolist(),
            'emotion_detection_times': np.random.lognormal(-1.5, 0.4, 100).tolist(),
            'memory_usage': np.random.normal(200, 50, 100).tolist(),
            'cpu_usage': np.random.beta(2, 5, 100).tolist() * 100,
            'gpu_usage': np.random.beta(1.5, 3, 100).tolist() * 100
        }
    
    @staticmethod
    def save_test_config(filename, config_type='minimal'):
        """Сохранение тестовой конфигурации"""
        if config_type == 'minimal':
            config = {
                'processing': {
                    'chunk_size': 10,
                    'enable_gpu': False
                },
                'video': {
                    'fps': 1,
                    'max_faces': 1
                },
                'audio': {
                    'sample_rate': 16000,
                    'segment_duration': 2.0
                },
                'logging': {
                    'level': 'WARNING'
                }
            }
        elif config_type == 'full':
            config = {
                'processing': {
                    'chunk_size': 30,
                    'overlap': 0.1,
                    'enable_gpu': True,
                    'gpu_fallback': True,
                    'cpu_fallback': True
                },
                'video': {
                    'fps': 2,
                    'max_faces': 5,
                    'face_confidence': 0.5,
                    'max_resolution': (1920, 1080)
                },
                'audio': {
                    'sample_rate': 44100,
                    'segment_duration': 5.0,
                    'overlap': 0.5,
                    'min_segment_duration': 1.0
                },
                'models': {
                    'yolo_model': 'yolo11m.pt',
                    'confidence_threshold': 0.7,
                    'primary_emotion_model': 'deepface',
                    'fallback_emotion_model': 'fer'
                },
                'openai': {
                    'model': 'gpt-4',
                    'max_tokens': 1000,
                    'temperature': 0.3
                },
                'logging': {
                    'level': 'INFO',
                    'file': 'test.log'
                }
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return config


if __name__ == "__main__":
    # Пример использования
    generator = TestDataGenerator()
    
    # Создаем тестовые файлы
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Создаем видео
        video_file = temp_path / "test_video.mp4"
        generator.create_test_video(str(video_file), duration=3, fps=2)
        print(f"Создано тестовое видео: {video_file}")
        
        # Создаем аудио
        audio_file = temp_path / "test_audio.wav"
        generator.create_test_audio(str(audio_file), duration=3, audio_type='speech')
        print(f"Создано тестовое аудио: {audio_file}")
        
        # Создаем конфиг
        config_file = temp_path / "test_config.json"
        generator.save_test_config(str(config_file), 'minimal')
        print(f"Создан тестовый конфиг: {config_file}")
        
        # Создаем датасет эмоций
        emotion_data = generator.create_emotion_dataset(10)
        emotion_file = temp_path / "emotion_data.json"
        with open(emotion_file, 'w', encoding='utf-8') as f:
            json.dump(emotion_data, f, indent=2, ensure_ascii=False)
        print(f"Создан датасет эмоций: {emotion_file}")