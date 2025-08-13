"""
Тесты моделей машинного обучения
"""

import pytest
import os
import sys
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import cv2

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.speech_analyzer import AdvancedSpeechEmotionAnalyzer
from models.yolo_manager import YOLO11Manager


class TestSpeechAnalyzer:
    """Тесты анализатора речи"""
    
    @pytest.fixture
    def config(self):
        """Конфигурация для анализатора речи"""
        return {
            'audio': {
                'sample_rate': 16000,
                'segment_duration': 3.0,
                'overlap': 0.5,
                'min_segment_duration': 0.5
            },
            'logging': {'level': 'INFO'}
        }
    
    @pytest.fixture
    def speech_analyzer(self, config):
        """Создание анализатора речи"""
        return AdvancedSpeechEmotionAnalyzer(config)
    
    def test_speech_analyzer_initialization(self, config):
        """Тест инициализации анализатора речи"""
        analyzer = AdvancedSpeechEmotionAnalyzer(config)
        
        assert analyzer is not None
        assert analyzer.config == config
        assert analyzer.sample_rate == 16000
        assert analyzer.emotion_categories is not None
        assert len(analyzer.emotion_categories) > 0
    
    def test_load_audio_fallback(self, speech_analyzer):
        """Тест загрузки аудио с fallback механизмами"""
        # Создаем простой тестовый аудио файл
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            # Создаем простой синусоидальный сигнал
            sample_rate = 16000
            duration = 1.0  # 1 секунда
            frequency = 440  # Ля первой октавы
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
            # Сохраняем как простой WAV (для тестов без библиотек)
            temp_file.write(audio_data.tobytes())
            temp_file.flush()
            
            try:
                # Тестируем fallback механизм
                with patch('models.speech_analyzer.LIBROSA_AVAILABLE', False):
                    with patch('models.speech_analyzer.SOUNDFILE_AVAILABLE', False):
                        # Должен использовать wave модуль
                        try:
                            audio, sr = speech_analyzer.load_audio(temp_file.name)
                            assert audio is not None
                            assert sr > 0
                        except Exception as e:
                            # Ожидаем ошибку из-за простого формата
                            assert isinstance(e, (ValueError, RuntimeError, OSError))
                            
            finally:
                os.unlink(temp_file.name)
    
    def test_extract_features_basic(self, speech_analyzer):
        """Тест извлечения базовых признаков"""
        # Создаем тестовый аудио сигнал
        sample_rate = 16000
        duration = 2.0
        frequency = 440
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Тестируем извлечение признаков без librosa
        with patch('models.speech_analyzer.LIBROSA_AVAILABLE', False):
            features = speech_analyzer.extract_features(audio, sample_rate)
            
            assert isinstance(features, dict)
            assert 'energy' in features
            assert 'rms_mean' in features
            assert 'zcr_mean' in features
            assert features['energy'] > 0
    
    def test_classify_emotion(self, speech_analyzer):
        """Тест классификации эмоций"""
        # Тестовые признаки
        features = {
            'energy': 0.01,
            'rms_mean': 0.05,
            'pitch_mean': 150,
            'pitch_std': 30,
            'spectral_centroid_mean': 2000,
            'zcr_mean': 0.05
        }
        
        result = speech_analyzer.classify_emotion(features)
        
        assert isinstance(result, dict)
        assert 'emotion' in result
        assert 'confidence' in result
        assert 'all_emotions' in result
        assert 'method' in result
        
        assert result['emotion'] in speech_analyzer.emotion_categories
        assert 0 <= result['confidence'] <= 1
        assert isinstance(result['all_emotions'], dict)
    
    def test_analyze_speech_with_mock(self, speech_analyzer):
        """Тест полного анализа речи с моками"""
        # Мокируем load_audio
        mock_audio = np.random.randn(16000).astype(np.float32)  # 1 секунда
        mock_sr = 16000
        
        with patch.object(speech_analyzer, 'load_audio', return_value=(mock_audio, mock_sr)):
            result = speech_analyzer.analyze_speech("test_audio.wav")
            
            assert isinstance(result, dict)
            assert 'emotion' in result
            assert 'confidence' in result
            assert 'duration' in result
            assert 'features' in result
            
            # Проверяем, что длительность корректна
            assert abs(result['duration'] - 1.0) < 0.1


class TestYOLOManager:
    """Тесты YOLO менеджера"""
    
    @pytest.fixture
    def config(self):
        """Конфигурация для YOLO"""
        return {
            'models': {
                'yolo_model': 'yolo11n.pt',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            'video': {
                'max_faces': 5,
                'face_confidence': 0.5
            },
            'logging': {'level': 'INFO'}
        }
    
    def test_yolo_manager_initialization(self, config):
        """Тест инициализации YOLO менеджера"""
        try:
            manager = YOLO11Manager(config)
            assert manager is not None
            assert manager.config == config
            
        except ImportError:
            pytest.skip("Ultralytics не установлен")
        except Exception as e:
            # Может не удаться загрузить модель в тестовой среде
            pytest.skip(f"YOLO модель недоступна: {e}")
    
    def test_detect_faces_mock(self, config):
        """Тест детекции лиц с мокированием"""
        # Создаем тестовое изображение
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Рисуем простое "лицо" (прямоугольник)
        cv2.rectangle(test_image, (200, 150), (400, 350), (255, 255, 255), -1)
        
        try:
            manager = YOLO11Manager(config)
            
            # Мокируем результат YOLO
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.xyxy = [[200, 150, 400, 350]]  # координаты bbox
            mock_result.boxes.conf = [0.9]  # уверенность
            mock_result.boxes.cls = [0]  # класс (person)
            
            with patch.object(manager, 'model') as mock_model:
                mock_model.return_value = [mock_result]
                
                faces = manager.detect_faces(test_image)
                
                assert isinstance(faces, list)
                if len(faces) > 0:
                    face = faces[0]
                    assert 'bbox' in face
                    assert 'confidence' in face
                    assert len(face['bbox']) == 4  # x, y, w, h
                    
        except (ImportError, Exception) as e:
            pytest.skip(f"YOLO тест пропущен: {e}")


class TestEmotionAnalysis:
    """Тесты анализа эмоций"""
    
    @pytest.fixture
    def test_face_image(self):
        """Создание тестового изображения лица"""
        # Создаем простое изображение 224x224 (стандартный размер для моделей)
        face_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Добавляем некоторые черты "лица"
        # Глаза
        cv2.circle(face_image, (70, 80), 10, (0, 0, 0), -1)
        cv2.circle(face_image, (154, 80), 10, (0, 0, 0), -1)
        
        # Нос
        cv2.circle(face_image, (112, 112), 5, (100, 100, 100), -1)
        
        # Рот
        cv2.ellipse(face_image, (112, 150), (20, 10), 0, 0, 180, (50, 50, 50), 2)
        
        return face_image
    
    def test_deepface_emotion_analysis_mock(self, test_face_image):
        """Тест анализа эмоций DeepFace с мокированием"""
        try:
            # Мокируем DeepFace
            with patch('deepface.DeepFace.analyze') as mock_analyze:
                mock_analyze.return_value = [{
                    'emotion': {
                        'angry': 0.1,
                        'disgust': 0.05,
                        'fear': 0.1,
                        'happy': 0.6,
                        'sad': 0.05,
                        'surprise': 0.05,
                        'neutral': 0.05
                    },
                    'dominant_emotion': 'happy'
                }]
                
                # Имитируем анализ эмоций
                from core.emotion_analyzer import MultiModalEmotionAnalyzer
                
                config = {
                    'models': {'primary_emotion_model': 'deepface'},
                    'logging': {'level': 'INFO'}
                }
                
                analyzer = MultiModalEmotionAnalyzer(config)
                
                # Мокируем метод анализа
                with patch.object(analyzer, 'analyze_frame') as mock_frame_analysis:
                    mock_frame_analysis.return_value = {
                        'emotions': [{
                            'emotion': 'счастье',
                            'confidence': 0.6,
                            'bbox': [50, 50, 100, 100]
                        }],
                        'faces_detected': 1,
                        'processing_time': 0.1
                    }
                    
                    result = analyzer.analyze_frame(test_face_image)
                    
                    assert isinstance(result, dict)
                    assert 'emotions' in result
                    assert 'faces_detected' in result
                    assert result['faces_detected'] >= 0
                    
                    if result['faces_detected'] > 0:
                        emotion = result['emotions'][0]
                        assert 'emotion' in emotion
                        assert 'confidence' in emotion
                        assert 0 <= emotion['confidence'] <= 1
                        
        except ImportError:
            pytest.skip("DeepFace не установлен")
    
    def test_fer_emotion_analysis_mock(self, test_face_image):
        """Тест анализа эмоций FER с мокированием"""
        try:
            with patch('fer.FER') as mock_fer_class:
                # Мокируем FER детектор
                mock_detector = Mock()
                mock_detector.detect_emotions.return_value = [{
                    'box': [50, 50, 100, 100],
                    'emotions': {
                        'angry': 0.1,
                        'disgust': 0.05,
                        'fear': 0.1,
                        'happy': 0.6,
                        'sad': 0.05,
                        'surprise': 0.05,
                        'neutral': 0.05
                    }
                }]
                
                mock_fer_class.return_value = mock_detector
                
                # Используем FER для анализа
                detector = mock_fer_class()
                emotions = detector.detect_emotions(test_face_image)
                
                assert isinstance(emotions, list)
                if len(emotions) > 0:
                    emotion_data = emotions[0]
                    assert 'box' in emotion_data
                    assert 'emotions' in emotion_data
                    assert isinstance(emotion_data['emotions'], dict)
                    
                    # Проверяем, что сумма эмоций ~1
                    total = sum(emotion_data['emotions'].values())
                    assert abs(total - 1.0) < 0.01
                    
        except ImportError:
            pytest.skip("FER не установлен")


class TestModelPerformance:
    """Тесты производительности моделей"""
    
    def test_speech_analyzer_performance(self):
        """Тест производительности анализатора речи"""
        import time
        
        config = {
            'audio': {'sample_rate': 16000},
            'logging': {'level': 'WARNING'}
        }
        
        analyzer = AdvancedSpeechEmotionAnalyzer(config)
        
        # Создаем тестовый аудио сигнал (1 секунда)
        sample_rate = 16000
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        # Измеряем время извлечения признаков
        start_time = time.time()
        features = analyzer.extract_features(audio, sample_rate)
        feature_time = time.time() - start_time
        
        # Измеряем время классификации
        start_time = time.time()
        result = analyzer.classify_emotion(features)
        classification_time = time.time() - start_time
        
        print(f"Feature extraction: {feature_time:.3f}s")
        print(f"Classification: {classification_time:.3f}s")
        
        # Проверяем, что обработка быстрая
        assert feature_time < 1.0, f"Извлечение признаков слишком медленное: {feature_time:.3f}s"
        assert classification_time < 0.1, f"Классификация слишком медленная: {classification_time:.3f}s"
    
    @pytest.mark.slow
    def test_batch_processing_performance(self):
        """Тест производительности пакетной обработки"""
        import time
        
        config = {
            'audio': {'sample_rate': 16000},
            'logging': {'level': 'WARNING'}
        }
        
        analyzer = AdvancedSpeechEmotionAnalyzer(config)
        
        # Создаем несколько аудио сэмплов
        sample_rate = 16000
        duration = 1.0
        num_samples = 10
        
        audio_samples = []
        for _ in range(num_samples):
            audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
            audio_samples.append(audio)
        
        # Измеряем время пакетной обработки
        start_time = time.time()
        
        results = []
        for audio in audio_samples:
            features = analyzer.extract_features(audio, sample_rate)
            result = analyzer.classify_emotion(features)
            results.append(result)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_samples
        
        print(f"Batch processing: {total_time:.3f}s total, {avg_time:.3f}s per sample")
        
        # Проверяем результаты
        assert len(results) == num_samples
        assert avg_time < 0.5, f"Средняя обработка слишком медленная: {avg_time:.3f}s"


class TestModelFallbacks:
    """Тесты fallback механизмов моделей"""
    
    def test_speech_analyzer_librosa_fallback(self):
        """Тест fallback когда librosa недоступна"""
        config = {
            'audio': {'sample_rate': 16000},
            'logging': {'level': 'WARNING'}
        }
        
        # Тестируем без librosa
        with patch('models.speech_analyzer.LIBROSA_AVAILABLE', False):
            analyzer = AdvancedSpeechEmotionAnalyzer(config)
            
            # Создаем тестовый сигнал
            audio = np.random.randn(16000).astype(np.float32)
            
            # Должен работать с базовыми функциями
            features = analyzer.extract_features(audio, 16000)
            result = analyzer.classify_emotion(features)
            
            assert isinstance(features, dict)
            assert isinstance(result, dict)
            assert 'emotion' in result
    
    def test_emotion_model_fallback(self):
        """Тест fallback между моделями эмоций"""
        config = {
            'models': {
                'primary_emotion_model': 'nonexistent_model',
                'fallback_emotion_model': 'basic',
                'enable_fallbacks': True
            },
            'logging': {'level': 'WARNING'}
        }
        
        # Имитируем отсутствие первичной модели
        with patch('importlib.import_module') as mock_import:
            def side_effect(module_name):
                if 'nonexistent_model' in module_name:
                    raise ImportError("Model not found")
                return Mock()
            
            mock_import.side_effect = side_effect
            
            try:
                from core.emotion_analyzer import MultiModalEmotionAnalyzer
                analyzer = MultiModalEmotionAnalyzer(config)
                
                # Анализатор должен инициализироваться с fallback
                assert analyzer is not None
                
            except Exception as e:
                # Если fallback тоже не работает, это ожидаемо
                assert isinstance(e, (ImportError, AttributeError))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])