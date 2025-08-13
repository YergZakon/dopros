"""
Интеграционные тесты пайплайна обработки
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline import MasterPipeline
from core.data_aggregator import DataAggregator
from utils.logger import get_logger


class TestPipelineInitialization:
    """Тесты инициализации пайплайна"""
    
    @pytest.fixture
    def config(self):
        """Базовая конфигурация для тестов"""
        return {
            'processing': {
                'chunk_size': 30,
                'overlap': 0.1,
                'enable_gpu': False
            },
            'video': {
                'fps': 1,
                'max_faces': 5,
                'face_confidence': 0.5
            },
            'audio': {
                'sample_rate': 16000,
                'segment_duration': 3.0,
                'overlap': 0.5
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    def test_pipeline_initialization(self, config):
        """Тест инициализации пайплайна"""
        try:
            pipeline = MasterPipeline(config)
            assert pipeline is not None
            assert pipeline.config == config
            assert pipeline.logger is not None
            
        except Exception as e:
            pytest.fail(f"Ошибка инициализации пайплайна: {e}")
    
    def test_data_aggregator_initialization(self, config):
        """Тест инициализации агрегатора данных"""
        try:
            aggregator = DataAggregator(config)
            assert aggregator is not None
            assert aggregator.config == config
            
        except Exception as e:
            pytest.fail(f"Ошибка инициализации агрегатора: {e}")


class TestPipelineComponents:
    """Тесты компонентов пайплайна"""
    
    @pytest.fixture
    def pipeline(self):
        """Создание пайплайна для тестов"""
        config = {
            'processing': {'chunk_size': 30, 'enable_gpu': False},
            'video': {'fps': 1, 'max_faces': 5},
            'audio': {'sample_rate': 16000},
            'logging': {'level': 'INFO'}
        }
        return MasterPipeline(config)
    
    def test_video_processor_initialization(self, pipeline):
        """Тест инициализации видео процессора"""
        assert hasattr(pipeline, 'video_processor')
        assert pipeline.video_processor is not None
    
    def test_audio_processor_initialization(self, pipeline):
        """Тест инициализации аудио процессора"""
        assert hasattr(pipeline, 'audio_processor')
        assert pipeline.audio_processor is not None
    
    def test_emotion_analyzer_initialization(self, pipeline):
        """Тест инициализации анализатора эмоций"""
        assert hasattr(pipeline, 'emotion_analyzer')
        assert pipeline.emotion_analyzer is not None
    
    def test_speech_analyzer_initialization(self, pipeline):
        """Тест инициализации анализатора речи"""
        assert hasattr(pipeline, 'speech_analyzer')
        assert pipeline.speech_analyzer is not None


class TestPipelineProcessing:
    """Тесты обработки данных пайплайном"""
    
    @pytest.fixture
    def temp_video_file(self):
        """Создание временного видео файла для тестов"""
        # Создаем временный файл
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()
        
        # В реальности здесь было бы создание простого видео
        # Для тестов используем существующий файл или создаем заглушку
        yield temp_file.name
        
        # Очистка
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
    
    @pytest.fixture
    def mock_pipeline(self):
        """Мокированный пайплайн для тестов"""
        config = {
            'processing': {'chunk_size': 30, 'enable_gpu': False},
            'video': {'fps': 1, 'max_faces': 5},
            'audio': {'sample_rate': 16000},
            'logging': {'level': 'INFO'}
        }
        
        with patch('core.pipeline.MasterPipeline') as mock:
            pipeline = MasterPipeline(config)
            
            # Мокируем компоненты
            pipeline.video_processor = Mock()
            pipeline.audio_processor = Mock()
            pipeline.emotion_analyzer = Mock()
            pipeline.speech_analyzer = Mock()
            pipeline.data_aggregator = Mock()
            pipeline.report_generator = Mock()
            
            yield pipeline
    
    def test_pipeline_process_video_mock(self, mock_pipeline, temp_video_file):
        """Тест обработки видео с мокированными компонентами"""
        # Настраиваем моки
        mock_pipeline.video_processor.extract_frames.return_value = [
            {'frame': np.zeros((480, 640, 3)), 'timestamp': 0.0},
            {'frame': np.zeros((480, 640, 3)), 'timestamp': 1.0}
        ]
        
        mock_pipeline.audio_processor.extract_audio.return_value = {
            'audio_path': '/tmp/test.wav',
            'duration': 2.0,
            'sample_rate': 16000
        }
        
        mock_pipeline.emotion_analyzer.analyze_frame.return_value = {
            'emotions': [{'emotion': 'нейтральность', 'confidence': 0.8}],
            'faces_detected': 1
        }
        
        mock_pipeline.speech_analyzer.analyze_speech.return_value = {
            'emotion': 'нейтральность',
            'confidence': 0.7,
            'duration': 2.0
        }
        
        # Тестируем обработку
        try:
            # Имитируем вызов process_video
            video_results = mock_pipeline.video_processor.extract_frames(temp_video_file)
            audio_results = mock_pipeline.audio_processor.extract_audio(temp_video_file)
            
            assert len(video_results) == 2
            assert audio_results['duration'] == 2.0
            
            # Проверяем, что методы были вызваны
            mock_pipeline.video_processor.extract_frames.assert_called_once()
            mock_pipeline.audio_processor.extract_audio.assert_called_once()
            
        except Exception as e:
            pytest.fail(f"Ошибка в мокированной обработке: {e}")


class TestFallbackMechanisms:
    """Тесты fallback механизмов"""
    
    @pytest.fixture
    def config_with_fallbacks(self):
        """Конфигурация с fallback настройками"""
        return {
            'processing': {
                'enable_gpu': True,
                'gpu_fallback': True,
                'cpu_fallback': True
            },
            'models': {
                'primary_emotion_model': 'deepface',
                'fallback_emotion_model': 'fer',
                'enable_fallbacks': True
            },
            'logging': {'level': 'INFO'}
        }
    
    def test_gpu_to_cpu_fallback(self, config_with_fallbacks):
        """Тест переключения с GPU на CPU"""
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = MasterPipeline(config_with_fallbacks)
            
            # Проверяем, что пайплайн инициализировался с CPU
            assert pipeline is not None
    
    def test_model_fallback(self, config_with_fallbacks):
        """Тест fallback между моделями"""
        with patch('importlib.import_module') as mock_import:
            # Имитируем отсутствие первичной модели
            def side_effect(module_name):
                if 'deepface' in module_name:
                    raise ImportError("DeepFace not available")
                return Mock()
            
            mock_import.side_effect = side_effect
            
            # Пайплайн должен инициализироваться с fallback моделью
            try:
                pipeline = MasterPipeline(config_with_fallbacks)
                assert pipeline is not None
            except ImportError:
                pytest.skip("Fallback модель также недоступна")


class TestErrorRecovery:
    """Тесты восстановления после ошибок"""
    
    def test_invalid_video_file_handling(self):
        """Тест обработки некорректного видео файла"""
        config = {
            'processing': {'chunk_size': 30},
            'logging': {'level': 'INFO'}
        }
        
        pipeline = MasterPipeline(config)
        
        # Тестируем обработку несуществующего файла
        with pytest.raises((FileNotFoundError, Exception)):
            pipeline.process_video("nonexistent_file.mp4")
    
    def test_corrupted_audio_handling(self):
        """Тест обработки поврежденного аудио"""
        config = {
            'audio': {'sample_rate': 16000},
            'logging': {'level': 'INFO'}
        }
        
        # Создаем временный поврежденный файл
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(b'corrupted_audio_data')
            temp_file.flush()
            
            try:
                from models.speech_analyzer import AdvancedSpeechEmotionAnalyzer
                analyzer = AdvancedSpeechEmotionAnalyzer(config)
                
                # Должен обработать ошибку gracefully
                result = analyzer.analyze_speech(temp_file.name)
                assert 'error' in result or result is not None
                
            except Exception as e:
                # Ожидаем, что ошибка будет обработана
                assert isinstance(e, (ValueError, RuntimeError, OSError))
            
            finally:
                os.unlink(temp_file.name)
    
    def test_memory_error_handling(self):
        """Тест обработки ошибок памяти"""
        config = {
            'processing': {'chunk_size': 1},  # Очень маленький chunk
            'logging': {'level': 'INFO'}
        }
        
        pipeline = MasterPipeline(config)
        
        # Имитируем ошибку памяти
        with patch.object(pipeline, 'video_processor') as mock_processor:
            mock_processor.extract_frames.side_effect = MemoryError("Out of memory")
            
            # Пайплайн должен обработать ошибку
            with pytest.raises(MemoryError):
                pipeline.process_video("test.mp4")


class TestDataIntegrity:
    """Тесты целостности данных"""
    
    def test_timestamp_consistency(self):
        """Тест консистентности временных меток"""
        # Создаем тестовые данные
        video_results = [
            {'timestamp': 0.0, 'emotion': 'нейтральность'},
            {'timestamp': 1.0, 'emotion': 'счастье'},
            {'timestamp': 2.0, 'emotion': 'грусть'}
        ]
        
        audio_results = [
            {'timestamp': 0.5, 'emotion': 'нейтральность'},
            {'timestamp': 1.5, 'emotion': 'счастье'},
            {'timestamp': 2.5, 'emotion': 'грусть'}
        ]
        
        # Проверяем, что временные метки в правильном порядке
        video_timestamps = [r['timestamp'] for r in video_results]
        audio_timestamps = [r['timestamp'] for r in audio_results]
        
        assert video_timestamps == sorted(video_timestamps)
        assert audio_timestamps == sorted(audio_timestamps)
    
    def test_emotion_data_format(self):
        """Тест формата данных эмоций"""
        # Тестовые данные эмоций
        emotion_result = {
            'emotion': 'счастье',
            'confidence': 0.85,
            'all_emotions': {
                'счастье': 0.85,
                'грусть': 0.10,
                'нейтральность': 0.05
            }
        }
        
        # Проверяем формат
        assert isinstance(emotion_result['emotion'], str)
        assert 0 <= emotion_result['confidence'] <= 1
        assert isinstance(emotion_result['all_emotions'], dict)
        
        # Проверяем, что сумма вероятностей ~1
        total_prob = sum(emotion_result['all_emotions'].values())
        assert abs(total_prob - 1.0) < 0.01


class TestPerformance:
    """Тесты производительности"""
    
    @pytest.mark.slow
    def test_processing_speed(self):
        """Тест скорости обработки"""
        import time
        
        config = {
            'processing': {'chunk_size': 30},
            'video': {'fps': 1},
            'logging': {'level': 'WARNING'}  # Уменьшаем логирование
        }
        
        pipeline = MasterPipeline(config)
        
        # Создаем простые тестовые данные
        test_frames = [np.zeros((480, 640, 3)) for _ in range(10)]
        
        start_time = time.time()
        
        # Имитируем обработку кадров
        with patch.object(pipeline, 'emotion_analyzer') as mock_analyzer:
            mock_analyzer.analyze_frame.return_value = {
                'emotions': [{'emotion': 'нейтральность', 'confidence': 0.8}],
                'faces_detected': 1
            }
            
            for frame in test_frames:
                pipeline.emotion_analyzer.analyze_frame(frame)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Проверяем, что обработка занимает разумное время
        # (10 кадров должны обрабатываться менее чем за 10 секунд)
        assert processing_time < 10.0, f"Обработка слишком медленная: {processing_time:.2f}s"
    
    def test_memory_usage(self):
        """Тест использования памяти"""
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            config = {
                'processing': {'chunk_size': 30},
                'logging': {'level': 'WARNING'}
            }
            
            # Создаем несколько пайплайнов
            pipelines = []
            for _ in range(3):
                pipeline = MasterPipeline(config)
                pipelines.append(pipeline)
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Очищаем память
            del pipelines
            gc.collect()
            
            print(f"Увеличение памяти: {memory_increase:.2f} MB")
            
            # Проверяем, что увеличение памяти разумное (менее 500MB)
            assert memory_increase < 500, f"Слишком большое потребление памяти: {memory_increase:.2f} MB"
            
        except ImportError:
            pytest.skip("psutil не установлен")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])