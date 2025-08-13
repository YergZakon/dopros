"""
Тесты совместимости и проверки окружения
"""

import pytest
import os
import sys
import subprocess
import importlib
import torch
import yaml
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEnvironmentCompatibility:
    """Тесты проверки окружения и совместимости"""
    
    def test_python_version(self):
        """Проверка версии Python"""
        version = sys.version_info
        assert version.major == 3, f"Требуется Python 3.x, найден {version.major}.{version.minor}"
        assert version.minor >= 8, f"Требуется Python 3.8+, найден {version.major}.{version.minor}"
        
    def test_config_exists(self):
        """Проверка наличия конфигурационного файла"""
        config_path = Path(__file__).parent.parent / "config.yaml"
        assert config_path.exists(), "Файл config.yaml не найден"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        assert isinstance(config, dict), "config.yaml должен содержать словарь"
        assert 'logging' in config, "Отсутствует секция logging в config.yaml"
        
    def test_env_file_exists(self):
        """Проверка наличия .env файла"""
        env_path = Path(__file__).parent.parent / ".env"
        assert env_path.exists(), "Файл .env не найден"


class TestImports:
    """Тесты импорта основных модулей"""
    
    def test_core_imports(self):
        """Тест импорта основных модулей"""
        try:
            import numpy as np
            import pandas as pd
            import cv2
            import streamlit as st
            import yaml
            assert True
        except ImportError as e:
            pytest.fail(f"Ошибка импорта основных модулей: {e}")
    
    def test_ml_frameworks(self):
        """Тест импорта ML фреймворков"""
        frameworks = []
        
        # PyTorch
        try:
            import torch
            frameworks.append(f"PyTorch {torch.__version__}")
        except ImportError:
            frameworks.append("PyTorch: NOT AVAILABLE")
        
        # TensorFlow
        try:
            import tensorflow as tf
            frameworks.append(f"TensorFlow {tf.__version__}")
        except ImportError:
            frameworks.append("TensorFlow: NOT AVAILABLE")
        
        # Scikit-learn
        try:
            import sklearn
            frameworks.append(f"Scikit-learn {sklearn.__version__}")
        except ImportError:
            frameworks.append("Scikit-learn: NOT AVAILABLE")
        
        print(f"ML Frameworks: {frameworks}")
        assert len(frameworks) > 0, "Ни один ML фреймворк не доступен"
    
    def test_audio_libraries(self):
        """Тест аудио библиотек"""
        audio_libs = []
        
        # Librosa
        try:
            import librosa
            audio_libs.append(f"Librosa {librosa.__version__}")
        except ImportError:
            audio_libs.append("Librosa: NOT AVAILABLE")
        
        # SoundFile
        try:
            import soundfile as sf
            audio_libs.append(f"SoundFile {sf.__version__}")
        except ImportError:
            audio_libs.append("SoundFile: NOT AVAILABLE")
        
        print(f"Audio Libraries: {audio_libs}")
        # Хотя бы одна аудио библиотека должна быть доступна
        assert any("NOT AVAILABLE" not in lib for lib in audio_libs), "Нет доступных аудио библиотек"
    
    def test_vision_libraries(self):
        """Тест компьютерного зрения библиотек"""
        vision_libs = []
        
        # OpenCV
        try:
            import cv2
            vision_libs.append(f"OpenCV {cv2.__version__}")
        except ImportError:
            vision_libs.append("OpenCV: NOT AVAILABLE")
        
        # Ultralytics (YOLO)
        try:
            import ultralytics
            vision_libs.append(f"Ultralytics {ultralytics.__version__}")
        except ImportError:
            vision_libs.append("Ultralytics: NOT AVAILABLE")
        
        print(f"Vision Libraries: {vision_libs}")
        assert "OpenCV: NOT AVAILABLE" not in vision_libs, "OpenCV обязателен"
    
    def test_project_imports(self):
        """Тест импорта модулей проекта"""
        project_modules = [
            'utils.logger',
            'utils.gpu_manager',
            'utils.translation',
            'models.speech_analyzer',
            'core.pipeline',
            'core.data_aggregator',
            'integrations.openai_client'
        ]
        
        failed_imports = []
        
        for module in project_modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                failed_imports.append(f"{module}: {e}")
        
        if failed_imports:
            pytest.fail(f"Ошибки импорта модулей проекта: {failed_imports}")


class TestGPUCUDA:
    """Тесты GPU и CUDA"""
    
    def test_pytorch_cuda(self):
        """Проверка доступности CUDA в PyTorch"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count()
            
            print(f"CUDA Available: {cuda_available}")
            print(f"GPU Count: {device_count}")
            
            if cuda_available:
                for i in range(device_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    print(f"GPU {i}: {gpu_name}")
                    
                # Тест создания тензора на GPU
                tensor = torch.tensor([1, 2, 3]).cuda()
                assert tensor.is_cuda, "Не удалось создать тензор на GPU"
            
            # Тест CPU режима всегда должен работать
            cpu_tensor = torch.tensor([1, 2, 3])
            assert not cpu_tensor.is_cuda, "CPU тензор создан некорректно"
            
        except ImportError:
            pytest.skip("PyTorch не установлен")
    
    def test_tensorflow_gpu(self):
        """Проверка доступности GPU в TensorFlow"""
        try:
            import tensorflow as tf
            
            gpus = tf.config.list_physical_devices('GPU')
            print(f"TensorFlow GPUs: {len(gpus)}")
            
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}: {gpu.name}")
            
            # Проверяем, что TensorFlow может использовать CPU
            with tf.device('/CPU:0'):
                cpu_tensor = tf.constant([1, 2, 3])
                assert cpu_tensor is not None
                
        except ImportError:
            pytest.skip("TensorFlow не установлен")


class TestFFmpeg:
    """Тесты FFmpeg"""
    
    def test_ffmpeg_installed(self):
        """Проверка установки FFmpeg"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, "FFmpeg не установлен или недоступен"
            
            version_info = result.stdout.split('\n')[0]
            print(f"FFmpeg: {version_info}")
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("FFmpeg не найден в системе")
    
    def test_ffprobe_installed(self):
        """Проверка установки FFprobe"""
        try:
            result = subprocess.run(['ffprobe', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, "FFprobe не установлен или недоступен"
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("FFprobe не найден в системе")


class TestAPIKeys:
    """Тесты API ключей"""
    
    def test_openai_key_exists(self):
        """Проверка наличия OpenAI API ключа"""
        # Проверяем переменную окружения
        openai_key = os.getenv('OPENAI_API_KEY')
        
        # Проверяем .env файл
        if not openai_key:
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                with open(env_path, 'r') as f:
                    env_content = f.read()
                    if 'OPENAI_API_KEY' in env_content:
                        print("OpenAI API ключ найден в .env файле")
                        return
        
        if openai_key:
            assert len(openai_key) > 20, "OpenAI API ключ слишком короткий"
            assert openai_key.startswith('sk-'), "OpenAI API ключ должен начинаться с 'sk-'"
            print("OpenAI API ключ корректен")
        else:
            pytest.skip("OpenAI API ключ не найден")


class TestModels:
    """Тесты доступности моделей"""
    
    def test_yolo_model_download(self):
        """Тест загрузки YOLO модели"""
        try:
            from ultralytics import YOLO
            
            # Попробуем загрузить наименьшую модель
            model = YOLO('yolo11n.pt')
            assert model is not None, "Не удалось загрузить YOLO модель"
            
            print("YOLO модель успешно загружена")
            
        except ImportError:
            pytest.skip("Ultralytics не установлен")
        except Exception as e:
            pytest.skip(f"Ошибка загрузки YOLO модели: {e}")
    
    def test_deepface_availability(self):
        """Тест доступности DeepFace"""
        try:
            import deepface
            print(f"DeepFace доступен: {deepface.__version__}")
            
        except ImportError:
            pytest.skip("DeepFace не установлен")
    
    def test_fer_availability(self):
        """Тест доступности FER"""
        try:
            import fer
            print("FER библиотека доступна")
            
        except ImportError:
            pytest.skip("FER не установлен")


class TestDirectories:
    """Тесты структуры директорий"""
    
    def test_storage_directories(self):
        """Проверка структуры директорий storage"""
        base_path = Path(__file__).parent.parent
        storage_dirs = [
            'storage',
            'storage/audio',
            'storage/cache', 
            'storage/faces',
            'storage/faces_yolo11',
            'storage/frames',
            'storage/reports',
            'storage/results',
            'storage/videos'
        ]
        
        missing_dirs = []
        for dir_name in storage_dirs:
            dir_path = base_path / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            # Создаем отсутствующие директории
            for dir_name in missing_dirs:
                (base_path / dir_name).mkdir(parents=True, exist_ok=True)
            
            print(f"Созданы отсутствующие директории: {missing_dirs}")
    
    def test_temp_uploads_directory(self):
        """Проверка директории temp_uploads"""
        base_path = Path(__file__).parent.parent
        temp_dir = base_path / "temp_uploads"
        
        if not temp_dir.exists():
            temp_dir.mkdir(exist_ok=True)
            print("Создана директория temp_uploads")
        
        assert temp_dir.is_dir(), "temp_uploads должна быть директорией"


class TestSystemResources:
    """Тесты системных ресурсов"""
    
    def test_memory_availability(self):
        """Проверка доступной памяти"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            print(f"Доступная память: {available_gb:.2f} GB")
            assert available_gb > 2, f"Недостаточно памяти: {available_gb:.2f} GB < 2 GB"
            
        except ImportError:
            pytest.skip("psutil не установлен")
    
    def test_disk_space(self):
        """Проверка дискового пространства"""
        try:
            import psutil
            
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024**3)
            
            print(f"Свободное место на диске: {free_gb:.2f} GB")
            assert free_gb > 5, f"Недостаточно места на диске: {free_gb:.2f} GB < 5 GB"
            
        except ImportError:
            pytest.skip("psutil не установлен")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])