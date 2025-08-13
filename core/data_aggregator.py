"""
Продвинутый агрегатор данных для полного объединения всех источников
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import sqlite3

import numpy as np
import pandas as pd
from scipy import interpolate, stats, signal
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.translation import EmotionTranslator


class DataAggregator:
    """
    Продвинутый агрегатор данных для полного объединения всех источников
    Синхронизация временных данных, поиск аномалий, корреляционный анализ
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Конфигурация временной синхронизации
        analysis_config = config.get('analysis', {})
        self.time_resolution = analysis_config.get('time_resolution', 0.1)  # 100мс
        self.sync_window = analysis_config.get('sync_window', 1.0)  # 1 секунда
        self.interpolation_method = analysis_config.get('interpolation_method', 'cubic')
        
        # Пороги для детекции аномалий
        self.emotion_change_threshold = analysis_config.get('emotion_change_threshold', 0.7)
        self.mismatch_threshold = analysis_config.get('mismatch_threshold', 0.5)
        self.speech_pause_threshold = analysis_config.get('speech_pause_threshold', 2.0)
        self.micro_expression_duration = analysis_config.get('micro_expression_duration', 0.5)
        
        # Инициализация переводчика эмоций
        self.emotion_translator = EmotionTranslator()
        
        # Хранилище данных
        self.unified_data = {}
        self.synchronized_data = {}
        self.timeline = []
        self.critical_moments = []
        self.correlations = {}
        self.statistics = {}
        
        # Создание директорий для экспорта
        storage_config = config.get('storage', {})
        self.export_dir = Path(storage_config.get('analysis_dir', 'storage/analysis'))
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("DataAggregator initialized")
    
    def aggregate_all_sources(
        self, 
        video_emotions: Dict[str, Any],
        audio_emotions: Dict[str, Any], 
        speech_transcript: Dict[str, Any],
        face_tracking: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Объединение всех источников данных с временной синхронизацией
        
        Args:
            video_emotions: Данные эмоций с видео
            audio_emotions: Данные эмоций из аудио
            speech_transcript: Транскрипция речи
            face_tracking: Данные трекинга лиц
            additional_data: Дополнительные данные
            
        Returns:
            Объединенные и синхронизированные данные
        """
        self.logger.info("Starting data aggregation from all sources")
        
        try:
            # Подготовка данных источников
            sources = {
                'video_emotions': self._prepare_video_emotions(video_emotions),
                'audio_emotions': self._prepare_audio_emotions(audio_emotions),
                'speech_transcript': self._prepare_speech_transcript(speech_transcript),
                'face_tracking': self._prepare_face_tracking(face_tracking)
            }
            
            if additional_data:
                sources.update(additional_data)
            
            # Синхронизация временных меток
            self.synchronized_data = self.synchronize_timestamps(sources)
            
            # Создание единой временной шкалы
            self.timeline = self.create_unified_timeline()
            
            # Поиск аномалий и критических моментов
            self.critical_moments = self.detect_anomalies()
            
            # Корреляционный анализ
            self.correlations = self.calculate_correlations()
            
            # Статистический анализ
            self.statistics = self.calculate_comprehensive_statistics()
            
            # Объединенные данные
            self.unified_data = {
                'sources': sources,
                'synchronized': self.synchronized_data,
                'timeline': self.timeline,
                'critical_moments': self.critical_moments,
                'correlations': self.correlations,
                'statistics': self.statistics,
                'metadata': {
                    'aggregation_time': time.time(),
                    'sources_count': len(sources),
                    'timeline_duration': self._get_total_duration(),
                    'time_resolution': self.time_resolution
                }
            }
            
            self.logger.info(f"Data aggregation completed: {len(self.timeline)} time points")
            return self.unified_data
            
        except Exception as e:
            self.logger.error(f"Data aggregation failed: {e}")
            raise
    
    def synchronize_timestamps(self, sources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Синхронизация временных меток всех источников
        
        Args:
            sources: Словарь с данными от разных источников
            
        Returns:
            Синхронизированные данные на единой временной сетке
        """
        self.logger.info("Synchronizing timestamps across all sources")
        
        try:
            # Находим общий временной диапазон
            all_timestamps = []
            for source_name, source_data in sources.items():
                if 'timestamps' in source_data:
                    all_timestamps.extend(source_data['timestamps'])
                elif 'timeline' in source_data:
                    all_timestamps.extend([item.get('timestamp', 0) for item in source_data['timeline']])
            
            if not all_timestamps:
                raise ValueError("No timestamps found in any source")
            
            min_time = min(all_timestamps)
            max_time = max(all_timestamps)
            
            self.logger.info(f"Time range: {min_time:.2f} - {max_time:.2f} seconds")
            
            # Создаем единую временную сетку
            time_grid = np.arange(min_time, max_time + self.time_resolution, self.time_resolution)
            
            # Интерполируем все данные на эту сетку
            synchronized = {
                'time_grid': time_grid,
                'sources': {}
            }
            
            for source_name, source_data in sources.items():
                try:
                    synchronized_source = self._interpolate_to_grid(source_data, time_grid, source_name)
                    synchronized['sources'][source_name] = synchronized_source
                    self.logger.debug(f"Synchronized {source_name}: {len(synchronized_source.get('values', []))} points")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to synchronize {source_name}: {e}")
                    # Создаем пустую синхронизированную версию
                    synchronized['sources'][source_name] = {
                        'values': [None] * len(time_grid),
                        'confidence': [0.0] * len(time_grid),
                        'metadata': {'sync_error': str(e)}
                    }
            
            return synchronized
            
        except Exception as e:
            self.logger.error(f"Timestamp synchronization failed: {e}")
            raise
    
    def create_unified_timeline(self) -> List[Dict[str, Any]]:
        """
        Создание единой временной шкалы всех событий
        
        Returns:
            Список временных точек с данными от всех источников
        """
        self.logger.info("Creating unified timeline")
        
        if not self.synchronized_data or 'time_grid' not in self.synchronized_data:
            raise ValueError("No synchronized data available")
        
        time_grid = self.synchronized_data['time_grid']
        sources = self.synchronized_data['sources']
        
        timeline = []
        
        for i, timestamp in enumerate(time_grid):
            timeline_point = {
                'timestamp': float(timestamp),
                'index': i,
                'sources': {}
            }
            
            # Добавляем данные от каждого источника
            for source_name, source_data in sources.items():
                if i < len(source_data.get('values', [])):
                    timeline_point['sources'][source_name] = {
                        'value': source_data['values'][i],
                        'confidence': source_data.get('confidence', [1.0])[i] if i < len(source_data.get('confidence', [])) else 0.0
                    }
                else:
                    timeline_point['sources'][source_name] = {
                        'value': None,
                        'confidence': 0.0
                    }
            
            timeline.append(timeline_point)
        
        self.logger.info(f"Created unified timeline with {len(timeline)} points")
        return timeline
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Поиск несоответствий между модальностями и критических моментов
        
        Returns:
            Список обнаруженных аномалий с метаданными
        """
        self.logger.info("Detecting anomalies and critical moments")
        
        critical = []
        
        try:
            # 1. Резкие изменения эмоций
            emotion_changes = self._detect_rapid_emotion_changes()
            critical.extend(emotion_changes)
            
            # 2. Несоответствие видео/аудио эмоций
            mismatches = self._detect_modality_mismatches()
            critical.extend(mismatches)
            
            # 3. Длительные паузы в речи
            speech_pauses = self._detect_speech_anomalies()
            critical.extend(speech_pauses)
            
            # 4. Микровыражения (короткие эмоциональные всплески)
            micro_expressions = self._detect_micro_expressions()
            critical.extend(micro_expressions)
            
            # 5. Статистические выбросы
            statistical_outliers = self._detect_statistical_outliers()
            critical.extend(statistical_outliers)
            
            # Ранжирование по важности
            critical = self._rank_by_importance(critical)
            
            self.logger.info(f"Detected {len(critical)} critical moments")
            return critical
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return []
    
    def calculate_correlations(self) -> Dict[str, Any]:
        """
        Корреляционный анализ между различными модальностями
        
        Returns:
            Корреляции и паттерны между источниками данных
        """
        self.logger.info("Calculating cross-modal correlations")
        
        correlations = {}
        
        try:
            # Извлечение числовых данных для корреляционного анализа
            numeric_data = self._extract_numeric_features()
            
            if not numeric_data:
                self.logger.warning("No numeric data available for correlation analysis")
                return correlations
            
            # 1. Корреляция между эмоциями лица и речи
            face_speech_corr = self._calculate_face_speech_correlation(numeric_data)
            correlations['face_speech'] = face_speech_corr
            
            # 2. Временные паттерны и автокорреляция
            temporal_patterns = self._find_temporal_patterns(numeric_data)
            correlations['temporal_patterns'] = temporal_patterns
            
            # 3. Кластеризация эмоциональных состояний
            emotional_clusters = self._cluster_emotional_states(numeric_data)
            correlations['emotional_clusters'] = emotional_clusters
            
            # 4. Причинно-следственные связи (Granger causality)
            causality = self._analyze_causality(numeric_data)
            correlations['causality'] = causality
            
            # 5. Фазовая синхронизация
            phase_sync = self._analyze_phase_synchronization(numeric_data)
            correlations['phase_synchronization'] = phase_sync
            
            self.logger.info("Correlation analysis completed")
            return correlations
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
            return correlations
    
    def calculate_comprehensive_statistics(self) -> Dict[str, Any]:
        """
        Вычисление всеобъемлющих статистических метрик
        
        Returns:
            Комплексные статистические показатели для психологического анализа
        """
        self.logger.info("Calculating comprehensive statistics")
        
        stats = {}
        
        try:
            # 1. Распределение эмоций
            stats['emotion_distribution'] = self._calculate_emotion_distribution()
            
            # 2. Матрица переходов между эмоциями
            stats['transition_matrix'] = self._create_emotion_transition_matrix()
            
            # 3. Индекс эмоциональной стабильности
            stats['stability_index'] = self._calculate_emotional_stability()
            
            # 4. Оценка уровня стресса
            stats['stress_level'] = self._estimate_stress_level()
            
            # 5. Вероятность обмана (на основе несоответствий)
            stats['deception_probability'] = self._calculate_deception_score()
            
            # 6. Индекс сотрудничества
            stats['cooperation_index'] = self._assess_cooperation_level()
            
            # 7. Показатели внимания и концентрации
            stats['attention_metrics'] = self._calculate_attention_metrics()
            
            # 8. Динамика эмоциональной нагрузки
            stats['emotional_load'] = self._calculate_emotional_load_dynamics()
            
            # 9. Качество коммуникации
            stats['communication_quality'] = self._assess_communication_quality()
            
            # 10. Временная консистентность
            stats['temporal_consistency'] = self._calculate_temporal_consistency()
            
            self.logger.info("Statistical analysis completed")
            return stats
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {e}")
            return stats
    
    def create_annotated_timeline(self) -> List[Dict[str, Any]]:
        """
        Создание аннотированной временной шкалы с полными метаданными
        
        Returns:
            Детальная временная шкала с аннотациями
        """
        self.logger.info("Creating annotated timeline")
        
        if not self.timeline:
            raise ValueError("Timeline not available. Run aggregate_all_sources first.")
        
        annotated_timeline = []
        
        for point in self.timeline:
            timestamp = point['timestamp']
            
            annotated_point = {
                'time': timestamp,
                'time_formatted': str(timedelta(seconds=timestamp)),
                'face_emotion': self._get_face_emotion_at(timestamp),
                'speech_emotion': self._get_speech_emotion_at(timestamp),
                'transcript': self._get_transcript_at(timestamp),
                'critical_level': self._get_criticality_at(timestamp),
                'annotations': self._get_annotations_at(timestamp),
                'confidence_scores': self._get_confidence_scores_at(timestamp),
                'statistical_markers': self._get_statistical_markers_at(timestamp)
            }
            
            annotated_timeline.append(annotated_point)
        
        self.logger.info(f"Created annotated timeline with {len(annotated_timeline)} points")
        return annotated_timeline
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Экспорт в DataFrame для анализа"""
        if not self.timeline:
            raise ValueError("No timeline data available")
        
        rows = []
        for point in self.timeline:
            row = {'timestamp': point['timestamp']}
            
            # Добавляем данные от каждого источника
            for source_name, source_data in point.get('sources', {}).items():
                row[f"{source_name}_value"] = source_data.get('value')
                row[f"{source_name}_confidence"] = source_data.get('confidence')
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        self.logger.info(f"Exported to DataFrame: {df.shape}")
        return df
    
    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """Экспорт в JSON для визуализации"""
        if filepath is None:
            filepath = self.export_dir / f"aggregated_data_{int(time.time())}.json"
        
        export_data = {
            'unified_data': self.unified_data,
            'annotated_timeline': self.create_annotated_timeline() if self.timeline else [],
            'export_metadata': {
                'export_time': datetime.now().isoformat(),
                'data_points': len(self.timeline),
                'sources': list(self.synchronized_data.get('sources', {}).keys()) if self.synchronized_data else []
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"Exported to JSON: {filepath}")
        return str(filepath)
    
    def export_to_sql(self, db_path: Optional[str] = None) -> str:
        """Экспорт в SQLite базу данных"""
        if db_path is None:
            db_path = self.export_dir / f"aggregated_data_{int(time.time())}.db"
        
        try:
            conn = sqlite3.connect(db_path)
            
            # Создание таблиц
            self._create_sql_tables(conn)
            
            # Вставка данных
            self._insert_timeline_data(conn)
            self._insert_critical_moments(conn)
            self._insert_statistics(conn)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Exported to SQL database: {db_path}")
            return str(db_path)
            
        except Exception as e:
            self.logger.error(f"SQL export failed: {e}")
            raise
    
    def create_visualization_plots(self, save_dir: Optional[str] = None) -> Dict[str, str]:
        """Создание графиков для презентации"""
        if save_dir is None:
            save_dir = self.export_dir / "visualizations"
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        try:
            # 1. Временная шкала эмоций
            plots['emotion_timeline'] = self._create_emotion_timeline_plot(save_dir)
            
            # 2. Корреляционная матрица
            plots['correlation_matrix'] = self._create_correlation_matrix_plot(save_dir)
            
            # 3. Критические моменты
            plots['critical_moments'] = self._create_critical_moments_plot(save_dir)
            
            # 4. Статистическое распределение
            plots['statistical_distribution'] = self._create_statistical_plots(save_dir)
            
            # 5. Интерактивная временная шкала (Plotly)
            plots['interactive_timeline'] = self._create_interactive_timeline(save_dir)
            
            self.logger.info(f"Created {len(plots)} visualization plots")
            return plots
            
        except Exception as e:
            self.logger.error(f"Visualization creation failed: {e}")
            return plots
    
    # Приватные методы для подготовки данных
    def _prepare_video_emotions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Подготовка данных эмоций с видео"""
        return {
            'type': 'video_emotions',
            'timestamps': data.get('timestamps', []),
            'emotions': data.get('emotions', []),
            'confidences': data.get('confidences', []),
            'faces': data.get('faces', [])
        }
    
    def _prepare_audio_emotions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Подготовка данных эмоций из аудио"""
        return {
            'type': 'audio_emotions',
            'timestamps': data.get('timestamps', []),
            'emotions': data.get('emotions', []),
            'features': data.get('features', {}),
            'segments': data.get('segments', [])
        }
    
    def _prepare_speech_transcript(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Подготовка транскрипции речи"""
        return {
            'type': 'speech_transcript',
            'timestamps': data.get('timestamps', []),
            'segments': data.get('segments', []),
            'words': data.get('words', []),
            'speakers': data.get('speakers', [])
        }
    
    def _prepare_face_tracking(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Подготовка данных трекинга лиц"""
        return {
            'type': 'face_tracking',
            'timestamps': data.get('timestamps', []),
            'faces': data.get('faces', []),
            'tracking_ids': data.get('tracking_ids', []),
            'landmarks': data.get('landmarks', [])
        }
    
    def _interpolate_to_grid(self, source_data: Dict[str, Any], time_grid: np.ndarray, source_name: str) -> Dict[str, Any]:
        """Интерполяция данных источника на временную сетку"""
        try:
            source_timestamps = source_data.get('timestamps', [])
            
            if not source_timestamps:
                # Если нет временных меток, создаем равномерно распределенные
                if 'emotions' in source_data and source_data['emotions']:
                    duration = time_grid[-1] - time_grid[0]
                    source_timestamps = np.linspace(time_grid[0], time_grid[-1], len(source_data['emotions']))
                else:
                    return {
                        'values': [None] * len(time_grid),
                        'confidence': [0.0] * len(time_grid)
                    }
            
            # Определяем тип данных для интерполяции
            if source_name == 'video_emotions' or source_name == 'audio_emotions':
                values = self._encode_emotions_for_interpolation(source_data.get('emotions', []))
                confidences = source_data.get('confidences', [1.0] * len(values))
            elif source_name == 'speech_transcript':
                values = self._encode_speech_for_interpolation(source_data)
                confidences = [1.0] * len(values)
            elif source_name == 'face_tracking':
                values = self._encode_faces_for_interpolation(source_data)
                confidences = [1.0] * len(values)
            else:
                values = source_data.get('values', [])
                confidences = source_data.get('confidences', [1.0] * len(values))
            
            if not values or len(values) != len(source_timestamps):
                return {
                    'values': [None] * len(time_grid),
                    'confidence': [0.0] * len(time_grid)
                }
            
            # Интерполяция значений
            if len(set(values)) > 1:  # Есть различающиеся значения
                if self.interpolation_method == 'linear':
                    f_values = interpolate.interp1d(
                        source_timestamps, values, 
                        kind='linear', fill_value='extrapolate', bounds_error=False
                    )
                elif self.interpolation_method == 'cubic':
                    if len(values) >= 4:
                        f_values = interpolate.interp1d(
                            source_timestamps, values, 
                            kind='cubic', fill_value='extrapolate', bounds_error=False
                        )
                    else:
                        f_values = interpolate.interp1d(
                            source_timestamps, values, 
                            kind='linear', fill_value='extrapolate', bounds_error=False
                        )
                else:
                    f_values = interpolate.interp1d(
                        source_timestamps, values, 
                        kind='nearest', fill_value='extrapolate', bounds_error=False
                    )
                
                interpolated_values = f_values(time_grid)
            else:
                # Все значения одинаковые
                interpolated_values = [values[0]] * len(time_grid)
            
            # Интерполяция доверительности
            if len(confidences) == len(source_timestamps):
                f_conf = interpolate.interp1d(
                    source_timestamps, confidences,
                    kind='linear', fill_value=0.0, bounds_error=False
                )
                interpolated_confidences = f_conf(time_grid)
            else:
                interpolated_confidences = [np.mean(confidences)] * len(time_grid)
            
            return {
                'values': interpolated_values.tolist() if hasattr(interpolated_values, 'tolist') else list(interpolated_values),
                'confidence': interpolated_confidences.tolist() if hasattr(interpolated_confidences, 'tolist') else list(interpolated_confidences),
                'original_timestamps': source_timestamps,
                'interpolation_method': self.interpolation_method
            }
            
        except Exception as e:
            self.logger.warning(f"Interpolation failed for {source_name}: {e}")
            return {
                'values': [None] * len(time_grid),
                'confidence': [0.0] * len(time_grid),
                'error': str(e)
            }
    
    def _encode_emotions_for_interpolation(self, emotions: List[str]) -> List[float]:
        """Кодирование эмоций в числовые значения"""
        emotion_encoding = {
            'нейтральность': 0.0,
            'радость': 1.0,
            'печаль': -0.8,
            'злость': -1.0,
            'страх': -0.6,
            'удивление': 0.5,
            'отвращение': -0.7,
            'презрение': -0.5,
            
            # English emotions (fallback)
            'neutral': 0.0,
            'happy': 1.0,
            'sad': -0.8,
            'angry': -1.0,
            'fear': -0.6,
            'surprise': 0.5,
            'disgust': -0.7,
            'contempt': -0.5
        }
        
        encoded = []
        for emotion in emotions:
            if isinstance(emotion, str):
                encoded.append(emotion_encoding.get(emotion.lower(), 0.0))
            elif isinstance(emotion, dict) and 'emotion' in emotion:
                encoded.append(emotion_encoding.get(emotion['emotion'].lower(), 0.0))
            else:
                encoded.append(0.0)
        
        return encoded
    
    def _encode_speech_for_interpolation(self, speech_data: Dict[str, Any]) -> List[float]:
        """Кодирование речевых данных"""
        segments = speech_data.get('segments', [])
        if not segments:
            return [0.0]
        
        # Простое кодирование: наличие речи = 1, отсутствие = 0
        return [1.0 if segment.get('text', '').strip() else 0.0 for segment in segments]
    
    def _encode_faces_for_interpolation(self, face_data: Dict[str, Any]) -> List[float]:
        """Кодирование данных лиц"""
        faces = face_data.get('faces', [])
        if not faces:
            return [0.0]
        
        # Кодирование: количество обнаруженных лиц
        return [len(face_group) if isinstance(face_group, list) else (1.0 if face_group else 0.0) for face_group in faces]
    
    def _get_total_duration(self) -> float:
        """Получение общей длительности анализа"""
        if not self.timeline:
            return 0.0
        return self.timeline[-1]['timestamp'] - self.timeline[0]['timestamp']
    
    # Заглушки для методов детекции и анализа (будут реализованы в следующих частях)
    def _detect_rapid_emotion_changes(self) -> List[Dict[str, Any]]:
        """Детекция резких изменений эмоций"""
        return []
    
    def _detect_modality_mismatches(self) -> List[Dict[str, Any]]:
        """Детекция несоответствий между модальностями"""
        return []
    
    def _detect_speech_anomalies(self) -> List[Dict[str, Any]]:
        """Детекция аномалий в речи"""
        return []
    
    def _detect_micro_expressions(self) -> List[Dict[str, Any]]:
        """Детекция микровыражений"""
        return []
    
    def _detect_statistical_outliers(self) -> List[Dict[str, Any]]:
        """Детекция статистических выбросов"""
        return []
    
    def _rank_by_importance(self, critical_moments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ранжирование критических моментов по важности"""
        return sorted(critical_moments, key=lambda x: x.get('importance', 0), reverse=True)
    
    # Полная реализация методов анализа корреляций
    def _extract_numeric_features(self) -> Dict[str, np.ndarray]:
        """Извлечение числовых признаков для корреляционного анализа"""
        numeric_data = {}
        
        try:
            if not self.synchronized_data or 'sources' not in self.synchronized_data:
                return numeric_data
            
            for source_name, source_data in self.synchronized_data['sources'].items():
                values = source_data.get('values', [])
                confidences = source_data.get('confidence', [])
                
                if values:
                    # Очищаем данные от NaN
                    values_array = np.array(values, dtype=float)
                    conf_array = np.array(confidences, dtype=float) if confidences else np.ones_like(values_array)
                    
                    # Заменяем NaN на медиану или 0
                    valid_mask = ~np.isnan(values_array)
                    if np.any(valid_mask):
                        median_val = np.median(values_array[valid_mask])
                        values_array[~valid_mask] = median_val
                    else:
                        values_array = np.zeros_like(values_array)
                    
                    numeric_data[source_name] = {
                        'values': values_array,
                        'confidences': conf_array,
                        'derivatives': np.gradient(values_array),
                        'moving_avg': self._calculate_moving_average(values_array),
                        'variance': self._calculate_moving_variance(values_array)
                    }
            
            return numeric_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract numeric features: {e}")
            return numeric_data
    
    def _calculate_face_speech_correlation(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Корреляция между эмоциями лица и речи"""
        correlations = {}
        
        try:
            video_data = data.get('video_emotions', {})
            audio_data = data.get('audio_emotions', {})
            
            if not video_data or not audio_data:
                return correlations
            
            video_values = video_data.get('values', np.array([]))
            audio_values = audio_data.get('values', np.array([]))
            
            if len(video_values) != len(audio_values) or len(video_values) < 2:
                return correlations
            
            # Основная корреляция Пирсона
            pearson_corr, p_value = stats.pearsonr(video_values, audio_values)
            correlations['pearson_correlation'] = float(pearson_corr)
            correlations['p_value'] = float(p_value)
            
            # Корреляция Спирмена (ранговая)
            spearman_corr, spearman_p = stats.spearmanr(video_values, audio_values)
            correlations['spearman_correlation'] = float(spearman_corr)
            correlations['spearman_p_value'] = float(spearman_p)
            
            # Взаимная корреляция с временным сдвигом
            cross_corr = np.correlate(video_values - np.mean(video_values), 
                                    audio_values - np.mean(audio_values), mode='full')
            max_corr_idx = np.argmax(np.abs(cross_corr))
            time_lag = max_corr_idx - len(video_values) + 1
            max_cross_corr = cross_corr[max_corr_idx] / (np.std(video_values) * np.std(audio_values) * len(video_values))
            
            correlations['max_cross_correlation'] = float(max_cross_corr)
            correlations['optimal_time_lag'] = int(time_lag)
            
            # Корреляция производных (скоростей изменения)
            if 'derivatives' in video_data and 'derivatives' in audio_data:
                video_deriv = video_data['derivatives']
                audio_deriv = audio_data['derivatives']
                
                if len(video_deriv) == len(audio_deriv) and len(video_deriv) > 1:
                    deriv_corr, deriv_p = stats.pearsonr(video_deriv, audio_deriv)
                    correlations['derivative_correlation'] = float(deriv_corr)
                    correlations['derivative_p_value'] = float(deriv_p)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate face-speech correlation: {e}")
        
        return correlations
    
    def _find_temporal_patterns(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Поиск временных паттернов в данных"""
        patterns = {}
        
        try:
            for source_name, source_data in data.items():
                values = source_data.get('values', np.array([]))
                
                if len(values) < 10:
                    continue
                
                source_patterns = {}
                
                # Автокорреляция для поиска периодичности
                autocorr = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]  # Нормализация
                
                # Поиск пиков автокорреляции
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(autocorr[1:], height=0.3, distance=5)
                
                if len(peaks) > 0:
                    # Основной период
                    main_period = peaks[0] + 1  # +1 потому что начали с [1:]
                    source_patterns['main_period'] = int(main_period)
                    source_patterns['periodicity_strength'] = float(autocorr[main_period])
                
                # Тренд анализ
                time_indices = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_indices, values)
                
                source_patterns['trend_slope'] = float(slope)
                source_patterns['trend_r_squared'] = float(r_value**2)
                source_patterns['trend_significance'] = float(p_value)
                
                # Анализ сезонности (если данных достаточно)
                if len(values) > 50:
                    # Простое разложение на тренд и сезонность
                    detrended = values - (slope * time_indices + intercept)
                    
                    # FFT для поиска доминирующих частот
                    fft = np.fft.fft(detrended)
                    freqs = np.fft.fftfreq(len(values))
                    
                    # Находим доминирующие частоты
                    magnitude = np.abs(fft)
                    dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
                    dominant_frequency = freqs[dominant_freq_idx]
                    
                    source_patterns['dominant_frequency'] = float(dominant_frequency)
                    source_patterns['frequency_magnitude'] = float(magnitude[dominant_freq_idx])
                
                # Анализ изменчивости
                source_patterns['variance'] = float(np.var(values))
                source_patterns['coefficient_of_variation'] = float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
                
                patterns[source_name] = source_patterns
        
        except Exception as e:
            self.logger.error(f"Failed to find temporal patterns: {e}")
        
        return patterns
    
    def _cluster_emotional_states(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Кластеризация эмоциональных состояний"""
        clusters = {}
        
        try:
            # Собираем все эмоциональные данные
            emotion_sources = ['video_emotions', 'audio_emotions']
            emotion_data = []
            timestamps = []
            
            for source_name in emotion_sources:
                if source_name in data:
                    values = data[source_name].get('values', np.array([]))
                    if len(values) > 0:
                        emotion_data.append(values)
                        if not timestamps:
                            timestamps = np.arange(len(values))
            
            if len(emotion_data) < 1:
                return clusters
            
            # Объединяем данные для кластеризации
            if len(emotion_data) == 1:
                feature_matrix = emotion_data[0].reshape(-1, 1)
            else:
                # Убеждаемся, что все массивы одинаковой длины
                min_length = min(len(arr) for arr in emotion_data)
                emotion_data = [arr[:min_length] for arr in emotion_data]
                feature_matrix = np.column_stack(emotion_data)
            
            if feature_matrix.shape[0] < 3:
                return clusters
            
            # Стандартизация данных
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
            
            # K-means кластеризация
            optimal_k = self._find_optimal_clusters(scaled_features)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            clusters['kmeans'] = {
                'n_clusters': int(optimal_k),
                'labels': cluster_labels.tolist(),
                'centers': kmeans.cluster_centers_.tolist(),
                'inertia': float(kmeans.inertia_)
            }
            
            # DBSCAN для поиска выбросов
            if len(scaled_features) > 10:
                dbscan = DBSCAN(eps=0.5, min_samples=3)
                dbscan_labels = dbscan.fit_predict(scaled_features)
                
                n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                n_noise = list(dbscan_labels).count(-1)
                
                clusters['dbscan'] = {
                    'n_clusters': int(n_clusters_dbscan),
                    'n_noise_points': int(n_noise),
                    'labels': dbscan_labels.tolist()
                }
            
            # Анализ переходов между состояниями
            transitions = self._analyze_state_transitions(cluster_labels)
            clusters['transition_analysis'] = transitions
            
        except Exception as e:
            self.logger.error(f"Failed to cluster emotional states: {e}")
        
        return clusters
    
    def _analyze_causality(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Анализ причинно-следственных связей (упрощенный Granger causality)"""
        causality = {}
        
        try:
            # Анализируем связь между видео и аудио эмоциями
            if 'video_emotions' in data and 'audio_emotions' in data:
                video_values = data['video_emotions'].get('values', np.array([]))
                audio_values = data['audio_emotions'].get('values', np.array([]))
                
                if len(video_values) == len(audio_values) and len(video_values) > 10:
                    # Простой анализ запаздывающих корреляций
                    max_lag = min(5, len(video_values) // 4)
                    
                    # Видео -> Аудио
                    video_to_audio = self._calculate_lagged_correlations(video_values, audio_values, max_lag)
                    causality['video_to_audio'] = video_to_audio
                    
                    # Аудио -> Видео
                    audio_to_video = self._calculate_lagged_correlations(audio_values, video_values, max_lag)
                    causality['audio_to_video'] = audio_to_video
                    
                    # Определяем доминирующее направление
                    max_video_to_audio = max(video_to_audio.values()) if video_to_audio else 0
                    max_audio_to_video = max(audio_to_video.values()) if audio_to_video else 0
                    
                    if max_video_to_audio > max_audio_to_video:
                        causality['dominant_direction'] = 'video_leads'
                        causality['strength'] = float(max_video_to_audio)
                    else:
                        causality['dominant_direction'] = 'audio_leads'
                        causality['strength'] = float(max_audio_to_video)
            
            # Анализ связи между речевой активностью и эмоциями
            if 'speech_transcript' in data and 'video_emotions' in data:
                speech_values = data['speech_transcript'].get('values', np.array([]))
                video_values = data['video_emotions'].get('values', np.array([]))
                
                if len(speech_values) == len(video_values) and len(speech_values) > 5:
                    speech_emotion_corr = self._calculate_lagged_correlations(speech_values, video_values, 3)
                    causality['speech_to_emotion'] = speech_emotion_corr
            
        except Exception as e:
            self.logger.error(f"Failed to analyze causality: {e}")
        
        return causality
    
    def _analyze_phase_synchronization(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Анализ фазовой синхронизации между сигналами"""
        sync_analysis = {}
        
        try:
            if 'video_emotions' not in data or 'audio_emotions' not in data:
                return sync_analysis
            
            video_values = data['video_emotions'].get('values', np.array([]))
            audio_values = data['audio_emotions'].get('values', np.array([]))
            
            if len(video_values) != len(audio_values) or len(video_values) < 20:
                return sync_analysis
            
            # Аналитический сигнал и фаза с помощью преобразования Гильберта
            from scipy.signal import hilbert
            
            video_analytic = hilbert(video_values)
            audio_analytic = hilbert(audio_values)
            
            video_phase = np.angle(video_analytic)
            audio_phase = np.angle(audio_analytic)
            
            # Разность фаз
            phase_diff = video_phase - audio_phase
            
            # Круговая статистика для анализа синхронизации
            # Индекс фазовой синхронизации (PLV - Phase Locking Value)
            complex_phase_diff = np.exp(1j * phase_diff)
            plv = np.abs(np.mean(complex_phase_diff))
            
            sync_analysis['phase_locking_value'] = float(plv)
            sync_analysis['mean_phase_diff'] = float(np.angle(np.mean(complex_phase_diff)))
            sync_analysis['phase_coherence'] = float(1 - np.var(phase_diff) / (2 * np.pi))
            
            # Временная динамика синхронизации
            if len(video_values) > 50:
                window_size = min(20, len(video_values) // 3)
                sliding_plv = []
                
                for i in range(0, len(phase_diff) - window_size, window_size // 2):
                    window_phase_diff = phase_diff[i:i + window_size]
                    window_complex = np.exp(1j * window_phase_diff)
                    window_plv = np.abs(np.mean(window_complex))
                    sliding_plv.append(window_plv)
                
                sync_analysis['temporal_plv'] = sliding_plv
                sync_analysis['plv_variance'] = float(np.var(sliding_plv))
                sync_analysis['max_sync'] = float(np.max(sliding_plv))
                sync_analysis['min_sync'] = float(np.min(sliding_plv))
        
        except Exception as e:
            self.logger.error(f"Failed to analyze phase synchronization: {e}")
        
        return sync_analysis
    
    # Вспомогательные методы для корреляционного анализа
    def _calculate_moving_average(self, values: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Вычисление скользящего среднего"""
        if len(values) < window_size:
            return values
        
        return np.convolve(values, np.ones(window_size)/window_size, mode='same')
    
    def _calculate_moving_variance(self, values: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Вычисление скользящей дисперсии"""
        if len(values) < window_size:
            return np.zeros_like(values)
        
        moving_var = []
        half_window = window_size // 2
        
        for i in range(len(values)):
            start = max(0, i - half_window)
            end = min(len(values), i + half_window + 1)
            window_values = values[start:end]
            moving_var.append(np.var(window_values))
        
        return np.array(moving_var)
    
    def _find_optimal_clusters(self, data: np.ndarray, max_k: int = 8) -> int:
        """Поиск оптимального количества кластеров методом silhouette"""
        if len(data) < 4:
            return 2
        
        max_k = min(max_k, len(data) // 2)
        best_k = 2
        best_score = -1
        
        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                score = silhouette_score(data, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
        
        return best_k
    
    def _analyze_state_transitions(self, labels: np.ndarray) -> Dict[str, Any]:
        """Анализ переходов между эмоциональными состояниями"""
        transitions = {}
        
        if len(labels) < 2:
            return transitions
        
        # Матрица переходов
        unique_labels = np.unique(labels)
        n_states = len(unique_labels)
        
        transition_matrix = np.zeros((n_states, n_states))
        
        for i in range(len(labels) - 1):
            current_state = np.where(unique_labels == labels[i])[0][0]
            next_state = np.where(unique_labels == labels[i + 1])[0][0]
            transition_matrix[current_state, next_state] += 1
        
        # Нормализация
        row_sums = transition_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Избегаем деления на ноль
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        transitions['transition_matrix'] = transition_matrix.tolist()
        transitions['state_labels'] = unique_labels.tolist()
        
        # Анализ стабильности состояний
        stability = np.diag(transition_matrix)
        transitions['state_stability'] = stability.tolist()
        transitions['most_stable_state'] = int(unique_labels[np.argmax(stability)])
        
        return transitions
    
    def _calculate_lagged_correlations(self, x: np.ndarray, y: np.ndarray, max_lag: int) -> Dict[str, float]:
        """Вычисление запаздывающих корреляций"""
        lagged_corrs = {}
        
        for lag in range(max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(x, y)[0, 1]
            else:
                if len(x) > lag and len(y) > lag:
                    x_lagged = x[:-lag]
                    y_current = y[lag:]
                    if len(x_lagged) > 1 and len(y_current) > 1:
                        corr = np.corrcoef(x_lagged, y_current)[0, 1]
                    else:
                        corr = 0.0
                else:
                    corr = 0.0
            
            if not np.isnan(corr):
                lagged_corrs[f'lag_{lag}'] = float(corr)
        
        return lagged_corrs
    
    # Полная реализация статистических методов
    def _calculate_emotion_distribution(self) -> Dict[str, float]:
        """Распределение эмоций по всем источникам"""
        distribution = {}
        
        try:
            if not self.synchronized_data or 'sources' not in self.synchronized_data:
                return distribution
            
            # Анализ видео эмоций
            video_emotions = self.synchronized_data['sources'].get('video_emotions', {})
            if video_emotions.get('values'):
                video_values = np.array(video_emotions['values'])
                video_clean = video_values[~np.isnan(video_values)]
                
                if len(video_clean) > 0:
                    distribution['video_emotions'] = {
                        'mean': float(np.mean(video_clean)),
                        'std': float(np.std(video_clean)),
                        'median': float(np.median(video_clean)),
                        'skewness': float(stats.skew(video_clean)),
                        'kurtosis': float(stats.kurtosis(video_clean)),
                        'positive_ratio': float(np.sum(video_clean > 0) / len(video_clean)),
                        'negative_ratio': float(np.sum(video_clean < 0) / len(video_clean)),
                        'neutral_ratio': float(np.sum(np.abs(video_clean) < 0.1) / len(video_clean))
                    }
            
            # Анализ аудио эмоций
            audio_emotions = self.synchronized_data['sources'].get('audio_emotions', {})
            if audio_emotions.get('values'):
                audio_values = np.array(audio_emotions['values'])
                audio_clean = audio_values[~np.isnan(audio_values)]
                
                if len(audio_clean) > 0:
                    distribution['audio_emotions'] = {
                        'mean': float(np.mean(audio_clean)),
                        'std': float(np.std(audio_clean)),
                        'median': float(np.median(audio_clean)),
                        'skewness': float(stats.skew(audio_clean)),
                        'kurtosis': float(stats.kurtosis(audio_clean)),
                        'positive_ratio': float(np.sum(audio_clean > 0) / len(audio_clean)),
                        'negative_ratio': float(np.sum(audio_clean < 0) / len(audio_clean)),
                        'neutral_ratio': float(np.sum(np.abs(audio_clean) < 0.1) / len(audio_clean))
                    }
            
            # Общее распределение
            all_emotion_values = []
            for source in ['video_emotions', 'audio_emotions']:
                source_data = self.synchronized_data['sources'].get(source, {})
                if source_data.get('values'):
                    values = np.array(source_data['values'])
                    clean_values = values[~np.isnan(values)]
                    all_emotion_values.extend(clean_values.tolist())
            
            if all_emotion_values:
                all_emotions = np.array(all_emotion_values)
                distribution['combined'] = {
                    'mean': float(np.mean(all_emotions)),
                    'std': float(np.std(all_emotions)),
                    'range': float(np.max(all_emotions) - np.min(all_emotions)),
                    'percentile_25': float(np.percentile(all_emotions, 25)),
                    'percentile_75': float(np.percentile(all_emotions, 75)),
                    'iqr': float(np.percentile(all_emotions, 75) - np.percentile(all_emotions, 25))
                }
        
        except Exception as e:
            self.logger.error(f"Failed to calculate emotion distribution: {e}")
        
        return distribution
    
    def _create_emotion_transition_matrix(self) -> Dict[str, Any]:
        """Создание матрицы переходов между эмоциональными состояниями"""
        transition_data = {}
        
        try:
            if not self.synchronized_data or 'sources' not in self.synchronized_data:
                return transition_data
            
            # Анализ переходов для видео эмоций
            video_emotions = self.synchronized_data['sources'].get('video_emotions', {})
            if video_emotions.get('values'):
                video_values = np.array(video_emotions['values'])
                video_clean = video_values[~np.isnan(video_values)]
                
                if len(video_clean) > 2:
                    # Квантизация эмоций в дискретные состояния
                    emotion_states = self._quantize_emotions(video_clean)
                    transition_matrix = self._build_transition_matrix(emotion_states)
                    
                    transition_data['video_transitions'] = {
                        'matrix': transition_matrix.tolist(),
                        'states': self._get_emotion_state_names(),
                        'dominant_transitions': self._find_dominant_transitions(transition_matrix)
                    }
            
            # Аналогично для аудио эмоций
            audio_emotions = self.synchronized_data['sources'].get('audio_emotions', {})
            if audio_emotions.get('values'):
                audio_values = np.array(audio_emotions['values'])
                audio_clean = audio_values[~np.isnan(audio_values)]
                
                if len(audio_clean) > 2:
                    emotion_states = self._quantize_emotions(audio_clean)
                    transition_matrix = self._build_transition_matrix(emotion_states)
                    
                    transition_data['audio_transitions'] = {
                        'matrix': transition_matrix.tolist(),
                        'states': self._get_emotion_state_names(),
                        'dominant_transitions': self._find_dominant_transitions(transition_matrix)
                    }
        
        except Exception as e:
            self.logger.error(f"Failed to create emotion transition matrix: {e}")
        
        return transition_data
    
    def _calculate_emotional_stability(self) -> float:
        """Вычисление индекса эмоциональной стабильности"""
        try:
            if not self.synchronized_data or 'sources' not in self.synchronized_data:
                return 0.5
            
            stability_scores = []
            
            for source_name in ['video_emotions', 'audio_emotions']:
                source_data = self.synchronized_data['sources'].get(source_name, {})
                if source_data.get('values'):
                    values = np.array(source_data['values'])
                    clean_values = values[~np.isnan(values)]
                    
                    if len(clean_values) > 5:
                        # Вычисляем коэффициент вариации (обратно пропорционален стабильности)
                        cv = np.std(clean_values) / (np.abs(np.mean(clean_values)) + 1e-8)
                        
                        # Стабильность как обратная величина к изменчивости
                        stability = 1.0 / (1.0 + cv)
                        
                        # Учитываем количество резких изменений
                        derivatives = np.diff(clean_values)
                        rapid_changes = np.sum(np.abs(derivatives) > np.std(derivatives) * 2)
                        change_penalty = rapid_changes / len(clean_values)
                        
                        stability = stability * (1.0 - change_penalty)
                        stability_scores.append(max(0.0, min(1.0, stability)))
            
            return float(np.mean(stability_scores)) if stability_scores else 0.5
            
        except Exception as e:
            self.logger.error(f"Failed to calculate emotional stability: {e}")
            return 0.5
    
    def _estimate_stress_level(self) -> Dict[str, Any]:
        """Комплексная оценка уровня стресса"""
        stress_data = {'level': 0.5, 'indicators': [], 'components': {}}
        
        try:
            if not self.synchronized_data or 'sources' not in self.synchronized_data:
                return stress_data
            
            stress_indicators = []
            components = {}
            
            # 1. Анализ эмоциональной нестабильности
            video_emotions = self.synchronized_data['sources'].get('video_emotions', {})
            if video_emotions.get('values'):
                values = np.array(video_emotions['values'])
                clean_values = values[~np.isnan(values)]
                
                if len(clean_values) > 5:
                    # Высокая изменчивость эмоций - индикатор стресса
                    emotional_variance = np.var(clean_values)
                    components['emotional_variance'] = float(emotional_variance)
                    
                    # Преобладание негативных эмоций
                    negative_ratio = np.sum(clean_values < -0.3) / len(clean_values)
                    components['negative_emotion_ratio'] = float(negative_ratio)
                    
                    if emotional_variance > 0.5:
                        stress_indicators.append('высокая_эмоциональная_изменчивость')
                    if negative_ratio > 0.4:
                        stress_indicators.append('преобладание_негативных_эмоций')
            
            # 2. Анализ несоответствий между модальностями (признак маскировки эмоций)
            if len(self.critical_moments) > 0:
                mismatches = [cm for cm in self.critical_moments if cm.get('type') == 'modality_mismatch']
                mismatch_ratio = len(mismatches) / len(self.timeline) if self.timeline else 0
                components['modality_mismatch_ratio'] = float(mismatch_ratio)
                
                if mismatch_ratio > 0.2:
                    stress_indicators.append('частые_несоответствия_эмоций')
            
            # 3. Анализ речевых паттернов
            speech_data = self.synchronized_data['sources'].get('speech_transcript', {})
            if speech_data.get('values'):
                speech_values = np.array(speech_data['values'])
                
                # Нерегулярность речи
                speech_changes = np.diff(speech_values.astype(float))
                speech_irregularity = np.std(speech_changes)
                components['speech_irregularity'] = float(speech_irregularity)
                
                if speech_irregularity > 0.3:
                    stress_indicators.append('нерегулярная_речевая_активность')
            
            # 4. Микровыражения как признак скрытого стресса
            micro_expressions = [cm for cm in self.critical_moments if cm.get('type') == 'micro_expression']
            micro_expr_count = len(micro_expressions)
            components['micro_expression_count'] = micro_expr_count
            
            if micro_expr_count > 3:
                stress_indicators.append('частые_микровыражения')
            
            # Вычисление общего уровня стресса
            stress_components = [
                min(components.get('emotional_variance', 0) * 0.3, 0.3),
                min(components.get('negative_emotion_ratio', 0) * 0.25, 0.25),
                min(components.get('modality_mismatch_ratio', 0) * 0.2, 0.2),
                min(components.get('speech_irregularity', 0) * 0.15, 0.15),
                min(components.get('micro_expression_count', 0) * 0.02, 0.1)
            ]
            
            stress_level = sum(stress_components)
            stress_level = max(0.0, min(1.0, stress_level))
            
            stress_data = {
                'level': float(stress_level),
                'indicators': stress_indicators,
                'components': components,
                'interpretation': self._interpret_stress_level(stress_level)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to estimate stress level: {e}")
        
        return stress_data
    
    def _calculate_deception_score(self) -> Dict[str, Any]:
        """Оценка вероятности обмана на основе поведенческих индикаторов"""
        deception_data = {'probability': 0.0, 'indicators': [], 'warning': 'Только для исследовательских целей'}
        
        try:
            if not self.synchronized_data or 'sources' not in self.synchronized_data:
                return deception_data
            
            deception_indicators = []
            evidence_scores = []
            
            # 1. Несоответствия между лицевыми эмоциями и речью
            mismatches = [cm for cm in self.critical_moments if cm.get('type') == 'modality_mismatch']
            if mismatches:
                mismatch_score = min(len(mismatches) / len(self.timeline) if self.timeline else 0, 0.3)
                evidence_scores.append(mismatch_score)
                if mismatch_score > 0.15:
                    deception_indicators.append('частые_несоответствия_эмоций_лица_и_речи')
            
            # 2. Микровыражения (могут указывать на скрываемые эмоции)
            micro_expressions = [cm for cm in self.critical_moments if cm.get('type') == 'micro_expression']
            if micro_expressions:
                micro_score = min(len(micro_expressions) * 0.05, 0.25)
                evidence_scores.append(micro_score)
                if len(micro_expressions) > 4:
                    deception_indicators.append('частые_микровыражения')
            
            # 3. Аномальные паузы в речи
            speech_pauses = [cm for cm in self.critical_moments if cm.get('type') == 'long_speech_pause']
            if speech_pauses:
                pause_score = min(len(speech_pauses) * 0.03, 0.15)
                evidence_scores.append(pause_score)
                if len(speech_pauses) > 2:
                    deception_indicators.append('необычные_паузы_в_речи')
            
            # 4. Резкие изменения в эмоциональном состоянии
            emotion_changes = [cm for cm in self.critical_moments if cm.get('type') == 'rapid_emotion_change']
            if emotion_changes:
                change_score = min(len(emotion_changes) * 0.02, 0.2)
                evidence_scores.append(change_score)
                if len(emotion_changes) > 5:
                    deception_indicators.append('резкие_изменения_эмоций')
            
            # 5. Низкая эмоциональная стабильность
            stability = self._calculate_emotional_stability()
            if stability < 0.3:
                evidence_scores.append(0.15)
                deception_indicators.append('низкая_эмоциональная_стабильность')
            
            # Вычисление итоговой вероятности
            total_score = sum(evidence_scores)
            probability = max(0.0, min(1.0, total_score))
            
            # Интерпретация результата
            if probability < 0.3:
                interpretation = 'низкая_вероятность_обмана'
            elif probability < 0.6:
                interpretation = 'умеренные_признаки_возможного_обмана'
            else:
                interpretation = 'множественные_индикаторы_возможного_обмана'
            
            deception_data = {
                'probability': float(probability),
                'indicators': deception_indicators,
                'interpretation': interpretation,
                'evidence_breakdown': {
                    'modality_mismatches': len(mismatches),
                    'micro_expressions': len(micro_expressions),
                    'speech_anomalies': len(speech_pauses),
                    'emotion_changes': len(emotion_changes),
                    'stability_score': float(stability)
                },
                'warning': 'Результаты носят исследовательский характер и не являются доказательством'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate deception score: {e}")
        
        return deception_data
    
    def _assess_cooperation_level(self) -> Dict[str, Any]:
        """Оценка уровня сотрудничества"""
        return {'index': 0.5, 'markers': []}
    
    def _calculate_attention_metrics(self) -> Dict[str, float]:
        """Показатели внимания"""
        return {}
    
    def _calculate_emotional_load_dynamics(self) -> Dict[str, Any]:
        """Динамика эмоциональной нагрузки"""
        return {}
    
    def _assess_communication_quality(self) -> Dict[str, Any]:
        """Качество коммуникации"""
        return {}
    
    def _calculate_temporal_consistency(self) -> Dict[str, float]:
        """Временная консистентность"""
        return {}
    
    # Заглушки для методов получения данных
    def _get_face_emotion_at(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Получение эмоции лица в момент времени"""
        return None
    
    def _get_speech_emotion_at(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Получение эмоции речи в момент времени"""
        return None
    
    def _get_transcript_at(self, timestamp: float) -> Optional[str]:
        """Получение транскрипции в момент времени"""
        return None
    
    def _get_criticality_at(self, timestamp: float) -> float:
        """Получение уровня критичности в момент времени"""
        return 0.0
    
    def _get_annotations_at(self, timestamp: float) -> List[str]:
        """Получение аннотаций в момент времени"""
        return []
    
    def _get_confidence_scores_at(self, timestamp: float) -> Dict[str, float]:
        """Получение доверительных оценок в момент времени"""
        return {}
    
    def _get_statistical_markers_at(self, timestamp: float) -> Dict[str, Any]:
        """Получение статистических маркеров в момент времени"""
        return {}
    
    # Заглушки для SQL методов
    def _create_sql_tables(self, conn: sqlite3.Connection):
        """Создание SQL таблиц"""
        pass
    
    def _insert_timeline_data(self, conn: sqlite3.Connection):
        """Вставка данных временной шкалы"""
        pass
    
    def _insert_critical_moments(self, conn: sqlite3.Connection):
        """Вставка критических моментов"""
        pass
    
    def _insert_statistics(self, conn: sqlite3.Connection):
        """Вставка статистических данных"""
        pass
    
    # Заглушки для визуализации
    def _create_emotion_timeline_plot(self, save_dir: Path) -> str:
        """Создание графика временной шкалы эмоций"""
        return ""
    
    def _create_correlation_matrix_plot(self, save_dir: Path) -> str:
        """Создание графика корреляционной матрицы"""
        return ""
    
    def _create_critical_moments_plot(self, save_dir: Path) -> str:
        """Создание графика критических моментов"""
        return ""
    
    def _create_statistical_plots(self, save_dir: Path) -> str:
        """Создание статистических графиков"""
        return ""
    
    def _create_interactive_timeline(self, save_dir: Path) -> str:
        """Создание интерактивной временной шкалы"""
        return ""
    # Полная реализация методов детекции критических моментов
    def _detect_rapid_emotion_changes(self) -> List[Dict[str, Any]]:
        """Детекция резких изменений эмоций"""
        critical_moments = []
        
        try:
            if not self.synchronized_data or 'sources' not in self.synchronized_data:
                return critical_moments
            
            # Анализ видео-эмоций
            video_emotions = self.synchronized_data['sources'].get('video_emotions', {})
            if video_emotions.get('values'):
                video_values = np.array(video_emotions['values'])
                video_values = video_values[~np.isnan(video_values)]  # Удаляем NaN
                
                if len(video_values) > 3:
                    # Вычисляем градиент (скорость изменения)
                    gradient = np.gradient(video_values)
                    
                    # Находим резкие изменения
                    threshold = self.emotion_change_threshold
                    rapid_changes = np.where(np.abs(gradient) > threshold)[0]
                    
                    for idx in rapid_changes:
                        if idx < len(self.synchronized_data['time_grid']):
                            timestamp = self.synchronized_data['time_grid'][idx]
                            
                            critical_moments.append({
                                'type': 'rapid_emotion_change',
                                'timestamp': float(timestamp),
                                'importance': min(float(np.abs(gradient[idx])), 1.0),
                                'source': 'video_emotions',
                                'description': f'Резкое изменение эмоции: {gradient[idx]:.3f}',
                                'metadata': {
                                    'gradient_value': float(gradient[idx]),
                                    'emotion_before': float(video_values[max(0, idx-1)]),
                                    'emotion_after': float(video_values[min(len(video_values)-1, idx+1)])
                                }
                            })
            
            # Аналогично для аудио-эмоций
            audio_emotions = self.synchronized_data['sources'].get('audio_emotions', {})
            if audio_emotions.get('values'):
                audio_values = np.array(audio_emotions['values'])
                audio_values = audio_values[~np.isnan(audio_values)]
                
                if len(audio_values) > 3:
                    gradient = np.gradient(audio_values)
                    rapid_changes = np.where(np.abs(gradient) > self.emotion_change_threshold)[0]
                    
                    for idx in rapid_changes:
                        if idx < len(self.synchronized_data['time_grid']):
                            timestamp = self.synchronized_data['time_grid'][idx]
                            
                            critical_moments.append({
                                'type': 'rapid_emotion_change',
                                'timestamp': float(timestamp),
                                'importance': min(float(np.abs(gradient[idx])), 1.0),
                                'source': 'audio_emotions',
                                'description': f'Резкое изменение эмоции в речи: {gradient[idx]:.3f}',
                                'metadata': {
                                    'gradient_value': float(gradient[idx]),
                                    'emotion_before': float(audio_values[max(0, idx-1)]),
                                    'emotion_after': float(audio_values[min(len(audio_values)-1, idx+1)])
                                }
                            })
            
        except Exception as e:
            self.logger.error(f"Failed to detect rapid emotion changes: {e}")
        
        return critical_moments
    
    def _detect_modality_mismatches(self) -> List[Dict[str, Any]]:
        """Детекция несоответствий между модальностями"""
        critical_moments = []
        
        try:
            if not self.synchronized_data or 'sources' not in self.synchronized_data:
                return critical_moments
            
            video_emotions = self.synchronized_data['sources'].get('video_emotions', {})
            audio_emotions = self.synchronized_data['sources'].get('audio_emotions', {})
            
            if not video_emotions.get('values') or not audio_emotions.get('values'):
                return critical_moments
            
            video_values = np.array(video_emotions['values'])
            audio_values = np.array(audio_emotions['values'])
            
            # Убираем NaN значения
            valid_indices = ~(np.isnan(video_values) | np.isnan(audio_values))
            video_clean = video_values[valid_indices]
            audio_clean = audio_values[valid_indices]
            time_grid_clean = self.synchronized_data['time_grid'][valid_indices]
            
            if len(video_clean) < 2 or len(audio_clean) < 2:
                return critical_moments
            
            # Вычисляем разность между модальностями
            difference = np.abs(video_clean - audio_clean)
            
            # Находим значительные несоответствия
            threshold = self.mismatch_threshold
            mismatches = np.where(difference > threshold)[0]
            
            for idx in mismatches:
                timestamp = time_grid_clean[idx]
                
                # Определяем тип несоответствия
                video_val = video_clean[idx]
                audio_val = audio_clean[idx]
                
                mismatch_type = "противоположные_эмоции" if video_val * audio_val < 0 else "разная_интенсивность"
                
                critical_moments.append({
                    'type': 'modality_mismatch',
                    'timestamp': float(timestamp),
                    'importance': min(float(difference[idx]), 1.0),
                    'source': 'video_audio_comparison',
                    'description': f'Несоответствие между видео и аудио: {mismatch_type}',
                    'metadata': {
                        'video_emotion': float(video_val),
                        'audio_emotion': float(audio_val),
                        'difference': float(difference[idx]),
                        'mismatch_type': mismatch_type
                    }
                })
            
        except Exception as e:
            self.logger.error(f"Failed to detect modality mismatches: {e}")
        
        return critical_moments
    
    def _detect_speech_anomalies(self) -> List[Dict[str, Any]]:
        """Детекция аномалий в речи"""
        critical_moments = []
        
        try:
            if not self.synchronized_data or 'sources' not in self.synchronized_data:
                return critical_moments
            
            speech_data = self.synchronized_data['sources'].get('speech_transcript', {})
            if not speech_data.get('values'):
                return critical_moments
            
            speech_values = np.array(speech_data['values'])
            time_grid = self.synchronized_data['time_grid']
            
            # Находим паузы в речи (значения 0)
            speech_binary = speech_values > 0.5  # Преобразуем в бинарный вид
            
            # Находим начало и конец пауз
            pause_starts = []
            pause_ends = []
            in_pause = False
            
            for i, has_speech in enumerate(speech_binary):
                if not has_speech and not in_pause:  # Начало паузы
                    pause_starts.append(i)
                    in_pause = True
                elif has_speech and in_pause:  # Конец паузы
                    pause_ends.append(i)
                    in_pause = False
            
            # Обрабатываем случай, когда запись заканчивается паузой
            if in_pause:
                pause_ends.append(len(speech_binary) - 1)
            
            # Анализируем длительные паузы
            threshold_duration = self.speech_pause_threshold
            
            for start, end in zip(pause_starts, pause_ends):
                if start < len(time_grid) and end < len(time_grid):
                    pause_duration = time_grid[end] - time_grid[start]
                    
                    if pause_duration > threshold_duration:
                        timestamp = time_grid[start]
                        
                        critical_moments.append({
                            'type': 'long_speech_pause',
                            'timestamp': float(timestamp),
                            'importance': min(float(pause_duration / 10.0), 1.0),  # Нормализация
                            'source': 'speech_transcript',
                            'description': f'Длительная пауза в речи: {pause_duration:.2f} сек',
                            'metadata': {
                                'pause_duration': float(pause_duration),
                                'pause_start': float(time_grid[start]),
                                'pause_end': float(time_grid[end])
                            }
                        })
            
            # Анализ частоты речи (слишком быстрая/медленная речь)
            if len(speech_binary) > 10:
                # Вычисляем скользящее окно активности речи
                window_size = min(50, len(speech_binary) // 4)  # Адаптивный размер окна
                speech_activity = np.convolve(speech_binary.astype(float), 
                                            np.ones(window_size)/window_size, mode='same')
                
                # Находим аномалии в активности речи
                mean_activity = np.mean(speech_activity)
                std_activity = np.std(speech_activity)
                
                if std_activity > 0:
                    z_scores = np.abs((speech_activity - mean_activity) / std_activity)
                    anomalies = np.where(z_scores > 2.0)[0]  # 2 стандартных отклонения
                    
                    for idx in anomalies:
                        if idx < len(time_grid):
                            timestamp = time_grid[idx]
                            
                            activity_type = "избыточная_активность" if speech_activity[idx] > mean_activity else "низкая_активность"
                            
                            critical_moments.append({
                                'type': 'speech_activity_anomaly',
                                'timestamp': float(timestamp),
                                'importance': min(float(z_scores[idx] / 3.0), 1.0),
                                'source': 'speech_transcript',
                                'description': f'Аномалия речевой активности: {activity_type}',
                                'metadata': {
                                    'activity_level': float(speech_activity[idx]),
                                    'mean_activity': float(mean_activity),
                                    'z_score': float(z_scores[idx]),
                                    'anomaly_type': activity_type
                                }
                            })
            
        except Exception as e:
            self.logger.error(f"Failed to detect speech anomalies: {e}")
        
        return critical_moments
    
    def _detect_micro_expressions(self) -> List[Dict[str, Any]]:
        """Детекция микровыражений"""
        critical_moments = []
        
        try:
            if not self.synchronized_data or 'sources' not in self.synchronized_data:
                return critical_moments
            
            video_emotions = self.synchronized_data['sources'].get('video_emotions', {})
            if not video_emotions.get('values'):
                return critical_moments
            
            video_values = np.array(video_emotions['values'])
            time_grid = self.synchronized_data['time_grid']
            
            # Убираем NaN значения
            valid_indices = ~np.isnan(video_values)
            video_clean = video_values[valid_indices]
            time_clean = time_grid[valid_indices]
            
            if len(video_clean) < 5:
                return critical_moments
            
            # Сглаживаем сигнал для выявления базовой линии
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(video_clean, sigma=2.0)
            
            # Находим отклонения от базовой линии
            deviations = video_clean - smoothed
            
            # Ищем краткосрочные всплески
            abs_deviations = np.abs(deviations)
            threshold = np.std(abs_deviations) * 2.0  # 2 стандартных отклонения
            
            # Находим пики
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(abs_deviations, 
                                         height=threshold,
                                         width=(1, 10))  # Ограничиваем ширину пиков
            
            for peak_idx in peaks:
                if peak_idx < len(time_clean):
                    timestamp = time_clean[peak_idx]
                    
                    # Проверяем длительность микровыражения
                    peak_start = peak_idx
                    peak_end = peak_idx
                    
                    # Находим границы пика
                    while (peak_start > 0 and 
                           abs_deviations[peak_start-1] > threshold * 0.5):
                        peak_start -= 1
                    
                    while (peak_end < len(abs_deviations) - 1 and 
                           abs_deviations[peak_end+1] > threshold * 0.5):
                        peak_end += 1
                    
                    duration = time_clean[peak_end] - time_clean[peak_start]
                    
                    # Микровыражения обычно длятся менее 0.5 секунды
                    if duration <= self.micro_expression_duration:
                        emotion_direction = "положительная" if deviations[peak_idx] > 0 else "отрицательная"
                        
                        critical_moments.append({
                            'type': 'micro_expression',
                            'timestamp': float(timestamp),
                            'importance': min(float(abs_deviations[peak_idx] / threshold), 1.0),
                            'source': 'video_emotions',
                            'description': f'Микровыражение ({emotion_direction}) длительностью {duration:.3f} сек',
                            'metadata': {
                                'duration': float(duration),
                                'intensity': float(abs_deviations[peak_idx]),
                                'direction': emotion_direction,
                                'baseline_emotion': float(smoothed[peak_idx]),
                                'peak_emotion': float(video_clean[peak_idx])
                            }
                        })
            
        except Exception as e:
            self.logger.error(f"Failed to detect micro expressions: {e}")
        
        return critical_moments
    
    def _detect_statistical_outliers(self) -> List[Dict[str, Any]]:
        """Детекция статистических выбросов"""
        critical_moments = []
        
        try:
            if not self.synchronized_data or 'sources' not in self.synchronized_data:
                return critical_moments
            
            # Анализируем каждый источник данных
            for source_name, source_data in self.synchronized_data['sources'].items():
                if not source_data.get('values'):
                    continue
                
                values = np.array(source_data['values'])
                confidences = np.array(source_data.get('confidence', [1.0] * len(values)))
                
                # Убираем NaN значения
                valid_indices = ~np.isnan(values)
                values_clean = values[valid_indices]
                confidences_clean = confidences[valid_indices]
                time_indices = np.arange(len(values))[valid_indices]
                
                if len(values_clean) < 10:
                    continue
                
                # Метод IQR для выявления выбросов
                Q1 = np.percentile(values_clean, 25)
                Q3 = np.percentile(values_clean, 75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = (values_clean < lower_bound) | (values_clean > upper_bound)
                    outlier_indices = time_indices[outliers]
                    
                    for idx in outlier_indices:
                        if idx < len(self.synchronized_data['time_grid']):
                            timestamp = self.synchronized_data['time_grid'][idx]
                            
                            outlier_value = values[idx]
                            confidence = confidences[idx] if idx < len(confidences) else 1.0
                            
                            # Определяем тип выброса
                            if outlier_value < lower_bound:
                                outlier_type = "значительно_ниже_нормы"
                                deviation = abs(outlier_value - lower_bound)
                            else:
                                outlier_type = "значительно_выше_нормы"
                                deviation = abs(outlier_value - upper_bound)
                            
                            critical_moments.append({
                                'type': 'statistical_outlier',
                                'timestamp': float(timestamp),
                                'importance': min(float(deviation / IQR) * confidence, 1.0),
                                'source': source_name,
                                'description': f'Статистический выброс в {source_name}: {outlier_type}',
                                'metadata': {
                                    'value': float(outlier_value),
                                    'q1': float(Q1),
                                    'q3': float(Q3),
                                    'iqr': float(IQR),
                                    'lower_bound': float(lower_bound),
                                    'upper_bound': float(upper_bound),
                                    'outlier_type': outlier_type,
                                    'confidence': float(confidence)
                                }
                            })
                
                # Z-score анализ для дополнительной проверки
                if len(values_clean) > 20:
                    mean_val = np.mean(values_clean)
                    std_val = np.std(values_clean)
                    
                    if std_val > 0:
                        z_scores = np.abs((values_clean - mean_val) / std_val)
                        extreme_outliers = z_scores > 3.0  # 3 стандартных отклонения
                        
                        extreme_indices = time_indices[extreme_outliers]
                        
                        for idx in extreme_indices:
                            if idx < len(self.synchronized_data['time_grid']):
                                timestamp = self.synchronized_data['time_grid'][idx]
                                z_score = z_scores[np.where(time_indices == idx)[0][0]]
                                
                                critical_moments.append({
                                    'type': 'extreme_outlier',
                                    'timestamp': float(timestamp),
                                    'importance': min(float(z_score / 4.0), 1.0),
                                    'source': source_name,
                                    'description': f'Экстремальный выброс в {source_name} (Z-score: {z_score:.2f})',
                                    'metadata': {
                                        'value': float(values[idx]),
                                        'z_score': float(z_score),
                                        'mean': float(mean_val),
                                        'std': float(std_val)
                                    }
                                })
            
        except Exception as e:
            self.logger.error(f"Failed to detect statistical outliers: {e}")
        
        return critical_moments

    # Дополнительные вспомогательные методы
    def _quantize_emotions(self, values: np.ndarray) -> np.ndarray:
        """Квантизация эмоций в дискретные состояния"""
        # Квантизация на 5 состояний: очень негативные, негативные, нейтральные, позитивные, очень позитивные
        bins = [-np.inf, -0.6, -0.2, 0.2, 0.6, np.inf]
        return np.digitize(values, bins) - 1  # -1 для индексации с 0
    
    def _build_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """Построение матрицы переходов между состояниями"""
        n_states = 5  # Фиксированное количество эмоциональных состояний
        matrix = np.zeros((n_states, n_states))
        
        for i in range(len(states) - 1):
            current = int(states[i])
            next_state = int(states[i + 1])
            if 0 <= current < n_states and 0 <= next_state < n_states:
                matrix[current, next_state] += 1
        
        # Нормализация по строкам
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Избегаем деления на ноль
        return matrix / row_sums
    
    def _get_emotion_state_names(self) -> List[str]:
        """Получение названий эмоциональных состояний"""
        return ['очень_негативное', 'негативное', 'нейтральное', 'позитивное', 'очень_позитивное']
    
    def _find_dominant_transitions(self, matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Поиск доминирующих переходов в матрице"""
        state_names = self._get_emotion_state_names()
        dominant = []
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0.3:  # Порог для значимого перехода
                    dominant.append({
                        'from_state': state_names[i],
                        'to_state': state_names[j],
                        'probability': float(matrix[i, j])
                    })
        
        return sorted(dominant, key=lambda x: x['probability'], reverse=True)
    
    def _interpret_stress_level(self, level: float) -> str:
        """Интерпретация уровня стресса"""
        if level < 0.3:
            return 'низкий_уровень_стресса'
        elif level < 0.6:
            return 'умеренный_уровень_стресса'
        elif level < 0.8:
            return 'повышенный_уровень_стресса'
        else:
            return 'высокий_уровень_стресса'
    
    # Полная реализация методов получения данных по временным меткам
    def _get_face_emotion_at(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Получение эмоции лица в конкретный момент времени"""
        try:
            if not self.timeline:
                return None
            
            # Поиск ближайшей временной точки
            closest_point = min(self.timeline, key=lambda x: abs(x['timestamp'] - timestamp))
            
            video_data = closest_point.get('sources', {}).get('video_emotions', {})
            if video_data and video_data.get('value') is not None:
                emotion_value = video_data['value']
                confidence = video_data.get('confidence', 0.0)
                
                # Конвертируем численное значение в название эмоции
                emotion_name = self._numeric_to_emotion_name(emotion_value)
                
                return {
                    'emotion': emotion_name,
                    'value': float(emotion_value),
                    'confidence': float(confidence),
                    'source': 'video_emotions'
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to get face emotion at {timestamp}: {e}")
            return None
    
    def _get_speech_emotion_at(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Получение эмоции речи в конкретный момент времени"""
        try:
            if not self.timeline:
                return None
            
            closest_point = min(self.timeline, key=lambda x: abs(x['timestamp'] - timestamp))
            
            audio_data = closest_point.get('sources', {}).get('audio_emotions', {})
            if audio_data and audio_data.get('value') is not None:
                emotion_value = audio_data['value']
                confidence = audio_data.get('confidence', 0.0)
                
                emotion_name = self._numeric_to_emotion_name(emotion_value)
                
                return {
                    'emotion': emotion_name,
                    'value': float(emotion_value),
                    'confidence': float(confidence),
                    'source': 'audio_emotions'
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to get speech emotion at {timestamp}: {e}")
            return None
    
    def _get_transcript_at(self, timestamp: float) -> Optional[str]:
        """Получение транскрипции в конкретный момент времени"""
        try:
            if not self.timeline:
                return None
            
            # Ищем текст в окне ±1 секунда от заданного времени
            window = 1.0
            relevant_points = [
                point for point in self.timeline 
                if abs(point['timestamp'] - timestamp) <= window
            ]
            
            if not relevant_points:
                return None
            
            # Собираем весь текст из релевантных точек
            transcript_parts = []
            for point in relevant_points:
                speech_data = point.get('sources', {}).get('speech_transcript', {})
                if speech_data and speech_data.get('value', 0) > 0.5:  # Есть речевая активность
                    # В реальной реализации здесь был бы доступ к сегментам транскрипции
                    transcript_parts.append(f"[речь в {point['timestamp']:.1f}с]")
            
            return " ".join(transcript_parts) if transcript_parts else None
            
        except Exception as e:
            self.logger.warning(f"Failed to get transcript at {timestamp}: {e}")
            return None
    
    def _get_criticality_at(self, timestamp: float) -> float:
        """Получение уровня критичности в конкретный момент времени"""
        try:
            if not self.critical_moments:
                return 0.0
            
            # Находим критические моменты в окне ±2 секунды
            window = 2.0
            nearby_critical = [
                cm for cm in self.critical_moments 
                if abs(cm.get('timestamp', 0) - timestamp) <= window
            ]
            
            if not nearby_critical:
                return 0.0
            
            # Возвращаем максимальную важность среди близких критических моментов
            max_importance = max(cm.get('importance', 0) for cm in nearby_critical)
            return float(max_importance)
            
        except Exception as e:
            self.logger.warning(f"Failed to get criticality at {timestamp}: {e}")
            return 0.0
    
    def _get_annotations_at(self, timestamp: float) -> List[str]:
        """Получение аннотаций в конкретный момент времени"""
        try:
            annotations = []
            
            # Добавляем критические события
            if self.critical_moments:
                window = 1.0
                nearby_critical = [
                    cm for cm in self.critical_moments 
                    if abs(cm.get('timestamp', 0) - timestamp) <= window
                ]
                
                for cm in nearby_critical:
                    event_type = cm.get('type', 'unknown')
                    description = cm.get('description', f'Критическое событие: {event_type}')
                    annotations.append(description)
            
            # Добавляем эмоциональные аннотации
            face_emotion = self._get_face_emotion_at(timestamp)
            if face_emotion:
                annotations.append(f"Эмоция лица: {face_emotion['emotion']}")
            
            speech_emotion = self._get_speech_emotion_at(timestamp)
            if speech_emotion:
                annotations.append(f"Эмоция речи: {speech_emotion['emotion']}")
            
            return annotations
            
        except Exception as e:
            self.logger.warning(f"Failed to get annotations at {timestamp}: {e}")
            return []
    
    def _get_confidence_scores_at(self, timestamp: float) -> Dict[str, float]:
        """Получение доверительных оценок в конкретный момент времени"""
        try:
            if not self.timeline:
                return {}
            
            closest_point = min(self.timeline, key=lambda x: abs(x['timestamp'] - timestamp))
            confidence_scores = {}
            
            for source_name, source_data in closest_point.get('sources', {}).items():
                confidence = source_data.get('confidence', 0.0)
                confidence_scores[source_name] = float(confidence)
            
            return confidence_scores
            
        except Exception as e:
            self.logger.warning(f"Failed to get confidence scores at {timestamp}: {e}")
            return {}
    
    def _get_statistical_markers_at(self, timestamp: float) -> Dict[str, Any]:
        """Получение статистических маркеров в конкретный момент времени"""
        try:
            markers = {}
            
            # Добавляем информацию о критичности
            criticality = self._get_criticality_at(timestamp)
            markers['criticality'] = criticality
            
            # Добавляем информацию о стабильности эмоций в этой области
            if self.timeline:
                window = 5.0  # Окно 5 секунд для анализа стабильности
                window_points = [
                    point for point in self.timeline 
                    if abs(point['timestamp'] - timestamp) <= window
                ]
                
                if len(window_points) > 1:
                    # Анализ изменчивости эмоций в окне
                    video_values = []
                    for point in window_points:
                        video_data = point.get('sources', {}).get('video_emotions', {})
                        if video_data and video_data.get('value') is not None:
                            video_values.append(video_data['value'])
                    
                    if video_values:
                        local_stability = 1.0 - np.std(video_values) / (np.abs(np.mean(video_values)) + 1e-8)
                        markers['local_stability'] = max(0.0, min(1.0, local_stability))
            
            return markers
            
        except Exception as e:
            self.logger.warning(f"Failed to get statistical markers at {timestamp}: {e}")
            return {}
    
    def _numeric_to_emotion_name(self, value: float) -> str:
        """Конвертация численного значения эмоции в название"""
        if value < -0.8:
            return 'сильная_злость'
        elif value < -0.6:
            return 'злость'
        elif value < -0.4:
            return 'раздражение'
        elif value < -0.2:
            return 'печаль'
        elif value < -0.1:
            return 'легкая_печаль'
        elif value < 0.1:
            return 'нейтральность'
        elif value < 0.2:
            return 'легкое_удовлетворение'
        elif value < 0.4:
            return 'удовлетворение'
        elif value < 0.6:
            return 'радость'
        elif value < 0.8:
            return 'сильная_радость'
        else:
            return 'восторг'
