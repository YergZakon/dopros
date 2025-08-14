"""
Transition Metrics Calculator

Расчет метрик и индикаторов переходов эмоций
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
from .transition_detector import EmotionTransition


class TransitionMetricsCalculator:
    """
    Калькулятор метрик переходов эмоций
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Психологические индикаторы
        self.psychological_indicators = {
            "deception_likelihood": {
                "patterns": [
                    ("спокойствие", "страх"),
                    ("злость", "спокойствие"), 
                    ("страх", "нейтральность"),
                    ("спокойствие", "злость")
                ],
                "weight": 0.8
            },
            "stress_level": {
                "patterns": [
                    ("нейтральность", "страх"),
                    ("спокойствие", "напряжение"),
                    ("нейтральность", "тревога"),
                    ("счастье", "грусть")
                ],
                "weight": 0.7
            },
            "emotional_control": {
                "patterns": [
                    ("злость", "нейтральность"),
                    ("страх", "спокойствие"),
                    ("напряжение", "спокойствие"),
                    ("тревога", "нейтральность")
                ],
                "weight": 0.6
            },
            "emotional_instability": {
                "patterns": [
                    ("счастье", "злость"),
                    ("грусть", "счастье"),
                    ("злость", "счастье"),
                    ("спокойствие", "тревога")
                ],
                "weight": 0.9
            }
        }
    
    def calculate_comprehensive_metrics(self, 
                                       video_transitions: List[EmotionTransition],
                                       speech_transitions: List[EmotionTransition]) -> Dict[str, Any]:
        """
        Расчет комплексных метрик переходов
        
        Args:
            video_transitions: Переходы в видео
            speech_transitions: Переходы в речи
            
        Returns:
            Словарь с метриками
        """
        try:
            # Базовые метрики
            video_metrics = self._calculate_basic_metrics(video_transitions, "video")
            speech_metrics = self._calculate_basic_metrics(speech_transitions, "speech")
            
            # Психологические индикаторы
            video_psychology = self._calculate_psychological_indicators(video_transitions)
            speech_psychology = self._calculate_psychological_indicators(speech_transitions)
            
            # Корреляционный анализ
            correlation_metrics = self._calculate_correlation_metrics(video_transitions, speech_transitions)
            
            # Временной анализ
            temporal_metrics = self._calculate_temporal_metrics(video_transitions, speech_transitions)
            
            # Матрица переходов
            transition_matrices = {
                'video': self._create_transition_matrix(video_transitions),
                'speech': self._create_transition_matrix(speech_transitions)
            }
            
            # Индекс эмоциональной нестабильности
            instability_index = self._calculate_instability_index(video_transitions, speech_transitions)
            
            return {
                'basic_metrics': {
                    'video': video_metrics,
                    'speech': speech_metrics
                },
                'psychological_indicators': {
                    'video': video_psychology,
                    'speech': speech_psychology,
                    'combined': self._combine_psychological_indicators(video_psychology, speech_psychology)
                },
                'correlation_metrics': correlation_metrics,
                'temporal_metrics': temporal_metrics,
                'transition_matrices': transition_matrices,
                'instability_index': instability_index,
                'summary': self._create_summary(video_metrics, speech_metrics, instability_index)
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета метрик: {e}")
            return self._create_fallback_metrics()
    
    def _calculate_basic_metrics(self, transitions: List[EmotionTransition], modality: str) -> Dict[str, Any]:
        """Расчет базовых метрик переходов"""
        if not transitions:
            return self._empty_metrics()
        
        # Базовая статистика
        total_count = len(transitions)
        critical_count = sum(1 for t in transitions if t.is_critical)
        
        # Средние значения
        avg_severity = np.mean([t.severity for t in transitions])
        avg_duration = np.mean([t.duration for t in transitions])
        avg_confidence_before = np.mean([t.confidence_before for t in transitions])
        avg_confidence_after = np.mean([t.confidence_after for t in transitions])
        
        # Временные характеристики
        time_span = transitions[-1].timestamp - transitions[0].timestamp if len(transitions) > 1 else 0
        transition_rate = (total_count / time_span * 60) if time_span > 0 else 0
        
        # Распределение по скорости
        speed_distribution = Counter([t.transition_speed for t in transitions])
        
        # Распределение по типам
        type_distribution = Counter([t.transition_type for t in transitions])
        
        # Топ переходы
        transition_pairs = Counter([(t.from_emotion, t.to_emotion) for t in transitions])
        top_transitions = transition_pairs.most_common(5)
        
        return {
            'total_transitions': total_count,
            'critical_transitions': critical_count,
            'critical_ratio': critical_count / total_count,
            'avg_severity': round(avg_severity, 2),
            'avg_duration': round(avg_duration, 2),
            'avg_confidence_before': round(avg_confidence_before, 3),
            'avg_confidence_after': round(avg_confidence_after, 3),
            'confidence_change': round(avg_confidence_after - avg_confidence_before, 3),
            'transition_rate': round(transition_rate, 2),
            'time_span': round(time_span, 1),
            'speed_distribution': dict(speed_distribution),
            'type_distribution': dict(type_distribution),
            'top_transitions': [f"{t[0][0]}→{t[0][1]} ({t[1]}x)" for t in top_transitions],
            'modality': modality
        }
    
    def _calculate_psychological_indicators(self, transitions: List[EmotionTransition]) -> Dict[str, float]:
        """Расчет психологических индикаторов"""
        if not transitions:
            return {indicator: 0.0 for indicator in self.psychological_indicators.keys()}
        
        indicators = {}
        transition_pairs = [(t.from_emotion, t.to_emotion) for t in transitions]
        
        for indicator_name, indicator_config in self.psychological_indicators.items():
            # Подсчитываем совпадения с паттернами
            pattern_matches = 0
            for pattern in indicator_config['patterns']:
                pattern_matches += transition_pairs.count(pattern)
            
            # Нормализуем по количеству переходов и весу
            if transition_pairs:
                raw_score = pattern_matches / len(transition_pairs)
                weighted_score = raw_score * indicator_config['weight']
                indicators[indicator_name] = min(1.0, weighted_score)  # Ограничиваем 1.0
            else:
                indicators[indicator_name] = 0.0
        
        return indicators
    
    def _calculate_correlation_metrics(self, 
                                     video_transitions: List[EmotionTransition],
                                     speech_transitions: List[EmotionTransition]) -> Dict[str, Any]:
        """Расчет корреляционных метрик между видео и речью"""
        if not video_transitions or not speech_transitions:
            return {
                'temporal_correlation': 0.0,
                'severity_correlation': 0.0,
                'type_similarity': 0.0,
                'synchronized_transitions': 0,
                'contradictory_transitions': 0
            }
        
        synchronized_count = 0
        contradictory_count = 0
        severity_pairs = []
        
        # Анализ синхронизации переходов (в пределах 5 секунд)
        for video_t in video_transitions:
            closest_speech = self._find_closest_transition(video_t, speech_transitions, max_distance=5.0)
            
            if closest_speech:
                # Проверяем синхронизацию
                if self._are_transitions_similar(video_t, closest_speech):
                    synchronized_count += 1
                elif self._are_transitions_contradictory(video_t, closest_speech):
                    contradictory_count += 1
                
                # Собираем пары для корреляции серьезности
                severity_pairs.append((video_t.severity, closest_speech.severity))
        
        # Корреляция серьезности
        if len(severity_pairs) > 1:
            video_severities, speech_severities = zip(*severity_pairs)
            severity_correlation = np.corrcoef(video_severities, speech_severities)[0, 1]
            if np.isnan(severity_correlation):
                severity_correlation = 0.0
        else:
            severity_correlation = 0.0
        
        # Временная корреляция
        total_comparisons = len([v for v in video_transitions 
                               if self._find_closest_transition(v, speech_transitions, 5.0)])
        temporal_correlation = synchronized_count / total_comparisons if total_comparisons > 0 else 0.0
        
        # Схожесть типов переходов
        video_types = set([t.transition_type for t in video_transitions])
        speech_types = set([t.transition_type for t in speech_transitions])
        type_similarity = len(video_types & speech_types) / len(video_types | speech_types) if video_types | speech_types else 0.0
        
        return {
            'temporal_correlation': round(temporal_correlation, 3),
            'severity_correlation': round(severity_correlation, 3),
            'type_similarity': round(type_similarity, 3),
            'synchronized_transitions': synchronized_count,
            'contradictory_transitions': contradictory_count,
            'total_comparisons': total_comparisons,
            'sync_ratio': round(synchronized_count / total_comparisons, 3) if total_comparisons > 0 else 0.0
        }
    
    def _calculate_temporal_metrics(self,
                                   video_transitions: List[EmotionTransition],
                                   speech_transitions: List[EmotionTransition]) -> Dict[str, Any]:
        """Расчет временных характеристик переходов"""
        all_transitions = video_transitions + speech_transitions
        
        if not all_transitions:
            return {'total_analysis_time': 0, 'peak_activity_periods': []}
        
        # Общее время анализа
        min_time = min(t.timestamp for t in all_transitions)
        max_time = max(t.timestamp for t in all_transitions)
        total_time = max_time - min_time
        
        # Поиск пиков активности (периодов с высокой частотой переходов)
        time_windows = self._create_time_windows(all_transitions, window_size=30)  # 30-секундные окна
        peak_periods = []
        
        if time_windows:
            avg_transitions_per_window = np.mean([w['transition_count'] for w in time_windows])
            threshold = avg_transitions_per_window + np.std([w['transition_count'] for w in time_windows])
            
            peak_periods = [
                {
                    'start_time': w['start_time'],
                    'end_time': w['end_time'],
                    'transition_count': w['transition_count'],
                    'avg_severity': w['avg_severity']
                }
                for w in time_windows if w['transition_count'] > threshold
            ]
        
        return {
            'total_analysis_time': round(total_time, 1),
            'peak_activity_periods': peak_periods,
            'avg_transitions_per_minute': round(len(all_transitions) / (total_time / 60), 2) if total_time > 0 else 0,
            'time_windows_analyzed': len(time_windows)
        }
    
    def _create_transition_matrix(self, transitions: List[EmotionTransition]) -> Dict[str, Any]:
        """Создание матрицы переходов"""
        if not transitions:
            return {'matrix': {}, 'emotions': []}
        
        # Получаем все уникальные эмоции
        emotions = sorted(set([t.from_emotion for t in transitions] + [t.to_emotion for t in transitions]))
        
        # Создаем матрицу
        matrix = defaultdict(lambda: defaultdict(int))
        for transition in transitions:
            matrix[transition.from_emotion][transition.to_emotion] += 1
        
        # Преобразуем в обычные словари для JSON
        matrix_dict = {}
        for from_emotion in emotions:
            matrix_dict[from_emotion] = {}
            for to_emotion in emotions:
                matrix_dict[from_emotion][to_emotion] = matrix[from_emotion][to_emotion]
        
        return {
            'matrix': matrix_dict,
            'emotions': emotions,
            'total_transitions': len(transitions)
        }
    
    def _calculate_instability_index(self,
                                   video_transitions: List[EmotionTransition],
                                   speech_transitions: List[EmotionTransition]) -> Dict[str, float]:
        """Расчет индекса эмоциональной нестабильности"""
        
        def calculate_modality_instability(transitions):
            if not transitions:
                return 0.0
            
            # Факторы нестабильности
            critical_ratio = sum(1 for t in transitions if t.is_critical) / len(transitions)
            avg_severity = np.mean([t.severity for t in transitions])
            speed_factor = sum(1 for t in transitions if t.transition_speed in ['мгновенный', 'быстрый']) / len(transitions)
            
            # Нормализуем и комбинируем
            instability = (critical_ratio * 0.4 + (avg_severity / 10) * 0.4 + speed_factor * 0.2)
            return min(1.0, instability)
        
        video_instability = calculate_modality_instability(video_transitions)
        speech_instability = calculate_modality_instability(speech_transitions)
        
        # Общий индекс как взвешенное среднее
        combined_instability = (video_instability * 0.6 + speech_instability * 0.4)
        
        return {
            'video_instability': round(video_instability, 3),
            'speech_instability': round(speech_instability, 3),
            'combined_instability': round(combined_instability, 3),
            'interpretation': self._interpret_instability(combined_instability)
        }
    
    def _interpret_instability(self, instability_score: float) -> str:
        """Интерпретация индекса нестабильности"""
        if instability_score < 0.2:
            return "стабильное эмоциональное состояние"
        elif instability_score < 0.4:
            return "умеренная эмоциональная изменчивость"
        elif instability_score < 0.6:
            return "повышенная эмоциональная нестабильность"
        elif instability_score < 0.8:
            return "высокая эмоциональная нестабильность"
        else:
            return "критическая эмоциональная нестабильность"
    
    # Вспомогательные методы
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Пустые метрики для случая отсутствия данных"""
        return {
            'total_transitions': 0,
            'critical_transitions': 0,
            'critical_ratio': 0.0,
            'avg_severity': 0.0,
            'avg_duration': 0.0,
            'transition_rate': 0.0,
            'time_span': 0.0
        }
    
    def _find_closest_transition(self, target: EmotionTransition, 
                               candidates: List[EmotionTransition],
                               max_distance: float) -> EmotionTransition:
        """Поиск ближайшего перехода по времени"""
        closest = None
        min_distance = float('inf')
        
        for candidate in candidates:
            distance = abs(candidate.timestamp - target.timestamp)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                closest = candidate
        
        return closest
    
    def _are_transitions_similar(self, t1: EmotionTransition, t2: EmotionTransition) -> bool:
        """Проверка схожести переходов"""
        # Схожие если тип перехода одинаковый или серьезность близкая
        return (t1.transition_type == t2.transition_type or 
                abs(t1.severity - t2.severity) <= 2)
    
    def _are_transitions_contradictory(self, t1: EmotionTransition, t2: EmotionTransition) -> bool:
        """Проверка противоречивости переходов"""
        # Противоречивые если один критический, другой нет, или кардинально разные типы
        return (t1.is_critical != t2.is_critical and 
                abs(t1.severity - t2.severity) >= 5)
    
    def _create_time_windows(self, transitions: List[EmotionTransition], window_size: float) -> List[Dict[str, Any]]:
        """Создание временных окон для анализа"""
        if not transitions:
            return []
        
        min_time = min(t.timestamp for t in transitions)
        max_time = max(t.timestamp for t in transitions)
        
        windows = []
        current_time = min_time
        
        while current_time < max_time:
            end_time = min(current_time + window_size, max_time)
            
            # Переходы в данном окне
            window_transitions = [t for t in transitions 
                                if current_time <= t.timestamp < end_time]
            
            if window_transitions:
                windows.append({
                    'start_time': current_time,
                    'end_time': end_time,
                    'transition_count': len(window_transitions),
                    'avg_severity': np.mean([t.severity for t in window_transitions])
                })
            
            current_time = end_time
        
        return windows
    
    def _combine_psychological_indicators(self, video_indicators: Dict[str, float],
                                        speech_indicators: Dict[str, float]) -> Dict[str, float]:
        """Комбинирование психологических индикаторов"""
        combined = {}
        
        for indicator in video_indicators.keys():
            # Взвешенное среднее (видео более важно для эмоций лица)
            combined[indicator] = round(
                video_indicators[indicator] * 0.7 + speech_indicators[indicator] * 0.3, 3
            )
        
        return combined
    
    def _create_summary(self, video_metrics: Dict[str, Any], 
                       speech_metrics: Dict[str, Any], 
                       instability_index: Dict[str, float]) -> Dict[str, Any]:
        """Создание итогового резюме"""
        return {
            'total_transitions': video_metrics['total_transitions'] + speech_metrics['total_transitions'],
            'total_critical': video_metrics['critical_transitions'] + speech_metrics['critical_transitions'],
            'dominant_modality': 'video' if video_metrics['total_transitions'] > speech_metrics['total_transitions'] else 'speech',
            'overall_instability': instability_index['combined_instability'],
            'instability_interpretation': instability_index['interpretation'],
            'analysis_quality': 'высокое' if (video_metrics['total_transitions'] > 5 and 
                                            speech_metrics['total_transitions'] > 3) else 'среднее'
        }
    
    def _create_fallback_metrics(self) -> Dict[str, Any]:
        """Запасные метрики в случае ошибки"""
        return {
            'error': True,
            'basic_metrics': {'video': self._empty_metrics(), 'speech': self._empty_metrics()},
            'psychological_indicators': {'video': {}, 'speech': {}, 'combined': {}},
            'correlation_metrics': {},
            'temporal_metrics': {'total_analysis_time': 0},
            'transition_matrices': {'video': {'matrix': {}, 'emotions': []}, 'speech': {'matrix': {}, 'emotions': []}},
            'instability_index': {'combined_instability': 0.0, 'interpretation': 'данные недоступны'},
            'summary': {'analysis_quality': 'недоступно'}
        }