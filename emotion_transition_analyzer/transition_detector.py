"""
Emotion Transition Detector

Детекция и классификация переходов эмоций во времени
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import timedelta
import numpy as np


@dataclass
class EmotionTransition:
    """Класс для представления перехода эмоций"""
    timestamp: float
    from_emotion: str
    to_emotion: str
    duration: float
    transition_type: str
    severity: int
    confidence_before: float
    confidence_after: float
    modality: str  # 'video' or 'speech'
    
    @property
    def transition_speed(self) -> str:
        """Скорость перехода"""
        if self.duration < 0.5:
            return "мгновенный"
        elif self.duration < 2.0:
            return "быстрый"
        elif self.duration < 5.0:
            return "умеренный"
        else:
            return "медленный"
    
    @property
    def is_critical(self) -> bool:
        """Является ли переход критическим"""
        return self.severity >= 7


class EmotionTransitionDetector:
    """
    Детектор переходов эмоций с психологической классификацией
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация детектора переходов
        
        Args:
            config: Конфигурация детектора
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Параметры детекции
        self.min_duration = self.config.get('min_transition_duration', 0.3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.3)
        self.min_confidence_diff = self.config.get('min_confidence_diff', 0.1)
        
        # Загружаем паттерны критических переходов
        self._load_critical_patterns()
        
        self.logger.info("EmotionTransitionDetector инициализирован")
    
    def _load_critical_patterns(self):
        """Загрузка паттернов критических переходов"""
        self.critical_transitions = {
            # Высокая критичность - признаки лжи/сокрытия
            ("спокойствие", "злость"): {"severity": 9, "type": "агрессивная защита"},
            ("нейтральность", "страх"): {"severity": 8, "type": "внезапная тревога"},
            ("счастье", "грусть"): {"severity": 7, "type": "эмоциональный срыв"},
            ("злость", "спокойствие"): {"severity": 8, "type": "подавление эмоций"},
            ("страх", "нейтральность"): {"severity": 7, "type": "маскировка тревоги"},
            
            # Средняя критичность - подозрительные паттерны  
            ("грусть", "счастье"): {"severity": 6, "type": "неконгруэнтность"},
            ("злость", "счастье"): {"severity": 7, "type": "резкая смена"},
            ("напряжение", "спокойствие"): {"severity": 6, "type": "искусственное успокоение"},
            ("тревога", "нейтральность"): {"severity": 6, "type": "подавление тревоги"},
            
            # Низкая критичность - нормальные паттерны
            ("грусть", "грусть"): {"severity": 2, "type": "стабильная печаль"},
            ("спокойствие", "спокойствие"): {"severity": 1, "type": "уверенность"},
            ("нейтральность", "нейтральность"): {"severity": 1, "type": "стабильность"},
            ("счастье", "счастье"): {"severity": 2, "type": "радость"},
            
            # Эмоциональные нарастания
            ("нейтральность", "напряжение"): {"severity": 5, "type": "нарастание напряжения"},
            ("напряжение", "злость"): {"severity": 6, "type": "эскалация конфликта"},
            ("спокойствие", "тревога"): {"severity": 7, "type": "внутреннее беспокойство"}
        }
        
        # Синонимы эмоций для нормализации
        self.emotion_synonyms = {
            'нейтральность': 'нейтральность',
            'нейтральный': 'нейтральность', 
            'счастье': 'счастье',
            'радость': 'счастье',
            'грусть': 'грусть',
            'печаль': 'грусть',
            'злость': 'злость',
            'гнев': 'злость',
            'раздражение': 'злость',
            'страх': 'страх',
            'испуг': 'страх',
            'удивление': 'удивление',
            'отвращение': 'отвращение',
            'презрение': 'презрение',
            'спокойствие': 'спокойствие',
            'возбуждение': 'возбуждение',
            'фрустрация': 'злость',  # группируем с злостью
            'замешательство': 'тревога',
            'напряжение': 'напряжение',
            'тревога': 'тревога'
        }
    
    def normalize_emotion(self, emotion: str) -> str:
        """Нормализация названия эмоции"""
        return self.emotion_synonyms.get(emotion.lower(), emotion.lower())
    
    def detect_transitions(self, emotions_timeline: List[Dict[str, Any]], modality: str = "video") -> List[EmotionTransition]:
        """
        Детекция всех переходов эмоций в временном ряду
        
        Args:
            emotions_timeline: Список эмоций с временными метками
            modality: Тип данных ('video' или 'speech')
            
        Returns:
            Список обнаруженных переходов
        """
        if not emotions_timeline or len(emotions_timeline) < 2:
            return []
        
        transitions = []
        
        try:
            for i in range(1, len(emotions_timeline)):
                prev_emotion = emotions_timeline[i-1]
                curr_emotion = emotions_timeline[i]
                
                # Извлекаем данные из эмоций
                prev_timestamp = self._get_timestamp(prev_emotion)
                curr_timestamp = self._get_timestamp(curr_emotion)
                
                prev_emotion_name = self.normalize_emotion(prev_emotion.get('emotion', 'нейтральность'))
                curr_emotion_name = self.normalize_emotion(curr_emotion.get('emotion', 'нейтральность'))
                
                prev_confidence = prev_emotion.get('confidence', 0.5)
                curr_confidence = curr_emotion.get('confidence', 0.5)
                
                # Пропускаем если уверенность слишком низкая
                if prev_confidence < self.confidence_threshold or curr_confidence < self.confidence_threshold:
                    continue
                
                # Проверяем, есть ли переход
                if prev_emotion_name != curr_emotion_name:
                    duration = curr_timestamp - prev_timestamp
                    
                    # Пропускаем слишком короткие переходы
                    if duration < self.min_duration:
                        continue
                    
                    # Классифицируем переход
                    transition_info = self._classify_transition(prev_emotion_name, curr_emotion_name)
                    
                    transition = EmotionTransition(
                        timestamp=curr_timestamp,
                        from_emotion=prev_emotion_name,
                        to_emotion=curr_emotion_name,
                        duration=duration,
                        transition_type=transition_info['type'],
                        severity=transition_info['severity'],
                        confidence_before=prev_confidence,
                        confidence_after=curr_confidence,
                        modality=modality
                    )
                    
                    transitions.append(transition)
            
            self.logger.info(f"Обнаружено {len(transitions)} переходов в {modality}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при детекции переходов: {e}")
        
        return transitions
    
    def _get_timestamp(self, emotion_data: Dict[str, Any]) -> float:
        """Извлечение временной метки из данных эмоции"""
        # Для видео эмоций
        if 'timestamp' in emotion_data:
            return float(emotion_data['timestamp'])
        
        # Для речевых эмоций - берем середину сегмента
        if 'start_time' in emotion_data and 'end_time' in emotion_data:
            return (float(emotion_data['start_time']) + float(emotion_data['end_time'])) / 2
        
        # Fallback
        return float(emotion_data.get('start_time', emotion_data.get('time', 0)))
    
    def _classify_transition(self, from_emotion: str, to_emotion: str) -> Dict[str, Any]:
        """
        Классификация перехода эмоций
        
        Args:
            from_emotion: Исходная эмоция
            to_emotion: Целевая эмоция
            
        Returns:
            Информация о типе перехода и его критичности
        """
        # Точное совпадение
        if (from_emotion, to_emotion) in self.critical_transitions:
            return self.critical_transitions[(from_emotion, to_emotion)]
        
        # Поиск обратного перехода (может иметь другую семантику)
        if (to_emotion, from_emotion) in self.critical_transitions:
            reverse_info = self.critical_transitions[(to_emotion, from_emotion)]
            return {
                "severity": max(3, reverse_info['severity'] - 2),  # Снижаем критичность
                "type": f"обратный {reverse_info['type']}"
            }
        
        # Базовая классификация по типу эмоций
        return self._basic_classification(from_emotion, to_emotion)
    
    def _basic_classification(self, from_emotion: str, to_emotion: str) -> Dict[str, Any]:
        """Базовая классификация неизвестных переходов"""
        
        # Категории эмоций
        positive_emotions = {'счастье', 'радость', 'спокойствие', 'удивление'}
        negative_emotions = {'грусть', 'злость', 'страх', 'отвращение', 'презрение', 'тревога', 'напряжение'}
        neutral_emotions = {'нейтральность'}
        
        from_type = 'positive' if from_emotion in positive_emotions else 'negative' if from_emotion in negative_emotions else 'neutral'
        to_type = 'positive' if to_emotion in positive_emotions else 'negative' if to_emotion in negative_emotions else 'neutral'
        
        # Правила классификации
        if from_type == 'positive' and to_type == 'negative':
            return {"severity": 6, "type": "позитивно-негативный переход"}
        elif from_type == 'negative' and to_type == 'positive':
            return {"severity": 5, "type": "негативно-позитивный переход"}
        elif from_type == 'neutral' and to_type == 'negative':
            return {"severity": 5, "type": "нарастание негатива"}
        elif from_type == 'negative' and to_type == 'neutral':
            return {"severity": 4, "type": "успокоение"}
        elif from_type == 'neutral' and to_type == 'positive':
            return {"severity": 3, "type": "позитивное развитие"}
        elif from_type == 'positive' and to_type == 'neutral':
            return {"severity": 3, "type": "затухание позитива"}
        else:
            return {"severity": 3, "type": f"переход {from_emotion}→{to_emotion}"}
    
    def get_transition_statistics(self, transitions: List[EmotionTransition]) -> Dict[str, Any]:
        """
        Расчет статистики переходов
        
        Args:
            transitions: Список переходов
            
        Returns:
            Словарь со статистикой
        """
        if not transitions:
            return {
                'total_transitions': 0,
                'critical_transitions': 0,
                'avg_severity': 0,
                'most_common_transition': None,
                'transition_rate': 0,
                'avg_duration': 0
            }
        
        # Основные метрики
        total_count = len(transitions)
        critical_count = sum(1 for t in transitions if t.is_critical)
        avg_severity = sum(t.severity for t in transitions) / total_count
        avg_duration = sum(t.duration for t in transitions) / total_count
        
        # Самый частый переход
        transition_counts = {}
        for t in transitions:
            key = f"{t.from_emotion}→{t.to_emotion}"
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        most_common = max(transition_counts.items(), key=lambda x: x[1]) if transition_counts else None
        
        # Частота переходов (переходов в минуту)
        if transitions:
            time_span = transitions[-1].timestamp - transitions[0].timestamp
            transition_rate = (total_count / time_span * 60) if time_span > 0 else 0
        else:
            transition_rate = 0
        
        return {
            'total_transitions': total_count,
            'critical_transitions': critical_count,
            'critical_ratio': critical_count / total_count if total_count > 0 else 0,
            'avg_severity': round(avg_severity, 2),
            'avg_duration': round(avg_duration, 2),
            'most_common_transition': most_common[0] if most_common else None,
            'most_common_count': most_common[1] if most_common else 0,
            'transition_rate': round(transition_rate, 2),
            'time_span': transitions[-1].timestamp - transitions[0].timestamp if transitions else 0,
            'transition_types': list(set(t.transition_type for t in transitions))
        }