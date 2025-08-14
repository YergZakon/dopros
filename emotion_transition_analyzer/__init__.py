"""
Emotion Transition Analyzer Module

Анализ переходов эмоций для психологической оценки в допросах
"""

from .transition_detector import EmotionTransitionDetector
from .transition_metrics import TransitionMetricsCalculator

__version__ = "1.0.0"
__author__ = "ДОПРОС MVP Team"

__all__ = [
    'EmotionTransitionDetector',
    'TransitionMetricsCalculator'
]