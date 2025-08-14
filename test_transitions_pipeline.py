#!/usr/bin/env python3
"""
Интеграционный тест системы анализа переходов эмоций
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_transition_analyzer import EmotionTransitionDetector, TransitionMetricsCalculator


def test_transition_integration():
    """Тест интеграции системы анализа переходов"""
    print("🧪 Тестирование интеграции системы анализа переходов эмоций...\n")
    
    # Инициализация компонентов
    detector = EmotionTransitionDetector()
    metrics_calculator = TransitionMetricsCalculator()
    
    print("✅ Компоненты инициализированы")
    
    # Симуляция данных как из реального pipeline
    mock_video_emotions = [
        {'timestamp': 0.0, 'emotion': 'нейтральность', 'confidence': 0.85},
        {'timestamp': 2.5, 'emotion': 'спокойствие', 'confidence': 0.78},
        {'timestamp': 5.0, 'emotion': 'напряжение', 'confidence': 0.72},
        {'timestamp': 7.8, 'emotion': 'злость', 'confidence': 0.88},  # Критический переход от спокойствия
        {'timestamp': 10.2, 'emotion': 'грусть', 'confidence': 0.65},
        {'timestamp': 12.5, 'emotion': 'нейтральность', 'confidence': 0.82},
        {'timestamp': 15.0, 'emotion': 'страх', 'confidence': 0.79},  # Критический переход от нейтральности
        {'timestamp': 17.3, 'emotion': 'спокойствие', 'confidence': 0.88},  # Подавление тревоги
        {'timestamp': 20.0, 'emotion': 'счастье', 'confidence': 0.45}   # Неконгруэнтность
    ]
    
    mock_speech_emotions = [
        {'start_time': 0.0, 'end_time': 3.0, 'emotion': 'нейтральность', 'confidence': 0.62},
        {'start_time': 3.0, 'end_time': 6.5, 'emotion': 'напряжение', 'confidence': 0.58},
        {'start_time': 6.5, 'end_time': 9.0, 'emotion': 'злость', 'confidence': 0.71},
        {'start_time': 9.0, 'end_time': 12.0, 'emotion': 'грусть', 'confidence': 0.49},
        {'start_time': 12.0, 'end_time': 15.5, 'emotion': 'тревога', 'confidence': 0.55},
        {'start_time': 15.5, 'end_time': 18.0, 'emotion': 'нейтральность', 'confidence': 0.67},
        {'start_time': 18.0, 'end_time': 21.0, 'emotion': 'грусть', 'confidence': 0.41}
    ]
    
    print(f"📊 Тестовые данные: {len(mock_video_emotions)} видео эмоций, {len(mock_speech_emotions)} речевых сегментов")
    
    # Детекция переходов
    try:
        video_transitions = detector.detect_transitions(mock_video_emotions, 'video')
        speech_transitions = detector.detect_transitions(mock_speech_emotions, 'speech')
        
        print(f"🔄 Обнаружено переходов: видео ({len(video_transitions)}), речь ({len(speech_transitions)})")
        
        # Подсчет критических переходов
        critical_video = [t for t in video_transitions if t.is_critical]
        critical_speech = [t for t in speech_transitions if t.is_critical]
        
        print(f"⚠️ Критические переходы: видео ({len(critical_video)}), речь ({len(critical_speech)})")
        
        # Расчет комплексных метрик
        metrics = metrics_calculator.calculate_comprehensive_metrics(video_transitions, speech_transitions)
        
        print(f"📈 Метрики рассчитаны: {len(metrics)} категорий")
        
        # Проверяем основные компоненты метрик
        required_sections = ['basic_metrics', 'psychological_indicators', 'correlation_metrics', 
                           'temporal_metrics', 'transition_matrices', 'instability_index', 'summary']
        
        for section in required_sections:
            if section in metrics:
                print(f"✅ {section}: присутствует")
            else:
                print(f"❌ {section}: отсутствует")
                
        # Детальный анализ психологических индикаторов
        psych_indicators = metrics.get('psychological_indicators', {})
        combined = psych_indicators.get('combined', {})
        
        if combined:
            print(f"\n🧠 Психологические индикаторы:")
            print(f"  • Вероятность лжи: {combined.get('deception_likelihood', 0):.3f}")
            print(f"  • Уровень стресса: {combined.get('stress_level', 0):.3f}")
            print(f"  • Эмоциональный контроль: {combined.get('emotional_control', 0):.3f}")
            print(f"  • Нестабильность: {combined.get('emotional_instability', 0):.3f}")
        
        # Индекс нестабильности
        instability = metrics.get('instability_index', {})
        if instability:
            combined_instability = instability.get('combined_instability', 0)
            interpretation = instability.get('interpretation', 'неопределено')
            print(f"\n🌡️ Индекс нестабильности: {combined_instability:.3f} ({interpretation})")
        
        # Корреляции
        correlations = metrics.get('correlation_metrics', {})
        if correlations:
            print(f"\n🔗 Корреляции:")
            print(f"  • Временная корреляция: {correlations.get('temporal_correlation', 0):.3f}")
            print(f"  • Синхронизированные переходы: {correlations.get('synchronized_transitions', 0)}")
            print(f"  • Противоречивые переходы: {correlations.get('contradictory_transitions', 0)}")
        
        # Тестируем JSON сериализацию (как в реальном pipeline)
        try:
            # Преобразуем переходы в словари для сериализации
            video_transitions_dict = []
            for t in video_transitions:
                video_transitions_dict.append({
                    'timestamp': float(t.timestamp),
                    'from_emotion': t.from_emotion,
                    'to_emotion': t.to_emotion,
                    'duration': float(t.duration),
                    'transition_type': t.transition_type,
                    'severity': int(t.severity),
                    'confidence_before': float(t.confidence_before),
                    'confidence_after': float(t.confidence_after),
                    'modality': t.modality,
                    'transition_speed': t.transition_speed,
                    'is_critical': bool(t.is_critical)
                })
            
            speech_transitions_dict = []
            for t in speech_transitions:
                speech_transitions_dict.append({
                    'timestamp': float(t.timestamp),
                    'from_emotion': t.from_emotion,
                    'to_emotion': t.to_emotion,
                    'duration': float(t.duration),
                    'transition_type': t.transition_type,
                    'severity': int(t.severity),
                    'confidence_before': float(t.confidence_before),
                    'confidence_after': float(t.confidence_after),
                    'modality': t.modality,
                    'transition_speed': t.transition_speed,
                    'is_critical': bool(t.is_critical)
                })
            
            # Создаем полный результат как в pipeline
            pipeline_result = {
                'video_transitions': video_transitions_dict,
                'speech_transitions': speech_transitions_dict,
                'transition_metrics': metrics,
                'critical_patterns': [t for t in video_transitions_dict + speech_transitions_dict if t['is_critical']],
                'summary': f"Обнаружено переходов: видео ({len(video_transitions_dict)}), речь ({len(speech_transitions_dict)}). Критических переходов: {len(critical_video + critical_speech)}",
                'total_transitions': len(video_transitions) + len(speech_transitions),
                'critical_count': len(critical_video) + len(critical_speech)
            }
            
            # Тестируем JSON сериализацию
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(pipeline_result, f, ensure_ascii=False, indent=2)
                temp_file = f.name
            
            # Проверяем что файл создался и читается
            with open(temp_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            print(f"✅ JSON сериализация успешна: {len(loaded_data)} разделов")
            
            # Удаляем временный файл
            os.unlink(temp_file)
            
        except Exception as e:
            print(f"❌ Ошибка JSON сериализации: {e}")
            return False
        
        print(f"\n🎉 Интеграционный тест завершен успешно!")
        print(f"📋 Резюме:")
        print(f"  • Всего переходов: {len(video_transitions) + len(speech_transitions)}")
        print(f"  • Критических: {len(critical_video) + len(critical_speech)}")
        print(f"  • Видео переходы: {len(video_transitions)}")
        print(f"  • Речевые переходы: {len(speech_transitions)}")
        print(f"  • Психологические индикаторы: {len(combined)} показателей")
        print(f"  • JSON сериализация: работает")
        print(f"  • Готово к интеграции в pipeline")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка интеграционного теста: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_integration():
    """Тест интеграции с конфигурацией"""
    print("\n🔧 Тестирование интеграции с конфигурацией...")
    
    # Тестовая конфигурация
    test_config = {
        'min_transition_duration': 0.5,
        'confidence_threshold': 0.4,
        'min_confidence_diff': 0.15,
        'psychological_analysis': {
            'enable_deception_detection': True,
            'enable_stress_analysis': True,
            'enable_control_assessment': True,
            'enable_instability_tracking': True
        },
        'correlation': {
            'max_time_distance': 3.0,
            'enable_cross_modal': True
        }
    }
    
    try:
        detector = EmotionTransitionDetector(test_config)
        print("✅ Конфигурация загружена успешно")
        
        # Проверяем что параметры применились
        assert detector.min_duration == 0.5, f"min_duration: expected 0.5, got {detector.min_duration}"
        assert detector.confidence_threshold == 0.4, f"confidence_threshold: expected 0.4, got {detector.confidence_threshold}"
        print("✅ Параметры конфигурации применены корректно")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка конфигурации: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Запуск интеграционных тестов системы анализа переходов эмоций\n")
    
    # Основной интеграционный тест
    success1 = test_transition_integration()
    
    # Тест конфигурации
    success2 = test_config_integration()
    
    print(f"\n{'='*60}")
    if success1 and success2:
        print("🎉 ВСЕ ИНТЕГРАЦИОННЫЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print("✅ Система анализа переходов эмоций готова к работе!")
        print("✅ Интеграция с pipeline будет работать корректно!")
        print("✅ JSON сериализация поддерживается!")
        print("✅ Конфигурационные параметры работают!")
        print("\n🚀 Система готова для тестирования в main.py!")
    else:
        print("❌ ЕСТЬ ПРОБЛЕМЫ В ИНТЕГРАЦИОННЫХ ТЕСТАХ!")
        print("🔧 Необходимо исправить ошибки перед использованием")