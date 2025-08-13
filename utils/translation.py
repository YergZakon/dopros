"""
Comprehensive emotion translation system with colors, descriptions, and formatting
"""

import logging
from typing import Dict, Optional, List, Tuple, Any, Union


# Emotion mapping dictionaries for different sources
DEEPFACE_EMOTIONS = {
    'angry': 'злость',
    'disgust': 'отвращение',
    'fear': 'страх',
    'happy': 'счастье',
    'sad': 'грусть',
    'surprise': 'удивление',
    'neutral': 'нейтральность'
}

FER_EMOTIONS = {
    'angry': 'злость',
    'disgust': 'отвращение', 
    'fear': 'страх',
    'happy': 'счастье',
    'sad': 'грусть',
    'surprise': 'удивление',
    'neutral': 'нейтральность'
}

YOLO_SPECIALIZED = {
    'comfortable': 'спокойствие',
    'uncomfortable': 'дискомфорт',
    'tension_brow': 'напряжение_бровей',
    'tension_eye': 'напряжение_глаз',
    'tension_mouth': 'напряжение_рта',
    'stress': 'стресс',
    'anxiety': 'тревожность',
    'defensive': 'защитная_реакция',
    'aggressive': 'агрессия',
    'deceptive': 'попытка_обмана'
}

SPEECH_EMOTIONS = {
    'angry': 'злость',
    'disgust': 'отвращение',
    'fearful': 'страх', 
    'happy': 'счастье',
    'neutral': 'нейтральность',
    'sad': 'грусть',
    'surprised': 'удивление',
    'calm': 'спокойствие',
    'excited': 'возбуждение',
    'frustration': 'фрустрация'
}

# Color scheme for emotions (hex colors)
EMOTION_COLORS = {
    'злость': '#FF4444',
    'отвращение': '#9ACD32',
    'страх': '#800080',
    'счастье': '#FFD700',
    'грусть': '#4169E1',
    'удивление': '#FF8C00',
    'нейтральность': '#808080',
    'спокойствие': '#90EE90',
    'дискомфорт': '#8B0000',
    'напряжение_бровей': '#FF6347',
    'напряжение_глаз': '#DC143C',
    'напряжение_рта': '#B22222',
    'стресс': '#FF1493',
    'тревожность': '#9400D3',
    'защитная_реакция': '#4B0082',
    'агрессия': '#8B0000',
    'попытка_обмана': '#2F4F4F',
    'возбуждение': '#FF69B4',
    'фрустрация': '#CD5C5C'
}

# Detailed descriptions for psychological analysis
EMOTION_DESCRIPTIONS = {
    'злость': {
        'short': 'Состояние гнева или раздражения',
        'detailed': 'Эмоциональное состояние, характеризующееся повышенной агрессивностью, раздражением. Может указывать на фрустрацию или защитную реакцию.',
        'indicators': ['сжатые губы', 'нахмуренные брови', 'напряженная челюсть'],
        'interrogation_meaning': 'Возможная защитная реакция на неудобные вопросы'
    },
    'отвращение': {
        'short': 'Чувство неприятия или отвращения',
        'detailed': 'Эмоция отторжения, неприязни к объекту, ситуации или информации. Может сигнализировать о нежелании обсуждать тему.',
        'indicators': ['морщины на носу', 'приподнятая верхняя губа', 'суженные глаза'],
        'interrogation_meaning': 'Возможное неприятие темы разговора или попытка скрыть информацию'
    },
    'страх': {
        'short': 'Состояние тревоги или опасения',
        'detailed': 'Эмоциональная реакция на реальную или воображаемую угрозу. Может указывать на беспокойство по поводу последствий.',
        'indicators': ['расширенные глаза', 'приоткрытый рот', 'напряжение мышц'],
        'interrogation_meaning': 'Страх перед раскрытием информации или последствиями'
    },
    'счастье': {
        'short': 'Положительное эмоциональное состояние',
        'detailed': 'Состояние радости, удовлетворения или позитивного настроя. Может указывать на комфорт или облегчение.',
        'indicators': ['улыбка', 'морщинки у глаз', 'расслабленные черты лица'],
        'interrogation_meaning': 'Возможное облегчение или удовлетворение от хода беседы'
    },
    'грусть': {
        'short': 'Состояние печали или подавленности',
        'detailed': 'Негативная эмоция, связанная с потерей, разочарованием или сожалением. Может указывать на эмоциональную нагрузку.',
        'indicators': ['опущенные уголки губ', 'потухший взгляд', 'общая вялость'],
        'interrogation_meaning': 'Возможные сожаления или эмоциональные переживания по теме'
    },
    'удивление': {
        'short': 'Реакция на неожиданность',
        'detailed': 'Кратковременная эмоция в ответ на неожиданную информацию или событие. Может указывать на искренность реакции.',
        'indicators': ['приподнятые брови', 'расширенные глаза', 'приоткрытый рот'],
        'interrogation_meaning': 'Неожиданность вопроса или новой информации'
    },
    'нейтральность': {
        'short': 'Нейтральное эмоциональное состояние',
        'detailed': 'Отсутствие выраженных эмоций, спокойное состояние. Может указывать на контроль эмоций или отстраненность.',
        'indicators': ['расслабленные черты', 'ровное выражение', 'спокойный взгляд'],
        'interrogation_meaning': 'Попытка контролировать эмоциональные проявления'
    },
    'спокойствие': {
        'short': 'Состояние внутреннего покоя',
        'detailed': 'Расслабленное, умиротворенное состояние. Указывает на отсутствие стресса и тревоги.',
        'indicators': ['расслабленная поза', 'ровное дыхание', 'мягкие черты лица'],
        'interrogation_meaning': 'Комфорт с ситуацией или уверенность в своих ответах'
    },
    'дискомфорт': {
        'short': 'Состояние неудобства или напряжения',
        'detailed': 'Ощущение неловкости, беспокойства или физического/эмоционального дискомфорта.',
        'indicators': ['суетливость', 'напряженная поза', 'избегание взгляда'],
        'interrogation_meaning': 'Неудобство от вопросов или попытка скрыть информацию'
    },
    'напряжение_бровей': {
        'short': 'Напряжение в области бровей',
        'detailed': 'Физическое проявление стресса или концентрации в области лба и бровей.',
        'indicators': ['нахмуренные брови', 'морщины на лбу', 'напряжение мышц'],
        'interrogation_meaning': 'Концентрация на формулировке ответа или внутреннее напряжение'
    },
    'напряжение_глаз': {
        'short': 'Напряжение в области глаз',
        'detailed': 'Стресс или усилие, проявляющееся через мимику глаз и окружающих мышц.',
        'indicators': ['прищуренные глаза', 'напряжение век', 'изменение взгляда'],
        'interrogation_meaning': 'Попытка контролировать выражение или скрыть эмоции'
    },
    'напряжение_рта': {
        'short': 'Напряжение в области рта',
        'detailed': 'Стресс или контроль эмоций, проявляющийся через мимику рта и челюсти.',
        'indicators': ['сжатые губы', 'напряженная челюсть', 'изменение речи'],
        'interrogation_meaning': 'Контроль над словами или сдерживание эмоций'
    },
    'стресс': {
        'short': 'Общее состояние стресса',
        'detailed': 'Физиологическая и психологическая реакция на давление или угрозу.',
        'indicators': ['общее напряжение', 'учащенное дыхание', 'беспокойные движения'],
        'interrogation_meaning': 'Высокий уровень стресса от ситуации допроса'
    },
    'тревожность': {
        'short': 'Состояние повышенной тревоги',
        'detailed': 'Беспокойство по поводу будущих событий или неопределенности.',
        'indicators': ['беспокойные движения', 'частые взгляды', 'нервозность'],
        'interrogation_meaning': 'Беспокойство о последствиях или развитии ситуации'
    },
    'защитная_реакция': {
        'short': 'Защитное поведение',
        'detailed': 'Поведенческая реакция, направленная на самозащиту от угрозы или дискомфорта.',
        'indicators': ['закрытая поза', 'избегание контакта', 'оборонительность'],
        'interrogation_meaning': 'Попытка защититься от неудобных вопросов'
    },
    'агрессия': {
        'short': 'Агрессивное состояние',
        'detailed': 'Враждебность или готовность к конфликту как реакция на угрозу.',
        'indicators': ['напряженная поза', 'сжатые кулаки', 'пристальный взгляд'],
        'interrogation_meaning': 'Враждебность к процессу допроса или следователю'
    },
    'попытка_обмана': {
        'short': 'Признаки возможного обмана',
        'detailed': 'Поведенческие паттерны, которые могут указывать на попытку ввести в заблуждение.',
        'indicators': ['избегание взгляда', 'неконгруэнтность', 'чрезмерный контроль'],
        'interrogation_meaning': 'Возможная попытка скрыть правду или исказить факты'
    },
    'возбуждение': {
        'short': 'Состояние повышенного возбуждения',
        'detailed': 'Высокий уровень эмоциональной или физиологической активации.',
        'indicators': ['учащенная речь', 'активные жесты', 'повышенная энергия'],
        'interrogation_meaning': 'Эмоциональная реакция на тему разговора'
    },
    'фрустрация': {
        'short': 'Состояние фрустрации',
        'detailed': 'Раздражение от препятствий или неспособности достичь цели.',
        'indicators': ['признаки раздражения', 'нетерпеливость', 'напряжение'],
        'interrogation_meaning': 'Недовольство процессом или неспособность контролировать ситуацию'
    }
}

# Emoji mapping for emotions
EMOTION_EMOJIS = {
    'злость': '😠',
    'отвращение': '🤢',
    'страх': '😨',
    'счастье': '😊',
    'грусть': '😢',
    'удивление': '😲',
    'нейтральность': '😐',
    'спокойствие': '😌',
    'дискомфорт': '😟',
    'напряжение_бровей': '😤',
    'напряжение_глаз': '🙄',
    'напряжение_рта': '😬',
    'стресс': '😰',
    'тревожность': '😧',
    'защитная_реакция': '🛡️',
    'агрессия': '😡',
    'попытка_обмана': '🤥',
    'возбуждение': '🤩',
    'фрустрация': '😤'
}


class EmotionTranslator:
    """Comprehensive emotion translator with colors, descriptions, and formatting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Combine all emotion mappings
        self.all_mappings = {
            'deepface': DEEPFACE_EMOTIONS,
            'fer': FER_EMOTIONS,
            'yolo': YOLO_SPECIALIZED,
            'speech': SPEECH_EMOTIONS
        }
        
        # Create comprehensive mapping
        self.en_to_ru = {}
        for mapping in self.all_mappings.values():
            self.en_to_ru.update(mapping)
        
        # Reverse mapping
        self.ru_to_en = {v: k for k, v in self.en_to_ru.items()}
        
        # Additional variations
        self.en_variations = {
            'anger': 'angry',
            'happiness': 'happy',
            'joy': 'happy',
            'sadness': 'sad',
            'fearful': 'fear',
            'surprised': 'surprise',
            'disgust': 'disgust',
            'disgusted': 'disgust',
            'neutral': 'neutral',
            'comfortable': 'comfortable',
            'uncomfortable': 'uncomfortable'
        }
        
        self.ru_variations = {
            'гнев': 'злость',
            'радость': 'счастье', 
            'печаль': 'грусть',
            'испуг': 'страх',
            'нейтральный': 'нейтральность',
            'спокойный': 'спокойствие',
            'стресс': 'стресс',
            'беспокойство': 'тревожность'
        }
    
    def translate_emotion(self, emotion: str, source: str = 'auto', target_lang: str = 'ru') -> str:
        """
        Translate emotion with context from specific source
        
        Args:
            emotion: Emotion term to translate
            source: Source of emotion ('deepface', 'fer', 'yolo', 'speech', 'auto')
            target_lang: Target language ('ru' or 'en')
            
        Returns:
            Translated emotion term
        """
        if not emotion:
            return emotion
            
        emotion = emotion.lower().strip()
        
        # Use specific mapping if source is specified
        if source != 'auto' and source in self.all_mappings:
            mapping = self.all_mappings[source]
            if emotion in mapping and target_lang == 'ru':
                return mapping[emotion]
        
        # Fall back to general translation
        if target_lang.lower() == 'ru':
            return self._translate_to_russian(emotion)
        elif target_lang.lower() == 'en':
            return self._translate_to_english(emotion)
        else:
            self.logger.warning(f"Unknown target language: {target_lang}")
            return emotion
    
    def _translate_to_russian(self, emotion_en: str) -> str:
        """Translate English emotion to Russian"""
        
        # Normalize variations first
        emotion_en = self.en_variations.get(emotion_en, emotion_en)
        
        # Direct translation
        if emotion_en in self.en_to_ru:
            return self.en_to_ru[emotion_en]
            
        # If already Russian, return as-is
        if emotion_en in self.ru_to_en:
            return emotion_en
            
        # Check Russian variations
        if emotion_en in self.ru_variations:
            return self.ru_variations[emotion_en]
        
        # Default fallback
        self.logger.warning(f"Unknown emotion for Russian translation: {emotion_en}")
        return 'нейтральность'
    
    def _translate_to_english(self, emotion_ru: str) -> str:
        """Translate Russian emotion to English"""
        
        # Normalize variations first
        emotion_ru = self.ru_variations.get(emotion_ru, emotion_ru)
        
        # Direct translation
        if emotion_ru in self.ru_to_en:
            return self.ru_to_en[emotion_ru]
            
        # If already English, return as-is  
        if emotion_ru in self.en_to_ru:
            return emotion_ru
            
        # Check English variations
        if emotion_ru in self.en_variations:
            return self.en_variations[emotion_ru]
        
        # Default fallback
        self.logger.warning(f"Unknown emotion for English translation: {emotion_ru}")
        return 'neutral'
    
    def translate_batch(self, emotions_list: List[str], source: str = 'auto', target_lang: str = 'ru') -> List[str]:
        """
        Translate multiple emotions at once
        
        Args:
            emotions_list: List of emotion terms
            source: Source of emotions
            target_lang: Target language
            
        Returns:
            List of translated emotions
        """
        return [self.translate_emotion(emotion, source, target_lang) for emotion in emotions_list]
    
    def get_emotion_color(self, emotion: str, intensity: float = 1.0) -> str:
        """
        Get color for emotion with intensity adjustment
        
        Args:
            emotion: Emotion name (in Russian)
            intensity: Color intensity (0.0 to 1.0)
            
        Returns:
            Hex color string
        """
        # Normalize emotion
        emotion = emotion.lower().strip()
        if emotion in self.ru_variations:
            emotion = self.ru_variations[emotion]
        
        # Get base color
        base_color = EMOTION_COLORS.get(emotion, '#808080')  # Default gray
        
        if intensity == 1.0:
            return base_color
        
        # Adjust intensity by mixing with white/gray
        try:
            # Convert hex to RGB
            hex_color = base_color.lstrip('#')
            rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
            
            # Mix with neutral gray based on intensity
            neutral = [128, 128, 128]  # Gray
            adjusted_rgb = [
                int(rgb[i] * intensity + neutral[i] * (1 - intensity))
                for i in range(3)
            ]
            
            # Convert back to hex
            return '#' + ''.join(f'{c:02x}' for c in adjusted_rgb)
            
        except ValueError:
            return base_color
    
    def get_emotion_description(self, emotion: str, detail_level: str = 'short') -> str:
        """
        Get description for emotion
        
        Args:
            emotion: Emotion name (in Russian)
            detail_level: Level of detail ('short', 'detailed', 'indicators', 'interrogation_meaning')
            
        Returns:
            Emotion description
        """
        emotion = emotion.lower().strip()
        if emotion in self.ru_variations:
            emotion = self.ru_variations[emotion]
        
        description_data = EMOTION_DESCRIPTIONS.get(emotion, {})
        
        if detail_level in description_data:
            return description_data[detail_level]
        
        return description_data.get('short', 'Неизвестная эмоция')
    
    def get_interrogation_interpretation(self, emotion: str) -> str:
        """
        Get interrogation-specific interpretation of emotion
        
        Args:
            emotion: Emotion name (in Russian)
            
        Returns:
            Interrogation interpretation
        """
        return self.get_emotion_description(emotion, 'interrogation_meaning')
    
    def reverse_translate(self, russian_emotion: str) -> str:
        """
        Translate Russian emotion back to English
        
        Args:
            russian_emotion: Russian emotion term
            
        Returns:
            English emotion term
        """
        russian_emotion = russian_emotion.lower().strip()
        if russian_emotion in self.ru_variations:
            russian_emotion = self.ru_variations[russian_emotion]
        
        return self.ru_to_en.get(russian_emotion, 'neutral')
    
    def get_original_emotion(self, translated: str, source: str) -> str:
        """
        Get original emotion term for specific source
        
        Args:
            translated: Translated (Russian) emotion
            source: Original source ('deepface', 'fer', 'yolo', 'speech')
            
        Returns:
            Original emotion term
        """
        if source not in self.all_mappings:
            return self.reverse_translate(translated)
        
        # Find original term in source mapping
        source_mapping = self.all_mappings[source]
        for en_emotion, ru_emotion in source_mapping.items():
            if ru_emotion == translated:
                return en_emotion
        
        return self.reverse_translate(translated)
    
    def format_for_report(self, emotion_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format emotion data for report generation
        
        Args:
            emotion_data: Raw emotion analysis data
            
        Returns:
            Formatted data for reports
        """
        emotion = emotion_data.get('emotion', 'нейтральность')
        confidence = emotion_data.get('confidence', 0.0)
        
        formatted = {
            'emotion': emotion,
            'emotion_en': self.reverse_translate(emotion),
            'confidence': confidence,
            'confidence_percent': f"{confidence * 100:.1f}%",
            'color': self.get_emotion_color(emotion, confidence),
            'emoji': EMOTION_EMOJIS.get(emotion, '❓'),
            'short_description': self.get_emotion_description(emotion, 'short'),
            'detailed_description': self.get_emotion_description(emotion, 'detailed'),
            'indicators': self.get_emotion_description(emotion, 'indicators'),
            'interrogation_meaning': self.get_interrogation_interpretation(emotion),
            'timestamp': emotion_data.get('timestamp', 0),
            'method': emotion_data.get('method', 'unknown')
        }
        
        return formatted
    
    def create_emotion_legend(self) -> List[Dict[str, str]]:
        """
        Create legend for emotion visualization
        
        Returns:
            List of emotion legend entries
        """
        legend = []
        
        # Get unique emotions from all mappings
        all_emotions = set()
        for mapping in self.all_mappings.values():
            all_emotions.update(mapping.values())
        
        for emotion in sorted(all_emotions):
            legend.append({
                'emotion': emotion,
                'color': EMOTION_COLORS.get(emotion, '#808080'),
                'emoji': EMOTION_EMOJIS.get(emotion, '❓'),
                'description': self.get_emotion_description(emotion, 'short')
            })
        
        return legend
    
    def get_emotion_emoji(self, emotion: str) -> str:
        """
        Get emoji for emotion
        
        Args:
            emotion: Emotion name (in Russian)
            
        Returns:
            Emoji string
        """
        emotion = emotion.lower().strip()
        if emotion in self.ru_variations:
            emotion = self.ru_variations[emotion]
        
        return EMOTION_EMOJIS.get(emotion, '❓')
    
    def get_emotion_intensity_scale(self, emotion: str, num_levels: int = 5) -> List[Dict[str, str]]:
        """
        Generate intensity scale for emotion visualization
        
        Args:
            emotion: Base emotion
            num_levels: Number of intensity levels
            
        Returns:
            List of intensity levels with colors
        """
        scale = []
        for i in range(num_levels):
            intensity = (i + 1) / num_levels
            scale.append({
                'level': i + 1,
                'intensity': intensity,
                'color': self.get_emotion_color(emotion, intensity),
                'label': f"{emotion} ({intensity * 100:.0f}%)"
            })
        
        return scale
    
    def analyze_emotion_distribution(self, emotions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze distribution of emotions for statistical reporting
        
        Args:
            emotions_data: List of emotion analysis results
            
        Returns:
            Statistical analysis of emotions
        """
        if not emotions_data:
            return {'error': 'No emotion data provided'}
        
        emotion_counts = {}
        total_confidence = {}
        
        # Count emotions and sum confidences
        for data in emotions_data:
            emotion = data.get('emotion')
            confidence = data.get('confidence', 0)
            
            if emotion:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                total_confidence[emotion] = total_confidence.get(emotion, 0) + confidence
        
        # Calculate statistics
        total_emotions = sum(emotion_counts.values())
        distribution = {}
        
        for emotion, count in emotion_counts.items():
            avg_confidence = total_confidence[emotion] / count if count > 0 else 0
            
            distribution[emotion] = {
                'count': count,
                'percentage': (count / total_emotions) * 100 if total_emotions > 0 else 0,
                'avg_confidence': avg_confidence,
                'color': self.get_emotion_color(emotion),
                'emoji': self.get_emotion_emoji(emotion),
                'description': self.get_emotion_description(emotion, 'short')
            }
        
        return {
            'total_analyzed': total_emotions,
            'unique_emotions': len(emotion_counts),
            'distribution': distribution,
            'dominant_emotion': max(emotion_counts.keys(), key=emotion_counts.get) if emotion_counts else None
        }
    
    def get_emotion_transitions_colors(self, transitions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Get colors for emotion transitions visualization
        
        Args:
            transitions: List of emotion transitions
            
        Returns:
            Transitions with color information
        """
        colored_transitions = []
        
        for transition in transitions:
            from_emotion = transition.get('from_emotion')
            to_emotion = transition.get('to_emotion')
            
            colored_transition = transition.copy()
            colored_transition.update({
                'from_color': self.get_emotion_color(from_emotion) if from_emotion else '#808080',
                'to_color': self.get_emotion_color(to_emotion) if to_emotion else '#808080',
                'from_emoji': self.get_emotion_emoji(from_emotion) if from_emotion else '❓',
                'to_emoji': self.get_emotion_emoji(to_emotion) if to_emotion else '❓'
            })
            
            colored_transitions.append(colored_transition)
        
        return colored_transitions
    
    def get_emotion_mapping(self, lang: str = 'both') -> Dict[str, str]:
        """
        Get emotion mapping dictionary
        
        Args:
            lang: Language mapping to return ('ru_to_en', 'en_to_ru', or 'both')
            
        Returns:
            Emotion mapping dictionary
        """
        if lang == 'ru_to_en':
            return self.ru_to_en.copy()
        elif lang == 'en_to_ru':
            return self.en_to_ru.copy()
        elif lang == 'both':
            return {
                'ru_to_en': self.ru_to_en.copy(),
                'en_to_ru': self.en_to_ru.copy()
            }
        else:
            self.logger.warning(f"Unknown language mapping: {lang}")
            return {}
    
    def is_valid_emotion(self, emotion: str, lang: Optional[str] = None) -> bool:
        """
        Check if emotion is valid in given language
        
        Args:
            emotion: Emotion to check
            lang: Language to check ('ru', 'en', or None for both)
            
        Returns:
            True if emotion is valid
        """
        if not emotion:
            return False
            
        emotion = emotion.lower().strip()
        
        if lang == 'ru':
            return emotion in self.ru_to_en or emotion in self.ru_variations
        elif lang == 'en':
            return emotion in self.en_to_ru or emotion in self.en_variations
        else:
            # Check both languages
            return (emotion in self.ru_to_en or 
                   emotion in self.en_to_ru or
                   emotion in self.ru_variations or 
                   emotion in self.en_variations)
    
    def get_all_emotions(self, lang: str = 'ru') -> List[str]:
        """
        Get list of all supported emotions
        
        Args:
            lang: Language to get emotions for ('ru' or 'en')
            
        Returns:
            List of emotion terms
        """
        if lang == 'ru':
            return list(self.ru_to_en.keys())
        elif lang == 'en':
            return list(self.en_to_ru.keys()) 
        else:
            self.logger.warning(f"Unknown language: {lang}")
            return []
    
    def get_supported_sources(self) -> List[str]:
        """
        Get list of supported emotion sources
        
        Returns:
            List of source names
        """
        return list(self.all_mappings.keys())
    
    def get_source_emotions(self, source: str) -> Dict[str, str]:
        """
        Get emotion mapping for specific source
        
        Args:
            source: Source name
            
        Returns:
            Emotion mapping dictionary
        """
        return self.all_mappings.get(source, {}).copy()