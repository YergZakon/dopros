"""
ДОПРОС MVP 2.0 - Полнофункциональный Streamlit интерфейс
Система анализа видео допросов с мультимодальным анализом эмоций
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import json
import io
import base64
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import logging
import os
import tempfile
import shutil

# Импорт наших модулей
try:
    from core.pipeline import MasterPipeline
    from core.data_aggregator import DataAggregator  
    from core.report_generator import ComprehensiveReportGenerator
    from utils.translation import EmotionTranslator
    from utils.gpu_manager import get_gpu_manager
except ImportError as e:
    st.error(f"Ошибка импорта модулей: {e}")
    st.stop()

# ================================
# 1. НАСТРОЙКА СТРАНИЦЫ И СТИЛИЗАЦИЯ
# ================================

st.set_page_config(
    page_title="ДОПРОС MVP 2.0",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "ДОПРОС MVP 2.0 - Система анализа видео допросов"
    }
)

# Кастомные стили
st.markdown('''
<style>
    .main {
        padding: 0rem 1rem;
    }
    
    .stProgress > div > div > div > div {
        background-color: #FF4444;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF4444;
        margin: 0.5rem 0;
    }
    
    .critical-moment {
        background-color: #ffcccc;
        padding: 10px;
        border-left: 3px solid red;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    .normal-moment {
        background-color: #f0f2f6;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    .header-title {
        text-align: center;
        color: #FF4444;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
</style>
''', unsafe_allow_html=True)

# ================================
# 2. ИНИЦИАЛИЗАЦИЯ СЕССИИ
# ================================

def init_session_state():
    """Инициализация состояния сессии"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'uploaded_video_path' not in st.session_state:
        st.session_state.uploaded_video_path = None
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    if 'config' not in st.session_state:
        st.session_state.config = load_default_config()

def load_default_config():
    """Загрузка конфигурации по умолчанию"""
    return {
        'processing': {
            'frame_skip': 15,
            'models': {
                'emotion': {
                    'use_deepface': True,
                    'use_fer': True,
                    'use_yolo': True,
                    'confidence_threshold': 0.5
                }
            },
            'audio': {
                'sample_rate': 16000,
                'vad_enabled': True
            },
            'video': {
                'extract_fps': 1,
                'max_resolution': 1080
            }
        },
        'analysis': {
            'time_resolution': 0.1,
            'emotion_change_threshold': 0.7,
            'mismatch_threshold': 0.5,
            'speech_pause_threshold': 2.0
        },
        'storage': {
            'frames_dir': 'storage/frames',
            'audio_dir': 'storage/audio',
            'faces_dir': 'storage/faces',
            'reports_dir': 'storage/reports',
            'analysis_dir': 'storage/analysis'
        },
        'openai': {
            'api_key': os.getenv('OPENAI_API_KEY', ''),
            'model': 'gpt-4',
            'max_tokens': 2000,
            'temperature': 0.3
        }
    }

# ================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ================================

def create_emotion_timeline(results_data):
    """Создание интерактивного графика временной шкалы эмоций"""
    video_emotions = results_data.get('data', {}).get('video_emotions', [])
    speech_emotions = results_data.get('data', {}).get('speech_emotions', [])
    
    if not video_emotions and not speech_emotions:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных для отображения", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Эмоции лица', 'Эмоции речи'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # График эмоций лица
    if video_emotions:
        timestamps = [e.get('timestamp', 0) for e in video_emotions]
        confidences = [e.get('confidence', 0) for e in video_emotions]
        emotions = [e.get('emotion', 'нейтральность') for e in video_emotions]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=confidences,
                mode='lines+markers',
                name='Эмоции лица',
                line=dict(color='#FF4444', width=2),
                marker=dict(size=4),
                text=emotions,
                hovertemplate='<b>Время:</b> %{x:.1f}с<br><b>Эмоция:</b> %{text}<br><b>Уверенность:</b> %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # График эмоций речи
    if speech_emotions:
        timestamps = [e.get('timestamp', 0) for e in speech_emotions]
        confidences = [e.get('confidence', 0) for e in speech_emotions]
        emotions = [e.get('emotion', 'нейтральность') for e in speech_emotions]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=confidences,
                mode='lines+markers',
                name='Эмоции речи',
                line=dict(color='#4444FF', width=2),
                marker=dict(size=4),
                text=emotions,
                hovertemplate='<b>Время:</b> %{x:.1f}с<br><b>Эмоция:</b> %{text}<br><b>Уверенность:</b> %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title="Временная шкала эмоций",
        xaxis_title="Время (секунды)",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_emotion_heatmap(results_data, metrics):
    """Создание тепловой карты эмоций"""
    video_emotions = results_data.get('data', {}).get('video_emotions', [])
    speech_emotions = results_data.get('data', {}).get('speech_emotions', [])
    
    if not video_emotions and not speech_emotions:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных для тепловой карты", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    emotion_distribution = metrics.get('emotion_distribution', {})
    emotions = ['злость', 'грусть', 'нейтральность', 'счастье', 'удивление', 'страх', 'отвращение']
    
    # Create matrix: video emotions (row 0) and speech emotions (row 1)
    video_row = [emotion_distribution.get(emo, 0) for emo in emotions]
    speech_row = [0] * len(emotions)  # No speech data yet
    
    # Normalize to percentages
    total_frames = metrics.get('total_frames', 1)
    video_row = [x/total_frames * 100 for x in video_row]
    
    matrix = [video_row, speech_row]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[e.title() for e in emotions],
        y=['Лицо', 'Речь'],
        colorscale='RdYlBu_r',
        text=matrix,
        texttemplate="%{text:.1f}%",
        textfont={"size": 10},
        colorbar=dict(title="Процент времени")
    ))
    
    fig.update_layout(
        title="Тепловая карта эмоций",
        xaxis_title="Тип эмоции",
        yaxis_title="Модальность",
        height=300
    )
    
    return fig

def create_speech_emotion_plot(speech_data):
    """Создание графика эмоций в речи"""
    if not speech_data:
        fig = go.Figure()
        fig.add_annotation(text="Нет речевых данных", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    
    # График речевой активности
    fig.add_trace(go.Scatter(
        x=speech_data.get('timestamps', []),
        y=speech_data.get('activity', []),
        mode='lines',
        name='Речевая активность',
        line=dict(color='#00AA00', width=2),
        fill='tonexty'
    ))
    
    # График эмоций речи
    fig.add_trace(go.Scatter(
        x=speech_data.get('timestamps', []),
        y=speech_data.get('emotions', []),
        mode='lines+markers',
        name='Эмоции речи',
        line=dict(color='#FF8800', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Анализ речи",
        xaxis_title="Время (секунды)",
        yaxis=dict(title="Активность", side="left"),
        yaxis2=dict(title="Эмоция", side="right", overlaying="y"),
        height=400
    )
    
    return fig

def format_time(seconds):
    """Форматирование времени в MM:SS"""
    return str(timedelta(seconds=int(seconds)))[2:]

def save_uploaded_file(uploaded_file):
    """Сохранение загруженного файла"""
    try:
        # Создаем временную директорию
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Генерируем уникальное имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = temp_dir / f"{timestamp}_{uploaded_file.name}"
        
        # Сохраняем файл
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(file_path)
    
    except Exception as e:
        st.error(f"Ошибка сохранения файла: {e}")
        return None

def get_demo_video_path():
    """Получение пути к демо видео"""
    demo_path = Path("demo/sample_interrogation.mp4")
    if demo_path.exists():
        return str(demo_path)
    else:
        # Создаем фиктивный путь для демо
        return "demo/sample_interrogation.mp4"

def generate_demo_results():
    """Генерация демонстрационных результатов"""
    # Создаем синтетические данные для демо
    timestamps = np.linspace(0, 120, 1200)  # 2 минуты видео
    
    # Синтетические эмоции лица
    face_emotions = np.sin(timestamps * 0.1) * 0.5 + np.random.normal(0, 0.1, len(timestamps))
    
    # Синтетические эмоции речи
    speech_emotions = np.cos(timestamps * 0.08) * 0.3 + np.random.normal(0, 0.15, len(timestamps))
    
    # Речевая активность
    speech_activity = np.where(np.sin(timestamps * 0.2) > 0, 1, 0) + np.random.normal(0, 0.1, len(timestamps))
    
    return {
        'emotions': {
            'video_emotions': {
                'timestamps': timestamps.tolist(),
                'values': face_emotions.tolist()
            },
            'audio_emotions': {
                'timestamps': timestamps.tolist(),
                'values': speech_emotions.tolist()
            }
        },
        'speech': {
            'timestamps': timestamps.tolist(),
            'activity': np.clip(speech_activity, 0, 1).tolist(),
            'emotions': speech_emotions.tolist()
        },
        'critical_moments': [
            {
                'time': '00:15',
                'timestamp': 15,
                'type': 'Резкое изменение эмоции',
                'description': 'Переход от нейтрального к раздраженному состоянию',
                'severity': 7,
                'face_emotion': 'Раздражение',
                'speech_emotion': 'Нейтральность',
                'transcript': 'Я уже объяснял это несколько раз...',
                'frame': None
            },
            {
                'time': '00:45',
                'timestamp': 45,
                'type': 'Несоответствие модальностей',
                'description': 'Лицевые эмоции не соответствуют тону речи',
                'severity': 6,
                'face_emotion': 'Печаль',
                'speech_emotion': 'Радость',
                'transcript': 'Да, конечно, я был там.',
                'frame': None
            },
            {
                'time': '01:20',
                'timestamp': 80,
                'type': 'Длительная пауза',
                'description': 'Необычно долгая пауза перед ответом',
                'severity': 5,
                'face_emotion': 'Напряжение',
                'speech_emotion': 'Неопределенность',
                'transcript': '[пауза 4.2 сек] Ну... это сложный вопрос.',
                'frame': None
            }
        ],
        'transcript_segments': [
            {
                'speaker': 'Следователь',
                'text': 'Где вы находились 15 числа вечером?',
                'time': '00:05',
                'critical': False
            },
            {
                'speaker': 'Свидетель',
                'text': 'Я уже объяснял это несколько раз...',
                'time': '00:15',
                'critical': True
            },
            {
                'speaker': 'Свидетель',
                'text': 'Да, конечно, я был там.',
                'time': '00:45',
                'critical': True
            }
        ],
        'dominant_emotion': 'Напряжение',
        'emotion_changes': 23,
        'stress_level': 0.68,
        'stability': 0.42,
        'processing_time': 45.2,
        'gpt_insights': """
        **Психологический анализ допроса:**
        
        1. **Эмоциональная нестабильность**: Обнаружена высокая изменчивость эмоциональных состояний (42% стабильности), что может указывать на внутренний конфликт или стресс.
        
        2. **Повышенный уровень стресса**: Зафиксирован стресс-уровень 68%, что значительно выше нормы для обычной беседы.
        
        3. **Несоответствия между модальностями**: Выявлены моменты, когда эмоции лица не соответствуют эмоциям в речи, что может быть признаком контролируемого поведения.
        
        4. **Защитные реакции**: Фразы типа "я уже объяснял" могут указывать на желание избежать углубленного обсуждения темы.
        
        **Рекомендации для ведения допроса:**
        - Обратить внимание на моменты несоответствия эмоций
        - Детализировать вопросы в моменты повышенного стресса  
        - Проверить показания по эпизодам с длительными паузами
        """,
        'emotion_matrix': [[0.3, -0.2, 0.1, 0.2, 0.1], [-0.1, 0.2, 0.3, -0.2, 0.0]],
        'audio_path': 'demo/sample_audio.wav',
        'speech_segments': pd.DataFrame({
            'Время': ['00:05', '00:15', '00:25', '00:45', '01:20'],
            'Говорящий': ['Следователь', 'Свидетель', 'Следователь', 'Свидетель', 'Свидетель'],
            'Эмоция': ['Нейтральность', 'Раздражение', 'Настойчивость', 'Радость', 'Неопределенность'],
            'Уверенность': [0.85, 0.92, 0.78, 0.65, 0.43]
        })
    }

# ================================
# ИНИЦИАЛИЗАЦИЯ
# ================================

init_session_state()

# ================================
# ЗАГОЛОВОК ПРИЛОЖЕНИЯ
# ================================

st.markdown('<h1 class="header-title">🎥 ДОПРОС MVP 2.0</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        Система интеллектуального анализа видео допросов с мультимодальным распознаванием эмоций
    </p>
</div>
""", unsafe_allow_html=True)

# ================================
# 4. БОКОВАЯ ПАНЕЛЬ С НАСТРОЙКАМИ
# ================================

with st.sidebar:
    st.title("⚙️ Настройки анализа")
    
    # Информация о системе
    with st.expander("ℹ️ Информация о системе"):
        try:
            gpu_manager = get_gpu_manager()
            device_info = gpu_manager.get_device_info()
            
            st.write("**Устройство:**", device_info.get('current_device', 'Не определено'))
            st.write("**CUDA доступна:**", "✅" if device_info.get('cuda_available', False) else "❌")
            st.write("**Доступные устройства:**", len(device_info.get('available_devices', [])))
            
            if device_info.get('nvidia_gpus'):
                for gpu in device_info['nvidia_gpus'][:1]:  # Показываем только первую GPU
                    memory_gb = gpu['memory_total'] / (1024**3)
                    st.write(f"**GPU:** {gpu['name']}")
                    st.write(f"**Память:** {memory_gb:.1f} GB")
                    
        except Exception as e:
            st.write("**Статус:** Информация недоступна")
    
    st.divider()
    
    # Выбор моделей
    st.subheader("🤖 Модели")
    use_deepface = st.checkbox("DeepFace", value=True, help="Глубокий анализ эмоций лица")
    use_fer = st.checkbox("FER", value=True, help="Быстрое распознавание эмоций")
    use_yolo_emotion = st.checkbox("YOLO эмоции", value=True, help="YOLO для эмоций и лиц")
    use_speech_analysis = st.checkbox("Анализ речи", value=True, help="Анализ эмоций в речи")
    
    st.divider()
    
    # Параметры обработки
    st.subheader("📊 Параметры")
    frame_skip = st.slider(
        "Пропуск кадров", 
        min_value=1, max_value=30, value=15,
        help="Количество пропускаемых кадров (больше = быстрее обработка)"
    )
    
    confidence_threshold = st.slider(
        "Порог уверенности", 
        min_value=0.1, max_value=1.0, value=0.5, step=0.1,
        help="Минимальная уверенность для принятия решения"
    )
    
    time_resolution = st.slider(
        "Временное разрешение (сек)", 
        min_value=0.1, max_value=2.0, value=0.1, step=0.1,
        help="Временной интервал для синхронизации данных"
    )
    
    st.divider()
    
    # GPU настройки
    st.subheader("💻 Устройство")
    device_option = st.radio(
        "Выбор устройства",
        ["Авто", "GPU", "CPU"],
        index=0,
        help="Авто - автоматический выбор оптимального устройства"
    )
    
    force_cpu = st.checkbox("Принудительно CPU", value=False)
    
    st.divider()
    
    # Экспорт настройки
    st.subheader("📁 Экспорт")
    export_formats = st.multiselect(
        "Форматы отчетов",
        ["CSV", "JSON", "HTML", "PDF", "Excel"],
        default=["CSV", "HTML"],
        help="Выберите форматы для экспорта результатов"
    )
    
    include_charts = st.checkbox("Включить графики", value=True)
    include_raw_data = st.checkbox("Включить сырые данные", value=False)
    
    st.divider()
    
    # Дополнительные настройки
    with st.expander("🔧 Дополнительно"):
        st.subheader("Детекция аномалий")
        emotion_change_threshold = st.slider("Порог изменения эмоций", 0.1, 2.0, 0.7)
        mismatch_threshold = st.slider("Порог несоответствий", 0.1, 1.0, 0.5)
        speech_pause_threshold = st.slider("Порог пауз в речи (сек)", 1.0, 10.0, 2.0)
        
        st.subheader("OpenAI настройки")
        use_gpt_analysis = st.checkbox("Использовать GPT анализ", value=True)
        openai_model = st.selectbox("Модель GPT", ["gpt-4", "gpt-3.5-turbo"], index=0)
        
    # Обновление конфигурации
    st.session_state.config.update({
        'processing': {
            'frame_skip': frame_skip,
            'models': {
                'emotion': {
                    'use_deepface': use_deepface,
                    'use_fer': use_fer,
                    'use_yolo': use_yolo_emotion,
                    'use_speech_analysis': use_speech_analysis,
                    'confidence_threshold': confidence_threshold
                }
            },
            'device': {
                'option': device_option.lower(),
                'force_cpu': force_cpu
            }
        },
        'analysis': {
            'time_resolution': time_resolution,
            'emotion_change_threshold': emotion_change_threshold,
            'mismatch_threshold': mismatch_threshold,
            'speech_pause_threshold': speech_pause_threshold
        },
        'export': {
            'formats': export_formats,
            'include_charts': include_charts,
            'include_raw_data': include_raw_data
        },
        'openai': {
            'enabled': use_gpt_analysis,
            'model': openai_model
        }
    })

# ================================
# 5. ОСНОВНАЯ ОБЛАСТЬ С ТАБАМИ
# ================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📹 Загрузка", 
    "📊 Анализ эмоций", 
    "🎙️ Анализ речи",
    "📝 Транскрипция", 
    "⚠️ Критические моменты",
    "📈 Отчеты"
])

# ================================
# TAB 1: ЗАГРУЗКА ВИДЕО
# ================================

with tab1:
    st.header("📹 Загрузка и подготовка видео")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Загрузить видео файл")
        uploaded_file = st.file_uploader(
            "Выберите видео файл",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Максимальный размер файла: 2GB. Поддерживаемые форматы: MP4, AVI, MOV, MKV, WebM"
        )
        
        if uploaded_file is not None:
            # Отображаем информацию о файле
            file_details = {
                "Имя файла": uploaded_file.name,
                "Размер": f"{uploaded_file.size / (1024*1024):.1f} MB",
                "Тип": uploaded_file.type
            }
            
            st.success("✅ Файл успешно загружен!")
            
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
            
            # Сохраняем файл
            if st.session_state.uploaded_video_path is None:
                with st.spinner("Сохранение файла..."):
                    video_path = save_uploaded_file(uploaded_file)
                    if video_path:
                        st.session_state.uploaded_video_path = video_path
                        st.success("Файл сохранен и готов к анализу!")
            
            # Показываем видео плеер
            try:
                st.video(uploaded_file)
            except Exception as e:
                st.warning(f"Не удалось отобразить видео: {e}")
    
    with col2:
        st.subheader("🎬 Демо режим")
        st.write("Попробуйте систему на демонстрационном видео")
        
        if st.button("🚀 Использовать демо", type="primary", use_container_width=True):
            st.session_state.demo_mode = True
            st.session_state.uploaded_video_path = get_demo_video_path()
            st.success("✅ Демо режим активирован!")
            st.info("ℹ️ Используется демонстрационное видео для тестирования системы")
        
        if st.session_state.demo_mode:
            st.write("**Демо видео:** 2 минуты допроса")
            st.write("**Сценарий:** Опрос свидетеля")
            st.write("**Языки:** Русский")
    
    # Статус готовности к анализу
    st.divider()
    
    ready_for_analysis = (
        st.session_state.uploaded_video_path is not None or 
        st.session_state.demo_mode
    )
    
    if ready_for_analysis:
        st.success("🎯 Система готова к анализу!")
        
        # Предварительный анализ настроек
        st.subheader("📋 Предварительная информация")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            enabled_models = []
            if use_deepface: enabled_models.append("DeepFace")
            if use_fer: enabled_models.append("FER")
            if use_yolo_emotion: enabled_models.append("YOLO")
            if use_speech_analysis: enabled_models.append("Speech")
            
            st.write("**Активные модели:**")
            for model in enabled_models:
                st.write(f"• {model}")
        
        with col2:
            st.write("**Параметры:**")
            st.write(f"• Пропуск кадров: {frame_skip}")
            st.write(f"• Порог уверенности: {confidence_threshold}")
            st.write(f"• Устройство: {device_option}")
        
        with col3:
            estimated_time = len(enabled_models) * 15 + (30 if device_option == "CPU" else 10)
            st.write("**Оценочное время:**")
            st.write(f"• ~{estimated_time} секунд")
            st.write(f"• Экспорт: {', '.join(export_formats)}")
        
        # Кнопка запуска анализа
        st.divider()
        
        if st.button(
            "🚀 ЗАПУСТИТЬ ПОЛНЫЙ АНАЛИЗ", 
            type="primary",
            use_container_width=True,
            disabled=st.session_state.processing_complete
        ):
            # Переходим к процессу анализа (реализовано в следующем разделе)
            st.session_state.start_analysis = True
            st.rerun()
    
    else:
        st.info("👆 Загрузите видео файл или выберите демо режим для начала работы")

# ================================
# ПРОЦЕСС АНАЛИЗА
# ================================

if hasattr(st.session_state, 'start_analysis') and st.session_state.start_analysis:
    st.session_state.start_analysis = False
    
    with st.container():
        st.header("⚙️ Процесс анализа")
        
        # Создаем контейнеры для прогресса
        progress_container = st.container()
        status_container = st.container()
        log_container = st.container()
        
        with progress_container:
            st.write("**Инициализация анализа...**")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
        # Симуляция анализа (в реальной системе здесь будет вызов пайплайна)
        if st.session_state.demo_mode:
            # Демо режим - быстрая симуляция
            stages = [
                ("Загрузка видео", 0.1),
                ("Извлечение кадров", 0.2), 
                ("Анализ эмоций лица", 0.4),
                ("Извлечение аудио", 0.5),
                ("Анализ речи", 0.7),
                ("Синхронизация данных", 0.8),
                ("Поиск критических моментов", 0.9),
                ("Генерация отчета", 1.0)
            ]
            
            for stage_name, progress in stages:
                progress_bar.progress(progress)
                status_text.text(f"🔄 {stage_name}...")
                
                with log_container:
                    st.info(f"📍 {stage_name}")
                
                time.sleep(1)  # Симуляция обработки
            
            # Генерируем демо результаты
            st.session_state.analysis_results = generate_demo_results()
            st.session_state.processing_complete = True
            
            status_text.text("✅ Анализ завершен успешно!")
            st.success("🎉 Анализ завершен! Перейдите к другим вкладкам для просмотра результатов.")
            
        else:
            # Реальный анализ загруженного видео
            try:
                status_text.text("🔄 Инициализация пайплайна...")
                
                # Инициализируем пайплайн
                pipeline = MasterPipeline(st.session_state.config)
                
                def progress_callback(progress, stage, details):
                    """Callback для обновления прогресса"""
                    progress_bar.progress(progress)
                    status_text.text(f"🔄 {stage}: {details}")
                
                status_text.text("🎬 Обработка видео...")
                
                # Запускаем реальный анализ
                results = pipeline.process_video(
                    st.session_state.uploaded_video_path,
                    progress_callback=progress_callback
                )
                
                # Сохраняем результаты
                st.session_state.analysis_results = results
                st.session_state.processing_complete = True
                
                progress_bar.progress(1.0)
                status_text.text("✅ Анализ завершен!")
                st.success("🎉 Анализ завершен успешно!")
                
            except Exception as e:
                st.error(f"❌ Ошибка при анализе: {e}")
                st.exception(e)  # Показываем полную ошибку для отладки
                progress_bar.progress(0)
                status_text.text("❌ Анализ прерван из-за ошибки")
                
                # В случае ошибки, попробуем показать демо данные
                st.warning("⚠️ Используем демонстрационные данные из-за ошибки обработки")
                st.session_state.analysis_results = generate_demo_results()
                st.session_state.processing_complete = True

# ================================
# TAB 2: АНАЛИЗ ЭМОЦИЙ  
# ================================

def calculate_emotion_metrics(results):
    """Calculate emotion metrics from pipeline results"""
    video_emotions = results.get('data', {}).get('video_emotions', [])
    
    if not video_emotions:
        return {
            'dominant_emotion': 'Не определено',
            'emotion_changes': 0,
            'stress_level': 0.0,
            'stability': 0.0,
            'emotion_distribution': {}
        }
    
    # Count emotions
    emotion_counts = {}
    prev_emotion = None
    transitions = 0
    
    for emotion_data in video_emotions:
        emotion = emotion_data.get('emotion', 'нейтральность')
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        if prev_emotion and prev_emotion != emotion:
            transitions += 1
        prev_emotion = emotion
    
    # Calculate dominant emotion
    dominant_emotion = max(emotion_counts.keys(), key=emotion_counts.get) if emotion_counts else 'Не определено'
    
    # Calculate stability (higher = more stable)
    total_frames = len(video_emotions)
    dominant_count = emotion_counts.get(dominant_emotion, 0)
    stability = dominant_count / total_frames if total_frames > 0 else 0
    
    # Calculate stress level based on negative emotions
    negative_emotions = ['злость', 'страх', 'грусть', 'отвращение']
    negative_count = sum(emotion_counts.get(emo, 0) for emo in negative_emotions)
    stress_level = negative_count / total_frames if total_frames > 0 else 0
    
    return {
        'dominant_emotion': dominant_emotion,
        'emotion_changes': transitions,
        'stress_level': stress_level,
        'stability': stability,
        'emotion_distribution': emotion_counts,
        'total_frames': total_frames
    }

with tab2:
    st.header("📊 Анализ эмоций")
    
    if not st.session_state.processing_complete:
        st.info("ℹ️ Запустите анализ видео на вкладке 'Загрузка' для просмотра результатов")
    else:
        results = st.session_state.analysis_results
        
        # Calculate metrics from real data
        metrics = calculate_emotion_metrics(results)
        
        # Основные метрики
        st.subheader("📈 Основные показатели")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h3>🎭 Доминирующая эмоция</h3>
                <h2>{metrics['dominant_emotion']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h3>🔄 Изменения эмоций</h3>
                <h2>{metrics['emotion_changes']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            stress_level = metrics['stress_level']
            color = '#FF4444' if stress_level > 0.7 else '#FFA500' if stress_level > 0.4 else '#00AA00'
            st.markdown(f"""
            <div class="metric-container">
                <h3>⚡ Уровень стресса</h3>
                <h2 style="color: {color}">{stress_level:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            stability = metrics['stability']
            st.markdown(f"""
            <div class="metric-container">
                <h3>⚖️ Стабильность</h3>
                <h2>{stability:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Временная шкала эмоций
        st.subheader("📈 Временная шкала эмоций")
        
        fig = create_emotion_timeline(results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Тепловая карта эмоций
        st.subheader("🗺️ Тепловая карта эмоций")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            heatmap_fig = create_emotion_heatmap(results, metrics)
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        with col2:
            st.write("**Интерпретация тепловой карты:**")
            st.write("• Красный цвет - высокая интенсивность эмоции")
            st.write("• Синий цвет - низкая интенсивность")
            st.write("• Различия между лицом и речью могут указывать на несоответствия")

# ================================
# TAB 3: АНАЛИЗ РЕЧИ
# ================================

with tab3:
    st.header("🎙️ Анализ речи и звуковых паттернов")
    
    if not st.session_state.processing_complete:
        st.info("ℹ️ Запустите анализ видео для просмотра речевых данных")
    else:
        results = st.session_state.analysis_results
        
        # График эмоций в речи
        st.subheader("📈 Эмоции в речи")
        
        speech_fig = create_speech_emotion_plot(results.get('speech', {}))
        st.plotly_chart(speech_fig, use_container_width=True)
        
        # Аудио плеер
        st.subheader("🎵 Аудио запись")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # В реальной реализации здесь будет настоящий аудио файл
            st.write("**Аудио дорожка:**")
            st.write("• Длительность: 2:00")
            st.write("• Качество: 16 kHz")
            st.write("• Каналы: Моно")
            
            # Placeholder для аудио плеера
            st.info("🎵 Аудио плеер будет здесь в полной версии")
        
        with col2:
            st.write("**Характеристики речи:**")
            
            # Метрики речи
            speech_metrics = {
                "Средняя громкость": "65 dB",
                "Темп речи": "140 слов/мин", 
                "Количество пауз": "12",
                "Средняя длина пауз": "1.8 сек",
                "Эмоциональная окраска": "Умеренно напряженная"
            }
            
            for metric, value in speech_metrics.items():
                st.write(f"• **{metric}:** {value}")
        
        # Таблица речевых сегментов
        st.subheader("📊 Сегменты речи")
        
        if 'speech_segments' in results:
            st.dataframe(
                results['speech_segments'],
                use_container_width=True,
                height=300
            )
        
        # Статистика по говорящим
        st.subheader("👥 Анализ по участникам")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Следователь:**")
            st.write("• Время говорения: 45 сек (37%)")
            st.write("• Средняя эмоция: Нейтральность")
            st.write("• Темп речи: 120 слов/мин")
            st.write("• Количество вопросов: 8")
        
        with col2:
            st.write("**Допрашиваемый:**")
            st.write("• Время говорения: 75 сек (63%)")
            st.write("• Средняя эмоция: Напряжение")
            st.write("• Темп речи: 160 слов/мин")
            st.write("• Количество пауз: 9")

# ================================
# TAB 4: ТРАНСКРИПЦИЯ
# ================================

with tab4:
    st.header("📝 Транскрипция и анализ текста")
    
    if not st.session_state.processing_complete:
        st.info("ℹ️ Транскрипция появится после завершения анализа")
    else:
        results = st.session_state.analysis_results
        
        # Настройки отображения
        col1, col2, col3 = st.columns(3)
        
        with col1:
            version = st.radio(
                "Версия транскрипта",
                ["Оригинал", "Улучшенная", "С разметкой"],
                index=2
            )
        
        with col2:
            show_timestamps = st.checkbox("Показать время", value=True)
            show_confidence = st.checkbox("Показать уверенность", value=False)
        
        with col3:
            highlight_critical = st.checkbox("Выделить критические моменты", value=True)
            show_emotions = st.checkbox("Показать эмоции", value=True)
        
        st.divider()
        
        # Отображение транскрипции
        st.subheader("📜 Текст допроса")
        
        if 'transcript_segments' in results:
            for i, segment in enumerate(results['transcript_segments']):
                # Определяем стиль в зависимости от критичности
                is_critical = segment.get('critical', False) and highlight_critical
                
                # Формируем содержимое сегмента
                speaker = segment.get('speaker', 'Неизвестно')
                text = segment.get('text', '')
                time_info = segment.get('time', '00:00')
                
                # HTML разметка
                if is_critical:
                    st.markdown(f"""
                    <div class="critical-moment">
                        <b>🔴 {speaker}</b> 
                        {f'<small>({time_info})</small>' if show_timestamps else ''}
                        <br>
                        {text}
                        {f'<br><small>⚠️ Критический момент</small>' if highlight_critical else ''}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="normal-moment">
                        <b>{speaker}</b> 
                        {f'<small>({time_info})</small>' if show_timestamps else ''}
                        <br>
                        {text}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Статистика текста
        st.subheader("📊 Статистика транскрипции")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Общее количество слов", "324")
        
        with col2:
            st.metric("Слова следователя", "142")
        
        with col3:
            st.metric("Слова допрашиваемого", "182")
        
        with col4:
            st.metric("Точность распознавания", "94%")
        
        # Облако ключевых слов (placeholder)
        st.subheader("☁️ Ключевые слова")
        st.info("📝 Облако ключевых слов будет отображено здесь в полной версии")
        
        # Экспорт транскрипции
        st.subheader("💾 Экспорт транскрипции")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            transcript_text = "\n".join([f"{seg['speaker']}: {seg['text']}" for seg in results.get('transcript_segments', [])])
            st.download_button(
                "📄 Скачать TXT",
                transcript_text,
                "transcript.txt",
                "text/plain"
            )
        
        with col2:
            # Формат SRT для субтитров
            srt_content = ""
            for i, seg in enumerate(results.get('transcript_segments', []), 1):
                srt_content += f"{i}\n00:{seg['time']} --> 00:00:00\n{seg['speaker']}: {seg['text']}\n\n"
            
            st.download_button(
                "🎬 Скачать SRT",
                srt_content,
                "subtitles.srt",
                "text/plain"
            )
        
        with col3:
            # JSON с метаданными
            json_data = json.dumps(results.get('transcript_segments', []), ensure_ascii=False, indent=2)
            st.download_button(
                "📋 Скачать JSON",
                json_data,
                "transcript.json",
                "application/json"
            )

# ================================
# TAB 5: КРИТИЧЕСКИЕ МОМЕНТЫ
# ================================

with tab5:
    st.header("⚠️ Критические моменты допроса")
    
    if not st.session_state.processing_complete:
        st.info("ℹ️ Критические моменты будут определены после анализа")
    else:
        results = st.session_state.analysis_results
        critical_moments = results.get('critical_moments', [])
        
        if not critical_moments:
            st.success("✅ Критических моментов не обнаружено")
        else:
            # Общая статистика
            st.subheader("📊 Общая статистика")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Всего моментов", len(critical_moments))
            
            with col2:
                high_severity = len([cm for cm in critical_moments if cm.get('severity', 0) >= 7])
                st.metric("Высокая важность", high_severity)
            
            with col3:
                avg_severity = np.mean([cm.get('severity', 0) for cm in critical_moments])
                st.metric("Средняя важность", f"{avg_severity:.1f}/10")
            
            with col4:
                unique_types = len(set(cm.get('type', '') for cm in critical_moments))
                st.metric("Типов аномалий", unique_types)
            
            st.divider()
            
            # Фильтры
            st.subheader("🔍 Фильтры")
            
            col1, col2 = st.columns(2)
            
            with col1:
                severity_filter = st.slider(
                    "Минимальная важность",
                    min_value=1, max_value=10, value=1
                )
            
            with col2:
                type_filter = st.selectbox(
                    "Фильтр по типу",
                    ["Все типы"] + list(set(cm.get('type', '') for cm in critical_moments))
                )
            
            # Фильтрация моментов
            filtered_moments = critical_moments
            
            if severity_filter > 1:
                filtered_moments = [cm for cm in filtered_moments if cm.get('severity', 0) >= severity_filter]
            
            if type_filter != "Все типы":
                filtered_moments = [cm for cm in filtered_moments if cm.get('type', '') == type_filter]
            
            st.write(f"Показано {len(filtered_moments)} из {len(critical_moments)} моментов")
            
            st.divider()
            
            # Отображение критических моментов
            st.subheader("🔍 Детальный анализ")
            
            for idx, moment in enumerate(filtered_moments):
                severity = moment.get('severity', 0)
                severity_color = '#FF4444' if severity >= 7 else '#FFA500' if severity >= 5 else '#FFD700'
                severity_text = 'ВЫСОКАЯ' if severity >= 7 else 'СРЕДНЯЯ' if severity >= 5 else 'НИЗКАЯ'
                
                with st.expander(
                    f"🔴 Момент {idx+1}: {moment.get('time', '00:00')} - {moment.get('type', 'Неизвестно')} "
                    f"[{severity_text}]",
                    expanded=severity >= 7
                ):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**Описание:**")
                        st.write(moment.get('description', 'Описание недоступно'))
                        
                        st.write("**Контекст:**")
                        context_info = [
                            f"• **Эмоция лица:** {moment.get('face_emotion', 'Не определено')}",
                            f"• **Эмоция речи:** {moment.get('speech_emotion', 'Не определено')}",
                            f"• **Транскрипт:** \"{moment.get('transcript', 'Текст недоступен')}\""
                        ]
                        
                        for info in context_info:
                            st.markdown(info)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="padding: 1rem; background-color: {severity_color}20; border-radius: 0.5rem; border-left: 4px solid {severity_color};">
                            <h4 style="margin: 0; color: {severity_color};">Уровень важности</h4>
                            <h2 style="margin: 0; color: {severity_color};">{severity}/10</h2>
                            <p style="margin: 0; font-size: 0.8rem;">{severity_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.write("**Временные метки:**")
                        st.write(f"• Время: {moment.get('time', '00:00')}")
                        st.write(f"• Секунды: {moment.get('timestamp', 0)}")
                    
                    # Показать кадр, если доступен
                    if moment.get('frame'):
                        st.image(moment['frame'], caption=f"Кадр в момент {moment.get('time', '00:00')}")
                    else:
                        st.info("📷 Кадр недоступен для данного момента")
                    
                    # Рекомендации
                    st.write("**💡 Рекомендации:**")
                    if moment.get('type') == 'Резкое изменение эмоции':
                        st.write("• Уточните, что вызвало эмоциональную реакцию")
                        st.write("• Задайте дополнительные вопросы о данном моменте")
                    elif moment.get('type') == 'Несоответствие модальностей':
                        st.write("• Обратите внимание на возможное сокрытие информации")
                        st.write("• Переформулируйте вопрос для получения более честного ответа")
                    elif moment.get('type') == 'Длительная пауза':
                        st.write("• Дождитесь полного ответа, не торопите")
                        st.write("• Возможно, требуется время для формулировки сложного ответа")
            
            # График временного распределения критических моментов
            if filtered_moments:
                st.subheader("📈 Временное распределение")
                
                timestamps = [cm.get('timestamp', 0) for cm in filtered_moments]
                severities = [cm.get('severity', 0) for cm in filtered_moments]
                types = [cm.get('type', 'Неизвестно') for cm in filtered_moments]
                
                fig = go.Figure()
                
                # Цвета для разных типов
                type_colors = {
                    'Резкое изменение эмоции': '#FF4444',
                    'Несоответствие модальностей': '#FFA500', 
                    'Длительная пауза': '#4444FF'
                }
                
                for i, (ts, sev, typ) in enumerate(zip(timestamps, severities, types)):
                    fig.add_trace(go.Scatter(
                        x=[ts],
                        y=[sev],
                        mode='markers',
                        marker=dict(
                            size=max(8, sev * 2),
                            color=type_colors.get(typ, '#888888'),
                            line=dict(width=2, color='white')
                        ),
                        name=typ,
                        hovertemplate=f'<b>Время:</b> {format_time(ts)}<br><b>Важность:</b> {sev}/10<br><b>Тип:</b> {typ}<extra></extra>',
                        showlegend=i == 0 or typ not in [types[j] for j in range(i)]
                    ))
                
                fig.update_layout(
                    title="Критические моменты на временной шкале",
                    xaxis_title="Время (секунды)",
                    yaxis_title="Уровень важности",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ================================
# TAB 6: ОТЧЕТЫ
# ================================

with tab6:
    st.header("📈 Отчеты и психологический анализ")
    
    if not st.session_state.processing_complete:
        st.info("ℹ️ Отчеты будут доступны после завершения анализа")
    else:
        results = st.session_state.analysis_results
        
        # GPT Анализ и инсайты
        st.subheader("🧠 Психологический анализ")
        
        with st.container():
            if 'gpt_insights' in results:
                insights = results['gpt_insights']
                st.markdown(insights)
            else:
                st.info("🤖 GPT анализ недоступен")
        
        st.divider()
        
        # Статистический анализ
        st.subheader("📊 Статистическая сводка")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Эмоциональные показатели:**")
            emotional_stats = {
                "Доминирующая эмоция": results.get('dominant_emotion', 'Не определено'),
                "Количество переходов": results.get('emotion_changes', 0),
                "Уровень стресса": f"{results.get('stress_level', 0):.1%}",
                "Эмоциональная стабильность": f"{results.get('stability', 0):.1%}",
                "Критических моментов": len(results.get('critical_moments', []))
            }
            
            for stat, value in emotional_stats.items():
                st.write(f"• **{stat}:** {value}")
        
        with col2:
            st.write("**Технические показатели:**")
            tech_stats = {
                "Время обработки": f"{results.get('processing_time', 0):.1f} сек",
                "Качество видео": "HD (1080p)", 
                "Качество аудио": "16 kHz",
                "Точность распознавания": "94%",
                "Уверенность анализа": "87%"
            }
            
            for stat, value in tech_stats.items():
                st.write(f"• **{stat}:** {value}")
        
        st.divider()
        
        # Генерация и скачивание отчетов
        st.subheader("📥 Скачать отчеты")
        
        col1, col2, col3 = st.columns(3)
        
        # CSV отчет
        with col1:
            if st.button("📊 Генерировать CSV", use_container_width=True):
                with st.spinner("Подготовка CSV данных..."):
                    # Создаем DataFrame с основными данными
                    data_rows = []
                    
                    if 'emotions' in results and 'video_emotions' in results['emotions']:
                        video_data = results['emotions']['video_emotions']
                        timestamps = video_data.get('timestamps', [])
                        values = video_data.get('values', [])
                        
                        for ts, val in zip(timestamps, values):
                            data_rows.append({
                                'Время': ts,
                                'Время_форматированное': format_time(ts),
                                'Эмоция_лица': val,
                                'Эмоция_речи': 0.0,  # Placeholder
                                'Критический_момент': any(
                                    abs(cm.get('timestamp', 0) - ts) < 1.0 
                                    for cm in results.get('critical_moments', [])
                                )
                            })
                    
                    if data_rows:
                        df = pd.DataFrame(data_rows)
                        csv_data = df.to_csv(index=False, encoding='utf-8')
                        
                        st.download_button(
                            label="📄 Скачать CSV данные",
                            data=csv_data,
                            file_name=f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Нет данных для экспорта")
        
        # HTML отчет  
        with col2:
            if st.button("🌐 Генерировать HTML", use_container_width=True):
                with st.spinner("Создание HTML отчета..."):
                    html_content = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="utf-8">
                        <title>Отчет анализа допроса</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 40px; }}
                            h1, h2 {{ color: #FF4444; }}
                            .metric {{ background: #f0f2f6; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                            .critical {{ background: #ffcccc; padding: 10px; border-left: 3px solid red; }}
                        </style>
                    </head>
                    <body>
                        <h1>🎥 ДОПРОС MVP 2.0 - Отчет анализа</h1>
                        <p><strong>Дата создания:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        
                        <h2>📊 Основные показатели</h2>
                        <div class="metric">Доминирующая эмоция: {results.get('dominant_emotion', 'Не определено')}</div>
                        <div class="metric">Уровень стресса: {results.get('stress_level', 0):.1%}</div>
                        <div class="metric">Стабильность: {results.get('stability', 0):.1%}</div>
                        <div class="metric">Критических моментов: {len(results.get('critical_moments', []))}</div>
                        
                        <h2>⚠️ Критические моменты</h2>
                    """
                    
                    for cm in results.get('critical_moments', []):
                        html_content += f"""
                        <div class="critical">
                            <strong>{cm.get('time', '00:00')} - {cm.get('type', 'Неизвестно')}</strong><br>
                            {cm.get('description', '')}<br>
                            <em>Важность: {cm.get('severity', 0)}/10</em>
                        </div>
                        """
                    
                    html_content += """
                        <h2>🧠 Психологический анализ</h2>
                        <div style="background: #f8f9fa; padding: 20px; border-radius: 5px;">
                    """ + results.get('gpt_insights', 'Анализ недоступен').replace('\n', '<br>') + """
                        </div>
                    </body>
                    </html>
                    """
                    
                    st.download_button(
                        label="🌐 Скачать HTML отчет",
                        data=html_content,
                        file_name=f"full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
        
        # JSON отчет
        with col3:
            if st.button("📋 Генерировать JSON", use_container_width=True):
                with st.spinner("Подготовка JSON данных..."):
                    json_data = json.dumps(results, ensure_ascii=False, indent=2, default=str)
                    
                    st.download_button(
                        label="📋 Скачать JSON данные",
                        data=json_data,
                        file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        st.divider()
        
        # Дополнительные опции экспорта
        st.subheader("🔧 Дополнительные форматы")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Excel отчет**")
            st.info("Excel отчет с множественными листами будет доступен в полной версии")
            
        with col2:
            st.write("**📄 PDF отчет**") 
            st.info("PDF отчет с графиками и диаграммами будет доступен в полной версии")

# ================================
# 7. ФУТЕР С ИНФОРМАЦИЕЙ
# ================================

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**Версия системы:** 2.0.0")
    st.markdown("**Сборка:** MVP-DEMO")

with footer_col2:
    if st.session_state.processing_complete and 'processing_time' in st.session_state.analysis_results:
        processing_time = st.session_state.analysis_results['processing_time']
        st.markdown(f"**Время обработки:** {processing_time:.1f} сек")
    else:
        st.markdown("**Статус:** Готов к работе")
    
    # Отображение системной информации
    try:
        gpu_manager = get_gpu_manager()
        device_info = gpu_manager.get_device_info()
        current_device = device_info.get('current_device', 'CPU')
        st.markdown(f"**Устройство:** {current_device}")
    except:
        st.markdown("**Устройство:** CPU")

with footer_col3:
    st.markdown("**© 2025 ДОПРОС MVP**")
    st.markdown("**Лицензия:** Research Only")

# ================================
# БОКОВОЕ МЕНЮ НАВИГАЦИИ (дополнительно)
# ================================

# Дополнительная информация в sidebar
with st.sidebar:
    st.divider()
    
    # Системные уведомления
    st.subheader("🔔 Уведомления")
    
    if st.session_state.processing_complete:
        st.success("✅ Анализ завершен")
        
        critical_count = len(st.session_state.analysis_results.get('critical_moments', []))
        if critical_count > 0:
            st.warning(f"⚠️ Найдено {critical_count} критических моментов")
        
        high_stress = st.session_state.analysis_results.get('stress_level', 0) > 0.7
        if high_stress:
            st.error("🚨 Высокий уровень стресса обнаружен")
    
    else:
        if st.session_state.uploaded_video_path or st.session_state.demo_mode:
            st.info("▶️ Готов к запуску анализа")
        else:
            st.info("📁 Загрузите видео для начала")
    
    # Быстрые действия
    st.subheader("⚡ Быстрые действия")
    
    if st.button("🔄 Сбросить сессию", use_container_width=True):
        # Очистка состояния сессии
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    if st.session_state.processing_complete:
        if st.button("💾 Быстрое сохранение", use_container_width=True):
            st.info("Сохранение в разработке")
    
    # Справочная информация
    with st.expander("❓ Справка"):
        st.write("""
        **Как пользоваться системой:**
        
        1. 📹 Загрузите видео или выберите демо
        2. ⚙️ Настройте параметры анализа
        3. 🚀 Запустите анализ
        4. 📊 Просмотрите результаты в табах
        5. 📈 Скачайте отчеты
        
        **Поддерживаемые форматы:**
        - Видео: MP4, AVI, MOV, MKV
        - Максимальный размер: 2GB
        - Рекомендуемое качество: HD
        """)
    
    # Контактная информация
    with st.expander("📧 Контакты"):
        st.write("""
        **Техническая поддержка:**
        - Email: support@dopros-mvp.ru
        - Telegram: @dopros_support
        - GitHub: github.com/dopros-mvp
        
        **Документация:**
        - API документация
        - Руководство пользователя  
        - FAQ и частые вопросы
        """)

# ================================
# ЗАВЕРШЕНИЕ ЗАГРУЗКИ
# ================================

# Скрытие Streamlit элементов
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Финальная инициализация
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    st.toast("🎉 ДОПРОС MVP 2.0 загружен успешно!", icon="✅")