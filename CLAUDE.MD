# ДОПРОС MVP 2.0 - Полное руководство для Claude Code

## 🎯 Критические требования
- **НЕ УПРОЩАТЬ** функционал - все возможности должны быть реализованы
- **Коммитить** каждый успешный этап
- **Использовать** Ultralytics YOLO11 согласно документации
- **Сохранять** все промежуточные результаты
- **Полная** обратная совместимость с существующими данными

## 📁 Структура проекта (НЕ ИЗМЕНЯТЬ)

```
dopros-mvp-v2/
├── requirements.txt
├── config.yaml
├── .env
├── .gitignore
├── README.md
├── main.py                      # Главный Streamlit интерфейс
├── core/
│   ├── __init__.py
│   ├── pipeline.py             # Главный пайплайн обработки
│   ├── video_processor.py      # Извлечение и обработка кадров
│   ├── emotion_analyzer.py     # Мультимодальный анализ эмоций
│   ├── audio_processor.py      # Обработка аудио и речи
│   ├── report_generator.py     # Генерация отчетов
│   ├── data_aggregator.py      # Объединение всех данных
│   └── critical_analyzer.py    # Анализ критических моментов
├── models/
│   ├── __init__.py
│   ├── yolo_manager.py         # Управление YOLO11 моделями
│   ├── deepface_wrapper.py     # Обертка DeepFace с fallback
│   ├── speech_analyzer.py      # Анализ эмоций в речи
│   ├── fer_analyzer.py         # FER как альтернатива
│   └── simple_analyzer.py      # Простой анализ (последний fallback)
├── utils/
│   ├── __init__.py
│   ├── gpu_manager.py          # Управление GPU/CPU
│   ├── translation.py          # Переводы эмоций RU/EN
│   ├── file_manager.py         # Работа с файлами и хешами
│   ├── time_sync.py            # Синхронизация временных меток
│   ├── validators.py           # Валидация всех данных
│   └── logger.py               # Централизованное логирование
├── integrations/
│   ├── __init__.py
│   ├── openai_client.py        # OpenAI API (Whisper, GPT-4)
│   ├── ffmpeg_wrapper.py       # Работа с FFmpeg
│   └── api_manager.py          # Управление всеми API
├── storage/
│   ├── videos/                 # Исходные видео
│   ├── audio/                  # Извлеченное аудио
│   ├── frames/                 # Все кадры
│   ├── faces/                  # Вырезанные лица
│   ├── faces_yolo11/           # Кадры с людьми YOLO11
│   ├── results/                # CSV результаты
│   ├── reports/                # Финальные отчеты
│   └── cache/                  # Кэш промежуточных данных
├── assets/
│   ├── demo/
│   │   └── input_video.mp4    # Демо видео
│   ├── models/
│   │   ├── yolo11n.pt         # YOLO11 nano
│   │   ├── yolo11n-face.pt    # YOLO11 для лиц
│   │   └── haarcascade_frontalface.xml
│   └── styles/
│       └── custom.css          # Стили для Streamlit
└── tests/
    ├── __init__.py
    ├── test_compatibility.py   # Тест совместимости
    ├── test_pipeline.py        # Тест пайплайна
    ├── test_models.py          # Тест моделей
    └── fixtures/               # Тестовые данные
```

## 📝 ПРОМПТЫ ДЛЯ РЕАЛИЗАЦИИ

### ПРОМПТ 1: Базовая инициализация проекта
```
Создай полную структуру проекта ДОПРОС MVP 2.0 для анализа видео допросов.

requirements.txt должен содержать ВСЕ зависимости:
- streamlit
- ultralytics
- opencv-python
- opencv-contrib-python
- deepface
- tensorflow>=2
- torch>=2.1.0
- torchvision>=0.16.0
- pandas>=2.1.0
- numpy>=1.24.0
- plotly>=5.18.0
- openai>=1.10.0
- librosa>=0.10.1
- soundfile>=0.12.1
- pyyaml>=6.0.1
- python-dotenv>=1.0.0
- tqdm>=4.66.0
- Pillow>=10.2.0
- fer>=22.5.0
- scikit-learn>=1.3.0
- scipy>=1.11.0
- matplotlib>=3.8.0
- seaborn>=0.13.0
- python-ffmpeg>=2.0.0
- colorama>=0.4.6
- psutil>=5.9.0

.gitignore должен исключать:
- .env
- *.pt (модели)
- storage/* (кроме .gitkeep)
- __pycache__
- *.pyc
- .DS_Store
- *.log

Создай config.yaml с ПОЛНЫМИ настройками (не упрощай):
```yaml
app:
  name: "ДОПРОС MVP 2.0"
  version: "2.0.0"
  language: "ru"
  debug: false

processing:
  video:
    frame_skip: 15  # Каждый N-й кадр
    min_frames: 10  # Минимум кадров для анализа
    output_format: "jpg"
    quality: 95
    max_resolution: [1920, 1080]
    
  audio:
    sample_rate: 16000
    channels: 1
    format: "wav"
    segment_length: 30  # секунд для анализа эмоций
    
  models:
    yolo:
      version: "11"
      models:
        detection: "yolo11n.pt"
        face: "yolo11n-face.pt"
        emotion: "yolo11n-emotion.pt"
      confidence_threshold: 0.5
      iou_threshold: 0.45
      max_detections: 100
      device: "auto"  # auto, cpu, cuda, mps
      force_cpu: false
      
    deepface:
      enabled: true
      backend: "opencv"
      enforce_detection: false
      detector_backend: "opencv"
      model: "Emotion"
      
    fer:
      enabled: true
      mtcnn: true
      
    speech:
      model: "base"
      emotions: ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
      
    whisper:
      model: "whisper-1"
      language: "ru"
      temperature: 0
      
    gpt:
      model: "gpt-4-turbo-preview"
      max_tokens: 4000
      temperature: 0.3

storage:
  base_dir: "storage"
  videos_dir: "storage/videos"
  audio_dir: "storage/audio"
  frames_dir: "storage/frames"
  faces_dir: "storage/faces"
  faces_yolo11_dir: "storage/faces_yolo11"
  results_dir: "storage/results"
  reports_dir: "storage/reports"
  cache_dir: "storage/cache"
  
  cleanup:
    enabled: true
    max_age_days: 7
    max_size_gb: 50

analysis:
  emotion_categories:
    basic: ["злость", "отвращение", "страх", "счастье", "грусть", "удивление", "нейтральность"]
    specialized: ["спокойствие", "дискомфорт", "напряжение_бровей", "напряжение_глаз", "напряжение_рта"]
    
  critical_detection:
    rapid_change_threshold: 3  # изменений подряд
    confidence_threshold: 0.7
    time_window: 5  # секунд
    
  report:
    formats: ["csv", "json", "html", "pdf"]
    include_graphs: true
    include_timeline: true
    include_transitions: true
    include_statistics: true
    include_insights: true

api:
  openai:
    key: "${OPENAI_API_KEY}"
    timeout: 60
    max_retries: 3
    
  rate_limits:
    whisper: 50  # запросов в минуту
    gpt: 60  # запросов в минуту

performance:
  max_workers: 4
  batch_size: 32
  memory_limit_gb: 8
  gpu_memory_fraction: 0.8
  enable_caching: true
  
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "dopros.log"
  max_size_mb: 100
  backup_count: 5
```

Создай .env.example:
```
OPENAI_API_KEY=your_api_key_here
HUGGINGFACE_TOKEN=optional_token
LOG_LEVEL=INFO
```

Создай все папки с .gitkeep файлами в storage подпапках.
```

### ПРОМПТ 2: GPU/CPU менеджер с полной обработкой ошибок
```
Создай utils/gpu_manager.py для управления устройствами с ПОЛНЫМ функционалом:

1. Класс GPUManager должен:
   - Автоматически определять доступность CUDA, MPS (Mac), DirectML (Windows)
   - Проверять версию CUDA и совместимость с PyTorch
   - Тестировать проблему torchvision::nms для YOLO
   - Мониторить использование памяти GPU
   - Автоматически переключаться на CPU при ошибках
   - Поддерживать множественные GPU

2. Методы:
   - setup_device(force_cpu=False, preferred_gpu=0) - настройка устройства
   - test_cuda_nms() - специальный тест для YOLO NMS
   - get_device_info() - полная информация об устройстве
   - get_memory_info() - использование памяти
   - clear_cache() - очистка кэша GPU
   - benchmark_device() - тест производительности
   - auto_select_device() - автовыбор лучшего устройства

3. Контекстные менеджеры:
   - with gpu_manager.device_context('cuda:1') - временное переключение
   - with gpu_manager.memory_limit(4096) - ограничение памяти

4. Обработка специфичных ошибок YOLO:
   - RuntimeError с torchvision::nms
   - CUDA out of memory
   - CUDNN errors
   
5. Логирование всех операций и автоматических переключений

6. Singleton паттерн для единого экземпляра

Пример использования:
```python
gpu_manager = GPUManager.get_instance()
device = gpu_manager.setup_device()
if gpu_manager.test_cuda_nms():
    print("CUDA NMS работает")
info = gpu_manager.get_device_info()
```
```

### ПРОМПТ 3: YOLO11 менеджер согласно документации Ultralytics
```
Создай models/yolo_manager.py используя документацию Ultralytics для YOLO11:

1. Класс YOLO11Manager с полной поддержкой всех возможностей:
   - Загрузка моделей yolo11n/s/m/l/x.pt
   - Специальные модели для лиц и эмоций
   - Поддержка всех режимов: detect, segment, classify, pose
   - Автоматическое скачивание моделей если отсутствуют

2. Метод extract_frames_with_people():
   - Использовать model.predict() согласно документации
   - Параметры: conf, iou, imgsz, max_det, classes
   - Фильтрация только класса person (class_id=0)
   - Сохранение с временными метками в имени
   - Возврат результатов с boxes, masks, keypoints если есть
   
3. Метод detect_faces():
   - Использовать специальную модель для лиц
   - Вырезание и сохранение отдельных лиц
   - Tracking лиц между кадрами (model.track)
   - Сохранение метаданных (confidence, bbox, track_id)

4. Пакетная обработка:
   - model.predict(source=image_list, stream=True)
   - Оптимизация памяти через генераторы
   - Прогресс-бар с tqdm

5. Обработка ошибок CUDA:
   ```python
   try:
       results = model.predict(frame, device='cuda')
   except RuntimeError as e:
       if "torchvision::nms" in str(e):
           model.to('cpu')
           results = model.predict(frame, device='cpu')
   ```

6. Экспорт результатов:
   - results.save() для сохранения аннотированных изображений
   - results.to_df() для pandas DataFrame
   - results.to_json() для JSON

7. Настройки согласно документации:
   ```python
   model = YOLO('yolo11n.pt')
   model.info()  # информация о модели
   model.fuse()  # оптимизация для inference
   ```

8. Валидация модели:
   - model.val() для проверки точности
   - Метрики: mAP, precision, recall

Используй официальный API из документации, НЕ придумывай свои методы.
```

### ПРОМПТ 4: Мультимодальный анализатор эмоций с приоритетами
```
Создай core/emotion_analyzer.py с ПОЛНОЙ системой анализа эмоций:

1. Класс MultiModalEmotionAnalyzer с приоритетной системой:
   - Приоритет 1: DeepFace (если доступен)
   - Приоритет 2: FER (Face Emotion Recognition)
   - Приоритет 3: YOLO11 с моделью эмоций
   - Приоритет 4: OpenCV + простой классификатор
   - Автоматический выбор доступного метода

2. Метод analyze_frame() должен возвращать:
   ```python
   {
       'emotion': 'злость',
       'emotion_en': 'angry',
       'confidence': 0.85,
       'all_emotions': {
           'злость': 0.85,
           'страх': 0.10,
           'нейтральность': 0.05
       },
       'method': 'deepface',
       'face_bbox': [x1, y1, x2, y2],
       'timestamp': 12.5,
       'frame_number': 375
   }
   ```

3. DeepFace обертка:
   - try/except для импорта
   - Все 7 базовых эмоций
   - enforce_detection=False для стабильности
   - Обработка множественных лиц

4. FER анализатор:
   - MTCNN для детекции лиц
   - top_emotion и все scores
   - Обработка видео потока

5. YOLO11 эмоции:
   - Специализированные эмоции для допросов
   - Микровыражения (брови, глаза, рот)
   - Уровень комфорта/дискомфорта

6. Простой анализатор (fallback):
   - Haar Cascade для лиц
   - Анализ по цветовым гистограммам
   - Базовые геометрические признаки

7. Временной анализ:
   - detect_emotion_transitions() - переходы между эмоциями
   - find_rapid_changes() - резкие изменения
   - calculate_emotion_stability() - стабильность состояния

8. Batch processing:
   - Обработка списка кадров
   - Параллелизация через ThreadPoolExecutor
   - Прогресс с tqdm

9. Кэширование результатов:
   - Сохранение в storage/cache
   - Проверка по хешу изображения
   - Автоочистка старого кэша

10. Синхронизация с видео временем:
    - Точные временные метки
    - Интерполяция между кадрами
    - Сглаживание резких переходов
```

### ПРОМПТ 5: Полный переводчик эмоций с описаниями
```
Создай utils/translation.py с ПОЛНОЙ системой перевода:

1. Словари для ВСЕХ типов эмоций:
```python
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
```

2. Цветовая схема для КАЖДОЙ эмоции:
```python
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
```

3. Детальные описания для психологического анализа:
```python
EMOTION_DESCRIPTIONS = {
    'злость': {
        'short': 'Состояние гнева или раздражения',
        'detailed': 'Эмоциональное состояние, характеризующееся повышенной агрессивностью, раздражением. Может указывать на фрустрацию или защитную реакцию.',
        'indicators': ['сжатые губы', 'нахмуренные брови', 'напряженная челюсть'],
        'interrogation_meaning': 'Возможная защитная реакция на неудобные вопросы'
    },
    # ... для каждой эмоции
}
```

4. Функции перевода с контекстом:
   - translate_emotion(emotion, source='deepface', target_lang='ru')
   - translate_batch(emotions_list, source='deepface')
   - get_emotion_color(emotion, intensity=1.0) - с интенсивностью
   - get_emotion_description(emotion, detail_level='short')
   - get_interrogation_interpretation(emotion) - для допросов

5. Обратный перевод:
   - reverse_translate(russian_emotion)
   - get_original_emotion(translated, source)

6. Форматирование для отчетов:
   - format_for_report(emotion_data)
   - create_emotion_legend() - легенда для графиков
   - get_emotion_emoji(emotion) - эмодзи для UI
```

### ПРОМПТ 6: Продвинутый анализатор эмоций в речи
```
Создай models/speech_analyzer.py для ПОЛНОГО анализа речи:

1. Класс AdvancedSpeechEmotionAnalyzer:
   - Использование librosa для извлечения признаков
   - MFCC, спектральные характеристики, pitch, energy
   - Предобученные модели если доступны
   - Fallback на правила если модели недоступны

2. Извлечение аудио признаков:
```python
def extract_features(audio_segment):
    features = {
        'mfcc': librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13),
        'spectral_centroid': librosa.feature.spectral_centroid(y=audio, sr=sr),
        'zero_crossing_rate': librosa.feature.zero_crossing_rate(audio),
        'spectral_rolloff': librosa.feature.spectral_rolloff(y=audio, sr=sr),
        'pitch': librosa.piptrack(y=audio, sr=sr),
        'energy': np.sum(audio**2) / len(audio),
        'tempo': librosa.beat.tempo(y=audio, sr=sr)
    }
    return features
```

3. Сегментация аудио:
   - Разбиение на временные окна (1-3 секунды)
   - Детекция пауз и silence removal
   - Voice Activity Detection (VAD)
   - Синхронизация с видео кадрами

4. Классификация эмоций:
   - 7 базовых эмоций + дополнительные
   - Вероятности для каждой эмоции
   - Сглаживание между сегментами

5. Временная привязка:
```python
{
    'start_time': 10.5,
    'end_time': 13.2,
    'emotion': 'страх',
    'confidence': 0.78,
    'all_emotions': {...},
    'audio_features': {...},
    'intensity': 0.65,
    'voice_quality': 'trembling'
}
```

6. Анализ просодики:
   - Изменения тона голоса
   - Скорость речи
   - Громкость и интенсивность
   - Дрожание голоса

7. Детекция аномалий:
   - Резкие изменения характеристик
   - Неестественные паузы
   - Признаки стресса в голосе

8. Корреляция с видео:
   - Сопоставление эмоций видео/аудио
   - Выявление расхождений
   - Общий коэффициент согласованности

9. Экспорт результатов:
   - CSV с временными метками
   - JSON для визуализации
   - Аудио сегменты с метками
```

### ПРОМПТ 7: Полный процессор аудио с транскрипцией
```
Создай core/audio_processor.py со ВСЕМИ возможностями:

1. Класс CompleteAudioProcessor:
   - Извлечение аудио из видео (FFmpeg)
   - Транскрипция (Whisper API)
   - Улучшение текста (GPT-4)
   - Разделение говорящих
   - Анализ содержания

2. FFmpeg интеграция:
```python
def extract_audio(video_path):
    # Поиск FFmpeg в разных местах
    ffmpeg_paths = [
        'ffmpeg',
        shutil.which('ffmpeg'),
        '/usr/local/bin/ffmpeg',
        'C:\\ffmpeg\\bin\\ffmpeg.exe',
        os.path.expanduser('~/ffmpeg/ffmpeg')
    ]
    
    # Команда с оптимальными параметрами
    cmd = [
        ffmpeg_path,
        '-i', video_path,
        '-vn',  # без видео
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-y',  # перезапись
        output_path
    ]
```

3. Whisper транскрипция с таймкодами:
```python
def transcribe_with_timestamps(audio_path):
    response = openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="verbose_json",
        timestamp_granularities=["segment", "word"],
        language="ru"
    )
    # Возвращает сегменты с временными метками
```

4. GPT-4 обработка транскрипта:
   - enhance_transcript() - исправление ошибок, пунктуация
   - identify_speakers() - определение говорящих
   - extract_key_moments() - ключевые моменты
   - analyze_content() - психологический анализ

5. Разделение на реплики:
```python
def split_dialogue(transcript):
    return {
        'interrogator': [
            {'text': '...', 'start': 0.0, 'end': 5.2},
            ...
        ],
        'subject': [
            {'text': '...', 'start': 5.3, 'end': 12.1},
            ...
        ]
    }
```

6. Анализ содержания допроса:
   - Выявление противоречий
   - Поиск ключевых фраз
   - Определение тем разговора
   - Оценка кооперативности

7. Синхронизация с видео:
   - Привязка реплик к кадрам
   - Соответствие эмоций и слов
   - Временная шкала событий

8. Форматы экспорта:
   - SRT субтитры с эмоциями
   - Текстовая стенограмма
   - JSON с полной разметкой
   - Аналитический отчет
```

### ПРОМПТ 8: OpenAI интеграция с полным функционалом
```
Создай integrations/openai_client.py со ВСЕМИ возможностями API:

1. Класс OpenAIIntegration с полной обработкой:
   - Загрузка ключа из .env
   - Retry логика с exponential backoff
   - Rate limiting
   - Токен counting и оптимизация
   - Кэширование результатов

2. Whisper методы:
```python
def transcribe_audio(self, audio_path, **kwargs):
    # С повторами при ошибках
    for attempt in range(self.max_retries):
        try:
            with open(audio_path, 'rb') as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ru",
                    response_format="verbose_json",
                    timestamp_granularities=["segment", "word"],
                    **kwargs
                )
            return self._process_whisper_response(response)
        except openai.RateLimitError:
            time.sleep(2 ** attempt)
```

3. GPT-4 анализ с промптами для допросов:
```python
PROMPTS = {
    'enhance_transcript': '''
        Исправь ошибки транскрипции допроса.
        Добавь пунктуацию и форматирование.
        Раздели на реплики допрашивающего и допрашиваемого.
        Сохрани все слова и смысл без изменений.
    ''',
    
    'analyze_psychology': '''
        Проанализируй психологическое состояние допрашиваемого.
        Найди признаки стресса, обмана, уклонения.
        Определи критические моменты.
        Оцени достоверность показаний.
    ''',
    
    'extract_insights': '''
        На основе эмоций и транскрипта определи:
        1. Критические эпизоды эмоциональной дестабилизации
        2. Признаки психологического давления
        3. Попытки уклонения от ответов
        4. Противоречия в показаниях
        5. Рекомендации для дальнейшего допроса
    '''
}
```

4. Структурированный вывод:
```python
def get_structured_analysis(self, data):
    response = self.client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[...],
        response_format={"type": "json_object"},
        functions=[{
            "name": "analyze_interrogation",
            "parameters": {
                "type": "object",
                "properties": {
                    "critical_moments": {"type": "array"},
                    "deception_indicators": {"type": "array"},
                    "stress_level": {"type": "number"},
                    "cooperation_level": {"type": "number"},
                    "key_findings": {"type": "array"},
                    "recommendations": {"type": "array"}
                }
            }
        }]
    )
```

5. Токен оптимизация:
   - Подсчет токенов перед отправкой
   - Разбиение больших текстов
   - Суммаризация длинных транскриптов

6. Кэширование:
   - Хеширование запросов
   - Сохранение в storage/cache
   - TTL для разных типов запросов

7. Мониторинг использования:
   - Логирование всех запросов
   - Подсчет затрат
   - Алерты при превышении лимитов
```

### ПРОМПТ 9: Полный генератор отчетов
```
Создай core/report_generator.py со ВСЕМИ типами отчетов:

1. Класс ComprehensiveReportGenerator:
   - 5 форматов экспорта (CSV, JSON, HTML, PDF, DOCX)
   - Интерактивные графики Plotly
   - Временные шкалы
   - Статистические сводки
   - Психологические инсайты

2. CSV отчеты (множественные):
```python
def generate_csv_reports(self):
    reports = {
        'emotions_timeline.csv': self._create_emotions_timeline(),
        'transitions.csv': self._create_transitions_report(),
        'speech_emotions.csv': self._create_speech_report(),
        'critical_moments.csv': self._create_critical_moments(),
        'correlation_matrix.csv': self._create_correlation_matrix(),
        'summary_statistics.csv': self._create_statistics()
    }
```

3. Интерактивная временная шкала:
```python
def create_interactive_timeline(self):
    fig = go.Figure()
    
    # Видео эмоции
    fig.add_trace(go.Scatter(
        x=video_timestamps,
        y=video_emotions,
        mode='lines+markers',
        name='Эмоции лица',
        line=dict(color='blue', width=2),
        hovertemplate='%{y}<br>Время: %{x:.1f}с<br>Уверенность: %{customdata:.2f}'
    ))
    
    # Аудио эмоции
    fig.add_trace(go.Scatter(
        x=audio_timestamps,
        y=audio_emotions,
        mode='lines+markers',
        name='Эмоции речи',
        line=dict(color='red', width=2)
    ))
    
    # Критические моменты
    for moment in critical_moments:
        fig.add_vline(x=moment['time'], line_dash="dash", 
                     annotation_text=moment['description'])
```

4. Тепловая карта эмоций:
```python
def create_emotion_heatmap(self):
    # Матрица эмоций по времени
    fig = px.imshow(
        emotion_matrix,
        labels=dict(x="Время", y="Эмоция", color="Интенсивность"),
        x=time_labels,
        y=emotion_labels,
        color_continuous_scale="RdBu_r"
    )
```

5. Статистический анализ:
   - Распределение эмоций (pie charts)
   - Корреляции видео/аудио
   - Частота переходов (transition matrix)
   - Временные паттерны

6. HTML отчет с полной визуализацией:
```python
def generate_html_report(self):
    template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ДОПРОС MVP - Отчет</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>{custom_css}</style>
    </head>
    <body>
        <h1>Комплексный анализ допроса</h1>
        
        <section id="summary">
            <h2>Краткая сводка</h2>
            {summary_content}
        </section>
        
        <section id="timeline">
            <h2>Временная шкала эмоций</h2>
            {timeline_plot}
        </section>
        
        <section id="critical">
            <h2>Критические моменты</h2>
            {critical_moments_table}
        </section>
        
        <section id="transcript">
            <h2>Транскрипция с анализом</h2>
            {annotated_transcript}
        </section>
        
        <section id="insights">
            <h2>Психологические инсайты</h2>
            {gpt_insights}
        </section>
    </body>
    </html>
    '''
```

7. PDF генерация:
   - Использование reportlab или weasyprint
   - Включение всех графиков
   - Форматированный текст
   - Оглавление и нумерация

8. Экспорт для дальнейшего анализа:
   - JSON со всеми метаданными
   - Pickle для Python объектов
   - Excel с множественными листами
```

### ПРОМПТ 10: Главный пайплайн с полной координацией
```
Создай core/pipeline.py - центральный координатор ВСЕХ процессов:

1. Класс MasterPipeline с полным функционалом:
```python
class MasterPipeline:
    def __init__(self, config_path='config.yaml'):
        self.config = self._load_config(config_path)
        self.gpu_manager = GPUManager.get_instance()
        self.yolo_manager = YOLO11Manager(self.config)
        self.emotion_analyzer = MultiModalEmotionAnalyzer(self.config)
        self.audio_processor = CompleteAudioProcessor(self.config)
        self.speech_analyzer = AdvancedSpeechEmotionAnalyzer(self.config)
        self.data_aggregator = DataAggregator(self.config)
        self.report_generator = ComprehensiveReportGenerator(self.config)
        self.openai_client = OpenAIIntegration(self.config)
        
    def process_video(self, video_path, options=None):
        # Полный пайплайн обработки
```

2. Этапы обработки с checkpoints:
```python
stages = [
    ('validate_input', self._validate_input),
    ('setup_session', self._setup_session),
    ('extract_frames', self._extract_frames),
    ('detect_faces', self._detect_faces),
    ('analyze_emotions', self._analyze_emotions),
    ('extract_audio', self._extract_audio),
    ('transcribe_audio', self._transcribe_audio),
    ('analyze_speech', self._analyze_speech),
    ('synchronize_data', self._synchronize_data),
    ('detect_critical', self._detect_critical_moments),
    ('generate_insights', self._generate_insights),
    ('create_reports', self._create_reports)
]

for stage_name, stage_func in stages:
    try:
        checkpoint_file = f"cache/{session_id}/{stage_name}.pkl"
        if os.path.exists(checkpoint_file) and not force_reprocess:
            results[stage_name] = load_checkpoint(checkpoint_file)
        else:
            results[stage_name] = stage_func(previous_results)
            save_checkpoint(results[stage_name], checkpoint_file)
        
        if progress_callback:
            progress_callback(stage_index / total_stages, stage_name)
            
    except Exception as e:
        self._handle_stage_error(stage_name, e, results)
```

3. Параллельная обработка где возможно:
```python
with ThreadPoolExecutor(max_workers=self.config['performance']['max_workers']) as executor:
    futures = {
        'video': executor.submit(self._process_video_stream, video_path),
        'audio': executor.submit(self._process_audio_stream, video_path)
    }
    
    for name, future in futures.items():
        results[name] = future.result(timeout=timeout)
```

4. Управление памятью:
```python
def _manage_memory(self):
    if psutil.virtual_memory().percent > 80:
        self._cleanup_cache()
        gc.collect()
        if self.gpu_manager.device != 'cpu':
            torch.cuda.empty_cache()
```

5. Обработка ошибок с продолжением:
```python
def _handle_stage_error(self, stage, error, results):
    self.logger.error(f"Ошибка на этапе {stage}: {error}")
    
    if stage in self.critical_stages:
        raise PipelineError(f"Критическая ошибка: {stage}")
    
    # Попытка использовать fallback
    if stage in self.fallback_methods:
        results[stage] = self.fallback_methods[stage](results)
    else:
        results[stage] = None
        results['errors'].append({
            'stage': stage,
            'error': str(error),
            'timestamp': datetime.now()
        })
```

6. Кэширование и восстановление:
   - Сохранение после каждого этапа
   - Возможность продолжить прерванную обработку
   - Очистка старых кэшей

7. Метрики и мониторинг:
```python
metrics = {
    'processing_time': {},
    'memory_usage': {},
    'gpu_usage': {},
    'api_calls': {},
    'errors': []
}
```

8. Финализация и экспорт:
```python
def finalize(self, results):
    # Генерация всех отчетов
    # Архивация результатов
    # Очистка временных файлов
    # Логирование статистики
    return {
        'session_id': session_id,
        'video_path': video_path,
        'reports': report_paths,
        'statistics': final_stats,
        'processing_time': total_time
    }
```
```

### ПРОМПТ 11: Продвинутый агрегатор данных
```
Создай core/data_aggregator.py для ПОЛНОГО объединения данных:

1. Класс DataAggregator со всеми методами синхронизации:
```python
class DataAggregator:
    def aggregate_all_sources(self, video_emotions, audio_emotions, 
                            speech_transcript, face_tracking):
        # Объединение всех источников данных
        
    def create_unified_timeline(self):
        # Единая временная шкала всех событий
        
    def detect_anomalies(self):
        # Поиск несоответствий между модальностями
```

2. Синхронизация по времени:
```python
def synchronize_timestamps(self, sources):
    # Находим общий временной диапазон
    min_time = min(s['start_time'] for s in sources)
    max_time = max(s['end_time'] for s in sources)
    
    # Создаем единую временную сетку
    time_grid = np.arange(min_time, max_time, self.time_resolution)
    
    # Интерполируем все данные на эту сетку
    synchronized = {}
    for source_name, source_data in sources.items():
        synchronized[source_name] = self._interpolate_to_grid(
            source_data, time_grid
        )
```

3. Поиск критических моментов:
```python
def detect_critical_moments(self):
    critical = []
    
    # Резкие изменения эмоций
    emotion_changes = self._detect_rapid_emotion_changes()
    
    # Несоответствие видео/аудио
    mismatches = self._detect_modality_mismatches()
    
    # Длительные паузы в речи
    speech_pauses = self._detect_speech_anomalies()
    
    # Микровыражения
    micro_expressions = self._detect_micro_expressions()
    
    # Ранжирование по важности
    critical = self._rank_by_importance(
        emotion_changes + mismatches + speech_pauses + micro_expressions
    )
```

4. Корреляционный анализ:
```python
def calculate_correlations(self):
    # Корреляция между эмоциями лица и речи
    face_speech_corr = np.corrcoef(
        face_emotions_encoded, 
        speech_emotions_encoded
    )
    
    # Временные паттерны
    patterns = self._find_temporal_patterns()
    
    # Кластеризация эмоциональных состояний
    clusters = self._cluster_emotional_states()
```

5. Статистические метрики:
```python
def calculate_comprehensive_statistics(self):
    stats = {
        'emotion_distribution': self._calculate_distribution(),
        'transition_matrix': self._create_transition_matrix(),
        'stability_index': self._calculate_stability(),
        'stress_level': self._estimate_stress_level(),
        'deception_probability': self._calculate_deception_score(),
        'cooperation_index': self._assess_cooperation()
    }
```

6. Создание аннотированной временной шкалы:
```python
def create_annotated_timeline(self):
    timeline = []
    
    for timestamp in self.unified_timeline:
        timeline.append({
            'time': timestamp,
            'face_emotion': self.get_face_emotion_at(timestamp),
            'speech_emotion': self.get_speech_emotion_at(timestamp),
            'transcript': self.get_transcript_at(timestamp),
            'critical_level': self.get_criticality_at(timestamp),
            'annotations': self.get_annotations_at(timestamp)
        })
```

7. Экспорт в различные форматы:
   - DataFrame для анализа
   - JSON для визуализации
   - SQL для базы данных
   - График для презентации
```

### ПРОМПТ 12: Полнофункциональный Streamlit интерфейс
```
Создай main.py - ПОЛНЫЙ интерфейс без упрощений:

1. Настройка страницы и стилизация:
```python
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
    .main {padding: 0rem 1rem;}
    .stProgress > div > div > div > div {background-color: #FF4444;}
    [data-testid="stSidebar"] {background-color: #f0f2f6;}
    .css-1d391kg {padding-top: 1rem;}
</style>
''', unsafe_allow_html=True)
```

2. Боковая панель с настройками:
```python
with st.sidebar:
    st.title("⚙️ Настройки анализа")
    
    # Выбор моделей
    st.subheader("🤖 Модели")
    use_deepface = st.checkbox("DeepFace", value=True)
    use_fer = st.checkbox("FER", value=True)
    use_yolo_emotion = st.checkbox("YOLO эмоции", value=True)
    
    # Параметры обработки
    st.subheader("📊 Параметры")
    frame_skip = st.slider("Пропуск кадров", 1, 30, 15)
    confidence_threshold = st.slider("Порог уверенности", 0.1, 1.0, 0.5)
    
    # GPU настройки
    st.subheader("💻 Устройство")
    device_option = st.radio(
        "Выбор устройства",
        ["Авто", "GPU", "CPU"],
        index=0
    )
    
    # Экспорт настройки
    st.subheader("📁 Экспорт")
    export_formats = st.multiselect(
        "Форматы отчетов",
        ["CSV", "JSON", "HTML", "PDF", "Excel"],
        default=["CSV", "HTML"]
    )
```

3. Главная область с табами:
```python
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📹 Загрузка", 
    "📊 Анализ эмоций", 
    "🎙️ Анализ речи",
    "📝 Транскрипция", 
    "⚠️ Критические моменты",
    "📈 Отчеты"
])

with tab1:
    # Загрузка видео
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Загрузить видео")
        uploaded_file = st.file_uploader(
            "Выберите видео файл",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Максимальный размер: 2GB"
        )
        
    with col2:
        st.subheader("🎬 Демо видео")
        if st.button("Использовать демо", type="primary"):
            # Загрузка демо видео
```

4. Процесс анализа с детальным прогрессом:
```python
if st.button("🚀 ЗАПУСТИТЬ АНАЛИЗ", type="primary"):
    # Создаем контейнеры для прогресса
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.container()
    
    # Запускаем пайплайн
    pipeline = MasterPipeline(config)
    
    def update_progress(progress, stage_name, details=""):
        progress_bar.progress(progress)
        status_text.text(f"Этап: {stage_name}")
        if details:
            with log_container:
                st.info(f"📍 {details}")
    
    # Обработка с callback
    results = pipeline.process_video(
        video_path,
        progress_callback=update_progress
    )
```

5. Интерактивная визуализация результатов:
```python
with tab2:  # Анализ эмоций
    st.subheader("🎭 Временная шкала эмоций")
    
    # Интерактивный график
    fig = create_emotion_timeline(results['emotions'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Статистика
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Доминирующая эмоция", results['dominant_emotion'])
    with col2:
        st.metric("Изменений эмоций", results['emotion_changes'])
    with col3:
        st.metric("Уровень стресса", f"{results['stress_level']:.1%}")
    with col4:
        st.metric("Стабильность", f"{results['stability']:.1%}")
    
    # Тепловая карта
    st.subheader("🗺️ Тепловая карта эмоций")
    heatmap_fig = create_emotion_heatmap(results['emotion_matrix'])
    st.plotly_chart(heatmap_fig, use_container_width=True)
```

6. Анализ речи с воспроизведением:
```python
with tab3:  # Анализ речи
    st.subheader("🎙️ Эмоции в речи")
    
    # График эмоций в речи
    speech_fig = create_speech_emotion_plot(results['speech_emotions'])
    st.plotly_chart(speech_fig, use_container_width=True)
    
    # Аудио плеер с метками
    st.audio(results['audio_path'])
    
    # Таблица сегментов
    st.dataframe(
        results['speech_segments'],
        use_container_width=True,
        height=300
    )
```

7. Транскрипция с подсветкой:
```python
with tab4:  # Транскрипция
    st.subheader("📝 Расшифровка допроса")
    
    # Выбор версии
    version = st.radio(
        "Версия транскрипта",
        ["Оригинал", "Улучшенная", "С разметкой"]
    )
    
    # Отображение с подсветкой критических моментов
    if version == "С разметкой":
        for segment in results['transcript_segments']:
            if segment['critical']:
                st.markdown(f"""
                <div style="background-color: #ffcccc; padding: 10px; 
                           border-left: 3px solid red;">
                    <b>{segment['speaker']}:</b> {segment['text']}
                    <br><small>⏱️ {segment['time']}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"**{segment['speaker']}:** {segment['text']}")
```

8. Критические моменты:
```python
with tab5:  # Критические моменты
    st.subheader("⚠️ Обнаруженные критические моменты")
    
    for idx, moment in enumerate(results['critical_moments']):
        with st.expander(f"🔴 Момент {idx+1}: {moment['time']}"):
            st.write(f"**Тип:** {moment['type']}")
            st.write(f"**Описание:** {moment['description']}")
            st.write(f"**Уровень критичности:** {moment['severity']}/10")
            
            # Показать кадр
            if moment['frame']:
                st.image(moment['frame'], caption=f"Кадр в {moment['time']}")
            
            # Контекст
            st.write("**Контекст:**")
            st.write(f"- Эмоция лица: {moment['face_emotion']}")
            st.write(f"- Эмоция речи: {moment['speech_emotion']}")
            st.write(f"- Транскрипт: {moment['transcript']}")
```

9. Генерация и скачивание отчетов:
```python
with tab6:  # Отчеты
    st.subheader("📊 Генерация отчетов")
    
    # GPT инсайты
    with st.spinner("Генерация психологических инсайтов..."):
        insights = results['gpt_insights']
        
    st.markdown("### 🧠 Психологический анализ")
    st.write(insights)
    
    # Кнопки скачивания
    st.markdown("### 📥 Скачать отчеты")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📄 CSV данные",
            data=results['csv_data'],
            file_name="analysis_data.csv",
            mime="text/csv"
        )
    
    with col2:
        st.download_button(
            label="📊 HTML отчет",
            data=results['html_report'],
            file_name="full_report.html",
            mime="text/html"
        )
    
    with col3:
        st.download_button(
            label="📑 PDF отчет",
            data=results['pdf_report'],
            file_name="interrogation_analysis.pdf",
            mime="application/pdf"
        )
```

10. Футер с информацией:
```python
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Версия:** 2.0.0")
    
with col2:
    if 'processing_time' in results:
        st.markdown(f"**Время обработки:** {results['processing_time']:.1f} сек")
        
with col3:
    st.markdown("**© 2025 ДОПРОС MVP**")
```
```

### ПРОМПТ 13: Система тестирования и валидации
```
Создай полную систему тестирования в tests/:

1. test_compatibility.py - проверка окружения:
   - Тест всех импортов
   - Проверка GPU/CUDA
   - Наличие FFmpeg
   - API ключи
   - Модели YOLO
   - Версии библиотек

2. test_pipeline.py - интеграционные тесты:
   - Обработка демо видео
   - Все этапы пайплайна
   - Fallback механизмы
   - Восстановление после ошибок

3. test_models.py - тесты моделей:
   - YOLO детекция
   - DeepFace анализ
   - FER работа
   - Speech анализ

4. fixtures/ - тестовые данные:
   - Короткое видео (10 сек)
   - Аудио сэмпл
   - Эталонные результаты

Запуск: pytest tests/ -v --cov=core --cov=models
```

### ПРОМПТ 14: Документация и README
```
Создай полную документацию:

1. README.md с:
   - Описанием проекта
   - Требованиями к системе
   - Инструкцией по установке
   - Примерами использования
   - Troubleshooting
   - Архитектурой системы

2. docs/API.md:
   - Описание всех классов
   - Методы и параметры
   - Примеры кода

3. docs/MODELS.md:
   - Используемые модели
   - Их характеристики
   - Как добавить новые

4. docs/DEPLOYMENT.md:
   - Развертывание на сервере
   - Docker контейнер
   - Оптимизация производительности
```

## 🚀 ПОСЛЕДОВАТЕЛЬНОСТЬ РЕАЛИЗАЦИИ

### День 1: Фундамент
1. Используй ПРОМПТ 1 - создай структуру
2. Используй ПРОМПТ 2 - GPU менеджер
3. Коммит: "Initial setup with GPU management"

### День 2: YOLO и детекция
1. Используй ПРОМПТ 3 - YOLO менеджер
2. Тестируй на демо видео
3. Коммит: "YOLO11 integration complete"

### День 3: Анализ эмоций
1. Используй ПРОМПТ 4 - мультимодальный анализатор
2. Используй ПРОМПТ 5 - переводчик
3. Коммит: "Emotion analysis with fallbacks"

### День 4: Аудио обработка
1. Используй ПРОМПТ 6 - анализ речи
2. Используй ПРОМПТ 7 - процессор аудио
3. Коммит: "Audio processing complete"

### День 5: Интеграции
1. Используй ПРОМПТ 8 - OpenAI клиент
2. Тестируй транскрипцию
3. Коммит: "OpenAI integration"

### День 6: Отчеты и агрегация
1. Используй ПРОМПТ 9 - генератор отчетов
2. Используй ПРОМПТ 11 - агрегатор
3. Коммит: "Report generation system"

### День 7: Пайплайн
1. Используй ПРОМПТ 10 - главный пайплайн
2. Интеграция всех компонентов
3. Коммит: "Master pipeline complete"

### День 8: Интерфейс
1. Используй ПРОМПТ 12 - Streamlit UI
2. Тестируй весь флоу
3. Коммит: "Full UI implementation"

### День 9: Тестирование
1. Используй ПРОМПТ 13 - тесты
2. Исправь найденные проблемы
3. Коммит: "Testing and fixes"

### День 10: Финализация
1. Используй ПРОМПТ 14 - документация
2. Оптимизация производительности
3. Коммит: "v2.0.0 release"

## ⚠️ КРИТИЧЕСКИ ВАЖНО

1. **НЕ УПРОЩАЙ** - реализуй ВСЕ функции из промптов
2. **ТЕСТИРУЙ** после каждого этапа
3. **КОММИТЬ** только рабочий код
4. **ЛОГИРУЙ** все операции
5. **FALLBACK** для каждого компонента
6. **КЭШИРУЙ** промежуточные результаты
7. **ДОКУМЕНТИРУЙ** каждый метод

## 📋 ДОПОЛНИТЕЛЬНЫЕ КРИТИЧЕСКИЕ ПРОМПТЫ

### ПРОМПТ 15: Критический анализатор для допросов
```
Создай core/critical_analyzer.py для выявления критических моментов допроса:

1. Класс InterrogationCriticalAnalyzer:
   - Специализированный анализ для правоохранительных органов
   - Выявление признаков обмана
   - Определение психологического давления
   - Поиск противоречий

2. Методы детекции обмана:
```python
def detect_deception_indicators(self):
    indicators = {
        'micro_expressions': self._detect_micro_expressions(),
        'voice_stress': self._analyze_voice_stress(),
        'eye_movement': self._track_eye_patterns(),
        'body_language': self._analyze_body_language(),
        'verbal_cues': self._detect_verbal_indicators(),
        'response_latency': self._measure_response_time(),
        'emotional_leakage': self._detect_emotional_leakage()
    }
    
    # Взвешенная оценка вероятности обмана
    deception_score = self._calculate_weighted_score(indicators)
```

3. Анализ противоречий:
```python
def find_contradictions(self, transcript_segments):
    contradictions = []
    
    # Семантический анализ утверждений
    statements = self._extract_factual_statements(transcript_segments)
    
    # Поиск логических противоречий
    for i, stmt1 in enumerate(statements):
        for stmt2 in statements[i+1:]:
            if self._are_contradictory(stmt1, stmt2):
                contradictions.append({
                    'statement1': stmt1,
                    'statement2': stmt2,
                    'times': [stmt1['time'], stmt2['time']],
                    'type': self._classify_contradiction(stmt1, stmt2),
                    'significance': self._assess_significance(stmt1, stmt2)
                })
```

4. Паттерны уклонения:
```python
def detect_evasion_patterns(self):
    patterns = {
        'question_avoidance': [],  # Не отвечает на прямой вопрос
        'topic_shifting': [],       # Смена темы
        'vague_responses': [],      # Расплывчатые ответы
        'excessive_details': [],    # Излишние детали (признак лжи)
        'memory_issues': [],        # "Не помню" в ключевых моментах
        'deflection': []           # Перенаправление вины
    }
```

5. Анализ стресса и давления:
```python
def analyze_stress_dynamics(self):
    # Базовый уровень стресса (первые минуты)
    baseline = self._establish_baseline()
    
    # Динамика стресса по времени
    stress_timeline = []
    for segment in self.timeline:
        stress_level = self._calculate_stress(segment, baseline)
        stress_timeline.append({
            'time': segment['time'],
            'stress': stress_level,
            'factors': self._identify_stress_factors(segment),
            'recovery_time': self._measure_recovery(segment)
        })
```

6. Ключевые вопросы и реакции:
```python
def map_questions_to_reactions(self, transcript, emotions):
    qa_map = []
    
    questions = self._extract_questions(transcript)
    for question in questions:
        reaction = {
            'question': question['text'],
            'question_time': question['time'],
            'emotion_before': self._get_emotion_at(question['time'] - 2),
            'emotion_during': self._get_emotion_at(question['time']),
            'emotion_after': self._get_emotion_at(question['time'] + 5),
            'response_time': self._measure_response_latency(question),
            'stress_spike': self._detect_stress_spike(question['time']),
            'significance': self._assess_question_impact(question)
        }
        qa_map.append(reaction)
```

7. Генерация следственных рекомендаций:
```python
def generate_investigative_recommendations(self):
    recommendations = {
        'high_priority_topics': self._identify_sensitive_topics(),
        'followup_questions': self._suggest_followup_questions(),
        'behavioral_flags': self._summarize_behavioral_indicators(),
        'credibility_assessment': self._assess_overall_credibility(),
        'suggested_verification': self._suggest_fact_checking(),
        'psychological_profile': self._create_psychological_profile()
    }
```
```

### ПРОМПТ 16: Менеджер сессий и восстановления
```
Создай utils/session_manager.py для управления сессиями обработки:

1. Класс SessionManager для полного контроля:
   - Создание уникальных сессий
   - Сохранение состояния
   - Восстановление после сбоев
   - История обработок

2. Управление сессиями:
```python
class SessionManager:
    def create_session(self, video_path, user_id=None):
        session_id = self._generate_session_id(video_path)
        session_data = {
            'id': session_id,
            'video_path': video_path,
            'video_hash': self._calculate_file_hash(video_path),
            'user_id': user_id,
            'created_at': datetime.now(),
            'status': 'initialized',
            'stages_completed': [],
            'stages_pending': [],
            'results': {},
            'errors': [],
            'config': self._capture_config()
        }
        
        # Создание директорий сессии
        self._create_session_directories(session_id)
        
        # Сохранение метаданных
        self._save_session_metadata(session_data)
        
        return session_id

3. Checkpoint система:
```python
def save_checkpoint(self, session_id, stage_name, data):
    checkpoint_path = f"storage/sessions/{session_id}/checkpoints/{stage_name}.pkl"
    
    checkpoint = {
        'stage': stage_name,
        'timestamp': datetime.now(),
        'data': data,
        'data_hash': hashlib.md5(pickle.dumps(data)).hexdigest(),
        'stage_version': self._get_stage_version(stage_name)
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    # Обновление метаданных сессии
    self._update_session_status(session_id, stage_name, 'completed')

4. Восстановление сессии:
```python
def restore_session(self, session_id):
    metadata = self._load_session_metadata(session_id)
    
    if metadata['status'] == 'completed':
        return self._load_completed_session(session_id)
    
    # Определение точки восстановления
    last_checkpoint = self._find_last_valid_checkpoint(session_id)
    
    # Восстановление данных
    restored_data = {}
    for stage in metadata['stages_completed']:
        checkpoint = self._load_checkpoint(session_id, stage)
        if self._validate_checkpoint(checkpoint):
            restored_data[stage] = checkpoint['data']
    
    return {
        'session_id': session_id,
        'resume_from': last_checkpoint,
        'completed_stages': metadata['stages_completed'],
        'pending_stages': metadata['stages_pending'],
        'data': restored_data
    }

5. Управление историей:
```python
def get_session_history(self, user_id=None, days=30):
    sessions = []
    
    for session_dir in os.listdir('storage/sessions'):
        metadata = self._load_session_metadata(session_dir)
        
        if user_id and metadata.get('user_id') != user_id:
            continue
            
        if (datetime.now() - metadata['created_at']).days > days:
            continue
            
        sessions.append({
            'id': metadata['id'],
            'video_name': os.path.basename(metadata['video_path']),
            'created': metadata['created_at'],
            'status': metadata['status'],
            'progress': len(metadata['stages_completed']) / len(self.all_stages),
            'has_results': 'results' in metadata and bool(metadata['results'])
        })
    
    return sorted(sessions, key=lambda x: x['created'], reverse=True)

6. Очистка и архивация:
```python
def cleanup_old_sessions(self, max_age_days=7, archive=True):
    for session_dir in os.listdir('storage/sessions'):
        metadata = self._load_session_metadata(session_dir)
        age = (datetime.now() - metadata['created_at']).days
        
        if age > max_age_days:
            if archive:
                self._archive_session(session_dir)
            self._delete_session(session_dir)
```
```

### ПРОМПТ 17: Продвинутый валидатор данных
```
Создай utils/validators.py с ПОЛНОЙ валидацией всех данных:

1. Класс ComprehensiveValidator:
```python
class ComprehensiveValidator:
    def __init__(self, config):
        self.config = config
        self.validation_rules = self._load_validation_rules()
        
    def validate_all(self):
        """Полная валидация перед запуском"""
        validations = [
            ('system', self.validate_system_requirements()),
            ('dependencies', self.validate_dependencies()),
            ('models', self.validate_models()),
            ('api_keys', self.validate_api_keys()),
            ('storage', self.validate_storage()),
            ('gpu', self.validate_gpu_availability())
        ]
        
        report = ValidationReport()
        for name, result in validations:
            report.add_validation(name, result)
        
        return report

2. Валидация видео файла:
```python
def validate_video_file(self, video_path):
    checks = {
        'exists': os.path.exists(video_path),
        'readable': os.access(video_path, os.R_OK),
        'format': self._check_video_format(video_path),
        'codec': self._check_video_codec(video_path),
        'duration': self._check_duration(video_path),
        'resolution': self._check_resolution(video_path),
        'framerate': self._check_framerate(video_path),
        'size': self._check_file_size(video_path),
        'corruption': self._check_corruption(video_path)
    }
    
    # Детальная информация о видео
    if all(checks.values()):
        video_info = self._extract_video_metadata(video_path)
        return ValidationResult(True, video_info)
    else:
        failures = [k for k, v in checks.items() if not v]
        return ValidationResult(False, {'failed_checks': failures})

3. Валидация моделей ML:
```python
def validate_models(self):
    model_checks = {}
    
    # YOLO модели
    for model_name in ['yolo11n.pt', 'yolo11n-face.pt']:
        model_path = f"assets/models/{model_name}"
        model_checks[model_name] = {
            'exists': os.path.exists(model_path),
            'size_valid': os.path.getsize(model_path) > 1000000,  # > 1MB
            'loadable': self._try_load_yolo(model_path),
            'version': self._check_yolo_version(model_path)
        }
    
    # DeepFace
    model_checks['deepface'] = {
        'importable': self._can_import('deepface'),
        'models_downloaded': self._check_deepface_models(),
        'backend_available': self._check_deepface_backend()
    }
    
    # FER
    model_checks['fer'] = {
        'importable': self._can_import('fer'),
        'mtcnn_available': self._check_mtcnn()
    }
    
    return model_checks

4. Валидация API:
```python
def validate_api_keys(self):
    api_checks = {}
    
    # OpenAI
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        api_checks['openai'] = {
            'key_present': True,
            'key_format': self._validate_openai_key_format(openai_key),
            'api_accessible': self._test_openai_api(openai_key),
            'models_available': self._check_openai_models(openai_key)
        }
    else:
        api_checks['openai'] = {'key_present': False}
    
    return api_checks

5. Валидация результатов:
```python
def validate_analysis_results(self, results):
    """Проверка корректности результатов анализа"""
    
    validations = {
        'structure': self._validate_result_structure(results),
        'timestamps': self._validate_timestamps(results),
        'emotions': self._validate_emotion_data(results),
        'transcript': self._validate_transcript(results),
        'consistency': self._check_data_consistency(results),
        'completeness': self._check_completeness(results)
    }
    
    # Проверка на аномалии
    anomalies = self._detect_result_anomalies(results)
    if anomalies:
        validations['anomalies'] = anomalies
    
    return validations

6. Система требований:
```python
def validate_system_requirements(self):
    requirements = {
        'python_version': sys.version_info >= (3, 8),
        'ram_available': psutil.virtual_memory().total >= 8 * 1024**3,  # 8GB
        'disk_space': shutil.disk_usage('/').free >= 10 * 1024**3,  # 10GB
        'ffmpeg_installed': shutil.which('ffmpeg') is not None,
        'internet_connection': self._check_internet(),
        'write_permissions': self._check_write_permissions()
    }
    
    return requirements
```
```

### ПРОМПТ 18: Логирование и мониторинг
```
Создай utils/logger.py для полного логирования:

1. Настройка централизованного логгера:
```python
import logging
import logging.handlers
from datetime import datetime
import json
import traceback

class CustomLogger:
    def __init__(self, name='DOPROS_MVP', config=None):
        self.logger = logging.getLogger(name)
        self.config = config or self._load_config()
        self._setup_handlers()
        self._setup_formatters()
        
    def _setup_handlers(self):
        # Файловый handler с ротацией
        file_handler = logging.handlers.RotatingFileHandler(
            'logs/dopros.log',
            maxBytes=100*1024*1024,  # 100MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # Console handler с цветами
        console_handler = ColoredConsoleHandler()
        
        # JSON handler для структурированных логов
        json_handler = JSONLogHandler('logs/dopros.json')
        
        # Error handler отдельно
        error_handler = logging.handlers.RotatingFileHandler(
            'logs/errors.log',
            maxBytes=50*1024*1024,
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)

2. Контекстный логгер:
```python
class ContextLogger:
    def __init__(self, logger, context):
        self.logger = logger
        self.context = context
        
    def log(self, level, message, **kwargs):
        extra = {'context': self.context, **kwargs}
        self.logger.log(level, message, extra=extra)
    
    def log_stage(self, stage_name, status, duration=None, **details):
        self.log(
            logging.INFO,
            f"Stage: {stage_name} - Status: {status}",
            stage=stage_name,
            status=status,
            duration=duration,
            details=details
        )

3. Метрики производительности:
```python
class PerformanceMonitor:
    def __init__(self, logger):
        self.logger = logger
        self.metrics = {}
        
    @contextmanager
    def measure(self, operation_name):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            memory_delta = psutil.Process().memory_info().rss - start_memory
            
            self.metrics[operation_name] = {
                'duration': duration,
                'memory_delta': memory_delta,
                'timestamp': datetime.now()
            }
            
            self.logger.info(
                f"Performance: {operation_name}",
                extra={
                    'operation': operation_name,
                    'duration_sec': duration,
                    'memory_mb': memory_delta / 1024 / 1024
                }
            )

4. Логирование ошибок с контекстом:
```python
def log_exception(self, exception, context=None, critical=False):
    exc_info = {
        'type': type(exception).__name__,
        'message': str(exception),
        'traceback': traceback.format_exc(),
        'context': context or {},
        'timestamp': datetime.now().isoformat()
    }
    
    if critical:
        self.logger.critical(
            f"CRITICAL ERROR: {exception}",
            extra=exc_info,
            exc_info=True
        )
        # Отправка уведомления
        self._send_critical_alert(exc_info)
    else:
        self.logger.error(
            f"Error: {exception}",
            extra=exc_info,
            exc_info=True
        )

5. Аудит действий:
```python
class AuditLogger:
    def log_action(self, action, user=None, details=None):
        audit_entry = {
            'action': action,
            'user': user or 'system',
            'timestamp': datetime.now().isoformat(),
            'details': details or {},
            'session_id': self.get_current_session_id()
        }
        
        # Запись в отдельный файл аудита
        with open('logs/audit.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(audit_entry, ensure_ascii=False) + '\n')
```
```

### ПРОМПТ 19: Docker и развертывание
```
Создай файлы для контейнеризации и развертывания:

1. Dockerfile с multi-stage build:
```dockerfile
# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements и установка
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Копирование системных библиотек из builder
COPY --from=builder /usr/lib /usr/lib
COPY --from=builder /usr/bin/ffmpeg /usr/bin/ffmpeg

# Копирование Python пакетов
COPY --from=builder /root/.local /root/.local

# Копирование приложения
COPY . .

# Создание директорий
RUN mkdir -p storage/videos storage/audio storage/frames \
    storage/results storage/reports storage/cache logs

# Переменные окружения
ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:$PATH
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Порт
EXPOSE 8501

# Запуск
CMD ["streamlit", "run", "main.py", "--server.maxUploadSize=2000"]
```

2. docker-compose.yml для полного стека:
```yaml
version: '3.8'

services:
  dopros-app:
    build: .
    container_name: dopros-mvp
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    volumes:
      - ./storage:/app/storage
      - ./logs:/app/logs
      - ./assets/models:/app/assets/models
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    container_name: dopros-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - dopros-app
    restart: unless-stopped

  redis:
    image: redis:alpine
    container_name: dopros-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  redis-data:
```

3. Kubernetes deployment:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dopros-mvp
spec:
  replicas: 2
  selector:
    matchLabels:
      app: dopros
  template:
    metadata:
      labels:
        app: dopros
    spec:
      containers:
      - name: dopros
        image: dopros-mvp:2.0.0
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: dopros-secrets
              key: openai-key
        volumeMounts:
        - name: storage
          mountPath: /app/storage
        - name: models
          mountPath: /app/assets/models
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: dopros-storage
      - name: models
        persistentVolumeClaim:
          claimName: dopros-models
```

4. CI/CD pipeline (.gitlab-ci.yml):
```yaml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA

test:
  stage: test
  image: python:3.10
  script:
    - pip install -r requirements.txt
    - pytest tests/ -v --cov=core --cov=models
    - python tests/test_compatibility.py
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $DOCKER_IMAGE .
    - docker push $DOCKER_IMAGE

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/dopros-mvp dopros=$DOCKER_IMAGE
    - kubectl rollout status deployment/dopros-mvp
  only:
    - main
```
```

### ПРОМПТ 20: Оптимизация производительности
```
Создай utils/optimizer.py для оптимизации производительности:

1. Класс PerformanceOptimizer:
```python
class PerformanceOptimizer:
    def __init__(self, config):
        self.config = config
        self.profiler = cProfile.Profile()
        self.memory_tracker = tracemalloc
        
    def optimize_video_processing(self, video_path):
        """Оптимизация обработки видео"""
        optimizations = {
            'frame_skip': self._calculate_optimal_frame_skip(video_path),
            'batch_size': self._calculate_optimal_batch_size(),
            'resolution': self._calculate_optimal_resolution(video_path),
            'codec': self._select_optimal_codec(),
            'parallel_workers': self._calculate_workers_count()
        }
        
        return optimizations
    
    def optimize_memory_usage(self):
        """Оптимизация использования памяти"""
        # Очистка кэшей
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Настройка лимитов
        resource.setrlimit(
            resource.RLIMIT_AS,
            (self.config['performance']['memory_limit_gb'] * 1024**3, -1)
        )
    
    def profile_operation(self, func, *args, **kwargs):
        """Профилирование операции"""
        self.profiler.enable()
        result = func(*args, **kwargs)
        self.profiler.disable()
        
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        
        return result, stats
    
    def optimize_model_inference(self, model):
        """Оптимизация инференса модели"""
        if hasattr(model, 'fuse'):
            model.fuse()  # Fusion для YOLO
        
        if hasattr(model, 'half') and torch.cuda.is_available():
            model.half()  # FP16 для GPU
        
        # Компиляция для скорости
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
        
        return model

2. Кэширование результатов:
```python
class SmartCache:
    def __init__(self, max_size_gb=5, ttl_hours=24):
        self.cache_dir = "storage/cache"
        self.max_size = max_size_gb * 1024**3
        self.ttl = ttl_hours * 3600
        self.index = self._load_index()
    
    def get_or_compute(self, key, compute_func, *args, **kwargs):
        cache_path = self._get_cache_path(key)
        
        if self._is_cached(key) and not self._is_expired(key):
            return self._load_from_cache(cache_path)
        
        result = compute_func(*args, **kwargs)
        self._save_to_cache(key, result)
        
        return result
    
    def _manage_size(self):
        """Управление размером кэша"""
        total_size = self._calculate_cache_size()
        
        if total_size > self.max_size:
            # Удаление старых записей
            entries = sorted(
                self.index.items(),
                key=lambda x: x[1]['accessed'],
                reverse=True
            )
            
            while total_size > self.max_size * 0.8:  # 80% threshold
                oldest_key = entries.pop()[0]
                self._remove_from_cache(oldest_key)
                total_size = self._calculate_cache_size()

3. Параллельная обработка:
```python
class ParallelProcessor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or cpu_count()
        
    def process_frames_parallel(self, frames, process_func):
        """Параллельная обработка кадров"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i, frame in enumerate(frames):
                future = executor.submit(process_func, frame, index=i)
                futures.append(future)
            
            results = []
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Frame processing error: {e}")
                    results.append(None)
        
        return results

4. Батчевая обработка:
```python
def create_batches(data, batch_size=32):
    """Создание батчей для обработки"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def process_in_batches(self, data, model, batch_size=32):
    """Обработка данных батчами"""
    results = []
    
    for batch in create_batches(data, batch_size):
        batch_tensor = torch.stack([self.preprocess(item) for item in batch])
        
        with torch.no_grad():
            if self.device != 'cpu':
                batch_tensor = batch_tensor.to(self.device)
            
            batch_results = model(batch_tensor)
            results.extend(batch_results.cpu().numpy())
    
    return results
```
```

### ПРОМПТ 21: Финальная интеграция и запуск
```
Создай скрипты для финальной сборки и запуска:

1. setup.py для установки:
```python
#!/usr/bin/env python
import os
import sys
import subprocess
from pathlib import Path

def setup_project():
    """Полная настройка проекта"""
    
    print("🚀 ДОПРОС MVP 2.0 - Установка")
    print("=" * 50)
    
    # 1. Проверка Python версии
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8+")
        sys.exit(1)
    
    # 2. Создание виртуального окружения
    print("📦 Создание виртуального окружения...")
    subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # 3. Активация и установка пакетов
    pip_path = "venv/Scripts/pip" if os.name == 'nt' else "venv/bin/pip"
    
    print("📚 Установка зависимостей...")
    subprocess.run([pip_path, "install", "--upgrade", "pip"])
    subprocess.run([pip_path, "install", "-r", "requirements.txt"])
    
    # 4. Создание директорий
    print("📁 Создание структуры папок...")
    dirs = [
        "storage/videos", "storage/audio", "storage/frames",
        "storage/faces", "storage/faces_yolo11", "storage/results",
        "storage/reports", "storage/cache", "storage/sessions",
        "logs", "assets/models"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 5. Скачивание моделей
    print("🤖 Загрузка моделей...")
    download_models()
    
    # 6. Проверка FFmpeg
    if not check_ffmpeg():
        print("⚠️  FFmpeg не найден. Установите: https://ffmpeg.org/download.html")
    
    # 7. Настройка переменных окружения
    if not os.path.exists('.env'):
        print("🔑 Создание .env файла...")
        with open('.env', 'w') as f:
            f.write("OPENAI_API_KEY=your_key_here\n")
            f.write("LOG_LEVEL=INFO\n")
        print("⚠️  Добавьте ваш OpenAI API ключ в .env файл")
    
    # 8. Тестирование
    print("🧪 Запуск тестов совместимости...")
    test_result = subprocess.run(
        [pip_path.replace('pip', 'python'), "tests/test_compatibility.py"],
        capture_output=True
    )
    
    if test_result.returncode == 0:
        print("✅ Установка завершена успешно!")
        print("\n📌 Для запуска используйте:")
        print("   python run.py")
        print("   или")
        print("   streamlit run main.py")
    else:
        print("⚠️  Некоторые тесты не прошли. Проверьте логи.")

def download_models():
    """Загрузка необходимых моделей"""
    import urllib.request
    
    models = {
        "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt",
        # Добавить остальные модели
    }
    
    for model_name, url in models.items():
        model_path = f"assets/models/{model_name}"
        if not os.path.exists(model_path):
            print(f"  Загрузка {model_name}...")
            urllib.request.urlretrieve(url, model_path)

def check_ffmpeg():
    """Проверка наличия FFmpeg"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False

if __name__ == "__main__":
    setup_project()
```

2. run.py для запуска:
```python
#!/usr/bin/env python
import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='ДОПРОС MVP 2.0')
    parser.add_argument('--mode', choices=['web', 'cli', 'api'], 
                       default='web', help='Режим запуска')
    parser.add_argument('--video', type=str, help='Путь к видео для CLI режима')
    parser.add_argument('--port', type=int, default=8501, help='Порт для веб-интерфейса')
    parser.add_argument('--debug', action='store_true', help='Режим отладки')
    parser.add_argument('--gpu', action='store_true', help='Принудительно использовать GPU')
    parser.add_argument('--test', action='store_true', help='Запустить тесты')
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
        return
    
    if args.mode == 'web':
        run_web(args.port, args.debug)
    elif args.mode == 'cli':
        if not args.video:
            print("❌ Укажите видео файл: --video path/to/video.mp4")
            sys.exit(1)
        run_cli(args.video, args.gpu)
    elif args.mode == 'api':
        run_api(args.port)

def run_web(port, debug):
    """Запуск веб-интерфейса"""
    print(f"🌐 Запуск веб-интерфейса на порту {port}...")
    
    env = os.environ.copy()
    if debug:
        env['LOG_LEVEL'] = 'DEBUG'
        env['STREAMLIT_THEME_BASE'] = 'light'
    
    cmd = [
        "streamlit", "run", "main.py",
        f"--server.port={port}",
        "--server.maxUploadSize=2000"
    ]
    
    if debug:
        cmd.append("--logger.level=debug")
    
    subprocess.run(cmd, env=env)

def run_cli(video_path, use_gpu):
    """Запуск в CLI режиме"""
    print(f"🎬 Обработка видео: {video_path}")
    
    from core.pipeline import MasterPipeline
    
    # Настройка конфигурации
    config = load_config()
    if not use_gpu:
        config['processing']['models']['yolo']['force_cpu'] = True
    
    # Запуск пайплайна
    pipeline = MasterPipeline(config)
    
    def progress_callback(progress, stage):
        bar_length = 50
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'\r[{bar}] {progress:.1%} - {stage}', end='')
    
    results = pipeline.process_video(video_path, progress_callback=progress_callback)
    
    print(f"\n✅ Обработка завершена!")
    print(f"📁 Результаты сохранены в: {results['reports']['main']}")

def run_api(port):
    """Запуск API сервера"""
    print(f"🔌 Запуск API на порту {port}...")
    # Здесь можно добавить FastAPI сервер
    pass

def run_tests():
    """Запуск тестов"""
    print("🧪 Запуск тестов...")
    subprocess.run(["pytest", "tests/", "-v", "--cov=core", "--cov=models"])

def load_config():
    import yaml
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    main()
```

3. Makefile для удобства:
```makefile
.PHONY: install run test clean docker help

help:
	@echo "ДОПРОС MVP 2.0 - Команды:"
	@echo "  make install  - Установка проекта"
	@echo "  make run      - Запуск веб-интерфейса"
	@echo "  make test     - Запуск тестов"
	@echo "  make docker   - Сборка Docker образа"
	@echo "  make clean    - Очистка кэша и логов"

install:
	python setup.py

run:
	python run.py --mode web

run-debug:
	python run.py --mode web --debug

test:
	python run.py --test

docker:
	docker build -t dopros-mvp:2.0.0 .

docker-run:
	docker-compose up -d

clean:
	rm -rf storage/cache/*
	rm -rf logs/*
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

benchmark:
	python tests/benchmark.py

format:
	black core/ models/ utils/ integrations/
	isort core/ models/ utils/ integrations/

lint:
	pylint core/ models/ utils/ integrations/
	mypy core/ models/ utils/ integrations/
```
```

## 🏁 ФИНАЛЬНЫЙ ЧЕКЛИСТ

### Перед началом разработки:
- [ ] Создать новую папку проекта
- [ ] Инициализировать git репозиторий
- [ ] Создать виртуальное окружение
- [ ] Настроить .env с API ключами

### После каждого промпта:
- [ ] Проверить работоспособность кода
- [ ] Запустить соответствующие тесты
- [ ] Убедиться в отсутствии упрощений
- [ ] Закоммитить рабочую версию

### Критические проверки:
- [ ] YOLO11 работает с GPU и CPU
- [ ] DeepFace fallback работает
- [ ] FFmpeg извлекает аудио
- [ ] OpenAI API доступен
- [ ] Все переводы на русский работают
- [ ] Временные метки синхронизированы
- [ ] Критические моменты определяются
- [ ] Все 5 типов отчетов генерируются
- [ ] Streamlit интерфейс полностью функционален
- [ ] Docker контейнер собирается и запускается

### Финальная проверка:
- [ ] Обработка демо видео от начала до конца
- [ ] Все отчеты генерируются корректно
- [ ] Нет критических ошибок в логах
- [ ] Документация соответствует коду
- [ ] Тесты проходят успешно

## ⚠️ НАПОМИНАНИЕ

**НИКАКИХ УПРОЩЕНИЙ!** Каждый промпт должен быть реализован ПОЛНОСТЬЮ со всеми указанными функциями. Если что-то не работает - исправляй, но НЕ удаляй функционал.

**КОММИТЬ ЧАСТО!** После каждого успешного этапа делай коммит с понятным сообщением.

**ТЕСТИРУЙ ВСЁ!** Не переходи к следующему промпту пока текущий не работает полностью.

Успехов в реализации! 🚀