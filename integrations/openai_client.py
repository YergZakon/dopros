"""
Complete OpenAI Integration with full API functionality for interrogation analysis
"""

import logging
import os
import time
import json
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
from functools import wraps
import tiktoken

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


@dataclass
class Usage:
    """Track OpenAI API usage"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0


@dataclass 
class APIRequest:
    """API request metadata"""
    timestamp: float
    method: str
    model: str
    tokens_used: int
    cost: float
    success: bool
    error: Optional[str] = None


# OpenAI model pricing (as of 2024)
PRICING = {
    'gpt-4': {'input': 0.03, 'output': 0.06},  # per 1K tokens
    'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
    'gpt-4-turbo-preview': {'input': 0.01, 'output': 0.03},
    'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
    'whisper-1': {'audio': 0.006}  # per minute
}

# Specialized prompts for interrogation analysis
INTERROGATION_PROMPTS = {
    'enhance_transcript': """
Исправь ошибки транскрипции допроса. Добавь пунктуацию и правильное форматирование.
Сохрани временные метки и структуру разговора.

Требования:
- Исправь только ошибки распознавания речи, не добавляй новую информацию
- Добавь правильную пунктуацию и капитализацию
- Раздели реплики на допрашивающего и допрашиваемого где возможно
- Сохрани все слова и смысл без изменений
- Укажи паузы и особенности речи (заикание, повторы)
- Используй формат: [Время] ГОВОРЯЩИЙ: текст реплики

Транскрипт для исправления:
{transcript}

Исправленный транскрипт:""",

    'analyze_psychology': """
Проанализируй психологическое состояние допрашиваемого на основе транскрипта и эмоциональных данных.

Анализируй:
1. Признаки стресса и тревожности
2. Попытки обмана или сокрытия информации
3. Эмоциональные реакции на конкретные вопросы
4. Изменения поведения в ходе допроса
5. Признаки давления или принуждения

Данные для анализа:
Транскрипт: {transcript}
Эмоциональные данные: {emotions}

Психологический анализ:""",

    'extract_insights': """
На основе комплексного анализа эмоций, речи и транскрипта определи:

1. КРИТИЧЕСКИЕ ЭПИЗОДЫ: моменты эмоциональной дестабилизации
2. ПРИЗНАКИ ОБМАНА: попытки уклонения, противоречия, нервозность
3. ПСИХОЛОГИЧЕСКОЕ ДАВЛЕНИЕ: признаки стресса от допроса
4. ДОСТОВЕРНОСТЬ ПОКАЗАНИЙ: оценка правдивости утверждений
5. РЕКОМЕНДАЦИИ: тактика для дальнейшего допроса

Входные данные:
- Транскрипт: {transcript}
- Эмоции видео: {video_emotions}
- Эмоции речи: {speech_emotions}
- Анализ содержания: {content_analysis}

Экспертные выводы:""",

    'summarize_interrogation': """
Создай краткую сводку допроса для оперативных работников.

Включи:
- Основные темы и вопросы
- Ключевые ответы допрашиваемого
- Признания или отрицания
- Эмоциональные реакции
- Подозрительные моменты
- Рекомендации для продолжения

Данные допроса: {data}

Сводка допроса:""",

    'detect_contradictions': """
Найди противоречия в показаниях допрашиваемого.

Анализируй:
- Несовпадения в хронологии событий
- Противоречивые утверждения
- Изменения в деталях рассказа
- Нелогичные объяснения

Транскрипт: {transcript}

Найденные противоречия:""",

    'assess_credibility': """
Оцени достоверность показаний допрашиваемого по шкале от 0 до 10.

Критерии оценки:
- Последовательность изложения
- Эмоциональные реакции
- Детализация ответов
- Поведенческие признаки
- Логичность утверждений

Данные: {data}

Оценка достоверности:"""
}


def rate_limited(max_calls: int, time_window: int = 60):
    """Rate limiting decorator"""
    def decorator(func: Callable) -> Callable:
        calls = []
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                now = time.time()
                # Remove old calls
                calls[:] = [call_time for call_time in calls if now - call_time < time_window]
                
                if len(calls) >= max_calls:
                    sleep_time = time_window - (now - calls[0]) + 1
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        calls.clear()
                
                calls.append(now)
                return func(*args, **kwargs)
        return wrapper
    return decorator


def retry_on_error(max_retries: int = 3, backoff_factor: float = 2):
    """Retry decorator with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (
                    openai.RateLimitError, 
                    openai.APIConnectionError, 
                    openai.APITimeoutError
                ) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = backoff_factor ** attempt
                        logging.warning(f"API error (attempt {attempt + 1}): {e}. Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                    continue
                except openai.BadRequestError as e:
                    # Don't retry bad requests
                    logging.error(f"Bad request: {e}")
                    raise
            
            # All retries failed
            logging.error(f"All {max_retries} attempts failed. Last error: {last_exception}")
            raise last_exception
            
        return wrapper
    return decorator


class TokenCounter:
    """Count tokens for different models"""
    
    def __init__(self):
        self.encodings = {}
        self._load_encodings()
    
    def _load_encodings(self):
        """Load tokenizer encodings"""
        try:
            self.encodings['gpt-4'] = tiktoken.encoding_for_model('gpt-4')
            self.encodings['gpt-3.5-turbo'] = tiktoken.encoding_for_model('gpt-3.5-turbo')
        except Exception as e:
            logging.warning(f"Failed to load tiktoken encodings: {e}")
            # Fallback to approximate counting
            self.encodings = {}
    
    def count_tokens(self, text: str, model: str = 'gpt-4') -> int:
        """Count tokens in text"""
        if model in self.encodings:
            return len(self.encodings[model].encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def count_messages_tokens(self, messages: List[Dict], model: str = 'gpt-4') -> int:
        """Count tokens in message format"""
        if model in self.encodings:
            encoding = self.encodings[model]
            
            # Different models have different token overhead
            tokens_per_message = 3  # For GPT-4
            tokens_per_name = 1
            
            total_tokens = 0
            for message in messages:
                total_tokens += tokens_per_message
                for key, value in message.items():
                    total_tokens += len(encoding.encode(str(value)))
                    if key == "name":
                        total_tokens += tokens_per_name
            
            total_tokens += 3  # Every reply is primed with assistant
            return total_tokens
        else:
            # Rough approximation
            total_text = ' '.join([msg.get('content', '') for msg in messages])
            return self.count_tokens(total_text, model)


class CacheManager:
    """Manage response caching with TTL"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_config = {
            'whisper': 7 * 24 * 3600,      # 7 days
            'gpt_analysis': 24 * 3600,      # 1 day  
            'gpt_enhancement': 3 * 24 * 3600, # 3 days
            'default': 24 * 3600            # 1 day
        }
    
    def _get_cache_key(self, method: str, params: Dict[str, Any]) -> str:
        """Generate cache key from method and parameters"""
        # Create hashable representation
        cache_data = {
            'method': method,
            'params': self._make_hashable(params)
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(cache_string.encode('utf-8')).hexdigest()
    
    def _make_hashable(self, obj: Any) -> Any:
        """Convert object to hashable format"""
        if isinstance(obj, dict):
            return sorted((k, self._make_hashable(v)) for k, v in obj.items())
        elif isinstance(obj, list):
            return tuple(self._make_hashable(item) for item in obj)
        elif isinstance(obj, set):
            return tuple(sorted(self._make_hashable(item) for item in obj))
        else:
            return obj
    
    def get(self, method: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached result if valid"""
        cache_key = self._get_cache_key(method, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check TTL
            cache_age = time.time() - cached_data['timestamp']
            ttl = self.ttl_config.get(method.split('_')[0], self.ttl_config['default'])
            
            if cache_age > ttl:
                # Cache expired
                cache_file.unlink()
                return None
            
            return cached_data['result']
            
        except Exception as e:
            logging.warning(f"Cache read error for {cache_key}: {e}")
            return None
    
    def set(self, method: str, params: Dict[str, Any], result: Any):
        """Cache result"""
        cache_key = self._get_cache_key(method, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            cached_data = {
                'timestamp': time.time(),
                'method': method,
                'result': result
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
                
        except Exception as e:
            logging.warning(f"Cache write error for {cache_key}: {e}")
    
    def clear_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        removed_count = 0
        
        for cache_file in self.cache_dir.glob('*.pkl'):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                method = cached_data.get('method', 'default')
                cache_age = current_time - cached_data['timestamp']
                ttl = self.ttl_config.get(method.split('_')[0], self.ttl_config['default'])
                
                if cache_age > ttl:
                    cache_file.unlink()
                    removed_count += 1
                    
            except Exception:
                # Remove corrupted cache files
                cache_file.unlink()
                removed_count += 1
        
        if removed_count > 0:
            logging.info(f"Removed {removed_count} expired cache entries")


class UsageMonitor:
    """Monitor OpenAI API usage and costs"""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.usage_file = self.storage_dir / 'openai_usage.json'
        
        # Usage tracking
        self.current_usage = defaultdict(lambda: Usage())
        self.daily_requests = defaultdict(list)
        self.cost_alerts = {
            'daily_limit': 50.0,  # $50 per day
            'monthly_limit': 1000.0  # $1000 per month
        }
        
        self._load_usage()
    
    def _load_usage(self):
        """Load usage history"""
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                    # Load recent requests for rate limiting
                    for date, requests in data.get('daily_requests', {}).items():
                        self.daily_requests[date] = [
                            APIRequest(**req) for req in requests
                        ]
            except Exception as e:
                logging.warning(f"Failed to load usage history: {e}")
    
    def _save_usage(self):
        """Save usage history"""
        try:
            # Convert recent requests to serializable format
            serializable_requests = {}
            current_date = time.strftime('%Y-%m-%d')
            
            # Keep only last 30 days
            cutoff_time = time.time() - 30 * 24 * 3600
            for date, requests in self.daily_requests.items():
                recent_requests = [
                    asdict(req) for req in requests 
                    if req.timestamp > cutoff_time
                ]
                if recent_requests:
                    serializable_requests[date] = recent_requests
            
            data = {
                'daily_requests': serializable_requests,
                'last_updated': time.time()
            }
            
            with open(self.usage_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logging.warning(f"Failed to save usage history: {e}")
    
    def track_request(self, request: APIRequest):
        """Track API request"""
        date_str = time.strftime('%Y-%m-%d', time.localtime(request.timestamp))
        self.daily_requests[date_str].append(request)
        
        # Check cost alerts
        self._check_cost_alerts()
        
        # Periodic save
        if len(self.daily_requests[date_str]) % 10 == 0:
            self._save_usage()
    
    def _check_cost_alerts(self):
        """Check if usage exceeds limits"""
        current_date = time.strftime('%Y-%m-%d')
        
        # Daily cost check
        daily_cost = sum(
            req.cost for req in self.daily_requests[current_date]
            if req.success
        )
        
        if daily_cost > self.cost_alerts['daily_limit']:
            logging.warning(f"Daily OpenAI cost limit exceeded: ${daily_cost:.2f}")
        
        # Monthly cost check
        current_month = time.strftime('%Y-%m')
        monthly_cost = 0
        
        for date, requests in self.daily_requests.items():
            if date.startswith(current_month):
                monthly_cost += sum(req.cost for req in requests if req.success)
        
        if monthly_cost > self.cost_alerts['monthly_limit']:
            logging.warning(f"Monthly OpenAI cost limit exceeded: ${monthly_cost:.2f}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        current_date = time.strftime('%Y-%m-%d')
        current_month = time.strftime('%Y-%m')
        
        daily_requests = len(self.daily_requests[current_date])
        daily_cost = sum(
            req.cost for req in self.daily_requests[current_date]
            if req.success
        )
        
        monthly_cost = 0
        monthly_requests = 0
        
        for date, requests in self.daily_requests.items():
            if date.startswith(current_month):
                monthly_requests += len(requests)
                monthly_cost += sum(req.cost for req in requests if req.success)
        
        return {
            'daily_requests': daily_requests,
            'daily_cost': daily_cost,
            'monthly_requests': monthly_requests,
            'monthly_cost': monthly_cost,
            'cost_limits': self.cost_alerts
        }


class OpenAIIntegration:
    """
    Complete OpenAI API integration with advanced features for interrogation analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        if DOTENV_AVAILABLE:
            load_dotenv()
        
        # API configuration
        api_config = config.get('apis', {}).get('openai', {})
        self.api_key = api_config.get('api_key') or os.getenv('OPENAI_API_KEY')
        self.organization = api_config.get('organization') or os.getenv('OPENAI_ORG_ID')
        self.max_retries = api_config.get('max_retries', 3)
        self.rate_limit = api_config.get('rate_limit', 60)  # requests per minute
        
        # Model settings
        self.default_gpt_model = api_config.get('gpt_model', 'gpt-4-turbo-preview')
        self.whisper_model = api_config.get('whisper_model', 'whisper-1')
        self.max_tokens_per_request = api_config.get('max_tokens_per_request', 4000)
        
        # Storage paths
        storage_config = config.get('storage', {})
        self.cache_dir = Path(storage_config.get('cache_dir', 'storage/cache'))
        self.usage_dir = Path(storage_config.get('usage_dir', 'storage/usage'))
        
        # Initialize components
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            organization=self.organization
        )
        
        # Initialize supporting components
        self.token_counter = TokenCounter()
        self.cache_manager = CacheManager(self.cache_dir)
        self.usage_monitor = UsageMonitor(self.usage_dir)
        
        self.logger.info("OpenAI integration initialized successfully")
    
    def _calculate_cost(self, usage_data: Dict[str, Any], model: str, audio_duration: float = 0) -> float:
        """Calculate API call cost"""
        if model == 'whisper-1':
            # Whisper pricing is per minute
            return audio_duration * PRICING[model]['audio'] / 60
        
        # Text models pricing
        if model not in PRICING:
            model = 'gpt-4'  # Default pricing
        
        input_cost = usage_data.get('prompt_tokens', 0) * PRICING[model]['input'] / 1000
        output_cost = usage_data.get('completion_tokens', 0) * PRICING[model]['output'] / 1000
        
        return input_cost + output_cost
    
    def _create_api_request(self, method: str, model: str, success: bool, 
                          usage_data: Dict = None, error: str = None, 
                          audio_duration: float = 0) -> APIRequest:
        """Create API request record"""
        tokens_used = 0
        if usage_data:
            tokens_used = usage_data.get('total_tokens', 0)
        
        cost = self._calculate_cost(usage_data or {}, model, audio_duration) if success else 0
        
        return APIRequest(
            timestamp=time.time(),
            method=method,
            model=model,
            tokens_used=tokens_used,
            cost=cost,
            success=success,
            error=error
        )
    
    @rate_limited(max_calls=60, time_window=60)  # 60 calls per minute
    @retry_on_error(max_retries=3)
    def transcribe_audio(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio with Whisper API including timestamps and error handling
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional Whisper parameters
            
        Returns:
            Processed transcription with segments and metadata
        """
        
        # Check cache first
        cache_params = {
            'audio_path': audio_path,
            'model': self.whisper_model,
            **kwargs
        }
        
        cached_result = self.cache_manager.get('whisper_transcribe', cache_params)
        if cached_result:
            self.logger.info(f"Using cached transcription for {audio_path}")
            return cached_result
        
        # Get audio duration for cost calculation
        audio_duration = 0
        try:
            import librosa
            audio_duration = librosa.get_duration(path=audio_path)
        except Exception:
            pass
        
        try:
            self.logger.info(f"Transcribing audio: {audio_path}")
            
            # Check file size (Whisper has 25MB limit)
            file_size = os.path.getsize(audio_path)
            if file_size > 25 * 1024 * 1024:  # 25MB
                raise ValueError(f"Audio file too large: {file_size / 1024 / 1024:.1f}MB (max 25MB)")
            
            # Default parameters
            whisper_params = {
                'model': self.whisper_model,
                'language': 'ru',
                'response_format': 'verbose_json',
                'timestamp_granularities': ['segment', 'word'],
                **kwargs
            }
            
            with open(audio_path, 'rb') as audio_file:
                response = self.client.audio.transcriptions.create(
                    file=audio_file,
                    **whisper_params
                )
            
            # Process response
            result = self._process_whisper_response(response)
            
            # Track usage
            request = self._create_api_request(
                'whisper_transcribe', 
                self.whisper_model, 
                True,
                audio_duration=audio_duration
            )
            self.usage_monitor.track_request(request)
            
            # Cache result
            self.cache_manager.set('whisper_transcribe', cache_params, result)
            
            self.logger.info(f"Transcription completed: {len(result['segments'])} segments")
            return result
            
        except Exception as e:
            # Track failed request
            request = self._create_api_request(
                'whisper_transcribe',
                self.whisper_model,
                False,
                error=str(e)
            )
            self.usage_monitor.track_request(request)
            
            self.logger.error(f"Transcription failed: {e}")
            raise
    
    def _process_whisper_response(self, response: Any) -> Dict[str, Any]:
        """Process Whisper API response into structured format"""
        segments = []
        words = []
        
        # Extract segments
        if hasattr(response, 'segments') and response.segments:
            for segment in response.segments:
                segment_data = {
                    'id': segment.id,
                    'seek': segment.seek,
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'tokens': getattr(segment, 'tokens', []),
                    'temperature': getattr(segment, 'temperature', 0.0),
                    'avg_logprob': getattr(segment, 'avg_logprob', 0.0),
                    'compression_ratio': getattr(segment, 'compression_ratio', 0.0),
                    'no_speech_prob': getattr(segment, 'no_speech_prob', 0.0)
                }
                
                # Extract words if available
                if hasattr(segment, 'words') and segment.words:
                    segment_words = []
                    for word in segment.words:
                        word_data = {
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': getattr(word, 'probability', 1.0)
                        }
                        segment_words.append(word_data)
                        words.append(word_data)
                    
                    segment_data['words'] = segment_words
                
                segments.append(segment_data)
        
        return {
            'text': response.text,
            'language': getattr(response, 'language', 'ru'),
            'duration': getattr(response, 'duration', 0.0),
            'segments': segments,
            'words': words,
            'metadata': {
                'model': self.whisper_model,
                'timestamp': time.time()
            }
        }
    
    @rate_limited(max_calls=30, time_window=60)  # 30 calls per minute for GPT
    @retry_on_error(max_retries=3)
    def enhance_transcript(self, transcript: str, **kwargs) -> Dict[str, Any]:
        """
        Enhance transcript using GPT-4 with specialized interrogation prompts
        """
        
        # Check token count
        estimated_tokens = self.token_counter.count_tokens(transcript, self.default_gpt_model)
        if estimated_tokens > self.max_tokens_per_request:
            # Split into chunks
            return self._process_large_text(transcript, 'enhance_transcript', **kwargs)
        
        # Check cache
        cache_params = {
            'transcript': transcript,
            'model': self.default_gpt_model,
            **kwargs
        }
        
        cached_result = self.cache_manager.get('gpt_enhancement', cache_params)
        if cached_result:
            return cached_result
        
        try:
            prompt = INTERROGATION_PROMPTS['enhance_transcript'].format(transcript=transcript)
            
            messages = [
                {
                    "role": "system", 
                    "content": "Ты эксперт по обработке транскрипций допросов. Исправляй ошибки, сохраняя точность и структуру."
                },
                {"role": "user", "content": prompt}
            ]
            
            # Count tokens for the request
            request_tokens = self.token_counter.count_messages_tokens(messages, self.default_gpt_model)
            
            response = self.client.chat.completions.create(
                model=self.default_gpt_model,
                messages=messages,
                temperature=0.1,
                max_tokens=min(4000, 8000 - request_tokens),
                **kwargs
            )
            
            result = {
                'enhanced_text': response.choices[0].message.content.strip(),
                'original_length': len(transcript),
                'enhanced_length': len(response.choices[0].message.content),
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'model': self.default_gpt_model,
                'timestamp': time.time()
            }
            
            # Track usage
            request = self._create_api_request(
                'gpt_enhancement',
                self.default_gpt_model,
                True,
                usage_data=asdict(response.usage)
            )
            self.usage_monitor.track_request(request)
            
            # Cache result
            self.cache_manager.set('gpt_enhancement', cache_params, result)
            
            return result
            
        except Exception as e:
            request = self._create_api_request(
                'gpt_enhancement',
                self.default_gpt_model,
                False,
                error=str(e)
            )
            self.usage_monitor.track_request(request)
            raise
    
    @rate_limited(max_calls=20, time_window=60)  # 20 calls per minute for analysis
    @retry_on_error(max_retries=3)
    def analyze_psychology(self, transcript: str, emotions: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Analyze psychological state of the interrogated person
        """
        
        # Prepare input data
        emotions_summary = json.dumps(emotions, ensure_ascii=False, indent=2)
        
        # Check cache
        cache_params = {
            'transcript': transcript,
            'emotions': emotions_summary,
            'model': self.default_gpt_model
        }
        
        cached_result = self.cache_manager.get('gpt_psychology', cache_params)
        if cached_result:
            return cached_result
        
        try:
            prompt = INTERROGATION_PROMPTS['analyze_psychology'].format(
                transcript=transcript,
                emotions=emotions_summary
            )
            
            messages = [
                {
                    "role": "system",
                    "content": "Ты психолог-эксперт по анализу допросов. Анализируй поведение и эмоциональное состояние допрашиваемого."
                },
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.default_gpt_model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
                **kwargs
            )
            
            result = {
                'psychological_analysis': response.choices[0].message.content.strip(),
                'usage': asdict(response.usage),
                'model': self.default_gpt_model,
                'timestamp': time.time()
            }
            
            # Track usage
            request = self._create_api_request(
                'gpt_psychology',
                self.default_gpt_model,
                True,
                usage_data=asdict(response.usage)
            )
            self.usage_monitor.track_request(request)
            
            # Cache result
            self.cache_manager.set('gpt_psychology', cache_params, result)
            
            return result
            
        except Exception as e:
            request = self._create_api_request(
                'gpt_psychology',
                self.default_gpt_model,
                False,
                error=str(e)
            )
            self.usage_monitor.track_request(request)
            raise
    
    @rate_limited(max_calls=15, time_window=60)  # 15 calls per minute for complex analysis
    @retry_on_error(max_retries=3)
    def get_structured_analysis(self, transcript: str, video_emotions: Dict, 
                              speech_emotions: Dict, content_analysis: Dict, 
                              **kwargs) -> Dict[str, Any]:
        """
        Get structured interrogation analysis with JSON output
        """
        
        try:
            prompt = INTERROGATION_PROMPTS['extract_insights'].format(
                transcript=transcript,
                video_emotions=json.dumps(video_emotions, ensure_ascii=False, indent=2),
                speech_emotions=json.dumps(speech_emotions, ensure_ascii=False, indent=2),
                content_analysis=json.dumps(content_analysis, ensure_ascii=False, indent=2)
            )
            
            messages = [
                {
                    "role": "system",
                    "content": "Ты эксперт по анализу допросов. Предоставь структурированный анализ в формате JSON."
                },
                {"role": "user", "content": prompt}
            ]
            
            # Define the function schema for structured output
            functions = [
                {
                    "name": "analyze_interrogation",
                    "description": "Analyze interrogation and provide structured insights",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "critical_moments": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "timestamp": {"type": "number"},
                                        "description": {"type": "string"},
                                        "importance": {"type": "number", "minimum": 1, "maximum": 10},
                                        "emotions": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            },
                            "deception_indicators": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string"},
                                        "evidence": {"type": "string"},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                    }
                                }
                            },
                            "stress_level": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 10,
                                "description": "Overall stress level (0-10)"
                            },
                            "cooperation_level": {
                                "type": "number", 
                                "minimum": 0,
                                "maximum": 10,
                                "description": "Cooperation level (0-10)"
                            },
                            "credibility_score": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 10,
                                "description": "Overall credibility (0-10)"
                            },
                            "key_findings": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Main findings and insights"
                            },
                            "recommendations": {
                                "type": "array", 
                                "items": {"type": "string"},
                                "description": "Recommendations for further investigation"
                            },
                            "emotional_timeline": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "period": {"type": "string"},
                                        "dominant_emotions": {"type": "array", "items": {"type": "string"}},
                                        "significance": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "required": ["critical_moments", "deception_indicators", "stress_level", 
                                   "cooperation_level", "key_findings", "recommendations"]
                    }
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.default_gpt_model,
                messages=messages,
                functions=functions,
                function_call={"name": "analyze_interrogation"},
                temperature=0.2,
                **kwargs
            )
            
            # Parse function call result
            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "analyze_interrogation":
                structured_result = json.loads(function_call.arguments)
            else:
                # Fallback to regular response
                structured_result = {
                    "analysis": response.choices[0].message.content,
                    "parsing_failed": True
                }
            
            result = {
                'structured_analysis': structured_result,
                'usage': asdict(response.usage),
                'model': self.default_gpt_model,
                'timestamp': time.time()
            }
            
            # Track usage
            request = self._create_api_request(
                'gpt_structured',
                self.default_gpt_model,
                True,
                usage_data=asdict(response.usage)
            )
            self.usage_monitor.track_request(request)
            
            return result
            
        except Exception as e:
            request = self._create_api_request(
                'gpt_structured',
                self.default_gpt_model,
                False,
                error=str(e)
            )
            self.usage_monitor.track_request(request)
            raise
    
    def _process_large_text(self, text: str, method: str, **kwargs) -> Dict[str, Any]:
        """Process large text by splitting into manageable chunks"""
        
        # Split text into chunks that fit within token limits
        chunk_size = self.max_tokens_per_request // 2  # Leave room for prompt and response
        chunks = self._split_text_by_tokens(text, chunk_size)
        
        results = []
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            if method == 'enhance_transcript':
                chunk_result = self.enhance_transcript(chunk, **kwargs)
                results.append(chunk_result['enhanced_text'])
            # Add other methods as needed
        
        # Combine results
        if method == 'enhance_transcript':
            combined_text = '\n'.join(results)
            return {
                'enhanced_text': combined_text,
                'original_length': len(text),
                'enhanced_length': len(combined_text),
                'chunks_processed': len(chunks),
                'timestamp': time.time()
            }
        
        return {'processed_chunks': results}
    
    def _split_text_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks based on token count"""
        
        # Simple paragraph-based splitting first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            para_tokens = self.token_counter.count_tokens(paragraph, self.default_gpt_model)
            
            if current_tokens + para_tokens > max_tokens and current_chunk:
                # Start new chunk
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_tokens = para_tokens
            else:
                current_chunk.append(paragraph)
                current_tokens += para_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def summarize_interrogation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of interrogation"""
        
        try:
            prompt = INTERROGATION_PROMPTS['summarize_interrogation'].format(
                data=json.dumps(data, ensure_ascii=False, indent=2)
            )
            
            messages = [
                {
                    "role": "system",
                    "content": "Создай краткую сводку допроса для руководства следственной группы."
                },
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.default_gpt_model,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            return {
                'summary': response.choices[0].message.content.strip(),
                'usage': asdict(response.usage),
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            raise
    
    def detect_contradictions(self, transcript: str) -> Dict[str, Any]:
        """Detect contradictions in testimony"""
        
        try:
            prompt = INTERROGATION_PROMPTS['detect_contradictions'].format(
                transcript=transcript
            )
            
            messages = [
                {
                    "role": "system",
                    "content": "Ты эксперт по выявлению противоречий в показаниях. Найди все несостыковки."
                },
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.default_gpt_model,
                messages=messages,
                temperature=0.2,
                max_tokens=1500
            )
            
            return {
                'contradictions': response.choices[0].message.content.strip(),
                'usage': asdict(response.usage),
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Contradiction detection failed: {e}")
            raise
    
    def assess_credibility(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall credibility of testimony"""
        
        try:
            prompt = INTERROGATION_PROMPTS['assess_credibility'].format(
                data=json.dumps(data, ensure_ascii=False, indent=2)
            )
            
            messages = [
                {
                    "role": "system",
                    "content": "Оцени достоверность показаний на основе всех доступных данных."
                },
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.default_gpt_model,
                messages=messages,
                temperature=0.1,
                max_tokens=800
            )
            
            return {
                'credibility_assessment': response.choices[0].message.content.strip(),
                'usage': asdict(response.usage),
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Credibility assessment failed: {e}")
            raise
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get detailed usage statistics"""
        return self.usage_monitor.get_usage_stats()
    
    def clear_cache(self, older_than_days: int = 7):
        """Clear old cache entries"""
        self.cache_manager.clear_expired()
        self.logger.info(f"Cache cleanup completed")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.usage_monitor._save_usage()
        except:
            pass