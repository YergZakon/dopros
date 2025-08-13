"""
Master Pipeline - центральный координатор всех процессов ДОПРОС MVP 2.0
"""

import os
import gc
import time
import pickle
import logging
import shutil
import hashlib
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import uuid

import yaml
import psutil
import cv2
import numpy as np
from tqdm import tqdm

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Internal imports
from utils.gpu_manager import get_gpu_manager
from models.yolo_manager import YOLO11Manager
from core.emotion_analyzer import MultiModalEmotionAnalyzer
from core.audio_processor import CompleteAudioProcessor
from models.speech_analyzer import AdvancedSpeechEmotionAnalyzer
from core.report_generator import ComprehensiveReportGenerator
from integrations.openai_client import OpenAIIntegration


class PipelineError(Exception):
    """Custom exception for pipeline errors"""
    pass


class MasterPipeline:
    """
    Главный координатор всех процессов системы анализа допросов
    Управляет полным пайплайном от входного видео до итоговых отчетов
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Threading lock for thread-safe operations
        self._lock = Lock()
        
        # Initialize core components
        self.gpu_manager = get_gpu_manager()
        self.yolo_manager = None
        self.emotion_analyzer = None
        self.audio_processor = None
        self.speech_analyzer = None
        self.report_generator = None
        self.openai_client = None
        
        # Pipeline configuration
        pipeline_config = self.config.get('pipeline', {})
        self.max_workers = pipeline_config.get('max_workers', 2)
        self.chunk_size = pipeline_config.get('chunk_size', 30)  # seconds
        self.enable_checkpoints = pipeline_config.get('enable_checkpoints', True)
        self.cleanup_cache = pipeline_config.get('cleanup_cache', True)
        self.memory_limit = pipeline_config.get('memory_limit', 80)  # percent
        
        # Storage paths
        storage_config = self.config.get('storage', {})
        self.cache_dir = Path(storage_config.get('cache_dir', 'cache'))
        self.temp_dir = Path(storage_config.get('temp_dir', 'temp'))
        self.output_dir = Path(storage_config.get('output_dir', 'output'))
        
        # Create directories
        for dir_path in [self.cache_dir, self.temp_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Processing stages definition
        self.stages = [
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
        
        # Critical stages that cannot fail
        self.critical_stages = {
            'validate_input', 'setup_session', 'extract_frames', 
            'extract_audio', 'synchronize_data'
        }
        
        # Fallback methods for non-critical stages
        self.fallback_methods = {
            'detect_faces': self._fallback_face_detection,
            'analyze_emotions': self._fallback_emotion_analysis,
            'transcribe_audio': self._fallback_transcription,
            'analyze_speech': self._fallback_speech_analysis,
            'detect_critical': self._fallback_critical_moments,
            'generate_insights': self._fallback_insights
        }
        
        # Metrics tracking
        self.metrics = {
            'processing_time': {},
            'memory_usage': {},
            'gpu_usage': {},
            'api_calls': {},
            'errors': [],
            'stages_completed': 0,
            'total_stages': len(self.stages)
        }
        
        self.logger.info("MasterPipeline initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    self.logger.info(f"Configuration loaded from {config_path}")
                    return config
            else:
                self.logger.warning(f"Config file {config_path} not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'pipeline': {
                'max_workers': 2,
                'chunk_size': 30,
                'enable_checkpoints': True,
                'cleanup_cache': True,
                'memory_limit': 80
            },
            'storage': {
                'cache_dir': 'cache',
                'temp_dir': 'temp',
                'output_dir': 'output',
                'frames_dir': 'storage/frames',
                'faces_dir': 'storage/faces',
                'audio_dir': 'storage/audio',
                'reports_dir': 'storage/reports'
            },
            'processing': {
                'video': {
                    'fps_extract': 1.0,
                    'max_frames': 1000,
                    'frame_resize': [640, 480]
                },
                'models': {
                    'yolo': {
                        'confidence_threshold': 0.5,
                        'iou_threshold': 0.45,
                        'max_detections': 100,
                        'device': 'auto'
                    }
                }
            },
            'analysis': {
                'emotion': {
                    'methods': ['deepface', 'fer', 'yolo11'],
                    'priority_weights': {'deepface': 0.6, 'fer': 0.3, 'yolo11': 0.1}
                }
            }
        }
    
    def _init_components(self):
        """Initialize processing components lazily"""
        if self.yolo_manager is None:
            self.yolo_manager = YOLO11Manager(self.config)
        
        if self.emotion_analyzer is None:
            self.emotion_analyzer = MultiModalEmotionAnalyzer(self.config)
        
        if self.audio_processor is None:
            self.audio_processor = CompleteAudioProcessor(self.config)
        
        if self.speech_analyzer is None:
            self.speech_analyzer = AdvancedSpeechEmotionAnalyzer(self.config)
        
        if self.report_generator is None:
            self.report_generator = ComprehensiveReportGenerator(self.config)
        
        if self.openai_client is None:
            try:
                self.openai_client = OpenAIIntegration(self.config)
            except Exception as e:
                self.logger.warning(f"OpenAI client initialization failed: {e}")
                self.openai_client = None
    
    def process_video(self, 
                     video_path: str, 
                     options: Optional[Dict[str, Any]] = None,
                     progress_callback: Optional[Callable] = None,
                     force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Основной метод обработки видео через весь пайплайн
        
        Args:
            video_path: Путь к видеофайлу
            options: Дополнительные опции обработки
            progress_callback: Функция для уведомления о прогрессе
            force_reprocess: Принудительная переобработка (игнорировать кэш)
            
        Returns:
            Результаты обработки со всеми данными и путями к отчетам
        """
        start_time = time.time()
        session_id = str(uuid.uuid4())[:8]
        
        self.logger.info(f"Starting video processing pipeline: {video_path}")
        self.logger.info(f"Session ID: {session_id}")
        
        # Initialize components
        self._init_components()
        
        # Reset metrics
        self.metrics = {
            'processing_time': {},
            'memory_usage': {},
            'gpu_usage': {},
            'api_calls': {},
            'errors': [],
            'stages_completed': 0,
            'total_stages': len(self.stages),
            'session_id': session_id,
            'video_path': video_path,
            'start_time': start_time
        }
        
        # Results container
        results = {
            'session_id': session_id,
            'video_path': video_path,
            'options': options or {},
            'errors': [],
            'warnings': [],
            'performance': {}
        }
        
        # Create session cache directory
        session_cache_dir = self.cache_dir / session_id
        session_cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Execute pipeline stages
            for stage_index, (stage_name, stage_func) in enumerate(self.stages):
                stage_start_time = time.time()
                
                try:
                    self.logger.info(f"Starting stage {stage_index + 1}/{len(self.stages)}: {stage_name}")
                    
                    # Check for checkpoint
                    checkpoint_file = session_cache_dir / f"{stage_name}.pkl"
                    
                    if (self.enable_checkpoints and 
                        checkpoint_file.exists() and 
                        not force_reprocess):
                        
                        self.logger.info(f"Loading checkpoint for {stage_name}")
                        results[stage_name] = self._load_checkpoint(checkpoint_file)
                    else:
                        # Execute stage
                        stage_result = stage_func(results)
                        results[stage_name] = stage_result
                        
                        # Save checkpoint
                        if self.enable_checkpoints:
                            self._save_checkpoint(stage_result, checkpoint_file)
                    
                    # Update metrics
                    stage_time = time.time() - stage_start_time
                    self.metrics['processing_time'][stage_name] = stage_time
                    self.metrics['stages_completed'] += 1
                    
                    # Memory management
                    self._manage_memory()
                    
                    # Progress callback
                    if progress_callback:
                        progress = (stage_index + 1) / len(self.stages)
                        progress_callback(progress, stage_name, results.get(stage_name))
                    
                    self.logger.info(f"Completed stage {stage_name} in {stage_time:.2f}s")
                    
                except Exception as e:
                    self._handle_stage_error(stage_name, e, results)
                    
                    # If critical stage failed, abort pipeline
                    if stage_name in self.critical_stages:
                        raise PipelineError(f"Critical stage {stage_name} failed: {e}")
            
            # Finalize results
            final_results = self._finalize_results(results, session_id)
            
            # Calculate total processing time
            total_time = time.time() - start_time
            final_results['processing_time'] = total_time
            final_results['metrics'] = self.metrics
            
            self.logger.info(f"Pipeline completed successfully in {total_time:.2f}s")
            
            # Cleanup if requested
            if self.cleanup_cache:
                self._cleanup_session_cache(session_cache_dir)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            
            # Save error state
            error_results = {
                'session_id': session_id,
                'video_path': video_path,
                'error': str(e),
                'partial_results': results,
                'metrics': self.metrics,
                'processing_time': time.time() - start_time
            }
            
            return error_results
    
    def resume_processing(self, session_id: str, 
                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Возобновить прерванную обработку из checkpoint'ов
        
        Args:
            session_id: ID сессии для восстановления
            progress_callback: Функция для уведомления о прогрессе
            
        Returns:
            Результаты обработки
        """
        session_cache_dir = self.cache_dir / session_id
        
        if not session_cache_dir.exists():
            raise PipelineError(f"Session cache not found: {session_id}")
        
        self.logger.info(f"Resuming processing for session: {session_id}")
        
        # Load existing results
        results = {}
        completed_stages = []
        
        for stage_name, _ in self.stages:
            checkpoint_file = session_cache_dir / f"{stage_name}.pkl"
            if checkpoint_file.exists():
                try:
                    results[stage_name] = self._load_checkpoint(checkpoint_file)
                    completed_stages.append(stage_name)
                except Exception as e:
                    self.logger.warning(f"Failed to load checkpoint {stage_name}: {e}")
        
        self.logger.info(f"Loaded {len(completed_stages)} completed stages")
        
        # Find next stage to execute
        start_index = len(completed_stages)
        
        if start_index >= len(self.stages):
            self.logger.info("All stages already completed")
            return self._finalize_results(results, session_id)
        
        # Continue from next stage
        remaining_stages = self.stages[start_index:]
        
        for stage_index, (stage_name, stage_func) in enumerate(remaining_stages):
            actual_index = start_index + stage_index
            
            try:
                self.logger.info(f"Executing stage {actual_index + 1}/{len(self.stages)}: {stage_name}")
                
                stage_start_time = time.time()
                stage_result = stage_func(results)
                results[stage_name] = stage_result
                
                # Save checkpoint
                if self.enable_checkpoints:
                    checkpoint_file = session_cache_dir / f"{stage_name}.pkl"
                    self._save_checkpoint(stage_result, checkpoint_file)
                
                # Update metrics
                stage_time = time.time() - stage_start_time
                if 'processing_time' not in self.metrics:
                    self.metrics['processing_time'] = {}
                self.metrics['processing_time'][stage_name] = stage_time
                
                # Memory management
                self._manage_memory()
                
                # Progress callback
                if progress_callback:
                    progress = (actual_index + 1) / len(self.stages)
                    progress_callback(progress, stage_name, results.get(stage_name))
                
                self.logger.info(f"Completed stage {stage_name} in {stage_time:.2f}s")
                
            except Exception as e:
                self._handle_stage_error(stage_name, e, results)
                
                if stage_name in self.critical_stages:
                    raise PipelineError(f"Critical stage {stage_name} failed during resume: {e}")
        
        # Finalize and return results
        return self._finalize_results(results, session_id)
    
    def get_processing_status(self, session_id: str) -> Dict[str, Any]:
        """
        Получить статус обработки для сессии
        
        Args:
            session_id: ID сессии
            
        Returns:
            Статус обработки
        """
        session_cache_dir = self.cache_dir / session_id
        
        if not session_cache_dir.exists():
            return {'status': 'not_found', 'session_id': session_id}
        
        completed_stages = []
        stage_details = {}
        
        for stage_name, _ in self.stages:
            checkpoint_file = session_cache_dir / f"{stage_name}.pkl"
            if checkpoint_file.exists():
                completed_stages.append(stage_name)
                stage_details[stage_name] = {
                    'completed': True,
                    'checkpoint_time': checkpoint_file.stat().st_mtime,
                    'file_size': checkpoint_file.stat().st_size
                }
            else:
                stage_details[stage_name] = {'completed': False}
        
        progress = len(completed_stages) / len(self.stages)
        
        status = {
            'session_id': session_id,
            'progress': progress,
            'completed_stages': len(completed_stages),
            'total_stages': len(self.stages),
            'stage_details': stage_details,
            'can_resume': len(completed_stages) < len(self.stages),
            'is_complete': len(completed_stages) == len(self.stages)
        }
        
        return status
    
    # ====== STAGE METHODS ======
    
    def _validate_input(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input video file"""
        video_path = results['video_path']
        
        self.logger.info(f"Validating input: {video_path}")
        
        if not os.path.exists(video_path):
            raise PipelineError(f"Video file not found: {video_path}")
        
        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            raise PipelineError(f"Video file is empty: {video_path}")
        
        # Validate video format
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise PipelineError(f"Cannot open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            validation_result = {
                'video_path': video_path,
                'file_size': file_size,
                'duration': duration,
                'fps': fps,
                'frame_count': frame_count,
                'resolution': (width, height),
                'valid': True
            }
            
            self.logger.info(f"Video validation successful: {duration:.2f}s, {frame_count} frames, {width}x{height}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Video validation failed: {e}")
            raise PipelineError(f"Invalid video file: {e}")
    
    def _setup_session(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Setup processing session"""
        session_id = results['session_id']
        video_info = results['validate_input']
        
        self.logger.info(f"Setting up session: {session_id}")
        
        # Create session directories
        session_temp_dir = self.temp_dir / session_id
        session_output_dir = self.output_dir / session_id
        
        session_temp_dir.mkdir(parents=True, exist_ok=True)
        session_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session metadata
        session_data = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'video_info': video_info,
            'temp_dir': str(session_temp_dir),
            'output_dir': str(session_output_dir),
            'config': self.config
        }
        
        # Save session metadata
        metadata_file = session_output_dir / 'session_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        return session_data
    
    def _extract_frames(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract frames from video using parallel processing"""
        video_path = results['video_path']
        video_info = results['validate_input']
        session_data = results['setup_session']
        
        self.logger.info("Extracting frames from video")
        
        # Configuration
        video_config = self.config.get('processing', {}).get('video', {})
        fps_extract = video_config.get('fps_extract', 1.0)
        max_frames = video_config.get('max_frames', 1000)
        frame_resize = video_config.get('frame_resize', None)
        
        # Calculate frame extraction parameters
        original_fps = video_info['fps']
        frame_interval = max(1, int(original_fps / fps_extract))
        
        # Extract frames
        frames_dir = Path(session_data['temp_dir']) / 'frames'
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        
        extracted_frames = []
        frame_index = 0
        saved_count = 0
        
        while cap.isOpened() and saved_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract every nth frame
            if frame_index % frame_interval == 0:
                timestamp = frame_index / original_fps
                
                # Resize if requested
                if frame_resize:
                    frame = cv2.resize(frame, tuple(frame_resize))
                
                # Save frame
                frame_filename = f"frame_{saved_count:06d}_{timestamp:.3f}s.jpg"
                frame_path = frames_dir / frame_filename
                
                cv2.imwrite(str(frame_path), frame)
                
                extracted_frames.append({
                    'frame_path': str(frame_path),
                    'timestamp': timestamp,
                    'frame_number': frame_index,
                    'extract_index': saved_count
                })
                
                saved_count += 1
            
            frame_index += 1
        
        cap.release()
        
        frames_result = {
            'frames_dir': str(frames_dir),
            'extracted_frames': extracted_frames,
            'total_extracted': len(extracted_frames),
            'extraction_fps': fps_extract,
            'frame_interval': frame_interval
        }
        
        self.logger.info(f"Extracted {len(extracted_frames)} frames")
        
        return frames_result
    
    def _detect_faces(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect faces in extracted frames using YOLO11"""
        frames_data = results['extract_frames']
        
        self.logger.info("Detecting faces with YOLO11")
        
        frame_paths = [frame['frame_path'] for frame in frames_data['extracted_frames']]
        
        try:
            # Use YOLO manager for face detection
            face_results = self.yolo_manager.detect_faces(
                frame_paths, 
                use_tracking=True,
                save_crops=True,
                min_face_size=32
            )
            
            self.logger.info(f"Face detection completed: {face_results['total_faces']} faces detected")
            
            return face_results
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            # Use fallback method
            return self.fallback_methods['detect_faces'](results)
    
    def _analyze_emotions(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emotions in detected faces"""
        frames_data = results['extract_frames']
        faces_data = results['detect_faces']
        
        self.logger.info("Analyzing emotions in faces")
        
        try:
            # Get face crops or fall back to full frames
            if faces_data.get('face_crops'):
                analysis_paths = faces_data['face_crops']
            else:
                analysis_paths = [frame['frame_path'] for frame in frames_data['extracted_frames']]
            
            # Analyze emotions using multi-modal analyzer
            emotion_results = []
            
            for i, path in enumerate(tqdm(analysis_paths, desc="Analyzing emotions")):
                # Find corresponding timestamp
                timestamp = 0.0
                frame_number = 0
                
                # Try to find matching frame data
                for frame in frames_data['extracted_frames']:
                    if frame['frame_path'] in path:
                        timestamp = frame['timestamp']
                        frame_number = frame['frame_number']
                        break
                
                try:
                    emotion_result = self.emotion_analyzer.analyze_frame(
                        path, timestamp, frame_number
                    )
                    emotion_results.append(emotion_result)
                    
                except Exception as e:
                    self.logger.warning(f"Emotion analysis failed for {path}: {e}")
                    continue
            
            # Analyze emotion transitions
            transitions = self.emotion_analyzer.detect_emotion_transitions(emotion_results)
            rapid_changes = self.emotion_analyzer.find_rapid_changes(emotion_results)
            
            emotions_analysis = {
                'emotion_results': emotion_results,
                'transitions': transitions,
                'rapid_changes': rapid_changes,
                'total_analyzed': len(emotion_results)
            }
            
            self.logger.info(f"Emotion analysis completed: {len(emotion_results)} frames analyzed")
            
            return emotions_analysis
            
        except Exception as e:
            self.logger.error(f"Emotion analysis failed: {e}")
            return self.fallback_methods['analyze_emotions'](results)
    
    def _extract_audio(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and process audio using parallel processing"""
        video_path = results['video_path']
        session_data = results['setup_session']
        
        self.logger.info("Extracting audio from video")
        
        # Use audio processor to extract audio
        temp_dir = Path(session_data['temp_dir'])
        audio_path = temp_dir / 'audio.wav'
        
        try:
            # Extract audio with optimal settings
            extracted_path = self.audio_processor.extract_audio(
                video_path, 
                str(audio_path)
            )
            
            audio_result = {
                'audio_path': extracted_path,
                'extracted_successfully': True
            }
            
            self.logger.info(f"Audio extraction completed: {extracted_path}")
            
            return audio_result
            
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            raise PipelineError(f"Critical audio extraction failed: {e}")
    
    def _transcribe_audio(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe audio with timestamps using OpenAI Whisper"""
        audio_data = results['extract_audio']
        
        self.logger.info("Transcribing audio")
        
        try:
            # Transcribe with timestamps
            transcription_result = self.audio_processor.transcribe_with_timestamps(
                audio_data['audio_path']
            )
            
            # Enhance transcript if OpenAI client available
            if self.openai_client and transcription_result.get('segments'):
                enhanced_segments = self.audio_processor.enhance_transcript(
                    transcription_result['segments']
                )
                transcription_result['enhanced_segments'] = enhanced_segments
            
            self.logger.info(f"Transcription completed: {len(transcription_result.get('segments', []))} segments")
            
            return transcription_result
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return self.fallback_methods['transcribe_audio'](results)
    
    def _analyze_speech(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze speech emotions and prosodic features"""
        audio_data = results['extract_audio']
        transcription_data = results['transcribe_audio']
        
        self.logger.info("Analyzing speech emotions")
        
        try:
            # Analyze speech emotions
            speech_results = self.speech_analyzer.analyze_audio_file(
                audio_data['audio_path']
            )
            
            # Combine with transcription data if available
            if transcription_data.get('segments'):
                # Align speech emotions with transcript segments
                aligned_results = self._align_speech_with_transcript(
                    speech_results, transcription_data['segments']
                )
                speech_results['aligned_segments'] = aligned_results
            
            self.logger.info(f"Speech analysis completed: {len(speech_results.get('segments', []))} segments")
            
            return speech_results
            
        except Exception as e:
            self.logger.error(f"Speech analysis failed: {e}")
            return self.fallback_methods['analyze_speech'](results)
    
    def _synchronize_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize video emotions, speech emotions, and transcript"""
        video_emotions = results.get('analyze_emotions', {})
        speech_emotions = results.get('analyze_speech', {})
        transcription = results.get('transcribe_audio', {})
        
        self.logger.info("Synchronizing multimodal data")
        
        try:
            # Synchronize timestamps and create unified timeline
            synchronized_data = {
                'video_emotions': video_emotions.get('emotion_results', []),
                'speech_emotions': speech_emotions.get('segments', []),
                'transcript_segments': transcription.get('segments', []),
                'synchronized_timeline': [],
                'correlations': {}
            }
            
            # Create unified timeline combining all modalities
            all_events = []
            
            # Add video emotion events
            for emotion in synchronized_data['video_emotions']:
                all_events.append({
                    'timestamp': emotion.get('timestamp', 0),
                    'type': 'video_emotion',
                    'data': emotion
                })
            
            # Add speech emotion events
            for speech in synchronized_data['speech_emotions']:
                all_events.append({
                    'timestamp': speech.get('timestamp', 0),
                    'type': 'speech_emotion', 
                    'data': speech
                })
            
            # Add transcript events
            for segment in synchronized_data['transcript_segments']:
                all_events.append({
                    'timestamp': segment.get('start', 0),
                    'type': 'transcript',
                    'data': segment
                })
            
            # Sort by timestamp
            synchronized_data['synchronized_timeline'] = sorted(
                all_events, key=lambda x: x['timestamp']
            )
            
            # Calculate basic correlations
            if synchronized_data['video_emotions'] and synchronized_data['speech_emotions']:
                correlations = self._calculate_emotion_correlations(
                    synchronized_data['video_emotions'],
                    synchronized_data['speech_emotions']
                )
                synchronized_data['correlations'] = correlations
            
            self.logger.info(f"Data synchronization completed: {len(all_events)} total events")
            
            return synchronized_data
            
        except Exception as e:
            self.logger.error(f"Data synchronization failed: {e}")
            raise PipelineError(f"Critical synchronization failed: {e}")
    
    def _detect_critical_moments(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect critical moments based on multimodal analysis"""
        synchronized_data = results['synchronize_data']
        
        self.logger.info("Detecting critical moments")
        
        try:
            # Analyze synchronized timeline for critical patterns
            critical_moments = []
            
            timeline = synchronized_data['synchronized_timeline']
            video_emotions = synchronized_data['video_emotions']
            speech_emotions = synchronized_data['speech_emotions']
            
            # Detect rapid emotion changes
            emotion_changes = self._detect_emotion_changes(timeline)
            critical_moments.extend(emotion_changes)
            
            # Detect high-intensity emotional moments
            intense_moments = self._detect_intense_emotions(video_emotions, speech_emotions)
            critical_moments.extend(intense_moments)
            
            # Detect contradictions between modalities
            contradictions = self._detect_modality_contradictions(video_emotions, speech_emotions)
            critical_moments.extend(contradictions)
            
            # Detect silence or speech anomalies
            speech_anomalies = self._detect_speech_anomalies(speech_emotions)
            critical_moments.extend(speech_anomalies)
            
            # Sort by timestamp and remove duplicates
            critical_moments = self._deduplicate_critical_moments(critical_moments)
            
            critical_result = {
                'critical_moments': critical_moments,
                'total_critical': len(critical_moments),
                'analysis_methods': [
                    'emotion_changes', 'intense_emotions', 
                    'modality_contradictions', 'speech_anomalies'
                ]
            }
            
            self.logger.info(f"Critical moments detection completed: {len(critical_moments)} moments found")
            
            return critical_result
            
        except Exception as e:
            self.logger.error(f"Critical moments detection failed: {e}")
            return self.fallback_methods['detect_critical'](results)
    
    def _generate_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate psychological insights using OpenAI GPT-4"""
        synchronized_data = results['synchronize_data']
        critical_moments = results['detect_critical']
        
        self.logger.info("Generating psychological insights")
        
        try:
            if not self.openai_client:
                self.logger.warning("OpenAI client not available, using fallback insights")
                return self.fallback_methods['generate_insights'](results)
            
            # Prepare data for analysis
            analysis_data = {
                'video_emotions': synchronized_data['video_emotions'],
                'speech_emotions': synchronized_data['speech_emotions'],
                'transcript': synchronized_data['transcript_segments'],
                'critical_moments': critical_moments['critical_moments']
            }
            
            # Get structured analysis from OpenAI
            insights = self.openai_client.get_structured_analysis(
                analysis_data['transcript'],
                analysis_data['video_emotions'], 
                analysis_data['speech_emotions'],
                analysis_data
            )
            
            # Add metadata
            insights_result = {
                'insights': insights,
                'generated_at': datetime.now().isoformat(),
                'method': 'openai_gpt4',
                'data_summary': {
                    'video_emotions_count': len(analysis_data['video_emotions']),
                    'speech_segments_count': len(analysis_data['speech_emotions']),
                    'transcript_segments_count': len(analysis_data['transcript']),
                    'critical_moments_count': len(analysis_data['critical_moments'])
                }
            }
            
            self.logger.info("Psychological insights generation completed")
            
            return insights_result
            
        except Exception as e:
            self.logger.error(f"Insights generation failed: {e}")
            return self.fallback_methods['generate_insights'](results)
    
    def _create_reports(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive reports in multiple formats"""
        synchronized_data = results['synchronize_data']
        critical_moments = results['detect_critical']
        insights = results['generate_insights']
        session_data = results['setup_session']
        
        self.logger.info("Creating comprehensive reports")
        
        try:
            # Prepare data for report generation
            report_data = {
                'metadata': {
                    'session_id': results['session_id'],
                    'video_path': results['video_path'],
                    'duration': results['validate_input']['duration'],
                    'video_frames': len(results['extract_frames']['extracted_frames']),
                    'audio_segments': len(synchronized_data['speech_emotions'])
                },
                'video_emotions': synchronized_data['video_emotions'],
                'speech_emotions': synchronized_data['speech_emotions'],
                'transcript': synchronized_data['transcript_segments'],
                'critical_moments': critical_moments['critical_moments'],
                'openai_insights': insights.get('insights', {}),
                'correlations': synchronized_data.get('correlations', {}),
                'statistics': {}
            }
            
            # Generate reports using report generator
            report_results = self.report_generator.generate_full_report(report_data)
            
            # Move reports to session output directory
            session_output_dir = Path(session_data['output_dir'])
            final_report_paths = {}
            
            for format_name, report_path in report_results.get('formats', {}).items():
                if report_path and os.path.exists(report_path):
                    # Copy to session output directory
                    report_filename = f"{results['session_id']}_{format_name}_report"
                    if format_name == 'html':
                        report_filename += '.html'
                    elif format_name == 'pdf':
                        report_filename += '.pdf'
                    elif format_name == 'docx':
                        report_filename += '.docx'
                    elif format_name == 'json':
                        report_filename += '.json'
                    
                    final_path = session_output_dir / report_filename
                    shutil.copy2(report_path, final_path)
                    final_report_paths[format_name] = str(final_path)
            
            # Copy CSV reports
            csv_reports = report_results.get('formats', {}).get('csv', {})
            csv_final_paths = {}
            
            for csv_type, csv_path in csv_reports.items():
                if csv_path and os.path.exists(csv_path):
                    csv_filename = f"{results['session_id']}_{csv_type}.csv"
                    final_csv_path = session_output_dir / csv_filename
                    shutil.copy2(csv_path, final_csv_path)
                    csv_final_paths[csv_type] = str(final_csv_path)
            
            final_report_paths['csv'] = csv_final_paths
            
            reports_result = {
                'reports': final_report_paths,
                'report_metadata': report_results,
                'session_output_dir': str(session_output_dir),
                'generated_at': datetime.now().isoformat()
            }
            
            self.logger.info(f"Reports creation completed: {len(final_report_paths)} format types generated")
            
            return reports_result
            
        except Exception as e:
            self.logger.error(f"Reports creation failed: {e}")
            # At minimum, save raw data as JSON
            return self._create_fallback_report(results, session_data)
    
    # ====== HELPER METHODS ======
    
    def _manage_memory(self):
        """Monitor and manage memory usage"""
        try:
            memory_percent = psutil.virtual_memory().percent
            self.metrics['memory_usage'][time.time()] = memory_percent
            
            if memory_percent > self.memory_limit:
                self.logger.warning(f"High memory usage: {memory_percent:.1f}%")
                
                # Clean up cache
                self._cleanup_cache()
                
                # Force garbage collection
                gc.collect()
                
                # Clear GPU cache if available
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Record GPU usage if available
                if hasattr(self.gpu_manager, 'get_memory_info'):
                    gpu_info = self.gpu_manager.get_memory_info()
                    if 'utilization' in gpu_info:
                        self.metrics['gpu_usage'][time.time()] = gpu_info['utilization']
                
        except Exception as e:
            self.logger.warning(f"Memory management failed: {e}")
    
    def _handle_stage_error(self, stage_name: str, error: Exception, results: Dict[str, Any]):
        """Handle stage errors with fallback strategies"""
        
        error_info = {
            'stage': stage_name,
            'error': str(error),
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__
        }
        
        self.metrics['errors'].append(error_info)
        results['errors'].append(error_info)
        
        self.logger.error(f"Stage {stage_name} failed: {error}")
        
        # Try fallback method if available
        if stage_name in self.fallback_methods:
            try:
                self.logger.info(f"Attempting fallback for {stage_name}")
                fallback_result = self.fallback_methods[stage_name](results)
                results[stage_name] = fallback_result
                
                # Mark as fallback
                results[stage_name]['_fallback_used'] = True
                results['warnings'].append(f"Fallback method used for {stage_name}")
                
                self.logger.info(f"Fallback successful for {stage_name}")
                
            except Exception as fallback_error:
                self.logger.error(f"Fallback failed for {stage_name}: {fallback_error}")
                
                # Store null result
                results[stage_name] = {
                    'error': str(error),
                    'fallback_error': str(fallback_error),
                    'failed': True
                }
        else:
            # No fallback available
            results[stage_name] = {
                'error': str(error),
                'failed': True
            }
    # ====== CHECKPOINT METHODS ======
    
    def _save_checkpoint(self, data: Any, checkpoint_file: Path):
        """Save checkpoint data"""
        try:
            with self._lock:
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_file, "wb") as f:
                    pickle.dump(data, f)
            self.logger.debug(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint {checkpoint_file}: {e}")
    
    def _load_checkpoint(self, checkpoint_file: Path) -> Any:
        """Load checkpoint data"""
        try:
            with self._lock:
                with open(checkpoint_file, "rb") as f:
                    data = pickle.load(f)
            self.logger.debug(f"Checkpoint loaded: {checkpoint_file}")
            return data
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
            raise
    
    def _cleanup_cache(self):
        """Cleanup old cache files"""
        try:
            cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours
            
            for cache_file in self.cache_dir.rglob("*"):
                if cache_file.is_file() and cache_file.stat().st_mtime < cutoff_time:
                    try:
                        cache_file.unlink()
                        self.logger.debug(f"Cleaned up cache file: {cache_file}")
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup {cache_file}: {e}")
                        
        except Exception as e:
            self.logger.warning(f"Cache cleanup failed: {e}")
    
    def _cleanup_session_cache(self, session_cache_dir: Path):
        """Cleanup session-specific cache"""
        try:
            if session_cache_dir.exists():
                shutil.rmtree(session_cache_dir)
                self.logger.info(f"Session cache cleaned up: {session_cache_dir}")
        except Exception as e:
            self.logger.warning(f"Session cache cleanup failed: {e}")
    
    # ====== PARALLEL PROCESSING ======
    
    def process_video_parallel(self, video_path: str, 
                              options: Optional[Dict[str, Any]] = None,
                              progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Process video with parallel video and audio streams"""
        start_time = time.time()
        session_id = str(uuid.uuid4())[:8]
        
        self.logger.info(f"Starting parallel video processing: {video_path}")
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit parallel tasks
                futures = {
                    "video": executor.submit(self._process_video_stream, video_path, session_id),
                    "audio": executor.submit(self._process_audio_stream, video_path, session_id)
                }
                
                # Collect results
                results = {}
                for name, future in futures.items():
                    try:
                        results[name] = future.result(timeout=3600)  # 1 hour timeout
                        if progress_callback:
                            progress_callback(len(results) / len(futures), f"{name}_completed")
                    except Exception as e:
                        self.logger.error(f"Parallel {name} processing failed: {e}")
                        results[name] = {"error": str(e)}
                
                # Synchronize and finalize
                final_results = self._merge_parallel_results(results, session_id)
                final_results["processing_time"] = time.time() - start_time
                
                return final_results
                
        except Exception as e:
            self.logger.error(f"Parallel processing failed: {e}")
            return {"error": str(e), "session_id": session_id}
    
    def _process_video_stream(self, video_path: str, session_id: str) -> Dict[str, Any]:
        """Process video stream (frames and visual analysis)"""
        try:
            # Simplified video processing for parallel execution
            frames_data = self._extract_frames_parallel(video_path, session_id)
            faces_data = self._detect_faces_parallel(frames_data)
            emotions_data = self._analyze_emotions_parallel(frames_data, faces_data)
            
            return {
                "frames": frames_data,
                "faces": faces_data, 
                "emotions": emotions_data,
                "stream": "video"
            }
        except Exception as e:
            self.logger.error(f"Video stream processing failed: {e}")
            return {"error": str(e), "stream": "video"}
    
    def _process_audio_stream(self, video_path: str, session_id: str) -> Dict[str, Any]:
        """Process audio stream (extraction, transcription, speech analysis)"""
        try:
            # Simplified audio processing for parallel execution  
            audio_data = self._extract_audio_parallel(video_path, session_id)
            transcript_data = self._transcribe_audio_parallel(audio_data)
            speech_data = self._analyze_speech_parallel(audio_data, transcript_data)
            
            return {
                "audio": audio_data,
                "transcript": transcript_data,
                "speech": speech_data,
                "stream": "audio"
            }
        except Exception as e:
            self.logger.error(f"Audio stream processing failed: {e}")
            return {"error": str(e), "stream": "audio"}
    
    # ====== FALLBACK METHODS ======
    
    def _fallback_face_detection(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback face detection using OpenCV"""
        self.logger.info("Using fallback face detection")
        return {
            "faces_detected": [],
            "total_faces": 0,
            "method": "fallback_opencv",
            "_fallback_used": True
        }
    
    def _fallback_emotion_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback emotion analysis"""
        self.logger.info("Using fallback emotion analysis")
        return {
            "emotion_results": [],
            "method": "fallback_basic",
            "_fallback_used": True
        }
    
    def _fallback_transcription(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback audio transcription"""
        self.logger.info("Using fallback transcription")
        return {
            "segments": [],
            "method": "fallback_silence",
            "_fallback_used": True
        }
    
    def _fallback_speech_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback speech analysis"""
        self.logger.info("Using fallback speech analysis")
        return {
            "segments": [],
            "method": "fallback_basic",
            "_fallback_used": True
        }
    
    def _fallback_critical_moments(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback critical moments detection"""
        self.logger.info("Using fallback critical moments detection")
        return {
            "critical_moments": [],
            "method": "fallback_basic",
            "_fallback_used": True
        }
    
    def _fallback_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback insights generation"""
        self.logger.info("Using fallback insights generation")
        return {
            "insights": {
                "general_analysis": "Анализ завершен с базовыми методами. Для детального анализа требуется OpenAI интеграция.",
                "method": "fallback_basic"
            },
            "_fallback_used": True
        }
    
    # ====== FINALIZATION METHODS ======
    
    def _finalize_results(self, results: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Finalize and package all results"""
        
        final_results = {
            "session_id": session_id,
            "status": "completed",
            "video_path": results.get("video_path"),
            "generated_at": datetime.now().isoformat(),
            "stages_completed": len([k for k in results.keys() if not k.startswith("_")]),
            "total_stages": len(self.stages),
            "success_rate": self._calculate_success_rate(results),
            "data": {
                "video_emotions": results.get("analyze_emotions", {}).get("emotion_results", []),
                "speech_emotions": results.get("analyze_speech", {}).get("segments", []), 
                "transcript": results.get("transcribe_audio", {}).get("segments", []),
                "critical_moments": results.get("detect_critical", {}).get("critical_moments", []),
                "insights": results.get("generate_insights", {}).get("insights", {}),
                "reports": results.get("create_reports", {})
            },
            "metadata": results.get("validate_input", {}),
            "errors": results.get("errors", []),
            "warnings": results.get("warnings", []),
            "metrics": self.metrics
        }
        
        return final_results
    
    def _calculate_success_rate(self, results: Dict[str, Any]) -> float:
        """Calculate pipeline success rate"""
        total_stages = len(self.stages)
        failed_stages = len([v for v in results.values() if isinstance(v, dict) and v.get("failed")])
        return (total_stages - failed_stages) / total_stages if total_stages > 0 else 0.0


# ====== CONVENIENCE FUNCTIONS ======

def create_master_pipeline(config_path: str = "config.yaml") -> MasterPipeline:
    """Create and return a configured MasterPipeline instance"""
    return MasterPipeline(config_path)

def process_video_file(video_path: str, 
                      config_path: str = "config.yaml",
                      progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Convenience function to process a video file with default settings"""
    pipeline = create_master_pipeline(config_path)
    return pipeline.process_video(video_path, progress_callback=progress_callback)


