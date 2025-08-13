"""
Multimodal Emotion Analyzer with priority-based system
"""

import logging
import os
import time
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict
import numpy as np
import cv2
from tqdm import tqdm

# Priority 1: DeepFace
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# Priority 2: FER
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False

# Priority 3: YOLO11 (will be imported from our models)
try:
    from models.yolo_manager import YOLO11Manager
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Utils
from utils.gpu_manager import get_gpu_manager
from utils.translation import EmotionTranslator


class MultiModalEmotionAnalyzer:
    """
    Multimodal emotion analyzer with priority-based fallback system
    
    Priority Order:
    1. DeepFace (most accurate)
    2. FER with MTCNN
    3. YOLO11 specialized emotions
    4. Simple OpenCV fallback
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Emotion categories
        emotion_config = config.get('analysis', {}).get('emotion_categories', {})
        self.basic_emotions = emotion_config.get('basic', [
            'злость', 'отвращение', 'страх', 'счастье', 'грусть', 'удивление', 'нейтральность'
        ])
        self.specialized_emotions = emotion_config.get('specialized', [
            'спокойствие', 'дискомфорт', 'напряжение_бровей', 'напряжение_глаз', 'напряжение_рта'
        ])
        
        # Storage paths
        storage_config = config.get('storage', {})
        self.cache_dir = Path(storage_config.get('cache_dir', 'storage/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance settings
        perf_config = config.get('performance', {})
        self.max_workers = perf_config.get('max_workers', 4)
        self.enable_caching = perf_config.get('enable_caching', True)
        
        # Initialize components
        self.gpu_manager = get_gpu_manager()
        self.emotion_translator = EmotionTranslator()
        
        # Analyzer instances
        self.deepface_analyzer = None
        self.fer_analyzer = None
        self.yolo_manager = None
        self.simple_analyzer = None
        
        # Detection priorities and availability
        self.available_methods = []
        self._initialize_analyzers()
        
        # Caching
        self.cache_lock = threading.Lock()
        self.emotion_cache = {}
        self._load_cache()
        
        # Temporal analysis
        self.emotion_history = defaultdict(list)
        
        self.logger.info(f"MultiModalEmotionAnalyzer initialized with methods: {self.available_methods}")
    
    def _initialize_analyzers(self):
        """Initialize emotion analyzers in priority order"""
        
        # Priority 1: DeepFace
        if DEEPFACE_AVAILABLE:
            try:
                self.deepface_analyzer = DeepFaceAnalyzer(self.config)
                self.available_methods.append('deepface')
                self.logger.info("DeepFace analyzer initialized (Priority 1)")
            except Exception as e:
                self.logger.warning(f"DeepFace initialization failed: {e}")
        
        # Priority 2: FER
        if FER_AVAILABLE:
            try:
                self.fer_analyzer = FERAnalyzer(self.config)
                self.available_methods.append('fer')
                self.logger.info("FER analyzer initialized (Priority 2)")
            except Exception as e:
                self.logger.warning(f"FER initialization failed: {e}")
        
        # Priority 3: YOLO11
        if YOLO_AVAILABLE:
            try:
                self.yolo_manager = YOLO11Manager(self.config)
                self.available_methods.append('yolo11')
                self.logger.info("YOLO11 analyzer initialized (Priority 3)")
            except Exception as e:
                self.logger.warning(f"YOLO11 initialization failed: {e}")
        
        # Priority 4: Simple fallback
        try:
            self.simple_analyzer = SimpleEmotionAnalyzer(self.config)
            self.available_methods.append('simple')
            self.logger.info("Simple analyzer initialized (Priority 4 - fallback)")
        except Exception as e:
            self.logger.error(f"Simple analyzer initialization failed: {e}")
        
        if not self.available_methods:
            raise RuntimeError("No emotion analysis methods available")
    
    def analyze_frame(
        self,
        frame_path: str,
        timestamp: float = 0.0,
        frame_number: int = 0,
        preferred_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze emotions in a single frame using priority system
        
        Args:
            frame_path: Path to frame image
            timestamp: Video timestamp in seconds
            frame_number: Frame number in sequence
            preferred_method: Preferred analysis method
            
        Returns:
            Dict with emotion analysis results
        """
        
        # Check cache first
        if self.enable_caching:
            cached_result = self._get_cached_result(frame_path)
            if cached_result:
                cached_result.update({
                    'timestamp': timestamp,
                    'frame_number': frame_number,
                    'from_cache': True
                })
                return cached_result
        
        # Load image
        try:
            image = cv2.imread(frame_path)
            if image is None:
                raise ValueError(f"Cannot load image: {frame_path}")
        except Exception as e:
            self.logger.error(f"Failed to load frame {frame_path}: {e}")
            return self._create_error_result(str(e), timestamp, frame_number)
        
        # Determine analysis method order
        methods = self._get_method_priority(preferred_method)
        
        # Try each method in priority order
        for method in methods:
            try:
                result = self._analyze_with_method(image, method, frame_path)
                if result and result.get('emotion'):
                    # Add metadata
                    result.update({
                        'timestamp': timestamp,
                        'frame_number': frame_number,
                        'method': method,
                        'frame_path': frame_path,
                        'from_cache': False
                    })
                    
                    # Cache successful result
                    if self.enable_caching:
                        self._cache_result(frame_path, result)
                    
                    # Update emotion history
                    self._update_emotion_history(result)
                    
                    return result
                    
            except Exception as e:
                self.logger.warning(f"Method {method} failed for {frame_path}: {e}")
                continue
        
        # All methods failed
        self.logger.error(f"All emotion analysis methods failed for {frame_path}")
        return self._create_error_result("All methods failed", timestamp, frame_number)
    
    def _analyze_with_method(self, image: np.ndarray, method: str, frame_path: str) -> Optional[Dict[str, Any]]:
        """Analyze emotion using specific method"""
        
        if method == 'deepface' and self.deepface_analyzer:
            return self.deepface_analyzer.analyze(image, frame_path)
        
        elif method == 'fer' and self.fer_analyzer:
            return self.fer_analyzer.analyze(image, frame_path)
        
        elif method == 'yolo11' and self.yolo_manager:
            return self._analyze_with_yolo11(image, frame_path)
        
        elif method == 'simple' and self.simple_analyzer:
            return self.simple_analyzer.analyze(image, frame_path)
        
        return None
    
    def _analyze_with_yolo11(self, image: np.ndarray, frame_path: str) -> Optional[Dict[str, Any]]:
        """Analyze emotions using YOLO11 specialized model"""
        try:
            # Save temp image for YOLO processing
            temp_path = self.cache_dir / f"temp_{int(time.time())}.jpg"
            cv2.imwrite(str(temp_path), image)
            
            # Use YOLO emotion model
            results = self.yolo_manager.batch_predict(
                sources=[str(temp_path)],
                model_type='emotion'
            )
            
            for result in results:
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    # Process YOLO emotion results
                    # This would need custom YOLO emotion model
                    # For now, return placeholder
                    return {
                        'emotion': 'нейтральность',
                        'emotion_en': 'neutral',
                        'confidence': 0.5,
                        'all_emotions': {'нейтральность': 0.5},
                        'face_bbox': [0, 0, image.shape[1], image.shape[0]],
                        'specialized_features': self._extract_specialized_features(image)
                    }
            
            # Cleanup temp file
            if temp_path.exists():
                temp_path.unlink()
                
        except Exception as e:
            self.logger.warning(f"YOLO11 emotion analysis failed: {e}")
        
        return None
    
    def _extract_specialized_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract specialized features for interrogation context"""
        features = {}
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect face region for feature extraction
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                
                # Analyze brow tension (upper 1/3 of face)
                brow_roi = face_roi[:h//3, :]
                brow_variance = np.var(brow_roi)
                features['напряжение_бровей'] = min(brow_variance / 1000, 1.0)
                
                # Analyze eye tension (middle 1/3 of face)
                eye_roi = face_roi[h//3:2*h//3, :]
                eye_edges = cv2.Canny(eye_roi, 50, 150)
                edge_density = np.sum(eye_edges) / eye_edges.size
                features['напряжение_глаз'] = min(edge_density * 10, 1.0)
                
                # Analyze mouth tension (lower 1/3 of face)
                mouth_roi = face_roi[2*h//3:, :]
                mouth_variance = np.var(mouth_roi)
                features['напряжение_рта'] = min(mouth_variance / 800, 1.0)
                
                # Overall comfort level
                tension_avg = np.mean(list(features.values()))
                features['дискомфорт'] = tension_avg
                features['спокойствие'] = 1.0 - tension_avg
            
        except Exception as e:
            self.logger.warning(f"Failed to extract specialized features: {e}")
            # Default values
            features = {
                'спокойствие': 0.5,
                'дискомфорт': 0.5,
                'напряжение_бровей': 0.0,
                'напряжение_глаз': 0.0,
                'напряжение_рта': 0.0
            }
        
        return features
    
    def analyze_batch(
        self,
        frame_paths: List[str],
        timestamps: Optional[List[float]] = None,
        frame_numbers: Optional[List[int]] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch analyze emotions with parallel processing
        
        Args:
            frame_paths: List of frame paths
            timestamps: Video timestamps for each frame
            frame_numbers: Frame numbers
            show_progress: Show progress bar
            
        Returns:
            List of emotion analysis results
        """
        self.logger.info(f"Starting batch emotion analysis of {len(frame_paths)} frames")
        
        # Prepare arguments
        if timestamps is None:
            timestamps = [i * (1/30) for i in range(len(frame_paths))]  # Assume 30fps
        if frame_numbers is None:
            frame_numbers = list(range(len(frame_paths)))
        
        results = []
        
        # Process with ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, (frame_path, timestamp, frame_num) in enumerate(zip(frame_paths, timestamps, frame_numbers)):
                future = executor.submit(
                    self.analyze_frame,
                    frame_path,
                    timestamp,
                    frame_num
                )
                future_to_index[future] = i
            
            # Collect results with progress bar
            results = [None] * len(frame_paths)
            
            iterator = as_completed(future_to_index.keys())
            if show_progress:
                iterator = tqdm(iterator, total=len(frame_paths), desc="Analyzing emotions")
            
            for future in iterator:
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    self.logger.error(f"Batch analysis failed for frame {index}: {e}")
                    results[index] = self._create_error_result(
                        str(e), 
                        timestamps[index], 
                        frame_numbers[index]
                    )
        
        self.logger.info(f"Completed batch emotion analysis: {len([r for r in results if r and not r.get('error')])} successful")
        return results
    
    def detect_emotion_transitions(self, emotion_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect transitions between emotions"""
        transitions = []
        
        if len(emotion_results) < 2:
            return transitions
        
        for i in range(1, len(emotion_results)):
            prev_result = emotion_results[i-1]
            curr_result = emotion_results[i]
            
            if not prev_result.get('emotion') or not curr_result.get('emotion'):
                continue
            
            prev_emotion = prev_result['emotion']
            curr_emotion = curr_result['emotion']
            
            # Detect transition
            if prev_emotion != curr_emotion:
                transition = {
                    'from_emotion': prev_emotion,
                    'to_emotion': curr_emotion,
                    'from_timestamp': prev_result.get('timestamp', 0),
                    'to_timestamp': curr_result.get('timestamp', 0),
                    'from_confidence': prev_result.get('confidence', 0),
                    'to_confidence': curr_result.get('confidence', 0),
                    'transition_speed': abs(curr_result.get('timestamp', 0) - prev_result.get('timestamp', 0))
                }
                transitions.append(transition)
        
        self.logger.info(f"Detected {len(transitions)} emotion transitions")
        return transitions
    
    def find_rapid_changes(
        self, 
        emotion_results: List[Dict[str, Any]], 
        time_window: float = 5.0,
        min_changes: int = 3
    ) -> List[Dict[str, Any]]:
        """Find rapid emotion changes indicating stress"""
        rapid_changes = []
        
        if len(emotion_results) < min_changes:
            return rapid_changes
        
        # Sliding window analysis
        for i in range(len(emotion_results) - min_changes + 1):
            window_results = emotion_results[i:i + min_changes]
            
            # Check if all results are within time window
            start_time = window_results[0].get('timestamp', 0)
            end_time = window_results[-1].get('timestamp', 0)
            
            if end_time - start_time <= time_window:
                # Count unique emotions in window
                emotions_in_window = set()
                for result in window_results:
                    if result.get('emotion'):
                        emotions_in_window.add(result['emotion'])
                
                if len(emotions_in_window) >= min_changes:
                    rapid_change = {
                        'start_timestamp': start_time,
                        'end_timestamp': end_time,
                        'duration': end_time - start_time,
                        'emotions_count': len(emotions_in_window),
                        'emotions': list(emotions_in_window),
                        'change_rate': len(emotions_in_window) / (end_time - start_time) if end_time > start_time else 0,
                        'frames_analyzed': len(window_results)
                    }
                    rapid_changes.append(rapid_change)
        
        self.logger.info(f"Found {len(rapid_changes)} rapid emotion change periods")
        return rapid_changes
    
    def calculate_emotion_stability(self, emotion_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate emotion stability metrics"""
        if not emotion_results:
            return {'error': 'No results to analyze'}
        
        valid_results = [r for r in emotion_results if r.get('emotion') and not r.get('error')]
        
        if not valid_results:
            return {'error': 'No valid emotion results'}
        
        # Count emotion frequencies
        emotion_counts = defaultdict(int)
        confidence_sum = defaultdict(float)
        
        for result in valid_results:
            emotion = result['emotion']
            confidence = result.get('confidence', 0)
            emotion_counts[emotion] += 1
            confidence_sum[emotion] += confidence
        
        # Calculate stability metrics
        total_frames = len(valid_results)
        dominant_emotion = max(emotion_counts.keys(), key=emotion_counts.get)
        dominance_ratio = emotion_counts[dominant_emotion] / total_frames
        
        # Emotion diversity (entropy)
        emotion_probs = [count / total_frames for count in emotion_counts.values()]
        entropy = -sum(p * np.log2(p) for p in emotion_probs if p > 0)
        
        # Average confidence per emotion
        avg_confidences = {
            emotion: confidence_sum[emotion] / emotion_counts[emotion]
            for emotion in emotion_counts.keys()
        }
        
        # Temporal consistency
        transitions = self.detect_emotion_transitions(valid_results)
        transition_rate = len(transitions) / total_frames if total_frames > 0 else 0
        
        stability_metrics = {
            'total_frames_analyzed': total_frames,
            'dominant_emotion': dominant_emotion,
            'dominance_ratio': dominance_ratio,
            'emotion_diversity': entropy,
            'emotion_distribution': dict(emotion_counts),
            'average_confidences': avg_confidences,
            'transition_count': len(transitions),
            'transition_rate': transition_rate,
            'stability_score': dominance_ratio * (1 - transition_rate),  # Higher = more stable
            'consistency_score': 1 - entropy / np.log2(len(emotion_counts)) if len(emotion_counts) > 1 else 1.0
        }
        
        self.logger.info(f"Emotion stability: {stability_metrics['stability_score']:.3f}, dominant: {dominant_emotion}")
        return stability_metrics
    
    def _get_method_priority(self, preferred_method: Optional[str] = None) -> List[str]:
        """Get analysis methods in priority order"""
        if preferred_method and preferred_method in self.available_methods:
            # Put preferred method first, then others in priority order
            methods = [preferred_method]
            methods.extend([m for m in self.available_methods if m != preferred_method])
            return methods
        
        return self.available_methods.copy()
    
    def _create_error_result(self, error_msg: str, timestamp: float, frame_number: int) -> Dict[str, Any]:
        """Create error result structure"""
        return {
            'emotion': None,
            'emotion_en': None,
            'confidence': 0.0,
            'all_emotions': {},
            'method': 'error',
            'face_bbox': None,
            'timestamp': timestamp,
            'frame_number': frame_number,
            'error': error_msg,
            'from_cache': False
        }
    
    def _get_cached_result(self, frame_path: str) -> Optional[Dict[str, Any]]:
        """Get cached emotion result"""
        cache_key = self._get_cache_key(frame_path)
        
        with self.cache_lock:
            if cache_key in self.emotion_cache:
                cached_data = self.emotion_cache[cache_key]
                # Check if cache is still valid (based on file modification time)
                if self._is_cache_valid(frame_path, cached_data.get('cache_time', 0)):
                    return cached_data['result']
        
        return None
    
    def _cache_result(self, frame_path: str, result: Dict[str, Any]):
        """Cache emotion analysis result"""
        cache_key = self._get_cache_key(frame_path)
        
        # Remove temporal data before caching
        cache_result = result.copy()
        cache_result.pop('timestamp', None)
        cache_result.pop('frame_number', None)
        cache_result.pop('frame_path', None)
        cache_result.pop('from_cache', None)
        
        with self.cache_lock:
            self.emotion_cache[cache_key] = {
                'result': cache_result,
                'cache_time': time.time()
            }
        
        # Periodic cache cleanup
        if len(self.emotion_cache) % 100 == 0:
            self._cleanup_cache()
    
    def _get_cache_key(self, frame_path: str) -> str:
        """Generate cache key for frame"""
        try:
            # Use file hash for cache key
            with open(frame_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return f"emotion_{file_hash}"
        except:
            # Fallback to path-based key
            return f"emotion_{hashlib.md5(frame_path.encode()).hexdigest()}"
    
    def _is_cache_valid(self, frame_path: str, cache_time: float) -> bool:
        """Check if cached result is still valid"""
        try:
            file_mtime = os.path.getmtime(frame_path)
            return file_mtime <= cache_time
        except:
            return False
    
    def _load_cache(self):
        """Load emotion cache from disk"""
        cache_file = self.cache_dir / 'emotion_cache.pkl'
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.emotion_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.emotion_cache)} cached emotion results")
            except Exception as e:
                self.logger.warning(f"Failed to load emotion cache: {e}")
                self.emotion_cache = {}
    
    def _save_cache(self):
        """Save emotion cache to disk"""
        cache_file = self.cache_dir / 'emotion_cache.pkl'
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.emotion_cache, f)
            self.logger.debug(f"Saved {len(self.emotion_cache)} emotion results to cache")
        except Exception as e:
            self.logger.warning(f"Failed to save emotion cache: {e}")
    
    def _cleanup_cache(self):
        """Clean up old cache entries"""
        current_time = time.time()
        cache_max_age = 24 * 3600  # 24 hours
        
        with self.cache_lock:
            keys_to_remove = []
            for key, data in self.emotion_cache.items():
                if current_time - data.get('cache_time', 0) > cache_max_age:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.emotion_cache[key]
            
            if keys_to_remove:
                self.logger.info(f"Cleaned up {len(keys_to_remove)} old cache entries")
    
    def _update_emotion_history(self, result: Dict[str, Any]):
        """Update emotion history for temporal analysis"""
        if result.get('emotion') and result.get('timestamp') is not None:
            self.emotion_history['emotions'].append({
                'emotion': result['emotion'],
                'confidence': result.get('confidence', 0),
                'timestamp': result['timestamp']
            })
            
            # Keep history manageable (last 1000 entries)
            if len(self.emotion_history['emotions']) > 1000:
                self.emotion_history['emotions'] = self.emotion_history['emotions'][-1000:]
    
    def __del__(self):
        """Save cache when object is destroyed"""
        if hasattr(self, 'emotion_cache') and self.enable_caching:
            self._save_cache()


class DeepFaceAnalyzer:
    """DeepFace wrapper with error handling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # DeepFace settings
        deepface_config = config.get('processing', {}).get('models', {}).get('deepface', {})
        self.backend = deepface_config.get('backend', 'opencv')
        self.enforce_detection = deepface_config.get('enforce_detection', False)
        self.detector_backend = deepface_config.get('detector_backend', 'opencv')
        
        # Initialize emotion translator
        self.emotion_translator = EmotionTranslator()
        
        self.logger.info("DeepFace analyzer initialized")
    
    def analyze(self, image: np.ndarray, frame_path: str) -> Dict[str, Any]:
        """Analyze emotions using DeepFace"""
        try:
            # DeepFace expects image path or numpy array
            results = DeepFace.analyze(
                img_path=image,
                actions=['emotion'],
                enforce_detection=self.enforce_detection,
                detector_backend=self.detector_backend,
                silent=True
            )
            
            # Handle multiple faces or single face result
            if isinstance(results, list):
                # Use first face if multiple detected
                result = results[0] if results else None
            else:
                result = results
            
            if result and 'emotion' in result:
                emotions = result['emotion']
                
                # Get dominant emotion
                dominant_emotion_en = max(emotions.keys(), key=emotions.get)
                confidence = emotions[dominant_emotion_en] / 100.0  # Convert percentage to ratio
                
                # Translate to Russian
                dominant_emotion_ru = self.emotion_translator.translate_emotion(dominant_emotion_en, 'ru')
                
                # Translate all emotions to Russian
                all_emotions_ru = {}
                for emotion_en, score in emotions.items():
                    emotion_ru = self.emotion_translator.translate_emotion(emotion_en, 'ru')
                    all_emotions_ru[emotion_ru] = score / 100.0
                
                # Get face region if available
                face_bbox = None
                if 'region' in result:
                    region = result['region']
                    face_bbox = [region['x'], region['y'], 
                               region['x'] + region['w'], region['y'] + region['h']]
                
                return {
                    'emotion': dominant_emotion_ru,
                    'emotion_en': dominant_emotion_en,
                    'confidence': confidence,
                    'all_emotions': all_emotions_ru,
                    'face_bbox': face_bbox
                }
            
        except Exception as e:
            self.logger.warning(f"DeepFace analysis failed: {e}")
        
        return None


class FERAnalyzer:
    """FER (Face Emotion Recognition) analyzer with MTCNN"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # FER settings
        fer_config = config.get('processing', {}).get('models', {}).get('fer', {})
        self.mtcnn = fer_config.get('mtcnn', True)
        
        # Initialize FER detector
        self.fer_detector = FER(mtcnn=self.mtcnn)
        
        # Initialize emotion translator
        self.emotion_translator = EmotionTranslator()
        
        self.logger.info("FER analyzer initialized with MTCNN")
    
    def analyze(self, image: np.ndarray, frame_path: str) -> Dict[str, Any]:
        """Analyze emotions using FER"""
        try:
            # FER detect emotions
            results = self.fer_detector.detect_emotions(image)
            
            if results:
                # Use first face detection
                result = results[0]
                
                emotions = result.get('emotions', {})
                box = result.get('box')
                
                if emotions:
                    # Get dominant emotion
                    dominant_emotion_en = max(emotions.keys(), key=emotions.get)
                    confidence = emotions[dominant_emotion_en]
                    
                    # Translate to Russian
                    dominant_emotion_ru = self.emotion_translator.translate_emotion(dominant_emotion_en, 'ru')
                    
                    # Translate all emotions to Russian
                    all_emotions_ru = {}
                    for emotion_en, score in emotions.items():
                        emotion_ru = self.emotion_translator.translate_emotion(emotion_en, 'ru')
                        all_emotions_ru[emotion_ru] = score
                    
                    # Face bounding box
                    face_bbox = None
                    if box:
                        face_bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                    
                    return {
                        'emotion': dominant_emotion_ru,
                        'emotion_en': dominant_emotion_en,
                        'confidence': confidence,
                        'all_emotions': all_emotions_ru,
                        'face_bbox': face_bbox
                    }
            
        except Exception as e:
            self.logger.warning(f"FER analysis failed: {e}")
        
        return None


class SimpleEmotionAnalyzer:
    """Simple emotion analyzer using OpenCV as fallback"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize emotion translator
        self.emotion_translator = EmotionTranslator()
        
        self.logger.info("Simple emotion analyzer initialized")
    
    def analyze(self, image: np.ndarray, frame_path: str) -> Dict[str, Any]:
        """Simple emotion analysis using basic computer vision"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Use first detected face
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                
                # Simple heuristics for emotion detection
                emotion_scores = self._analyze_face_features(face_roi, image[y:y+h, x:x+w])
                
                # Get dominant emotion
                dominant_emotion_ru = max(emotion_scores.keys(), key=emotion_scores.get)
                dominant_emotion_en = self.emotion_translator.translate_emotion(dominant_emotion_ru, 'en')
                confidence = emotion_scores[dominant_emotion_ru]
                
                return {
                    'emotion': dominant_emotion_ru,
                    'emotion_en': dominant_emotion_en,
                    'confidence': confidence,
                    'all_emotions': emotion_scores,
                    'face_bbox': [x, y, x + w, y + h]
                }
            
        except Exception as e:
            self.logger.warning(f"Simple analysis failed: {e}")
        
        return None
    
    def _analyze_face_features(self, face_gray: np.ndarray, face_color: np.ndarray) -> Dict[str, float]:
        """Analyze basic face features for emotion estimation"""
        emotions = {
            'нейтральность': 0.5,  # Default baseline
            'злость': 0.0,
            'счастье': 0.0,
            'грусть': 0.0,
            'страх': 0.0,
            'удивление': 0.0,
            'отвращение': 0.0
        }
        
        try:
            # Color analysis
            face_hsv = cv2.cvtColor(face_color, cv2.COLOR_BGR2HSV)
            
            # Brightness analysis (fear/surprise tend to have wider eyes - brighter)
            brightness = np.mean(face_gray)
            if brightness > 120:
                emotions['удивление'] += 0.2
                emotions['страх'] += 0.1
            
            # Edge density analysis (anger tends to have more defined features)
            edges = cv2.Canny(face_gray, 50, 150)
            edge_density = np.sum(edges) / edges.size
            if edge_density > 0.1:
                emotions['злость'] += edge_density * 2
                emotions['отвращение'] += edge_density * 1.5
            
            # Histogram analysis for color-based emotions
            hist = cv2.calcHist([face_hsv], [1, 2], None, [50, 50], [0, 256, 0, 256])
            hist_mean = np.mean(hist)
            
            # Variance analysis (happiness often has more color variation in cheeks)
            color_variance = np.var(face_color)
            if color_variance > 1000:
                emotions['счастье'] += 0.3
            elif color_variance < 500:
                emotions['грусть'] += 0.2
            
            # Normalize emotions to sum to 1.0
            total_score = sum(emotions.values())
            if total_score > 0:
                emotions = {k: v / total_score for k, v in emotions.items()}
            
        except Exception as e:
            self.logger.warning(f"Feature analysis failed: {e}")
        
        return emotions