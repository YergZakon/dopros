"""
Advanced Speech Emotion Analyzer Module
Модуль анализа эмоций в речи с мультимодальной поддержкой
"""

import os
import sys
import json
import time
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Check for optional dependencies
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available. Using basic audio processing.")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger


class AdvancedSpeechEmotionAnalyzer:
    """Advanced speech emotion analysis with prosodic features"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize speech emotion analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Analysis parameters
        self.sample_rate = config.get('audio', {}).get('sample_rate', 16000)
        self.segment_duration = config.get('audio', {}).get('segment_duration', 3.0)
        self.overlap = config.get('audio', {}).get('overlap', 0.5)
        self.min_segment_duration = config.get('audio', {}).get('min_segment_duration', 0.5)
        
        # Emotion categories
        self.emotion_categories = [
            'нейтральность', 'счастье', 'грусть', 'злость',
            'страх', 'удивление', 'отвращение', 'презрение',
            'спокойствие', 'возбуждение', 'фрустрация', 'замешательство'
        ]
        
        # Feature extraction settings
        self.n_mfcc = 13
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        
        # Voice quality thresholds
        self.voice_quality_thresholds = {
            'trembling': {'pitch_var': 60, 'energy_var': 0.02},
            'strained': {'spectral_centroid': 3500, 'zcr': 0.15},
            'harsh': {'spectral_centroid': 4000, 'energy': 0.01},
            'breathy': {'spectral_rolloff': 2000}
        }
        
        # Initialize classifier (placeholder for future ML model)
        self.classifier = None
        
        self.logger.info("Speech emotion analyzer initialized")
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with fallback methods
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            if LIBROSA_AVAILABLE:
                audio, sr = librosa.load(audio_path, sr=self.sample_rate)
                return audio, sr
            elif SOUNDFILE_AVAILABLE:
                audio, sr = sf.read(audio_path)
                # Resample if needed
                if sr != self.sample_rate:
                    # Simple resampling (not as good as librosa)
                    ratio = self.sample_rate / sr
                    new_length = int(len(audio) * ratio)
                    indices = np.linspace(0, len(audio) - 1, new_length)
                    audio = np.interp(indices, np.arange(len(audio)), audio)
                    sr = self.sample_rate
                return audio, sr
            else:
                # Last resort: try with wave module
                import wave
                with wave.open(audio_path, 'rb') as wav_file:
                    sr = wav_file.getframerate()
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                    audio = audio / 32768.0  # Normalize
                    
                    # Simple resampling if needed
                    if sr != self.sample_rate:
                        ratio = self.sample_rate / sr
                        new_length = int(len(audio) * ratio)
                        indices = np.linspace(0, len(audio) - 1, new_length)
                        audio = np.interp(indices, np.arange(len(audio)), audio)
                        sr = self.sample_rate
                    
                    return audio, sr
                    
        except Exception as e:
            self.logger.error(f"Failed to load audio: {e}")
            raise
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract comprehensive audio features
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        try:
            if LIBROSA_AVAILABLE:
                # MFCC features
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
                features['mfcc_mean'] = np.mean(mfccs, axis=1)
                features['mfcc_std'] = np.std(mfccs, axis=1)
                features['mfcc_delta'] = np.mean(librosa.feature.delta(mfccs), axis=1)
                
                # Spectral features
                spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
                features['spectral_centroid_mean'] = np.mean(spectral_centroid)
                features['spectral_centroid_std'] = np.std(spectral_centroid)
                
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
                features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
                features['spectral_rolloff_std'] = np.std(spectral_rolloff)
                
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
                features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
                features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
                
                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(audio)
                features['zcr_mean'] = np.mean(zcr)
                features['zcr_std'] = np.std(zcr)
                
                # Energy features
                rms = librosa.feature.rms(y=audio)
                features['rms_mean'] = np.mean(rms)
                features['rms_std'] = np.std(rms)
                features['energy'] = np.sum(audio**2) / len(audio)
                
                # Pitch features
                pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if pitch_values:
                    features['pitch_mean'] = np.mean(pitch_values)
                    features['pitch_std'] = np.std(pitch_values)
                    features['pitch_min'] = np.min(pitch_values)
                    features['pitch_max'] = np.max(pitch_values)
                    features['pitch_range'] = features['pitch_max'] - features['pitch_min']
                else:
                    features['pitch_mean'] = 0
                    features['pitch_std'] = 0
                    features['pitch_range'] = 0
                
                # Tempo
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                features['tempo'] = tempo
                
            else:
                # Basic feature extraction without librosa
                # Energy
                features['energy'] = np.sum(audio**2) / len(audio)
                features['rms_mean'] = np.sqrt(np.mean(audio**2))
                features['rms_std'] = np.std(np.sqrt(np.abs(audio)))
                
                # Zero crossing rate (manual)
                zero_crossings = np.where(np.diff(np.sign(audio)))[0]
                features['zcr_mean'] = len(zero_crossings) / len(audio)
                features['zcr_std'] = 0
                
                # Basic spectral features using FFT
                fft = np.fft.fft(audio)
                magnitude = np.abs(fft[:len(fft)//2])
                frequencies = np.fft.fftfreq(len(audio), 1/sr)[:len(fft)//2]
                
                # Spectral centroid
                if np.sum(magnitude) > 0:
                    features['spectral_centroid_mean'] = np.sum(frequencies * magnitude) / np.sum(magnitude)
                else:
                    features['spectral_centroid_mean'] = 0
                
                features['spectral_centroid_std'] = 0
                
                # Placeholder values for other features
                features['mfcc_mean'] = np.zeros(self.n_mfcc)
                features['mfcc_std'] = np.zeros(self.n_mfcc)
                features['spectral_rolloff_mean'] = 0
                features['spectral_bandwidth_mean'] = 0
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['tempo'] = 0
                
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
            # Return default features
            features = {
                'mfcc_mean': np.zeros(self.n_mfcc),
                'mfcc_std': np.zeros(self.n_mfcc),
                'spectral_centroid_mean': 0,
                'spectral_centroid_std': 0,
                'spectral_rolloff_mean': 0,
                'spectral_bandwidth_mean': 0,
                'zcr_mean': 0,
                'zcr_std': 0,
                'rms_mean': 0,
                'rms_std': 0,
                'energy': 0,
                'pitch_mean': 0,
                'pitch_std': 0,
                'pitch_range': 0,
                'tempo': 0
            }
        
        return features
    
    def classify_emotion(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify emotion from extracted features
        
        Args:
            features: Extracted audio features
            
        Returns:
            Emotion classification results
        """
        # For now, use rule-based classification
        # In production, this would use a trained ML model
        
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_categories}
        
        try:
            # Energy-based rules
            energy = features.get('energy', 0)
            rms_mean = features.get('rms_mean', 0)
            
            # Pitch-based rules
            pitch_mean = features.get('pitch_mean', 0)
            pitch_std = features.get('pitch_std', 0)
            
            # Spectral features
            spectral_centroid = features.get('spectral_centroid_mean', 0)
            zcr_mean = features.get('zcr_mean', 0)
            
            # Rule-based classification
            if energy > 0.005 and pitch_mean > 200:
                if pitch_std > 50:  # High pitch variation
                    emotion_scores['страх'] += 0.6
                    emotion_scores['возбуждение'] += 0.4
                else:  # Stable high pitch
                    emotion_scores['счастье'] += 0.7
                    emotion_scores['удивление'] += 0.3
            
            elif energy > 0.003 and spectral_centroid > 2500:
                # High energy, high frequency content
                emotion_scores['злость'] += 0.7
                emotion_scores['фрустрация'] += 0.5
            
            elif energy < 0.001:
                # Low energy
                emotion_scores['грусть'] += 0.6
                emotion_scores['спокойствие'] += 0.4
            
            elif zcr_mean > 0.08:
                # High zero crossing rate
                emotion_scores['нейтральность'] += 0.5
                emotion_scores['спокойствие'] += 0.3
            
            else:
                # Default to neutral
                emotion_scores['нейтральность'] += 0.5
            
            # Add some randomness to avoid identical classifications
            for emotion in emotion_scores:
                emotion_scores[emotion] += np.random.normal(0, 0.1)
                emotion_scores[emotion] = max(0, emotion_scores[emotion])
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
            else:
                emotion_scores = {k: 1/len(emotion_scores) for k in emotion_scores}
            
            # Get dominant emotion
            dominant_emotion = max(emotion_scores.keys(), key=emotion_scores.get)
            confidence = emotion_scores[dominant_emotion]
            
            return {
                'emotion': dominant_emotion,
                'confidence': confidence,
                'all_emotions': emotion_scores,
                'method': 'rule_based'
            }
            
        except Exception as e:
            self.logger.warning(f"Emotion classification failed: {e}")
            return {
                'emotion': 'нейтральность',
                'confidence': 0.5,
                'all_emotions': {emotion: 1/len(self.emotion_categories) 
                               for emotion in self.emotion_categories},
                'method': 'fallback'
            }
    
    def analyze_speech(self, audio_path: str) -> Dict[str, Any]:
        """
        Complete speech emotion analysis pipeline
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Complete analysis results
        """
        self.logger.info(f"Starting speech emotion analysis: {audio_path}")
        
        try:
            # Load audio
            audio, sr = self.load_audio(audio_path)
            
            # Extract features
            features = self.extract_features(audio, sr)
            
            # Classify emotion
            emotion_result = self.classify_emotion(features)
            
            # Compile results
            analysis_result = {
                'audio_path': audio_path,
                'duration': len(audio) / sr,
                'sample_rate': sr,
                'emotion': emotion_result['emotion'],
                'confidence': emotion_result['confidence'],
                'all_emotions': emotion_result['all_emotions'],
                'method': emotion_result['method'],
                'features': {
                    'energy': features.get('energy', 0),
                    'pitch_mean': features.get('pitch_mean', 0),
                    'pitch_std': features.get('pitch_std', 0),
                    'spectral_centroid': features.get('spectral_centroid_mean', 0),
                    'tempo': features.get('tempo', 0)
                },
                'processing_info': {
                    'librosa_available': LIBROSA_AVAILABLE,
                    'sklearn_available': SKLEARN_AVAILABLE
                }
            }
            
            self.logger.info(f"Speech analysis completed: {emotion_result['emotion']}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Speech analysis failed: {e}")
            return {
                'error': str(e),
                'audio_path': audio_path
            }