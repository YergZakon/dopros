"""
Complete Audio Processor with transcription, speaker separation, and content analysis
"""

import logging
import os
import shutil
import subprocess
import time
import json
import tempfile
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import whisper
    LOCAL_WHISPER_AVAILABLE = True
except ImportError:
    LOCAL_WHISPER_AVAILABLE = False

try:
    import librosa
    import librosa.display
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# Internal imports
from models.speech_analyzer import AdvancedSpeechEmotionAnalyzer
from utils.translation import EmotionTranslator


@dataclass
class AudioSegment:
    """Audio segment with metadata"""
    start: float
    end: float
    text: str
    confidence: float
    speaker: Optional[str] = None
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    audio_path: Optional[str] = None


@dataclass
class TranscriptionSegment:
    """Transcription segment with timestamps"""
    text: str
    start: float
    end: float
    confidence: float
    words: List[Dict[str, Any]] = None


@dataclass
class SpeakerSegment:
    """Speaker identification segment"""
    speaker_id: str
    start: float
    end: float
    confidence: float


@dataclass
class DialogueAnalysis:
    """Dialogue analysis results"""
    contradictions: List[Dict[str, Any]]
    key_phrases: List[Dict[str, Any]]
    topics: List[Dict[str, Any]]
    cooperation_score: float
    stress_indicators: List[Dict[str, Any]]
    deception_indicators: List[Dict[str, Any]]


class CompleteAudioProcessor:
    """
    Complete audio processing pipeline with transcription, speaker separation, 
    emotion analysis, and content analysis for interrogation scenarios
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Audio settings
        audio_config = config.get('processing', {}).get('audio', {})
        self.sample_rate = audio_config.get('sample_rate', 16000)
        self.channels = audio_config.get('channels', 1)
        self.audio_format = audio_config.get('format', 'wav')
        
        # API settings
        api_config = config.get('apis', {})
        self.openai_api_key = api_config.get('openai_api_key')
        self.whisper_model = api_config.get('whisper_model', 'whisper-1')
        self.gpt_model = api_config.get('gpt_model', 'gpt-4')
        
        # Storage paths
        storage_config = config.get('storage', {})
        self.audio_dir = Path(storage_config.get('audio_dir', 'storage/audio'))
        self.transcripts_dir = Path(storage_config.get('transcripts_dir', 'storage/transcripts'))
        self.temp_dir = Path(storage_config.get('temp_dir', 'storage/temp'))
        
        # Create directories
        for dir_path in [self.audio_dir, self.transcripts_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Language settings
        lang_config = config.get('language', {})
        self.primary_language = lang_config.get('primary', 'ru')
        self.fallback_language = lang_config.get('fallback', 'en')
        
        # Initialize components
        self.speech_analyzer = None
        self.emotion_translator = EmotionTranslator()
        
        # FFmpeg path detection
        self.ffmpeg_path = self._find_ffmpeg()
        
        # Initialize OpenAI
        if OPENAI_AVAILABLE and self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Initialize local Whisper model
        self.local_whisper_model = None
        if LOCAL_WHISPER_AVAILABLE:
            try:
                whisper_config = config.get('processing', {}).get('models', {}).get('whisper', {})
                model_size = whisper_config.get('model_size', 'base')  # base, small, medium, large
                self.local_whisper_model = whisper.load_model(model_size)
                self.logger.info(f"Local Whisper model '{model_size}' loaded successfully")
            except Exception as e:
                self.logger.warning(f"Local Whisper initialization failed: {e}")
        
        # Initialize speech emotion analyzer
        try:
            self.speech_analyzer = AdvancedSpeechEmotionAnalyzer(config)
            self.logger.info("Speech emotion analyzer initialized")
        except Exception as e:
            self.logger.warning(f"Speech analyzer initialization failed: {e}")
        
        # Speaker identification pipeline
        self.speaker_pipeline = None
        if PYANNOTE_AVAILABLE:
            try:
                self.speaker_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
                self.logger.info("Speaker diarization pipeline initialized")
            except Exception as e:
                self.logger.warning(f"Speaker diarization initialization failed: {e}")
        
        self.logger.info("CompleteAudioProcessor initialized")
    
    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable in various locations"""
        ffmpeg_paths = [
            'ffmpeg',
            shutil.which('ffmpeg'),
            '/usr/local/bin/ffmpeg',
            '/usr/bin/ffmpeg',
            '/opt/homebrew/bin/ffmpeg',  # macOS Homebrew
            'C:\\ffmpeg\\bin\\ffmpeg.exe',  # Windows
            'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe',
            os.path.expanduser('~/ffmpeg/ffmpeg'),
            os.path.expanduser('~/bin/ffmpeg')
        ]
        
        for path in ffmpeg_paths:
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                self.logger.info(f"Found FFmpeg at: {path}")
                return path
            elif path == 'ffmpeg':
                # Check if ffmpeg is in PATH
                try:
                    subprocess.run(['ffmpeg', '-version'], 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL, 
                                 check=True)
                    self.logger.info("Found FFmpeg in PATH")
                    return 'ffmpeg'
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
        
        self.logger.warning("FFmpeg not found. Audio extraction may fail.")
        return None
    
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from video using FFmpeg with optimal parameters
        
        Args:
            video_path: Path to input video file
            output_path: Optional output path (auto-generated if None)
            
        Returns:
            Path to extracted audio file
        """
        if not self.ffmpeg_path:
            raise RuntimeError("FFmpeg not available for audio extraction")
        
        # Generate output path if not provided
        if output_path is None:
            video_name = Path(video_path).stem
            timestamp = int(time.time())
            output_path = str(self.audio_dir / f"{video_name}_{timestamp}.wav")
        
        # FFmpeg command with optimal parameters for speech processing
        cmd = [
            self.ffmpeg_path,
            '-i', video_path,
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # 16-bit PCM for compatibility
            '-ar', str(self.sample_rate),  # Sample rate
            '-ac', str(self.channels),  # Mono for speech
            '-af', 'highpass=f=85,lowpass=f=8000',  # Filter for speech frequency range
            '-y',  # Overwrite output
            output_path
        ]
        
        try:
            self.logger.info(f"Extracting audio from {video_path}")
            
            # Run FFmpeg
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            # Verify output file exists
            if not os.path.exists(output_path):
                raise RuntimeError("Audio extraction failed - output file not created")
            
            # Get audio duration for verification
            duration_cmd = [
                self.ffmpeg_path,
                '-i', output_path,
                '-f', 'null',
                '-'
            ]
            
            duration_result = subprocess.run(
                duration_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Extract duration from stderr
            duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})', 
                                     duration_result.stderr)
            if duration_match:
                h, m, s = duration_match.groups()
                duration = int(h) * 3600 + int(m) * 60 + float(s)
                self.logger.info(f"Audio extracted successfully: {duration:.2f}s")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg failed: {e.stderr}")
            raise RuntimeError(f"Audio extraction failed: {e.stderr}")
        except Exception as e:
            self.logger.error(f"Audio extraction error: {e}")
            raise
    
    def transcribe_with_timestamps(self, audio_path: str) -> List[TranscriptionSegment]:
        """
        Transcribe audio with timestamps using OpenAI Whisper API or local Whisper
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of transcription segments with timestamps
        """
        # Try OpenAI API first if available and configured
        if OPENAI_AVAILABLE and self.openai_api_key:
            try:
                return self._transcribe_with_openai(audio_path)
            except Exception as e:
                self.logger.warning(f"OpenAI transcription failed: {e}. Falling back to local Whisper")
        
        # Fallback to local Whisper
        if LOCAL_WHISPER_AVAILABLE and self.local_whisper_model:
            return self._transcribe_with_local_whisper(audio_path)
        
        raise RuntimeError("No transcription method available. Need either OpenAI API key or local Whisper model")
    
    def _transcribe_with_openai(self, audio_path: str) -> List[TranscriptionSegment]:
        """Transcribe using OpenAI API"""
        self.logger.info(f"Starting OpenAI transcription of {audio_path}")
        
        try:
            # Check file size (Whisper has 25MB limit)
            file_size = os.path.getsize(audio_path)
            if file_size > 25 * 1024 * 1024:  # 25MB
                self.logger.warning(f"Audio file too large ({file_size / 1024 / 1024:.1f}MB), splitting...")
                return self._transcribe_large_file(audio_path)
            
            with open(audio_path, 'rb') as audio_file:
                # Whisper API call with detailed timestamps
                response = openai.audio.transcriptions.create(
                    model=self.whisper_model,
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment", "word"],
                    language=self.primary_language if self.primary_language != 'auto' else None
                )
            
            # Convert response to TranscriptionSegment objects
            segments = []
            
            if hasattr(response, 'segments') and response.segments:
                for segment in response.segments:
                    # Extract word-level timestamps if available
                    words = []
                    if hasattr(segment, 'words') and segment.words:
                        words = [
                            {
                                'word': word.word,
                                'start': word.start,
                                'end': word.end,
                                'probability': getattr(word, 'probability', 1.0)
                            }
                            for word in segment.words
                        ]
                    
                    segments.append(TranscriptionSegment(
                        text=segment.text.strip(),
                        start=segment.start,
                        end=segment.end,
                        confidence=getattr(segment, 'avg_logprob', 0.0),
                        words=words
                    ))
            else:
                # Fallback: single segment
                segments.append(TranscriptionSegment(
                    text=response.text,
                    start=0.0,
                    end=self._get_audio_duration(audio_path),
                    confidence=0.0,
                    words=[]
                ))
            
            self.logger.info(f"Transcription completed: {len(segments)} segments")
            return segments
                
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
    
    def _transcribe_with_local_whisper(self, audio_path: str) -> List[TranscriptionSegment]:
        """Transcribe using local Whisper model"""
        self.logger.info(f"Starting local Whisper transcription of {audio_path}")
        
        try:
            # Transcribe with local Whisper
            result = self.local_whisper_model.transcribe(
                audio_path,
                language=self.primary_language if self.primary_language != 'auto' else None,
                verbose=False,
                word_timestamps=True
            )
            
            # Convert to TranscriptionSegment objects
            segments = []
            
            if 'segments' in result:
                for segment in result['segments']:
                    # Extract word-level timestamps if available
                    words = []
                    if 'words' in segment and segment['words']:
                        words = [
                            {
                                'word': word['word'],
                                'start': word['start'],
                                'end': word['end'],
                                'probability': word.get('probability', 1.0)
                            }
                            for word in segment['words']
                        ]
                    
                    segments.append(TranscriptionSegment(
                        text=segment['text'].strip(),
                        start=segment['start'],
                        end=segment['end'],
                        confidence=segment.get('avg_logprob', 0.0),
                        words=words
                    ))
            else:
                # Fallback: single segment
                segments.append(TranscriptionSegment(
                    text=result['text'],
                    start=0.0,
                    end=self._get_audio_duration(audio_path),
                    confidence=0.0,
                    words=[]
                ))
            
            self.logger.info(f"Local Whisper transcription completed: {len(segments)} segments")
            return segments
            
        except Exception as e:
            self.logger.error(f"Local Whisper transcription failed: {e}")
            raise
    
    def _transcribe_large_file(self, audio_path: str, chunk_duration: float = 600) -> List[TranscriptionSegment]:
        """Split large audio file and transcribe in chunks"""
        try:
            audio_duration = self._get_audio_duration(audio_path)
            num_chunks = int(np.ceil(audio_duration / chunk_duration))
            
            self.logger.info(f"Splitting audio into {num_chunks} chunks")
            all_segments = []
            
            for i in range(num_chunks):
                start_time = i * chunk_duration
                end_time = min((i + 1) * chunk_duration, audio_duration)
                
                # Extract chunk
                chunk_path = str(self.temp_dir / f"chunk_{i}.wav")
                self._extract_audio_chunk(audio_path, chunk_path, start_time, end_time)
                
                try:
                    # Transcribe chunk
                    chunk_segments = self.transcribe_with_timestamps(chunk_path)
                    
                    # Adjust timestamps
                    for segment in chunk_segments:
                        segment.start += start_time
                        segment.end += start_time
                        if segment.words:
                            for word in segment.words:
                                word['start'] += start_time
                                word['end'] += start_time
                    
                    all_segments.extend(chunk_segments)
                    
                finally:
                    # Cleanup chunk
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
            
            return all_segments
            
        except Exception as e:
            self.logger.error(f"Large file transcription failed: {e}")
            raise
    
    def _extract_audio_chunk(self, input_path: str, output_path: str, 
                           start_time: float, end_time: float):
        """Extract specific time range from audio"""
        if not self.ffmpeg_path:
            raise RuntimeError("FFmpeg not available")
        
        cmd = [
            self.ffmpeg_path,
            '-i', input_path,
            '-ss', str(start_time),
            '-t', str(end_time - start_time),
            '-acodec', 'copy',
            '-y',
            output_path
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio file duration"""
        if not LIBROSA_AVAILABLE:
            return 0.0
        
        try:
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception:
            return 0.0
    
    def enhance_transcript(self, segments: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
        """
        Enhance transcript using GPT-4: fix errors, add punctuation, improve formatting
        
        Args:
            segments: Raw transcription segments
            
        Returns:
            Enhanced transcription segments
        """
        if not OPENAI_AVAILABLE or not self.openai_api_key:
            self.logger.warning("OpenAI API not available, returning original transcript")
            return segments
        
        try:
            self.logger.info("Enhancing transcript with GPT-4")
            
            # Combine segments into chunks for processing
            enhanced_segments = []
            chunk_size = 10  # Process 10 segments at a time
            
            for i in range(0, len(segments), chunk_size):
                chunk_segments = segments[i:i + chunk_size]
                chunk_text = "\n".join([
                    f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}"
                    for seg in chunk_segments
                ])
                
                # GPT-4 enhancement prompt
                prompt = f"""
Улучши эту транскрипцию аудио из допроса. Исправь ошибки распознавания речи, добавь правильную пунктуацию, 
сохрани временные метки в том же формате.

Оригинальная транскрипция:
{chunk_text}

Требования:
- Сохрани формат временных меток [00.0s - 00.0s]
- Исправь ошибки OCR и распознавания речи
- Добавь правильную пунктуацию
- Сохрани разговорный стиль
- НЕ добавляй новую информацию, только исправляй ошибки

Улучшенная транскрипция:
"""
                
                try:
                    response = openai.chat.completions.create(
                        model=self.gpt_model,
                        messages=[
                            {"role": "system", "content": "Ты эксперт по обработке транскрипций. Исправляй только ошибки, не добавляй новую информацию."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=2000
                    )
                    
                    enhanced_text = response.choices[0].message.content.strip()
                    
                    # Parse enhanced text back into segments
                    enhanced_chunk = self._parse_enhanced_transcript(enhanced_text, chunk_segments)
                    enhanced_segments.extend(enhanced_chunk)
                    
                except Exception as e:
                    self.logger.warning(f"GPT-4 enhancement failed for chunk: {e}")
                    enhanced_segments.extend(chunk_segments)  # Use original
            
            self.logger.info("Transcript enhancement completed")
            return enhanced_segments
            
        except Exception as e:
            self.logger.error(f"Transcript enhancement failed: {e}")
            return segments  # Return original on error
    
    def _parse_enhanced_transcript(self, enhanced_text: str, 
                                 original_segments: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
        """Parse enhanced transcript back into segments"""
        enhanced_segments = []
        
        try:
            # Extract lines with timestamps
            lines = enhanced_text.split('\n')
            segment_idx = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Match timestamp format
                match = re.match(r'\[(\d+\.?\d*)s - (\d+\.?\d*)s\] (.+)', line)
                if match and segment_idx < len(original_segments):
                    start, end, text = match.groups()
                    original_seg = original_segments[segment_idx]
                    
                    enhanced_segments.append(TranscriptionSegment(
                        text=text,
                        start=float(start),
                        end=float(end),
                        confidence=original_seg.confidence,
                        words=original_seg.words  # Keep original word timing
                    ))
                    segment_idx += 1
            
            # Add any remaining original segments
            while segment_idx < len(original_segments):
                enhanced_segments.append(original_segments[segment_idx])
                segment_idx += 1
                
        except Exception as e:
            self.logger.warning(f"Failed to parse enhanced transcript: {e}")
            return original_segments
        
        return enhanced_segments
    
    def identify_speakers(self, audio_path: str, segments: List[TranscriptionSegment]) -> List[SpeakerSegment]:
        """
        Identify speakers using pyannote.audio or fallback methods
        
        Args:
            audio_path: Path to audio file
            segments: Transcription segments
            
        Returns:
            List of speaker segments
        """
        speaker_segments = []
        
        try:
            if self.speaker_pipeline is not None:
                # Use pyannote.audio for speaker diarization
                self.logger.info("Running speaker diarization")
                
                diarization = self.speaker_pipeline(audio_path)
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    speaker_segments.append(SpeakerSegment(
                        speaker_id=speaker,
                        start=turn.start,
                        end=turn.end,
                        confidence=1.0  # pyannote doesn't provide confidence scores
                    ))
            
            else:
                # Fallback: simple heuristic-based speaker identification
                self.logger.info("Using fallback speaker identification")
                speaker_segments = self._identify_speakers_fallback(segments)
            
            self.logger.info(f"Speaker identification completed: {len(speaker_segments)} segments")
            return speaker_segments
            
        except Exception as e:
            self.logger.error(f"Speaker identification failed: {e}")
            return self._identify_speakers_fallback(segments)
    
    def _identify_speakers_fallback(self, segments: List[TranscriptionSegment]) -> List[SpeakerSegment]:
        """Simple fallback speaker identification based on pauses and content"""
        speaker_segments = []
        current_speaker = "SPEAKER_1"
        speaker_counter = 1
        last_end = 0.0
        
        for segment in segments:
            # If there's a significant pause, potentially new speaker
            pause_duration = segment.start - last_end
            
            # Simple heuristic: pauses > 2 seconds might indicate speaker change
            if pause_duration > 2.0 and segment.start > 0:
                # Check if text indicates question (interrogator) vs answer (subject)
                text_lower = segment.text.lower()
                if any(marker in text_lower for marker in ['?', 'что', 'как', 'где', 'когда', 'почему']):
                    current_speaker = "INTERROGATOR"
                else:
                    current_speaker = "SUBJECT"
            
            speaker_segments.append(SpeakerSegment(
                speaker_id=current_speaker,
                start=segment.start,
                end=segment.end,
                confidence=0.5  # Low confidence for fallback method
            ))
            
            last_end = segment.end
        
        return speaker_segments
    
    def split_dialogue(self, segments: List[TranscriptionSegment], 
                      speaker_segments: List[SpeakerSegment]) -> Dict[str, List[AudioSegment]]:
        """
        Split transcription into dialogue by speaker
        
        Args:
            segments: Transcription segments
            speaker_segments: Speaker identification segments
            
        Returns:
            Dictionary mapping speaker IDs to their audio segments
        """
        dialogue = defaultdict(list)
        
        try:
            # Map speakers to transcription segments
            for transcript_seg in segments:
                # Find overlapping speaker segment
                best_speaker = None
                best_overlap = 0.0
                
                for speaker_seg in speaker_segments:
                    # Calculate overlap
                    overlap_start = max(transcript_seg.start, speaker_seg.start)
                    overlap_end = min(transcript_seg.end, speaker_seg.end)
                    overlap_duration = max(0, overlap_end - overlap_start)
                    
                    if overlap_duration > best_overlap:
                        best_overlap = overlap_duration
                        best_speaker = speaker_seg.speaker_id
                
                # Default speaker if no overlap found
                if best_speaker is None:
                    best_speaker = "UNKNOWN"
                
                # Create audio segment
                audio_segment = AudioSegment(
                    start=transcript_seg.start,
                    end=transcript_seg.end,
                    text=transcript_seg.text,
                    confidence=transcript_seg.confidence,
                    speaker=best_speaker
                )
                
                dialogue[best_speaker].append(audio_segment)
            
            # Convert to regular dict and sort by timestamp
            result = {}
            for speaker, segments_list in dialogue.items():
                result[speaker] = sorted(segments_list, key=lambda x: x.start)
            
            self.logger.info(f"Dialogue split completed: {len(result)} speakers")
            return result
            
        except Exception as e:
            self.logger.error(f"Dialogue splitting failed: {e}")
            # Fallback: return all segments under unknown speaker
            return {
                "UNKNOWN": [
                    AudioSegment(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                        confidence=seg.confidence,
                        speaker="UNKNOWN"
                    ) for seg in segments
                ]
            }
    
    def analyze_content(self, dialogue: Dict[str, List[AudioSegment]]) -> DialogueAnalysis:
        """
        Analyze dialogue content for interrogation insights
        
        Args:
            dialogue: Speaker-separated dialogue
            
        Returns:
            Dialogue analysis results
        """
        try:
            self.logger.info("Starting content analysis")
            
            # Extract all text for analysis
            all_text = []
            for speaker, segments in dialogue.items():
                for segment in segments:
                    all_text.append({
                        'speaker': speaker,
                        'text': segment.text,
                        'start': segment.start,
                        'end': segment.end
                    })
            
            # Sort by timestamp
            all_text.sort(key=lambda x: x['start'])
            
            # Analyze contradictions
            contradictions = self._find_contradictions(all_text)
            
            # Extract key phrases
            key_phrases = self._extract_key_phrases(all_text)
            
            # Identify topics
            topics = self._identify_topics(all_text)
            
            # Assess cooperation
            cooperation_score = self._assess_cooperation(dialogue)
            
            # Find stress indicators
            stress_indicators = self._find_stress_indicators(all_text)
            
            # Detect deception indicators
            deception_indicators = self._find_deception_indicators(all_text)
            
            analysis = DialogueAnalysis(
                contradictions=contradictions,
                key_phrases=key_phrases,
                topics=topics,
                cooperation_score=cooperation_score,
                stress_indicators=stress_indicators,
                deception_indicators=deception_indicators
            )
            
            self.logger.info("Content analysis completed")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            return DialogueAnalysis(
                contradictions=[],
                key_phrases=[],
                topics=[],
                cooperation_score=0.5,
                stress_indicators=[],
                deception_indicators=[]
            )
    
    def _find_contradictions(self, text_segments: List[Dict]) -> List[Dict[str, Any]]:
        """Find potential contradictions in speech"""
        contradictions = []
        
        # Simple keyword-based contradiction detection
        contradiction_patterns = [
            (r'не.*помню', r'помню'),
            (r'не.*знаю', r'знаю'),
            (r'не.*был', r'был'),
            (r'не.*делал', r'делал'),
            (r'никогда.*не', r'всегда')
        ]
        
        for i, seg1 in enumerate(text_segments):
            for j, seg2 in enumerate(text_segments[i+1:], i+1):
                text1_lower = seg1['text'].lower()
                text2_lower = seg2['text'].lower()
                
                for neg_pattern, pos_pattern in contradiction_patterns:
                    if re.search(neg_pattern, text1_lower) and re.search(pos_pattern, text2_lower):
                        contradictions.append({
                            'type': 'potential_contradiction',
                            'segment1': {
                                'text': seg1['text'],
                                'timestamp': seg1['start'],
                                'speaker': seg1['speaker']
                            },
                            'segment2': {
                                'text': seg2['text'],
                                'timestamp': seg2['start'],
                                'speaker': seg2['speaker']
                            },
                            'confidence': 0.6
                        })
        
        return contradictions[:10]  # Limit to prevent spam
    
    def _extract_key_phrases(self, text_segments: List[Dict]) -> List[Dict[str, Any]]:
        """Extract key phrases relevant to interrogation"""
        key_phrases = []
        
        # Important phrase patterns for interrogation context
        important_patterns = {
            'denial': [r'не.*помню', r'не.*знаю', r'не.*был', r'не.*делал'],
            'admission': [r'да.*был', r'да.*делал', r'признаю', r'виноват'],
            'evasion': [r'может.*быть', r'не.*уверен', r'возможно', r'наверное'],
            'emotional': [r'извини', r'прости', r'сожалею', r'стыдно'],
            'temporal': [r'в.*тот.*день', r'тогда', r'потом', r'после', r'до']
        }
        
        for segment in text_segments:
            text_lower = segment['text'].lower()
            
            for category, patterns in important_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text_lower)
                    for match in matches:
                        key_phrases.append({
                            'phrase': match.group(),
                            'category': category,
                            'context': segment['text'],
                            'timestamp': segment['start'],
                            'speaker': segment['speaker'],
                            'relevance_score': 0.8
                        })
        
        # Sort by relevance and return top phrases
        key_phrases.sort(key=lambda x: x['relevance_score'], reverse=True)
        return key_phrases[:20]
    
    def _identify_topics(self, text_segments: List[Dict]) -> List[Dict[str, Any]]:
        """Identify main conversation topics"""
        topics = []
        
        # Topic keywords for interrogation context
        topic_keywords = {
            'time_place': ['когда', 'где', 'время', 'место', 'дата'],
            'people': ['кто', 'люди', 'человек', 'друг', 'знакомый'],
            'actions': ['что', 'делать', 'происходить', 'случиться'],
            'emotions': ['чувство', 'эмоция', 'злость', 'страх', 'грусть'],
            'evidence': ['доказательство', 'улика', 'свидетель', 'документ']
        }
        
        topic_scores = defaultdict(float)
        topic_segments = defaultdict(list)
        
        for segment in text_segments:
            text_lower = segment['text'].lower()
            
            for topic, keywords in topic_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    topic_scores[topic] += score
                    topic_segments[topic].append(segment)
        
        # Convert to result format
        for topic, score in topic_scores.items():
            if score > 0:
                topics.append({
                    'topic': topic,
                    'relevance_score': min(score / len(text_segments), 1.0),
                    'segments': topic_segments[topic][:5],  # Top 5 segments
                    'total_mentions': int(score)
                })
        
        # Sort by relevance
        topics.sort(key=lambda x: x['relevance_score'], reverse=True)
        return topics
    
    def _assess_cooperation(self, dialogue: Dict[str, List[AudioSegment]]) -> float:
        """Assess subject's cooperation level"""
        cooperation_score = 0.5  # Default neutral
        
        try:
            # Find subject speaker (usually not interrogator)
            subject_segments = []
            for speaker, segments in dialogue.items():
                if speaker not in ['INTERROGATOR', 'INVESTIGATOR']:
                    subject_segments.extend(segments)
            
            if not subject_segments:
                return cooperation_score
            
            total_segments = len(subject_segments)
            cooperative_indicators = 0
            uncooperative_indicators = 0
            
            for segment in subject_segments:
                text_lower = segment.text.lower()
                
                # Cooperative indicators
                if any(word in text_lower for word in ['да', 'конечно', 'согласен', 'помню', 'знаю']):
                    cooperative_indicators += 1
                
                # Uncooperative indicators
                if any(word in text_lower for word in ['нет', 'не помню', 'не знаю', 'отказываюсь']):
                    uncooperative_indicators += 1
            
            # Calculate cooperation score
            if total_segments > 0:
                cooperative_ratio = cooperative_indicators / total_segments
                uncooperative_ratio = uncooperative_indicators / total_segments
                cooperation_score = max(0, min(1, 0.5 + (cooperative_ratio - uncooperative_ratio)))
            
        except Exception as e:
            self.logger.warning(f"Cooperation assessment failed: {e}")
        
        return cooperation_score
    
    def _find_stress_indicators(self, text_segments: List[Dict]) -> List[Dict[str, Any]]:
        """Find linguistic stress indicators"""
        stress_indicators = []
        
        stress_patterns = {
            'repetition': r'(\b\w+\b).*\1',
            'filler_words': r'\b(эээ|ммм|ааа|это|значит|как бы)\b',
            'incomplete_sentences': r'\.\.\.|\-\-|[А-Я][а-я]*\s*\-$',
            'emotional_language': r'\b(боже|черт|блин|ужас|кошмар)\b'
        }
        
        for segment in text_segments:
            text_lower = segment['text'].lower()
            
            for indicator_type, pattern in stress_patterns.items():
                matches = list(re.finditer(pattern, text_lower))
                if matches:
                    stress_indicators.append({
                        'type': indicator_type,
                        'matches': [match.group() for match in matches],
                        'timestamp': segment['start'],
                        'speaker': segment['speaker'],
                        'context': segment['text']
                    })
        
        return stress_indicators
    
    def _find_deception_indicators(self, text_segments: List[Dict]) -> List[Dict[str, Any]]:
        """Find linguistic deception indicators"""
        deception_indicators = []
        
        deception_patterns = {
            'memory_gaps': r'\b(не помню|забыл|не припомню)\b',
            'vague_language': r'\b(может быть|наверное|кажется|вроде)\b',
            'over_detailed': r'^.{200,}$',  # Very long responses
            'defensive': r'\b(почему вы|зачем|не понимаю почему)\b'
        }
        
        for segment in text_segments:
            text = segment['text']
            text_lower = text.lower()
            
            for indicator_type, pattern in deception_patterns.items():
                if re.search(pattern, text_lower if indicator_type != 'over_detailed' else text):
                    deception_indicators.append({
                        'type': indicator_type,
                        'timestamp': segment['start'],
                        'speaker': segment['speaker'],
                        'context': segment['text'][:200] + '...' if len(segment['text']) > 200 else segment['text'],
                        'confidence': 0.6
                    })
        
        return deception_indicators
    
    def synchronize_with_video(self, dialogue: Dict[str, List[AudioSegment]], 
                             video_timestamps: List[float]) -> Dict[str, Any]:
        """
        Synchronize audio dialogue with video frames
        
        Args:
            dialogue: Speaker-separated dialogue
            video_timestamps: List of video frame timestamps
            
        Returns:
            Synchronized timeline data
        """
        try:
            self.logger.info("Synchronizing audio with video")
            
            synchronized_timeline = []
            
            # Create events from all audio segments
            events = []
            for speaker, segments in dialogue.items():
                for segment in segments:
                    events.append({
                        'type': 'speech',
                        'speaker': speaker,
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text,
                        'emotion': segment.emotion,
                        'confidence': segment.confidence
                    })
            
            # Sort events by timestamp
            events.sort(key=lambda x: x['start'])
            
            # Map to video frames
            for frame_timestamp in video_timestamps:
                frame_events = []
                
                for event in events:
                    if event['start'] <= frame_timestamp <= event['end']:
                        frame_events.append(event)
                
                synchronized_timeline.append({
                    'video_timestamp': frame_timestamp,
                    'audio_events': frame_events
                })
            
            return {
                'timeline': synchronized_timeline,
                'total_video_frames': len(video_timestamps),
                'total_audio_events': len(events),
                'synchronized_frames': len([f for f in synchronized_timeline if f['audio_events']])
            }
            
        except Exception as e:
            self.logger.error(f"Video synchronization failed: {e}")
            return {'timeline': [], 'error': str(e)}
    
    def add_emotion_analysis(self, audio_path: str, dialogue: Dict[str, List[AudioSegment]]) -> Dict[str, List[AudioSegment]]:
        """Add emotion analysis to dialogue segments"""
        if not self.speech_analyzer:
            self.logger.warning("Speech analyzer not available")
            return dialogue
        
        try:
            self.logger.info("Adding emotion analysis to dialogue")
            
            enhanced_dialogue = {}
            
            for speaker, segments in dialogue.items():
                enhanced_segments = []
                
                for segment in segments:
                    # Extract audio segment for emotion analysis
                    temp_audio = str(self.temp_dir / f"segment_{hash(segment.text)}.wav")
                    
                    try:
                        # Extract segment audio
                        self._extract_audio_chunk(
                            audio_path, temp_audio, 
                            segment.start, segment.end
                        )
                        
                        # Analyze emotions
                        emotion_result = self.speech_analyzer.analyze_audio_file(temp_audio)
                        
                        # Update segment with emotion
                        segment.emotion = emotion_result.get('dominant_emotion', 'нейтральность')
                        segment.emotion_confidence = emotion_result.get('confidence', 0.0)
                        
                        enhanced_segments.append(segment)
                        
                    except Exception as e:
                        self.logger.warning(f"Emotion analysis failed for segment: {e}")
                        enhanced_segments.append(segment)
                    
                    finally:
                        # Cleanup temp file
                        if os.path.exists(temp_audio):
                            os.remove(temp_audio)
                
                enhanced_dialogue[speaker] = enhanced_segments
            
            return enhanced_dialogue
            
        except Exception as e:
            self.logger.error(f"Emotion analysis addition failed: {e}")
            return dialogue
    
    def export_srt_with_emotions(self, dialogue: Dict[str, List[AudioSegment]], 
                               output_path: str) -> str:
        """Export SRT subtitle file with emotion annotations"""
        try:
            # Collect all segments and sort by timestamp
            all_segments = []
            for speaker, segments in dialogue.items():
                all_segments.extend(segments)
            
            all_segments.sort(key=lambda x: x.start)
            
            srt_content = []
            
            for i, segment in enumerate(all_segments, 1):
                # Format timestamps for SRT
                start_time = self._seconds_to_srt_time(segment.start)
                end_time = self._seconds_to_srt_time(segment.end)
                
                # Format text with speaker and emotion
                emotion_emoji = self.emotion_translator.get_emotion_emoji(segment.emotion or 'нейтральность')
                speaker_text = f"[{segment.speaker}] {emotion_emoji} {segment.text}"
                
                # SRT format
                srt_content.extend([
                    str(i),
                    f"{start_time} --> {end_time}",
                    speaker_text,
                    ""
                ])
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(srt_content))
            
            self.logger.info(f"SRT exported to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"SRT export failed: {e}")
            raise
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    def export_transcript(self, dialogue: Dict[str, List[AudioSegment]], 
                        output_path: str, format_type: str = 'txt') -> str:
        """Export transcript in various formats"""
        try:
            if format_type == 'json':
                return self._export_json_transcript(dialogue, output_path)
            elif format_type == 'txt':
                return self._export_text_transcript(dialogue, output_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            self.logger.error(f"Transcript export failed: {e}")
            raise
    
    def _export_json_transcript(self, dialogue: Dict[str, List[AudioSegment]], 
                              output_path: str) -> str:
        """Export detailed JSON transcript"""
        transcript_data = {
            'metadata': {
                'export_timestamp': time.time(),
                'total_speakers': len(dialogue),
                'total_segments': sum(len(segments) for segments in dialogue.values())
            },
            'speakers': {}
        }
        
        for speaker, segments in dialogue.items():
            transcript_data['speakers'][speaker] = [
                {
                    'start': seg.start,
                    'end': seg.end,
                    'duration': seg.end - seg.start,
                    'text': seg.text,
                    'confidence': seg.confidence,
                    'emotion': seg.emotion,
                    'emotion_confidence': seg.emotion_confidence
                }
                for seg in segments
            ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def _export_text_transcript(self, dialogue: Dict[str, List[AudioSegment]], 
                              output_path: str) -> str:
        """Export readable text transcript"""
        # Collect and sort all segments
        all_segments = []
        for speaker, segments in dialogue.items():
            for seg in segments:
                all_segments.append((seg.start, speaker, seg.text, seg.emotion))
        
        all_segments.sort(key=lambda x: x[0])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("ТРАНСКРИПЦИЯ АУДИО\n")
            f.write("=" * 50 + "\n\n")
            
            for start_time, speaker, text, emotion in all_segments:
                time_str = f"[{start_time/60:.0f}:{start_time%60:05.2f}]"
                emotion_str = f" ({emotion})" if emotion else ""
                f.write(f"{time_str} {speaker}{emotion_str}: {text}\n\n")
        
        return output_path
    
    def generate_analysis_report(self, dialogue: Dict[str, List[AudioSegment]], 
                               analysis: DialogueAnalysis, output_path: str) -> str:
        """Generate comprehensive analysis report"""
        try:
            report_data = {
                'summary': self._generate_summary(dialogue, analysis),
                'speakers': self._analyze_speakers(dialogue),
                'content_analysis': asdict(analysis),
                'timeline': self._create_timeline_summary(dialogue),
                'recommendations': self._generate_recommendations(analysis)
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Analysis report generated: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise
    
    def _generate_summary(self, dialogue: Dict[str, List[AudioSegment]], 
                        analysis: DialogueAnalysis) -> Dict[str, Any]:
        """Generate dialogue summary"""
        total_segments = sum(len(segments) for segments in dialogue.values())
        total_duration = 0
        if dialogue:
            all_segments = [seg for segments in dialogue.values() for seg in segments]
            if all_segments:
                total_duration = max(seg.end for seg in all_segments)
        
        return {
            'total_duration': total_duration,
            'total_speakers': len(dialogue),
            'total_segments': total_segments,
            'cooperation_score': analysis.cooperation_score,
            'contradictions_found': len(analysis.contradictions),
            'key_topics': len(analysis.topics),
            'stress_indicators': len(analysis.stress_indicators)
        }
    
    def _analyze_speakers(self, dialogue: Dict[str, List[AudioSegment]]) -> Dict[str, Dict[str, Any]]:
        """Analyze individual speakers"""
        speaker_analysis = {}
        
        for speaker, segments in dialogue.items():
            if not segments:
                continue
            
            total_duration = sum(seg.end - seg.start for seg in segments)
            avg_confidence = np.mean([seg.confidence for seg in segments])
            emotions = [seg.emotion for seg in segments if seg.emotion]
            emotion_distribution = {}
            
            if emotions:
                for emotion in set(emotions):
                    emotion_distribution[emotion] = emotions.count(emotion) / len(emotions)
            
            speaker_analysis[speaker] = {
                'total_segments': len(segments),
                'total_duration': total_duration,
                'average_confidence': avg_confidence,
                'emotion_distribution': emotion_distribution,
                'dominant_emotion': max(emotion_distribution.keys(), 
                                      key=emotion_distribution.get) if emotion_distribution else None
            }
        
        return speaker_analysis
    
    def _create_timeline_summary(self, dialogue: Dict[str, List[AudioSegment]]) -> List[Dict[str, Any]]:
        """Create timeline summary"""
        # Collect all segments
        all_segments = []
        for speaker, segments in dialogue.items():
            all_segments.extend(segments)
        
        # Sort by time
        all_segments.sort(key=lambda x: x.start)
        
        # Create timeline events
        timeline = []
        for seg in all_segments:
            timeline.append({
                'timestamp': seg.start,
                'speaker': seg.speaker,
                'duration': seg.end - seg.start,
                'emotion': seg.emotion,
                'text_preview': seg.text[:100] + '...' if len(seg.text) > 100 else seg.text
            })
        
        return timeline
    
    def _generate_recommendations(self, analysis: DialogueAnalysis) -> List[str]:
        """Generate investigation recommendations"""
        recommendations = []
        
        if analysis.cooperation_score < 0.3:
            recommendations.append("Низкий уровень кооперативности. Рекомендуется изменить тактику допроса.")
        
        if len(analysis.contradictions) > 3:
            recommendations.append("Обнаружены множественные противоречия. Требуется детальная проверка.")
        
        if len(analysis.stress_indicators) > 5:
            recommendations.append("Высокий уровень стресса. Возможно, требуется перерыв или изменение подхода.")
        
        if len(analysis.deception_indicators) > 3:
            recommendations.append("Обнаружены признаки возможного обмана. Необходима дополнительная верификация.")
        
        if not recommendations:
            recommendations.append("Допрос проходит в нормальном режиме. Продолжить по плану.")
        
        return recommendations

    def process_complete_audio(self, video_path: str) -> Dict[str, Any]:
        """
        Complete audio processing pipeline
        
        Args:
            video_path: Path to video file
            
        Returns:
            Complete processing results
        """
        try:
            self.logger.info(f"Starting complete audio processing for {video_path}")
            
            # Step 1: Extract audio
            audio_path = self.extract_audio(video_path)
            
            # Step 2: Transcribe with timestamps
            transcript_segments = self.transcribe_with_timestamps(audio_path)
            
            # Step 3: Enhance transcript
            enhanced_segments = self.enhance_transcript(transcript_segments)
            
            # Step 4: Identify speakers
            speaker_segments = self.identify_speakers(audio_path, enhanced_segments)
            
            # Step 5: Split dialogue by speaker
            dialogue = self.split_dialogue(enhanced_segments, speaker_segments)
            
            # Step 6: Add emotion analysis
            dialogue_with_emotions = self.add_emotion_analysis(audio_path, dialogue)
            
            # Step 7: Analyze content
            content_analysis = self.analyze_content(dialogue_with_emotions)
            
            # Generate outputs
            base_name = Path(video_path).stem
            timestamp = int(time.time())
            
            # Export formats
            srt_path = str(self.transcripts_dir / f"{base_name}_{timestamp}.srt")
            json_path = str(self.transcripts_dir / f"{base_name}_{timestamp}_transcript.json")
            txt_path = str(self.transcripts_dir / f"{base_name}_{timestamp}_transcript.txt")
            report_path = str(self.transcripts_dir / f"{base_name}_{timestamp}_analysis.json")
            
            self.export_srt_with_emotions(dialogue_with_emotions, srt_path)
            self.export_transcript(dialogue_with_emotions, json_path, 'json')
            self.export_transcript(dialogue_with_emotions, txt_path, 'txt')
            self.generate_analysis_report(dialogue_with_emotions, content_analysis, report_path)
            
            results = {
                'audio_path': audio_path,
                'transcript_segments': len(enhanced_segments),
                'speakers_detected': len(set(seg.speaker_id for seg in speaker_segments)),
                'dialogue': dialogue_with_emotions,
                'content_analysis': content_analysis,
                'exports': {
                    'srt': srt_path,
                    'json_transcript': json_path,
                    'text_transcript': txt_path,
                    'analysis_report': report_path
                },
                'processing_completed': True
            }
            
            self.logger.info("Complete audio processing finished successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Complete audio processing failed: {e}")
            return {
                'error': str(e),
                'processing_completed': False
            }