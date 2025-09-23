# utils/stt_module.py - Enhanced Streaming VAD Architecture for Real-time STT
import whisper
import asyncio
import numpy as np
import io
import threading
import time
import logging
from typing import Optional, Tuple, Dict, Any, List, Callable, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import aiohttp
import json
from enum import Enum
import os
from dotenv import load_dotenv
import struct
import wave
from collections import deque
import torch

load_dotenv()

# Configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
MAX_AUDIO_DURATION = 30.0  # seconds
MIN_AUDIO_DURATION = 0.5   # seconds - increased for better sentence detection

# Enhanced Streaming VAD Configuration for Continuous Microphone
VAD_FRAME_SIZE_MS = 30  # 30ms frames
VAD_SAMPLE_RATE = 16000
VAD_FRAME_SIZE = int(VAD_SAMPLE_RATE * VAD_FRAME_SIZE_MS / 1000)  # 480 samples
VAD_SPEECH_START_FRAMES = 2   # 60ms to start recording (faster start)
VAD_SILENCE_END_FRAMES = 30   # 900ms of silence to end recording (longer for complete sentences)
VAD_MIN_SPEECH_DURATION_MS = 500  # Minimum 500ms speech to process
VAD_MAX_PAUSE_BETWEEN_WORDS_MS = 300  # Maximum pause within a sentence

# Enhanced thresholds for better sentence detection
VAD_ENERGY_THRESHOLD = 0.01  # Minimum energy level
VAD_CONSECUTIVE_SILENCE_FOR_END = 25  # Frames of silence to confirm sentence end
VAD_MIN_FRAMES_FOR_PROCESSING = 15  # Minimum frames before processing

# Global instances
_WHISPER_MODEL = None
_MODEL_LOCK = threading.Lock()
_EXECUTOR = ThreadPoolExecutor(max_workers=3, thread_name_prefix="STT")
_HTTP_SESSION = None
_SESSION_LOCK = threading.Lock()
_SILERO_VAD = None
_WEBRTC_VAD = None

# Logging setup
logger = logging.getLogger(__name__)

class STTProvider(Enum):
    DEEPGRAM = "deepgram"
    WHISPER = "whisper"

class VADProvider(Enum):
    SILERO = "silero"
    WEBRTC = "webrtc"
    ENERGY = "energy"  # Simple energy-based VAD

class SpeechState(Enum):
    SILENCE = "silence"
    SPEECH_STARTING = "speech_starting"
    SPEECH_ACTIVE = "speech_active"
    SPEECH_ENDING = "speech_ending"
    PROCESSING = "processing"

@dataclass
class VADFrame:
    """Single VAD frame result"""
    audio: np.ndarray
    is_speech: bool
    confidence: float
    energy: float
    timestamp_ms: int
    frame_index: int
    state: SpeechState

@dataclass
class STTResult:
    """Result container for STT operations"""
    text: str
    provider: STTProvider
    confidence: float
    duration_ms: int
    audio_duration_ms: int
    error: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        return self.error is None and bool(self.text.strip())

class EnhancedStreamingVAD:
    """Enhanced Streaming VAD processor for continuous microphone input"""
    
    def __init__(self, provider: VADProvider = VADProvider.SILERO):
        self.provider = provider
        self.vad_model = None
        self.frame_counter = 0
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        self.consecutive_silence_frames = 0
        self.state = SpeechState.SILENCE
        self.recorded_frames = deque()
        self.max_recorded_frames = int(MAX_AUDIO_DURATION * 1000 / VAD_FRAME_SIZE_MS)
        
        # Enhanced tracking for sentence detection
        self.speech_start_timestamp = None
        self.last_speech_timestamp = None
        self.energy_history = deque(maxlen=10)
        self.speech_confidence_history = deque(maxlen=5)
        
        # Callbacks
        self.on_speech_start_callback = None
        self.on_speech_end_callback = None
        self.on_processing_callback = None
        
        self._init_vad_model()
    
    def set_callbacks(self, 
                     on_speech_start: Callable = None,
                     on_speech_end: Callable = None,
                     on_processing: Callable = None):
        """Set callback functions for speech events"""
        self.on_speech_start_callback = on_speech_start
        self.on_speech_end_callback = on_speech_end
        self.on_processing_callback = on_processing
    
    def _init_vad_model(self):
        """Initialize VAD model with fallback options"""
        try:
            if self.provider == VADProvider.SILERO:
                self._init_silero_vad()
            elif self.provider == VADProvider.WEBRTC:
                self._init_webrtc_vad()
            elif self.provider == VADProvider.ENERGY:
                self._init_energy_vad()
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider.value} VAD: {e}")
            # Fallback chain: Silero -> WebRTC -> Energy
            if self.provider == VADProvider.SILERO:
                logger.info("Falling back to WebRTC VAD")
                self.provider = VADProvider.WEBRTC
                try:
                    self._init_webrtc_vad()
                except:
                    logger.info("Falling back to Energy VAD")
                    self.provider = VADProvider.ENERGY
                    self._init_energy_vad()
            elif self.provider == VADProvider.WEBRTC:
                logger.info("Falling back to Energy VAD")
                self.provider = VADProvider.ENERGY
                self._init_energy_vad()
    
    def _init_silero_vad(self):
        """Initialize Silero VAD"""
        try:
            import torch
            global _SILERO_VAD
            if _SILERO_VAD is None:
                logger.info("Loading Silero VAD model...")
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                _SILERO_VAD = {
                    'model': model,
                    'utils': utils
                }
                logger.info("Silero VAD loaded successfully")
            
            self.vad_model = _SILERO_VAD['model']
            
        except Exception as e:
            logger.error(f"Silero VAD initialization failed: {e}")
            raise
    
    def _init_webrtc_vad(self):
        """Initialize WebRTC VAD"""
        try:
            import webrtcvad
            global _WEBRTC_VAD
            if _WEBRTC_VAD is None:
                logger.info("Initializing WebRTC VAD...")
                _WEBRTC_VAD = webrtcvad.Vad()
                _WEBRTC_VAD.set_mode(2)  # Balanced mode for continuous input
                logger.info("WebRTC VAD initialized successfully")
            
            self.vad_model = _WEBRTC_VAD
            
        except ImportError:
            logger.error("webrtcvad package not installed. Install with: pip install webrtcvad")
            raise
        except Exception as e:
            logger.error(f"WebRTC VAD initialization failed: {e}")
            raise
    
    def _init_energy_vad(self):
        """Initialize simple energy-based VAD as fallback"""
        logger.info("Using Energy-based VAD")
        self.vad_model = "energy"
    
    def process_frame(self, audio_frame: np.ndarray, timestamp_ms: int) -> VADFrame:
        """Process single audio frame with enhanced state management"""
        self.frame_counter += 1
        
        # Ensure correct frame size
        if len(audio_frame) != VAD_FRAME_SIZE:
            if len(audio_frame) < VAD_FRAME_SIZE:
                audio_frame = np.pad(audio_frame, (0, VAD_FRAME_SIZE - len(audio_frame)))
            else:
                audio_frame = audio_frame[:VAD_FRAME_SIZE]
        
        # Calculate energy
        energy = np.mean(np.square(audio_frame))
        self.energy_history.append(energy)
        
        # Detect speech with enhanced logic
        is_speech, confidence = self._detect_speech_frame(audio_frame, energy)
        self.speech_confidence_history.append(confidence if is_speech else 0.0)
        
        # Enhanced state management
        previous_state = self.state
        self._update_speech_state(is_speech, confidence, energy, timestamp_ms)
        
        # Handle state transitions
        if previous_state != self.state:
            self._handle_state_transition(previous_state, self.state, timestamp_ms)
        
        # Store frame if in recording states
        if self.state in [SpeechState.SPEECH_STARTING, SpeechState.SPEECH_ACTIVE, SpeechState.SPEECH_ENDING]:
            self.recorded_frames.append(audio_frame.copy())
            
            # Prevent memory overflow
            if len(self.recorded_frames) > self.max_recorded_frames:
                self.recorded_frames.popleft()
        
        return VADFrame(
            audio=audio_frame,
            is_speech=is_speech,
            confidence=confidence,
            energy=energy,
            timestamp_ms=timestamp_ms,
            frame_index=self.frame_counter,
            state=self.state
        )
    
    def _detect_speech_frame(self, audio_frame: np.ndarray, energy: float) -> Tuple[bool, float]:
        """Enhanced speech detection with energy gating"""
        try:
            # Energy gate - if too quiet, definitely not speech
            if energy < VAD_ENERGY_THRESHOLD:
                return False, 0.0
            
            if self.provider == VADProvider.SILERO:
                return self._detect_silero(audio_frame)
            elif self.provider == VADProvider.WEBRTC:
                return self._detect_webrtc(audio_frame)
            elif self.provider == VADProvider.ENERGY:
                return self._detect_energy(energy)
            else:
                return False, 0.0
                
        except Exception as e:
            logger.warning(f"VAD detection failed: {e}")
            return False, 0.0
    
    def _detect_silero(self, audio_frame: np.ndarray) -> Tuple[bool, float]:
        """Silero VAD detection with enhanced threshold"""
        try:
            audio_tensor = torch.from_numpy(audio_frame).float()
            speech_prob = self.vad_model(audio_tensor, VAD_SAMPLE_RATE).item()
            is_speech = speech_prob > 0.4  # Lower threshold for continuous input
            return is_speech, speech_prob
        except Exception as e:
            logger.warning(f"Silero detection error: {e}")
            return False, 0.0
    
    def _detect_webrtc(self, audio_frame: np.ndarray) -> Tuple[bool, float]:
        """WebRTC VAD detection"""
        try:
            audio_int16 = (audio_frame * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            is_speech = self.vad_model.is_speech(audio_bytes, VAD_SAMPLE_RATE)
            confidence = 1.0 if is_speech else 0.0
            return is_speech, confidence
        except Exception as e:
            logger.warning(f"WebRTC detection error: {e}")
            return False, 0.0
    
    def _detect_energy(self, energy: float) -> Tuple[bool, float]:
        """Simple energy-based VAD"""
        try:
            # Dynamic threshold based on recent energy history
            if len(self.energy_history) > 5:
                avg_energy = np.mean(list(self.energy_history))
                threshold = max(VAD_ENERGY_THRESHOLD, avg_energy * 2)
            else:
                threshold = VAD_ENERGY_THRESHOLD * 10
            
            is_speech = energy > threshold
            confidence = min(1.0, energy / threshold) if is_speech else 0.0
            return is_speech, confidence
        except Exception as e:
            logger.warning(f"Energy detection error: {e}")
            return False, 0.0
    
    def _update_speech_state(self, is_speech: bool, confidence: float, energy: float, timestamp_ms: int):
        """Update speech state with enhanced logic for sentence detection"""
        
        if is_speech:
            self.speech_frames_count += 1
            self.silence_frames_count = 0
            self.consecutive_silence_frames = 0
            self.last_speech_timestamp = timestamp_ms
        else:
            self.speech_frames_count = max(0, self.speech_frames_count - 1)
            self.silence_frames_count += 1
            self.consecutive_silence_frames += 1
        
        # State transitions
        if self.state == SpeechState.SILENCE:
            if self.speech_frames_count >= VAD_SPEECH_START_FRAMES:
                self.state = SpeechState.SPEECH_STARTING
                self.speech_start_timestamp = timestamp_ms
        
        elif self.state == SpeechState.SPEECH_STARTING:
            if self.speech_frames_count >= VAD_MIN_FRAMES_FOR_PROCESSING:
                self.state = SpeechState.SPEECH_ACTIVE
            elif self.consecutive_silence_frames >= VAD_SPEECH_START_FRAMES * 2:
                self.state = SpeechState.SILENCE
                self._reset_recording()
        
        elif self.state == SpeechState.SPEECH_ACTIVE:
            if self.consecutive_silence_frames >= VAD_CONSECUTIVE_SILENCE_FOR_END:
                self.state = SpeechState.SPEECH_ENDING
            elif self.consecutive_silence_frames >= VAD_SILENCE_END_FRAMES * 2:
                # Very long silence - likely end of sentence
                self.state = SpeechState.SPEECH_ENDING
        
        elif self.state == SpeechState.SPEECH_ENDING:
            if is_speech and confidence > 0.5:
                # Speech resumed - back to active
                self.state = SpeechState.SPEECH_ACTIVE
                self.consecutive_silence_frames = 0
            elif self.consecutive_silence_frames >= 5:  # Short confirmation
                # Confirmed end of speech
                if len(self.recorded_frames) * VAD_FRAME_SIZE_MS >= VAD_MIN_SPEECH_DURATION_MS:
                    self.state = SpeechState.PROCESSING
                else:
                    self.state = SpeechState.SILENCE
                    self._reset_recording()
        
        elif self.state == SpeechState.PROCESSING:
            # Stay in processing until explicitly reset
            pass
    
    def _handle_state_transition(self, old_state: SpeechState, new_state: SpeechState, timestamp_ms: int):
        """Handle state transition callbacks"""
        logger.debug(f"VAD State: {old_state.value} -> {new_state.value} at {timestamp_ms}ms")
        
        if old_state == SpeechState.SILENCE and new_state == SpeechState.SPEECH_STARTING:
            if self.on_speech_start_callback:
                try:
                    self.on_speech_start_callback(timestamp_ms)
                except Exception as e:
                    logger.error(f"Speech start callback error: {e}")
        
        elif old_state in [SpeechState.SPEECH_ACTIVE, SpeechState.SPEECH_ENDING] and new_state == SpeechState.PROCESSING:
            if self.on_speech_end_callback:
                try:
                    self.on_speech_end_callback(timestamp_ms, self.get_recorded_audio())
                except Exception as e:
                    logger.error(f"Speech end callback error: {e}")
        
        elif new_state == SpeechState.PROCESSING:
            if self.on_processing_callback:
                try:
                    self.on_processing_callback()
                except Exception as e:
                    logger.error(f"Processing callback error: {e}")
    
    def get_recorded_audio(self) -> Optional[np.ndarray]:
        """Get accumulated recorded audio"""
        if not self.recorded_frames:
            return None
        
        duration_ms = len(self.recorded_frames) * VAD_FRAME_SIZE_MS
        if duration_ms < VAD_MIN_SPEECH_DURATION_MS:
            logger.debug(f"Recording too short: {duration_ms}ms < {VAD_MIN_SPEECH_DURATION_MS}ms")
            return None
        
        audio_data = np.concatenate(list(self.recorded_frames))
        logger.info(f"🎵 Recorded audio: {len(audio_data)} samples, {duration_ms:.0f}ms")
        return audio_data
    
    def _reset_recording(self):
        """Reset recording state"""
        self.recorded_frames.clear()
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        self.consecutive_silence_frames = 0
        self.speech_start_timestamp = None
        self.last_speech_timestamp = None
    
    def finish_processing(self):
        """Call this after STT processing is complete"""
        self._reset_recording()
        self.state = SpeechState.SILENCE
        logger.debug("🔄 VAD reset to silence state")
    
    def is_ready_for_processing(self) -> bool:
        """Check if audio is ready for STT processing"""
        return self.state == SpeechState.PROCESSING
    
    def get_speech_duration_ms(self) -> int:
        """Get current speech duration"""
        if self.speech_start_timestamp and self.last_speech_timestamp:
            return self.last_speech_timestamp - self.speech_start_timestamp
        return 0

class ContinuousStreamingSTT:
    """Enhanced Streaming STT processor for continuous microphone input"""
    
    def __init__(self, vad_provider: VADProvider = VADProvider.SILERO):
        self.vad = EnhancedStreamingVAD(vad_provider)
        self.transcript_callbacks = []
        self.status_callbacks = []
        self.is_processing = False
        
        # Set VAD callbacks
        self.vad.set_callbacks(
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
            on_processing=self._on_processing_start
        )
    
    def add_transcript_callback(self, callback: Callable[[str, float], None]):
        """Add callback for when transcript is ready"""
        self.transcript_callbacks.append(callback)
    
    def add_status_callback(self, callback: Callable[[str], None]):
        """Add callback for status updates (listening, processing, etc.)"""
        self.status_callbacks.append(callback)
    
    def _on_speech_start(self, timestamp_ms: int):
        """Called when speech starts"""
        logger.info("🎤 Speech started")
        self._notify_status_callbacks("speech_started")
    
    def _on_speech_end(self, timestamp_ms: int, audio_data: Optional[np.ndarray]):
        """Called when speech ends"""
        logger.info("🔇 Speech ended")
        self._notify_status_callbacks("speech_ended")
        
        if audio_data is not None and not self.is_processing:
            # Start processing asynchronously
            asyncio.create_task(self._process_recording(audio_data))
    
    def _on_processing_start(self):
        """Called when processing starts"""
        logger.info("⚙️ Processing started")
        self._notify_status_callbacks("processing")
    
    def _notify_status_callbacks(self, status: str):
        """Notify all status callbacks"""
        for callback in self.status_callbacks:
            try:
                callback(status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")
    
    async def process_continuous_audio_stream(self, audio_generator: AsyncGenerator[np.ndarray, None]):
        """Process continuous audio stream from frontend microphone"""
        frame_timestamp = 0
        
        logger.info("🎧 Starting continuous audio processing...")
        self._notify_status_callbacks("listening")
        
        try:
            async for audio_chunk in audio_generator:
                # Split chunk into VAD frames
                chunk_frames = self._split_to_vad_frames(audio_chunk)
                
                for frame in chunk_frames:
                    # Process each frame
                    vad_result = self.vad.process_frame(frame, frame_timestamp)
                    frame_timestamp += VAD_FRAME_SIZE_MS
                    
                    # Check if ready for processing (non-blocking)
                    if self.vad.is_ready_for_processing() and not self.is_processing:
                        recorded_audio = self.vad.get_recorded_audio()
                        if recorded_audio is not None:
                            # Start processing without blocking the stream
                            asyncio.create_task(self._process_recording(recorded_audio))
                            self.vad.finish_processing()
        
        except Exception as e:
            logger.error(f"Audio stream processing error: {e}")
            self._notify_status_callbacks("error")
        
        finally:
            logger.info("🔌 Audio stream ended")
            self._notify_status_callbacks("stopped")
    
    def _split_to_vad_frames(self, audio_chunk: np.ndarray) -> List[np.ndarray]:
        """Split audio chunk into VAD-sized frames"""
        frames = []
        for i in range(0, len(audio_chunk), VAD_FRAME_SIZE):
            frame = audio_chunk[i:i + VAD_FRAME_SIZE]
            if len(frame) > 0:
                frames.append(frame)
        return frames
    
    async def _process_recording(self, audio_data: np.ndarray):
        """Process complete recording with STT"""
        if self.is_processing:
            logger.debug("Already processing, skipping...")
            return
        
        self.is_processing = True
        start_time = time.perf_counter()
        
        try:
            # Convert to WAV bytes
            wav_bytes = self._numpy_to_wav_bytes(audio_data)
            
            # Transcribe
            result = await transcribe_wav_bytes(wav_bytes, use_fallback=True)
            
            processing_time = int((time.perf_counter() - start_time) * 1000)
            
            # Notify callbacks
            if result.is_success:
                logger.info(f"📝 Transcript ready: '{result.text}' (processed in {processing_time}ms)")
                for callback in self.transcript_callbacks:
                    try:
                        callback(result.text, result.confidence)
                    except Exception as e:
                        logger.error(f"Transcript callback error: {e}")
                
                self._notify_status_callbacks("transcript_ready")
            else:
                logger.warning(f"STT failed: {result.error}")
                self._notify_status_callbacks("stt_failed")
            
        except Exception as e:
            logger.error(f"Recording processing failed: {e}")
            self._notify_status_callbacks("processing_error")
        finally:
            self.is_processing = False
            self._notify_status_callbacks("listening")  # Back to listening
    
    def _numpy_to_wav_bytes(self, audio_data: np.ndarray) -> bytes:
        """Convert numpy array to WAV bytes"""
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(VAD_SAMPLE_RATE)
            wav_file.writeframes(audio_int16.tobytes())
        
        return wav_buffer.getvalue()
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current processor state"""
        return {
            "vad_state": self.vad.state.value,
            "is_processing": self.is_processing,
            "speech_duration_ms": self.vad.get_speech_duration_ms(),
            "frame_count": self.vad.frame_counter,
            "provider": self.vad.provider.value
        }

# Keep original STT functions unchanged
async def get_http_session() -> aiohttp.ClientSession:
    """Get reusable HTTP session"""
    global _HTTP_SESSION
    if _HTTP_SESSION is None or _HTTP_SESSION.closed:
        with _SESSION_LOCK:
            if _HTTP_SESSION is None or _HTTP_SESSION.closed:
                timeout = aiohttp.ClientTimeout(total=10.0, connect=5.0)
                connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
                _HTTP_SESSION = aiohttp.ClientSession(timeout=timeout, connector=connector)
    return _HTTP_SESSION

def get_whisper_model():
    """Get Whisper model with thread-safe lazy loading"""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        with _MODEL_LOCK:
            if _WHISPER_MODEL is None:
                logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
                start_time = time.perf_counter()
                _WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_SIZE)
                load_time = (time.perf_counter() - start_time) * 1000
                logger.info(f"Whisper model loaded in {load_time:.0f}ms")
    return _WHISPER_MODEL

async def transcribe_deepgram(wav_bytes: bytes) -> STTResult:
    """Transcribe using Deepgram API"""
    start_time = time.perf_counter()
    
    try:
        if not DEEPGRAM_API_KEY:
            return STTResult("", STTProvider.DEEPGRAM, 0.0, 0, 0, "Deepgram API key not configured")
        
        session = await get_http_session()
        
        params = {
            'model': 'nova-2',
            'language': 'en',
            'smart_format': 'true',
            'punctuate': 'true',
            'interim_results': 'false',
        }
        
        headers = {
            'Authorization': f'Token {DEEPGRAM_API_KEY}',
            'Content-Type': 'audio/wav',
        }
        
        url = "https://api.deepgram.com/v1/listen"
        
        async with session.post(url, params=params, headers=headers, data=wav_bytes) as response:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Deepgram API error {response.status}: {error_text}")
                return STTResult("", STTProvider.DEEPGRAM, 0.0, duration_ms, len(wav_bytes), f"API error: {response.status}")
            
            result = await response.json()
            alternatives = result.get('results', {}).get('channels', [{}])[0].get('alternatives', [])
            
            if not alternatives:
                return STTResult("", STTProvider.DEEPGRAM, 0.0, duration_ms, len(wav_bytes), "No alternatives")
            
            best = alternatives[0]
            transcript = best.get('transcript', '').strip()
            confidence = best.get('confidence', 0.0)
            
            logger.info(f"✅ Deepgram: '{transcript}' ({confidence:.3f}, {duration_ms}ms)")
            return STTResult(transcript, STTProvider.DEEPGRAM, confidence, duration_ms, len(wav_bytes))
            
    except Exception as e:
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"Deepgram error: {e}")
        return STTResult("", STTProvider.DEEPGRAM, 0.0, duration_ms, len(wav_bytes), str(e))

def _transcribe_whisper_sync(audio_array: np.ndarray) -> Tuple[str, float]:
    """Synchronous Whisper transcription"""
    try:
        model = get_whisper_model()
        
        result = model.transcribe(
            audio_array,
            language='en',
            fp16=False,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            word_timestamps=False,
        )
        
        transcript = result.get("text", "").strip()
        
        # Clean up transcript
        if transcript:
            artifacts = ["[Music]", "[Applause]", "[Laughter]", "[Silence]"]
            for artifact in artifacts:
                transcript = transcript.replace(artifact, "")
            transcript = " ".join(transcript.split())
        
        confidence = min(1.0, len(transcript) / 50.0 + 0.3)
        return transcript, confidence
        
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise

async def transcribe_whisper(wav_bytes: bytes) -> STTResult:
    """Transcribe using local Whisper"""
    start_time = time.perf_counter()
    
    try:
        audio_buffer = io.BytesIO(wav_bytes)
        
        try:
            import librosa
            audio, sr = librosa.load(audio_buffer, sr=16000, mono=True)
        except ImportError:
            # Fallback to wave module
            with wave.open(audio_buffer, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
        loop = asyncio.get_running_loop()
        transcript, confidence = await loop.run_in_executor(_EXECUTOR, _transcribe_whisper_sync, audio)
        
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(f"✅ Whisper: '{transcript}' ({confidence:.3f}, {duration_ms}ms)")
        
        return STTResult(transcript, STTProvider.WHISPER, confidence, duration_ms, len(wav_bytes))
        
    except Exception as e:
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"Whisper error: {e}")
        return STTResult("", STTProvider.WHISPER, 0.0, duration_ms, len(wav_bytes), str(e))

async def transcribe_wav_bytes(wav_bytes: bytes, use_fallback: bool = True) -> STTResult:
    """Main transcription function"""
    overall_start = time.perf_counter()
    
    # Basic validation
    if not wav_bytes or len(wav_bytes) < 1000:
        return STTResult("", STTProvider.DEEPGRAM, 0, 0, 0, "Invalid audio data")
    
    logger.info(f"🎯 STT processing {len(wav_bytes)} bytes")
    
    # Try Deepgram first
    deepgram_result = await transcribe_deepgram(wav_bytes)
    
    if deepgram_result.is_success and deepgram_result.confidence > 0.3:
        total_time = int((time.perf_counter() - overall_start) * 1000)
        logger.info(f"🚀 STT completed via Deepgram in {total_time}ms")
        return deepgram_result
    
    logger.warning(f"Deepgram failed/low-confidence, trying Whisper...")
    
    if not use_fallback:
        return deepgram_result
    
    whisper_result = await transcribe_whisper(wav_bytes)
    
    total_time = int((time.perf_counter() - overall_start) * 1000)
    
    if whisper_result.is_success:
        logger.info(f"🚀 STT completed via Whisper in {total_time}ms")
        return whisper_result
    else:
        logger.error("❌ Both STT providers failed")
        return whisper_result if whisper_result.text else deepgram_result

# Enhanced factory functions
def create_continuous_streaming_stt(vad_provider: VADProvider = VADProvider.SILERO) -> ContinuousStreamingSTT:
    """Create continuous streaming STT processor optimized for frontend microphone"""
    return ContinuousStreamingSTT(vad_provider)

# WebSocket/Real-time integration helpers
class MicrophoneStreamHandler:
    """Helper class for handling microphone stream from frontend"""
    
    def __init__(self, stt_processor: ContinuousStreamingSTT):
        self.stt_processor = stt_processor
        self.audio_queue = asyncio.Queue()
        self.is_streaming = False
        
    async def start_stream(self):
        """Start processing microphone stream"""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        logger.info("🎤 Starting microphone stream handler")
        
        # Start the audio processing task
        processing_task = asyncio.create_task(
            self.stt_processor.process_continuous_audio_stream(self._audio_generator())
        )
        
        return processing_task
    
    async def stop_stream(self):
        """Stop microphone stream"""
        self.is_streaming = False
        logger.info("🔇 Stopping microphone stream handler")
    
    async def add_audio_chunk(self, audio_data: bytes):
        """Add audio chunk from frontend (WebSocket, HTTP, etc.)"""
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            await self.audio_queue.put(audio_array)
        except Exception as e:
            logger.error(f"Error adding audio chunk: {e}")
    
    async def _audio_generator(self):
        """Internal audio generator for the STT processor"""
        while self.is_streaming:
            try:
                # Wait for audio chunks with timeout
                audio_chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                yield audio_chunk
            except asyncio.TimeoutError:
                # Yield silence to keep the stream alive
                silence = np.zeros(VAD_FRAME_SIZE * 2, dtype=np.float32)
                yield silence
            except Exception as e:
                logger.error(f"Audio generator error: {e}")
                break

# Enhanced usage example
async def example_continuous_microphone_usage():
    """Example of how to use continuous STT with frontend microphone"""
    
    def on_transcript_ready(text: str, confidence: float):
        print(f"📝 User said: '{text}' (confidence: {confidence:.3f})")
        # Here you would send this to your agent for processing
    
    def on_status_change(status: str):
        print(f"🔄 Status: {status}")
        # Send status updates to frontend
    
    # Create continuous STT processor
    stt_processor = create_continuous_streaming_stt(VADProvider.SILERO)
    stt_processor.add_transcript_callback(on_transcript_ready)
    stt_processor.add_status_callback(on_status_change)
    
    # Create microphone stream handler
    mic_handler = MicrophoneStreamHandler(stt_processor)
    
    # Start streaming
    stream_task = await mic_handler.start_stream()
    
    # Simulate receiving audio chunks from frontend
    # In real usage, these would come from WebSocket/HTTP requests
    try:
        for i in range(1000):  # Simulate continuous stream
            # Generate dummy audio (replace with actual microphone data)
            if i % 50 < 10:  # Simulate speech periods
                audio_chunk = np.random.normal(0, 0.3, VAD_FRAME_SIZE * 4).astype(np.float32)
            else:  # Simulate silence
                audio_chunk = np.random.normal(0, 0.05, VAD_FRAME_SIZE * 4).astype(np.float32)
            
            # Convert to bytes (as frontend would send)
            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
            await mic_handler.add_audio_chunk(audio_bytes)
            
            await asyncio.sleep(0.02)  # 20ms intervals
    
    finally:
        await mic_handler.stop_stream()

# Flask/FastAPI integration example
class WebSTTEndpoint:
    """Example integration for web frameworks"""
    
    def __init__(self):
        self.stt_processor = create_continuous_streaming_stt(VADProvider.SILERO)
        self.mic_handler = None
        self.current_session_id = None
        
        # Set up callbacks
        self.stt_processor.add_transcript_callback(self._on_transcript)
        self.stt_processor.add_status_callback(self._on_status_change)
    
    async def start_session(self, session_id: str):
        """Start STT session for a user"""
        if self.mic_handler:
            await self.stop_session()
        
        self.current_session_id = session_id
        self.mic_handler = MicrophoneStreamHandler(self.stt_processor)
        stream_task = await self.mic_handler.start_stream()
        
        logger.info(f"🎧 Started STT session for {session_id}")
        return {"status": "started", "session_id": session_id}
    
    async def stop_session(self):
        """Stop current STT session"""
        if self.mic_handler:
            await self.mic_handler.stop_stream()
            self.mic_handler = None
        
        session_id = self.current_session_id
        self.current_session_id = None
        
        logger.info(f"🔇 Stopped STT session for {session_id}")
        return {"status": "stopped", "session_id": session_id}
    
    async def process_audio_chunk(self, audio_data: bytes):
        """Process audio chunk from frontend"""
        if not self.mic_handler:
            return {"error": "No active session"}
        
        await self.mic_handler.add_audio_chunk(audio_data)
        return {"status": "processed"}
    
    def _on_transcript(self, text: str, confidence: float):
        """Handle transcript ready - override in your implementation"""
        # Send to your agent or WebSocket to frontend
        logger.info(f"📝 Transcript for {self.current_session_id}: '{text}'")
    
    def _on_status_change(self, status: str):
        """Handle status change - override in your implementation"""
        # Send status to frontend via WebSocket
        logger.info(f"🔄 Status for {self.current_session_id}: {status}")
    
    def get_session_state(self):
        """Get current session state"""
        if not self.stt_processor:
            return {"error": "No processor"}
        
        return {
            "session_id": self.current_session_id,
            "active": self.mic_handler is not None,
            **self.stt_processor.get_current_state()
        }

# Cleanup and warmup functions
async def cleanup_stt():
    """Clean up STT resources"""
    global _HTTP_SESSION, _EXECUTOR
    
    if _HTTP_SESSION and not _HTTP_SESSION.closed:
        await _HTTP_SESSION.close()
        _HTTP_SESSION = None
    
    if _EXECUTOR:
        _EXECUTOR.shutdown(wait=True)
    
    logger.info("🧹 STT resources cleaned up")

def warmup_stt(vad_provider: VADProvider = VADProvider.SILERO):
    """Warmup STT and VAD components"""
    try:
        logger.info("🔥 Warming up Enhanced STT components...")
        
        # Load Whisper model
        get_whisper_model()
        
        # Initialize VAD
        try:
            vad = EnhancedStreamingVAD(vad_provider)
            dummy_frame = np.zeros(VAD_FRAME_SIZE, dtype=np.float32)
            vad.process_frame(dummy_frame, 0)
            logger.info(f"✅ Enhanced {vad_provider.value} VAD initialized")
        except Exception as e:
            logger.warning(f"Enhanced VAD initialization failed: {e}")
        
        if DEEPGRAM_API_KEY:
            logger.info("✅ Deepgram API configured")
        else:
            logger.warning("⚠️ Deepgram API key not found")
        
        logger.info("🚀 Enhanced STT warmup completed")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced STT warmup failed: {e}")
        return False

# Legacy compatibility
async def transcribe_wav_bytes_legacy(wav_bytes: bytes) -> str:
    """Legacy function for backward compatibility"""
    result = await transcribe_wav_bytes(wav_bytes)
    return result.text if result.is_success else ""

# Main entry point for testing
if __name__ == "__main__":
    # Test the enhanced system
    logging.basicConfig(level=logging.INFO)
    
    async def test_system():
        print("🧪 Testing Enhanced Continuous STT System...")
        
        # Warmup
        if warmup_stt(VADProvider.SILERO):
            print("✅ System ready")
            
            # Run example
            await example_continuous_microphone_usage()
        else:
            print("❌ System warmup failed")
    
    asyncio.run(test_system())