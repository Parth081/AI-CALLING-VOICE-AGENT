#!/usr/bin/env python3
"""
Production Voice Assistant with Emotional Intelligence v4.0
Complete, properly structured, production-ready implementation
"""
import os
import uvicorn
import json
import asyncio
import time
import signal
import sys
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import core modules
from utils.booking_module import is_booking_query, process_booking_query
from chromadb_pipeline import train_ai_agent, query_ai_agent, get_agent_stats
from utils.stt_module import (
    transcribe_wav_bytes, cleanup_stt, warmup_stt,
    create_continuous_streaming_stt, VADProvider, MicrophoneStreamHandler
)
from utils.tts_module import synthesize_speech, cleanup_tts, warmup_tts

# Import emotional intelligence
from emotion_memory import (
    start_emotional_session,
    record_conversation_turn,
    end_emotional_session,
    get_adaptive_greeting,
    get_user_emotion_stats,
    get_emotion_engine
)
from emotion_aware_chromadb import enhance_ai_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & DATA MODELS
# ============================================================================

class RequestStage(Enum):
    """Processing stages for request tracking"""
    RECEIVED = "received"
    USER_VALIDATION = "user_validation"
    STT_START = "stt_start"
    STT_COMPLETE = "stt_complete"
    BOOKING_CHECK = "booking_check"
    AI_AGENT_START = "ai_agent_start"
    AI_AGENT_COMPLETE = "ai_agent_complete"
    EMOTION_ENHANCEMENT = "emotion_enhancement"
    TTS_START = "tts_start"
    TTS_COMPLETE = "tts_complete"
    RESPONSE_SENT = "response_sent"
    ERROR = "error"
    USER_NOT_FOUND = "user_not_found"


class ConnectionMode(Enum):
    """WebSocket connection modes"""
    SINGLE_SHOT = "single_shot"
    CONTINUOUS = "continuous"


class UserBusinessData(BaseModel):
    """API model for business data submission"""
    user_id: str
    business_name: Optional[str] = None
    industry: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    additional_content: Optional[str] = None


@dataclass
class RequestMetrics:
    """Complete request metrics for monitoring"""
    request_id: str
    client_id: str
    user_id: str = ""
    audio_size_bytes: int = 0
    mode: ConnectionMode = ConnectionMode.SINGLE_SHOT
    
    # Timing (milliseconds)
    total_duration: int = 0
    user_validation_duration: int = 0
    stt_duration: int = 0
    booking_check_duration: int = 0
    ai_agent_duration: int = 0
    emotion_enhancement_duration: int = 0
    tts_duration: int = 0
    response_duration: int = 0
    
    # Results
    transcript: str = ""
    response_text: str = ""
    emotion_enhanced: bool = False
    detected_mood: str = "neutral"
    sentiment_score: float = 0.0
    stt_provider: str = ""
    stt_confidence: float = 0.0
    user_exists: bool = False
    user_has_data: bool = False
    is_booking: bool = False
    
    # Status
    stage: RequestStage = RequestStage.RECEIVED
    error: Optional[str] = None
    success: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['mode'] = self.mode.value
        result['stage'] = self.stage.value
        return result


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor:
    """Comprehensive performance monitoring"""
    
    def __init__(self):
        self.request_count = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.user_validation_failures = 0
        self.booking_requests = 0
        self.emotion_enhanced_requests = 0
        self.total_time = 0.0
        
        self.continuous_sessions = 0
        self.single_shot_requests = 0
        
        self.avg_user_validation_time = 0.0
        self.avg_stt_time = 0.0
        self.avg_ai_agent_time = 0.0
        self.avg_emotion_time = 0.0
        self.avg_tts_time = 0.0
        self.avg_response_time = 0.0
        
        self.recent_requests: List[RequestMetrics] = []
        self.max_recent_requests = 100
        
        self.errors_by_stage: Dict[str, int] = {}
        self.errors_by_type: Dict[str, int] = {}
    
    def record_request(self, metrics: RequestMetrics):
        """Record completed request with all metrics"""
        self.request_count += 1
        
        if metrics.mode == ConnectionMode.CONTINUOUS:
            self.continuous_sessions += 1
        else:
            self.single_shot_requests += 1
        
        if metrics.is_booking:
            self.booking_requests += 1
        
        if metrics.emotion_enhanced:
            self.emotion_enhanced_requests += 1
        
        if not metrics.user_exists or not metrics.user_has_data:
            self.user_validation_failures += 1
        
        if metrics.success:
            self.successful_requests += 1
            self.total_time += metrics.total_duration
            
            count = self.successful_requests
            self.avg_user_validation_time = ((self.avg_user_validation_time * (count - 1)) + metrics.user_validation_duration) / count
            self.avg_stt_time = ((self.avg_stt_time * (count - 1)) + metrics.stt_duration) / count
            self.avg_ai_agent_time = ((self.avg_ai_agent_time * (count - 1)) + metrics.ai_agent_duration) / count
            self.avg_emotion_time = ((self.avg_emotion_time * (count - 1)) + metrics.emotion_enhancement_duration) / count
            self.avg_tts_time = ((self.avg_tts_time * (count - 1)) + metrics.tts_duration) / count
            self.avg_response_time = ((self.avg_response_time * (count - 1)) + metrics.response_duration) / count
        else:
            self.failed_requests += 1
            
            stage_key = metrics.stage.value
            self.errors_by_stage[stage_key] = self.errors_by_stage.get(stage_key, 0) + 1
            
            if metrics.error:
                error_type = "Unknown"
                if hasattr(metrics.error, '__class__'):
                    error_type = type(metrics.error).__name__
                self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1
        
        self.recent_requests.append(metrics)
        if len(self.recent_requests) > self.max_recent_requests:
            self.recent_requests.pop(0)
        
        logger.info(f"Request completed: {metrics.request_id} | Success: {metrics.success} | Duration: {metrics.total_duration}ms")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        success_rate = (self.successful_requests / max(1, self.request_count)) * 100
        avg_total_time = self.total_time / max(1, self.successful_requests)
        user_validation_failure_rate = (self.user_validation_failures / max(1, self.request_count)) * 100
        emotion_enhancement_rate = (self.emotion_enhanced_requests / max(1, self.successful_requests)) * 100
        
        return {
            "requests": {
                "total": self.request_count,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate_percent": round(success_rate, 2),
                "user_validation_failures": self.user_validation_failures,
                "user_validation_failure_rate_percent": round(user_validation_failure_rate, 2),
                "booking_requests": self.booking_requests,
                "emotion_enhanced_requests": self.emotion_enhanced_requests,
                "emotion_enhancement_rate_percent": round(emotion_enhancement_rate, 2),
                "continuous_sessions": self.continuous_sessions,
                "single_shot_requests": self.single_shot_requests
            },
            "timing_ms": {
                "avg_total": round(avg_total_time, 0),
                "avg_user_validation": round(self.avg_user_validation_time, 0),
                "avg_stt": round(self.avg_stt_time, 0),
                "avg_ai_agent": round(self.avg_ai_agent_time, 0),
                "avg_emotion_enhancement": round(self.avg_emotion_time, 0),
                "avg_tts": round(self.avg_tts_time, 0),
                "avg_response": round(self.avg_response_time, 0)
            },
            "errors": {
                "by_stage": dict(self.errors_by_stage),
                "by_type": dict(self.errors_by_type)
            }
        }
    
    def get_recent_performance(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent request metrics"""
        recent = self.recent_requests[-limit:] if len(self.recent_requests) > limit else self.recent_requests
        return [req.to_dict() for req in recent]


# ============================================================================
# USER VALIDATION & LOCKS
# ============================================================================

user_query_locks: Dict[str, asyncio.Lock] = {}


async def get_user_lock(user_id: str) -> asyncio.Lock:
    """Get or create user-specific lock for thread safety"""
    if user_id not in user_query_locks:
        user_query_locks[user_id] = asyncio.Lock()
    return user_query_locks[user_id]


async def validate_user_and_data(user_id: str) -> Tuple[bool, bool, str]:
    """
    Validate user exists and has trained data
    Returns: (user_exists, user_has_data, message)
    """
    try:
        logger.info(f"Validating user: {user_id}")
        
        stats = await asyncio.get_running_loop().run_in_executor(
            None,
            get_agent_stats,
            user_id
        )
        
        if not stats or "error" in stats:
            return False, False, f"User {user_id} has not been trained yet. Please submit business data via /data endpoint first."
        
        qa_pairs = stats.get("qa_pairs", 0)
        
        if qa_pairs == 0:
            return False, False, f"User {user_id} has not been trained yet. Please provide business data first."
        
        business_name = stats.get("business_name", "Unknown Business")
        logger.info(f"User {user_id} validated: {qa_pairs} Q&A pairs for {business_name}")
        
        return True, True, f"User validated with {qa_pairs} Q&A pairs"
        
    except Exception as e:
        logger.error(f"User validation failed for {user_id}: {e}", exc_info=True)
        return False, False, f"Error validating user: {str(e)}"


# ============================================================================
# EMOTION-AWARE AI QUERY
# ============================================================================

async def query_ai_agent_with_emotion(
    transcript: str, 
    user_id: str,
    client_id: str,
    connection_manager: 'EnhancedConnectionManager'
) -> Tuple[str, bool, str, float]:
    """
    Thread-safe AI agent query WITH emotional intelligence
    Returns: (response_text, emotion_enhanced, detected_mood, sentiment_score)
    """
    user_lock = await get_user_lock(user_id)
    
    async with user_lock:
        try:
            # Validate user
            user_exists, user_has_data, validation_msg = await validate_user_and_data(user_id)
            
            if not user_exists or not user_has_data:
                logger.warning(f"Query blocked for user {user_id}: {validation_msg}")
                return f"Sorry, I can't answer your question. {validation_msg}", False, "neutral", 0.0
            
            # Get emotion context from current session
            detected_mood = "neutral"
            sentiment_score = 0.0
            conversation_context = None
            
            try:
                if client_id in connection_manager.emotion_sessions:
                    emotion_session_id = connection_manager.emotion_sessions[client_id]
                    
                    engine = await asyncio.get_running_loop().run_in_executor(
                        None,
                        get_emotion_engine
                    )
                    
                    if emotion_session_id in engine.active_sessions:
                        turns = engine.active_sessions[emotion_session_id]
                        
                        if turns:
                            last_turn = turns[-1]
                            detected_mood = last_turn.detected_mood.value
                            sentiment_score = last_turn.sentiment_score
                            
                            recent_turns = turns[-3:] if len(turns) >= 3 else turns
                            context_parts = []
                            for turn in recent_turns:
                                context_parts.append(
                                    f"User ({turn.detected_mood.value}): {turn.user_message[:50]}..."
                                )
                            conversation_context = " | ".join(context_parts)
                            
                            logger.info(
                                f"🎭 Emotion context for {client_id}: mood={detected_mood}, "
                                f"sentiment={sentiment_score:.2f}"
                            )
            except Exception as e:
                logger.error(f"Failed to get emotion context: {e}")
            
            # Query AI agent (base response)
            logger.info(f"Querying AI agent for user {user_id}: '{transcript[:50]}...'")
            base_response = await asyncio.get_running_loop().run_in_executor(
                None, 
                query_ai_agent, 
                transcript,
                user_id
            )
            
            if not base_response or not base_response.strip():
                base_response = "I'm not sure how to respond to that. Could you please rephrase your question?"
            
            # Enhance response with emotional intelligence
            enhanced_response = await asyncio.get_running_loop().run_in_executor(
                None,
                enhance_ai_response,
                base_response,
                transcript,
                detected_mood,
                sentiment_score,
                conversation_context
            )
            
            emotion_enhanced = enhanced_response != base_response
            
            if emotion_enhanced:
                logger.info(f"✨ Response emotionally enhanced for mood: {detected_mood}")
            
            logger.info(f"AI response ready: '{enhanced_response[:100]}...'")
            
            return enhanced_response, emotion_enhanced, detected_mood, sentiment_score
            
        except Exception as e:
            logger.error(f"AI agent query failed: {e}", exc_info=True)
            return "Sorry, I'm having trouble accessing my knowledge base right now.", False, "neutral", 0.0


# ============================================================================
# CONNECTION MANAGER WITH EMOTION TRACKING
# ============================================================================

class EnhancedConnectionManager:
    """WebSocket connection manager with full emotional intelligence"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_stats: Dict[str, Dict] = {}
        self.connection_modes: Dict[str, ConnectionMode] = {}
        self.connection_users: Dict[str, str] = {}
        self.streaming_sessions: Dict[str, Any] = {}
        self.microphone_handlers: Dict[str, MicrophoneStreamHandler] = {}
        self.emotion_sessions: Dict[str, str] = {}  # client_id -> emotion_session_id
        self.connection_greetings: Dict[str, str] = {}  # client_id -> greeting sent
    
    async def connect(self, websocket: WebSocket, client_id: str, user_id: str = None):
        """Accept connection and setup emotional intelligence"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        if not user_id or user_id == client_id:
            logger.warning(f"No valid user_id for connection {client_id}")
            user_id = "no_user"
        
        self.connection_users[client_id] = user_id
        
        self.connection_stats[client_id] = {
            "connected_at": time.time(),
            "requests_processed": 0,
            "last_activity": time.time(),
            "mode": ConnectionMode.SINGLE_SHOT.value,
            "user_id": user_id,
            "valid_user": user_id != "no_user"
        }
        
        # Start emotional intelligence session and get adaptive greeting
        greeting = "Hello! How can I help you today?"  # Default
        emotion_session_id = None
        
        if user_id != "no_user":
            try:
                # Start emotion session
                emotion_session_id = await asyncio.get_running_loop().run_in_executor(
                    None,
                    start_emotional_session,
                    user_id
                )
                self.emotion_sessions[client_id] = emotion_session_id
                logger.info(f"🎭 Emotion session started: {emotion_session_id}")
                
                # Get adaptive greeting based on history
                greeting = await asyncio.get_running_loop().run_in_executor(
                    None,
                    get_adaptive_greeting,
                    user_id
                )
                self.connection_greetings[client_id] = greeting
                logger.info(f"👋 Adaptive greeting generated: '{greeting[:50]}...'")
                
            except Exception as e:
                logger.error(f"Failed to start emotion session: {e}")
        
        # Send connection confirmation with greeting
        await self.safe_send_text(client_id, json.dumps({
            "type": "connected",
            "client_id": client_id,
            "user_id": user_id,
            "greeting": greeting,
            "emotion_tracking": emotion_session_id is not None,
            "emotion_session_id": emotion_session_id
        }))
        
        logger.info(f"✓ Client connected: {client_id} (user: {user_id})")
    
    def get_user_id(self, client_id: str) -> str:
        """Get user_id for client"""
        return self.connection_users.get(client_id, "no_user")
    
    def is_valid_user(self, client_id: str) -> bool:
        """Check if client has valid user_id"""
        user_id = self.get_user_id(client_id)
        return user_id != "no_user" and user_id is not None
    
    def disconnect(self, client_id: str):
        """Clean up connection and end emotional session"""
        # End emotional intelligence session
        if client_id in self.emotion_sessions:
            try:
                emotion_session_id = self.emotion_sessions[client_id]
                user_id = self.get_user_id(client_id)
                
                asyncio.create_task(self._end_emotion_session_async(emotion_session_id, user_id))
                
                del self.emotion_sessions[client_id]
            except Exception as e:
                logger.error(f"Failed to end emotion session: {e}")
        
        if client_id in self.streaming_sessions:
            del self.streaming_sessions[client_id]
        
        if client_id in self.microphone_handlers:
            handler = self.microphone_handlers[client_id]
            asyncio.create_task(handler.stop_stream())
            del self.microphone_handlers[client_id]
        
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        if client_id in self.connection_modes:
            del self.connection_modes[client_id]
            
        if client_id in self.connection_users:
            del self.connection_users[client_id]
        
        if client_id in self.connection_greetings:
            del self.connection_greetings[client_id]
        
        if client_id in self.connection_stats:
            stats = self.connection_stats.pop(client_id)
            duration = time.time() - stats["connected_at"]
            logger.info(f"✓ Client disconnected: {client_id} (duration: {duration:.0f}s, requests: {stats['requests_processed']})")
    
    async def _end_emotion_session_async(self, emotion_session_id: str, user_id: str):
        """End emotion session asynchronously with summary logging"""
        try:
            summary = await asyncio.get_running_loop().run_in_executor(
                None,
                end_emotional_session,
                emotion_session_id,
                user_id
            )
            outcome = summary.get('session_outcome', 'unknown')
            avg_sentiment = summary.get('average_sentiment', 0.0)
            total_turns = summary.get('total_turns', 0)
            
            logger.info(
                f"🎭 Emotion session ended: {emotion_session_id} | "
                f"Outcome: {outcome} | "
                f"Avg Sentiment: {avg_sentiment:.2f} | "
                f"Turns: {total_turns}"
            )
        except Exception as e:
            logger.error(f"Failed to end emotion session: {e}")
    
    async def record_emotion_turn(
        self,
        client_id: str,
        user_message: str,
        assistant_response: str,
        intent_fulfilled: bool,
        response_time_ms: int
    ):
        """Record conversation turn in emotional intelligence system"""
        if client_id not in self.emotion_sessions:
            return
        
        try:
            emotion_session_id = self.emotion_sessions[client_id]
            
            await asyncio.get_running_loop().run_in_executor(
                None,
                record_conversation_turn,
                emotion_session_id,
                user_message,
                assistant_response,
                intent_fulfilled,
                response_time_ms
            )
            
            logger.debug(f"📝 Emotion turn recorded for {client_id}")
            
        except Exception as e:
            logger.error(f"Failed to record emotion turn: {e}")
    
    async def start_continuous_session(self, client_id: str, vad_provider: VADProvider = VADProvider.SILERO):
        """Start continuous streaming session"""
        if client_id not in self.active_connections:
            return False
        
        if not self.is_valid_user(client_id):
            await self.safe_send_text(client_id, json.dumps({
                "type": "error",
                "message": "Cannot start continuous session: No valid user_id",
                "error_code": "INVALID_USER_ID"
            }))
            return False
        
        stt_processor = create_continuous_streaming_stt(vad_provider)
        
        stt_processor.add_transcript_callback(
            lambda text, confidence: asyncio.create_task(
                self._handle_transcript(client_id, text, confidence)
            )
        )
        stt_processor.add_status_callback(
            lambda status: asyncio.create_task(
                self._handle_status_update(client_id, status)
            )
        )
        
        mic_handler = MicrophoneStreamHandler(stt_processor)
        
        self.streaming_sessions[client_id] = stt_processor
        self.microphone_handlers[client_id] = mic_handler
        self.connection_modes[client_id] = ConnectionMode.CONTINUOUS
        
        if client_id in self.connection_stats:
            self.connection_stats[client_id]["mode"] = ConnectionMode.CONTINUOUS.value
        
        await mic_handler.start_stream()
        
        logger.info(f"🎤 Started continuous session for {client_id}")
        return True
    
    async def stop_continuous_session(self, client_id: str):
        """Stop continuous session"""
        if client_id in self.microphone_handlers:
            await self.microphone_handlers[client_id].stop_stream()
            del self.microphone_handlers[client_id]
        
        if client_id in self.streaming_sessions:
            del self.streaming_sessions[client_id]
        
        if client_id in self.connection_modes:
            self.connection_modes[client_id] = ConnectionMode.SINGLE_SHOT
        
        if client_id in self.connection_stats:
            self.connection_stats[client_id]["mode"] = ConnectionMode.SINGLE_SHOT.value
        
        logger.info(f"🛑 Stopped continuous session for {client_id}")
    
    async def add_audio_chunk_to_stream(self, client_id: str, audio_data: bytes):
        """Add audio to continuous stream"""
        if client_id not in self.microphone_handlers:
            return False
        
        handler = self.microphone_handlers[client_id]
        await handler.add_audio_chunk(audio_data)
        return True
    
    async def _handle_transcript(self, client_id: str, text: str, confidence: float):
        """Handle continuous transcript"""
        user_id = self.get_user_id(client_id)
        logger.info(f"📝 Transcript for {client_id}: '{text}' (confidence: {confidence:.3f})")
        
        request_id = f"{client_id}_continuous_{int(time.time())}"
        metrics = RequestMetrics(
            request_id=request_id,
            client_id=client_id,
            user_id=user_id,
            mode=ConnectionMode.CONTINUOUS,
            transcript=text,
            stt_confidence=confidence
        )
        
        success = await self._process_continuous_transcript(client_id, text, metrics)
        
        metrics.success = success
        perf_monitor.record_request(metrics)
    
    async def _handle_status_update(self, client_id: str, status: str):
        """Handle status updates"""
        await self.safe_send_text(client_id, json.dumps({
            "type": "status",
            "status": status,
            "timestamp": time.time()
        }))
    
    async def _process_continuous_transcript(self, client_id: str, transcript: str, metrics: RequestMetrics) -> bool:
        """Process continuous transcript with full emotion tracking"""
        overall_start = time.perf_counter()
        
        try:
            # User Validation
            metrics.stage = RequestStage.USER_VALIDATION
            validation_start = time.perf_counter()
            
            user_id = self.get_user_id(client_id)
            
            if user_id == "no_user":
                metrics.user_validation_duration = int((time.perf_counter() - validation_start) * 1000)
                metrics.error = "No user_id"
                metrics.stage = RequestStage.USER_NOT_FOUND
                
                await self.safe_send_text(client_id, json.dumps({
                    "type": "error", 
                    "message": "User ID not provided",
                    "error_code": "NO_USER_ID"
                }))
                return False
            
            user_exists, user_has_data, validation_msg = await validate_user_and_data(user_id)
            metrics.user_exists = user_exists
            metrics.user_has_data = user_has_data
            metrics.user_validation_duration = int((time.perf_counter() - validation_start) * 1000)
            
            if not user_exists or not user_has_data:
                metrics.error = validation_msg
                metrics.stage = RequestStage.USER_NOT_FOUND
                
                error_audio = await synthesize_speech(f"Sorry, I can't help you right now. {validation_msg}")
                if error_audio:
                    await self.safe_send_bytes(client_id, error_audio)
                
                await self.record_emotion_turn(
                    client_id, transcript, validation_msg, False,
                    int((time.perf_counter() - overall_start) * 1000)
                )
                return False
            
            # Check for greeting or booking
            metrics.stage = RequestStage.BOOKING_CHECK
            booking_start = time.perf_counter()
            
            # Check if this is a greeting query
            is_greeting_query = any(word in transcript.lower() for word in [
                'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'
            ]) and len(transcript.split()) <= 5
            
            if is_greeting_query:
                # Respond with adaptive greeting
                metrics.booking_check_duration = int((time.perf_counter() - booking_start) * 1000)
                
                metrics.stage = RequestStage.AI_AGENT_START
                ai_start = time.perf_counter()
                
                greeting = await asyncio.get_running_loop().run_in_executor(
                    None,
                    get_adaptive_greeting,
                    user_id
                )
                
                response_text = greeting
                metrics.ai_agent_duration = int((time.perf_counter() - ai_start) * 1000)
                metrics.response_text = response_text
                metrics.stage = RequestStage.AI_AGENT_COMPLETE
                
                metrics.emotion_enhanced = False
                metrics.detected_mood = "neutral"
                metrics.sentiment_score = 0.0
                
                logger.info(f"👋 Greeting response (continuous): '{response_text}'")
                
            elif await is_booking_query(transcript):
                metrics.is_booking = True
                metrics.booking_check_duration = int((time.perf_counter() - booking_start) * 1000)
                
                metrics.stage = RequestStage.AI_AGENT_START
                ai_start = time.perf_counter()
                
                response_text = await process_booking_query(transcript, user_id)
                metrics.ai_agent_duration = int((time.perf_counter() - ai_start) * 1000)
                metrics.response_text = response_text
                metrics.stage = RequestStage.AI_AGENT_COMPLETE
                
                # Booking responses don't get emotion enhancement
                metrics.emotion_enhanced = False
                metrics.detected_mood = "neutral"
                metrics.sentiment_score = 0.0
                
            else:
                metrics.booking_check_duration = int((time.perf_counter() - booking_start) * 1000)
                
                # Regular AI agent query with emotion
                metrics.stage = RequestStage.AI_AGENT_START
                ai_start = time.perf_counter()
                
                response_text, emotion_enhanced, detected_mood, sentiment_score = await query_ai_agent_with_emotion(
                    transcript, user_id, client_id, self
                )
                
                metrics.ai_agent_duration = int((time.perf_counter() - ai_start) * 1000)
                metrics.response_text = response_text
                metrics.emotion_enhanced = emotion_enhanced
                metrics.detected_mood = detected_mood
                metrics.sentiment_score = sentiment_score
                metrics.stage = RequestStage.AI_AGENT_COMPLETE
            
            # TTS
            metrics.stage = RequestStage.TTS_START
            tts_start = time.perf_counter()
            
            audio_bytes = await synthesize_speech(response_text)
            metrics.tts_duration = int((time.perf_counter() - tts_start) * 1000)
            
            if not audio_bytes or len(audio_bytes) == 0:
                raise ValueError("TTS returned empty audio")
            
            metrics.stage = RequestStage.TTS_COMPLETE
            
            # Send response
            response_start = time.perf_counter()
            
            await self.safe_send_text(client_id, json.dumps({
                "type": "response_metadata",
                "transcript": transcript,
                "response_text": response_text,
                "confidence": metrics.stt_confidence,
                "audio_size": len(audio_bytes),
                "user_id": user_id,
                "is_booking": metrics.is_booking,
                "emotion_enhanced": metrics.emotion_enhanced,
                "detected_mood": metrics.detected_mood,
                "sentiment_score": metrics.sentiment_score
            }))
            
            success = await self.safe_send_bytes(client_id, audio_bytes)
            
            metrics.response_duration = int((time.perf_counter() - response_start) * 1000)
            metrics.total_duration = int((time.perf_counter() - overall_start) * 1000)
            
            # Record emotional turn
            await self.record_emotion_turn(
                client_id,
                transcript,
                response_text,
                success,
                metrics.total_duration
            )
            
            if success:
                metrics.stage = RequestStage.RESPONSE_SENT
                logger.info(f"✓ Continuous request completed in {metrics.total_duration}ms")
                return True
            else:
                metrics.error = "Failed to send response"
                return False
                
        except Exception as e:
            metrics.stage = RequestStage.ERROR
            metrics.error = str(e)
            metrics.total_duration = int((time.perf_counter() - overall_start) * 1000)
            logger.error(f"Continuous processing error: {e}", exc_info=True)
            
            await self.record_emotion_turn(
                client_id, transcript, f"Error: {str(e)}", False,
                metrics.total_duration
            )
            return False
    
    def update_activity(self, client_id: str):
        """Update activity timestamp"""
        if client_id in self.connection_stats:
            self.connection_stats[client_id]["last_activity"] = time.time()
            self.connection_stats[client_id]["requests_processed"] += 1
    
    async def safe_send_bytes(self, client_id: str, data: bytes) -> bool:
        """Safely send bytes"""
        if client_id not in self.active_connections:
            return False
        
        try:
            await self.active_connections[client_id].send_bytes(data)
            return True
        except Exception as e:
            logger.error(f"Failed to send bytes to {client_id}: {e}")
            self.disconnect(client_id)
            return False
    
    async def safe_send_text(self, client_id: str, message: str) -> bool:
        """Safely send text"""
        if client_id not in self.active_connections:
            return False
        
        try:
            await self.active_connections[client_id].send_text(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send text to {client_id}: {e}")
            self.disconnect(client_id)
            return False
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        total = len(self.active_connections)
        if total == 0:
            return {"active_connections": 0}
        
        total_requests = sum(stats["requests_processed"] for stats in self.connection_stats.values())
        continuous = sum(1 for mode in self.connection_modes.values() if mode == ConnectionMode.CONTINUOUS)
        valid = sum(1 for stats in self.connection_stats.values() if stats.get("valid_user", False))
        emotion_tracked = len(self.emotion_sessions)
        
        return {
            "active_connections": total,
            "valid_user_connections": valid,
            "invalid_user_connections": total - valid,
            "continuous_sessions": continuous,
            "single_shot_connections": total - continuous,
            "emotion_tracked_sessions": emotion_tracked,
            "total_requests_processed": total_requests,
            "avg_requests_per_connection": round(total_requests / total, 1) if total > 0 else 0
        }


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

perf_monitor = PerformanceMonitor()
connection_manager = EnhancedConnectionManager()


# ============================================================================
# LIFECYCLE MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("=" * 80)
    logger.info("Starting Production Voice Assistant with Emotional Intelligence v4.0")
    logger.info("=" * 80)
    startup_start = time.perf_counter()
    
    try:
        await warmup_components()
        startup_time = (time.perf_counter() - startup_start) * 1000
        logger.info(f"✓ Backend ready in {startup_time:.0f}ms")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        raise
    
    yield
    
    logger.info("=" * 80)
    logger.info("Shutting down gracefully...")
    await cleanup_all_resources()
    logger.info("✓ Shutdown complete")
    logger.info("=" * 80)


async def warmup_components():
    """Warm up all system components"""
    logger.info("Warming up components...")
    
    warmup_tasks = [
        ("STT", warmup_stt_async()),
        ("TTS", warmup_tts()),
        ("Emotion Engine", warmup_emotion_engine())
    ]
    
    for name, task in warmup_tasks:
        try:
            start = time.perf_counter()
            await task
            duration = (time.perf_counter() - start) * 1000
            logger.info(f"  ✓ {name} ready in {duration:.0f}ms")
        except Exception as e:
            logger.error(f"  ✗ {name} warmup failed: {e}")


async def warmup_stt_async():
    """Async STT warmup"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, warmup_stt)


async def warmup_emotion_engine():
    """Warm up emotion engine"""
    try:
        engine = await asyncio.get_running_loop().run_in_executor(None, get_emotion_engine)
        logger.debug("Emotion engine initialized")
    except Exception as e:
        logger.error(f"Emotion engine warmup failed: {e}")


async def cleanup_all_resources():
    """Clean up all resources"""
    cleanup_tasks = [
        ("STT", cleanup_stt()),
        ("TTS", cleanup_tts())
    ]
    
    for name, task in cleanup_tasks:
        try:
            if asyncio.iscoroutine(task):
                await task
            logger.info(f"  ✓ {name} cleanup complete")
        except Exception as e:
            logger.error(f"  ✗ {name} cleanup failed: {e}")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Production AI Voice Assistant with Emotional Intelligence",
    description="Enterprise-grade voice assistant with emotion tracking, adaptive responses, and comprehensive monitoring",
    version="4.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Production AI Voice Assistant with Emotional Intelligence",
        "version": "4.0.0",
        "status": "running",
        "features": {
            "core": [
                "User-specific AI agents with ChromaDB",
                "Real-time WebSocket communication",
                "Single-shot and continuous microphone modes",
                "Deepgram + Whisper STT with fallback",
                "Byte-based TTS synthesis",
                "Booking system integration"
            ],
            "emotional_intelligence": [
                "Real-time sentiment analysis with Groq",
                "Mood trajectory tracking across sessions",
                "Session outcome classification",
                "Adaptive greetings based on history",
                "Emotion-enhanced AI responses",
                "Automatic repair strategies for poor experiences"
            ],
            "monitoring": [
                "Comprehensive performance metrics",
                "Request-level tracking",
                "Error analysis by stage and type",
                "Connection statistics"
            ]
        },
        "endpoints": {
            "training": "POST /data - Train AI agent with business data",
            "websocket": "WS /ws/voice?user_id=YOUR_USER_ID - Voice interaction",
            "stats": "GET /stats - Performance statistics",
            "user_validation": "GET /users/{user_id}/validate - Validate user",
            "emotion_stats": "GET /users/{user_id}/emotion-stats - Emotion statistics",
            "greeting": "GET /users/{user_id}/greeting - Get adaptive greeting"
        }
    }


@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    try:
        perf_stats = perf_monitor.get_current_stats()
        connection_stats = connection_manager.get_connection_stats()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {
                "chromadb": "ready",
                "booking_system": "active",
                "user_validation": "active",
                "emotion_engine": "active",
                "emotion_aware_responses": "active",
                "stt": "ready",
                "tts": "ready"
            },
            "performance": perf_stats,
            "connections": connection_stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e)
            }
        )


@app.post("/data")
async def save_user_data(request: UserBusinessData):
    """Train AI agent with business data"""
    try:
        logger.info(f"Training request from user: {request.user_id}")

        business_data = {
            "business_name": request.business_name or "Business",
            "industry": request.industry or "Service", 
            "phone": request.phone or "",
            "email": request.email or "",
            "content": request.additional_content or ""
        }

        logger.info(f"Initiating AI agent training for {request.user_id}...")
        training_success = await asyncio.get_running_loop().run_in_executor(
            None, 
            train_ai_agent, 
            request.user_id, 
            business_data
        )

        if training_success:
            stats = await asyncio.get_running_loop().run_in_executor(
                None,
                get_agent_stats,
                request.user_id
            )
            
            logger.info(f"✓ Training successful for {request.user_id}: {stats.get('qa_pairs', 0)} Q&A pairs")
            
            return {
                "success": True,
                "message": "AI agent trained successfully with emotional intelligence enabled",
                "user_id": request.user_id,
                "training_stats": {
                    "qa_pairs_generated": stats.get("qa_pairs", 0),
                    "business_name": stats.get("business_name", "Unknown"),
                    "comprehensive_coverage": stats.get("comprehensive_coverage", False)
                },
                "next_steps": {
                    "websocket": f"ws://localhost:5000/ws/voice?user_id={request.user_id}",
                    "note": "Voice queries with emotional intelligence tracking are now ready"
                }
            }
        else:
            logger.error(f"✗ Training failed for {request.user_id}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Training failed",
                    "user_id": request.user_id
                }
            )

    except Exception as e:
        logger.error(f"Data processing failed for {request.user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Failed to process user data"
            }
        )


@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for voice interactions"""
    client_id = f"{websocket.client.host}:{websocket.client.port}:{int(time.time())}"
    user_id = websocket.query_params.get("user_id")
    
    logger.info(f"WebSocket connection attempt: client={client_id}, user_id={user_id or 'NOT PROVIDED'}")
    
    try:
        await connection_manager.connect(websocket, client_id, user_id)
        
        # Send initial status with emotional context
        if user_id and user_id != "no_user":
            user_exists, user_has_data, validation_msg = await validate_user_and_data(user_id)
            
            # Get emotion stats
            emotion_stats = None
            try:
                emotion_stats = await asyncio.get_running_loop().run_in_executor(
                    None,
                    get_user_emotion_stats,
                    user_id
                )
            except Exception as e:
                logger.error(f"Failed to get emotion stats: {e}")
            
            if user_exists and user_has_data:
                await connection_manager.safe_send_text(client_id, json.dumps({
                    "type": "status",
                    "status": "ready",
                    "message": "Connected! Your AI agent is ready with emotional intelligence.",
                    "user_id": user_id,
                    "validation": validation_msg,
                    "emotion_stats": emotion_stats
                }))
            else:
                await connection_manager.safe_send_text(client_id, json.dumps({
                    "type": "status",
                    "status": "not_trained",
                    "error": "User not trained",
                    "message": f"Cannot process queries: {validation_msg}",
                    "action_required": "Train via POST /data endpoint first"
                }))
        else:
            await connection_manager.safe_send_text(client_id, json.dumps({
                "type": "status",
                "status": "invalid_user",
                "error": "No user ID provided",
                "message": "Cannot process queries without user_id",
                "action_required": "Reconnect with: ws://host/ws/voice?user_id=YOUR_USER_ID"
            }))
        
        request_counter = 0
        
        # Main WebSocket loop
        while True:
            try:
                # Check for text commands first
                try:
                    text_message = await asyncio.wait_for(
                        websocket.receive_text(), 
                        timeout=0.1
                    )
                    
                    if await handle_text_command(client_id, text_message):
                        continue
                    
                except asyncio.TimeoutError:
                    pass
                
                # Receive audio data
                message = await asyncio.wait_for(
                    websocket.receive_bytes(), 
                    timeout=60.0
                )
                
                if not connection_manager.is_valid_user(client_id):
                    await connection_manager.safe_send_text(
                        client_id, 
                        json.dumps({
                            "type": "error",
                            "error_code": "INVALID_USER_ID", 
                            "message": "Cannot process audio: No valid user_id"
                        })
                    )
                    continue
                
                # Determine mode
                current_mode = connection_manager.connection_modes.get(
                    client_id, ConnectionMode.SINGLE_SHOT
                )
                
                if current_mode == ConnectionMode.CONTINUOUS:
                    # Continuous mode - stream to handler
                    await connection_manager.add_audio_chunk_to_stream(client_id, message)
                    connection_manager.update_activity(client_id)
                    
                else:
                    # Single-shot mode - process immediately
                    if not message or len(message) < 100:
                        await connection_manager.safe_send_text(
                            client_id, 
                            json.dumps({"error": "Invalid audio data"})
                        )
                        continue
                    
                    request_counter += 1
                    request_id = f"{client_id}_req_{request_counter}"
                    
                    connection_manager.update_activity(client_id)
                    
                    user_id = connection_manager.get_user_id(client_id)
                    metrics = RequestMetrics(
                        request_id=request_id,
                        client_id=client_id,
                        user_id=user_id,
                        audio_size_bytes=len(message),
                        mode=ConnectionMode.SINGLE_SHOT
                    )
                    
                    success = await process_voice_request(
                        websocket, client_id, message, metrics
                    )
                    
                    metrics.success = success
                    perf_monitor.record_request(metrics)
                
            except asyncio.TimeoutError:
                logger.warning(f"Client {client_id} timed out (60s inactivity)")
                await connection_manager.safe_send_text(
                    client_id, 
                    json.dumps({"error": "timeout", "message": "No activity in 60s"})
                )
                break
                
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected normally")
                break
                
            except ConnectionResetError:
                logger.info(f"Client {client_id} connection reset")
                break
                
            except Exception as e:
                logger.error(f"WebSocket error for {client_id}: {e}", exc_info=True)
                await connection_manager.safe_send_text(
                    client_id, 
                    json.dumps({
                        "error": "processing_error",
                        "message": "An unexpected error occurred"
                    })
                )
                
    except Exception as e:
        logger.error(f"Fatal WebSocket error for {client_id}: {e}", exc_info=True)
    finally:
        connection_manager.disconnect(client_id)


async def handle_text_command(client_id: str, command: str) -> bool:
    """Handle text commands from WebSocket"""
    try:
        try:
            cmd_data = json.loads(command)
            cmd_type = cmd_data.get("type")
        except json.JSONDecodeError:
            cmd_type = command.strip().lower()
            cmd_data = {"type": cmd_type}
        
        if cmd_type == "start_continuous":
            if not connection_manager.is_valid_user(client_id):
                await connection_manager.safe_send_text(client_id, json.dumps({
                    "type": "error",
                    "message": "Cannot start continuous mode: Invalid user_id",
                    "error_code": "INVALID_USER_ID"
                }))
                return True
            
            vad_provider = VADProvider.SILERO
            if "vad_provider" in cmd_data:
                try:
                    vad_provider = VADProvider(cmd_data["vad_provider"])
                except ValueError:
                    pass
            
            success = await connection_manager.start_continuous_session(
                client_id, vad_provider
            )
            
            if success:
                user_id = connection_manager.get_user_id(client_id)
                await connection_manager.safe_send_text(client_id, json.dumps({
                    "type": "continuous_started",
                    "message": "Continuous mode activated with emotion tracking",
                    "vad_provider": vad_provider.value,
                    "user_id": user_id
                }))
            else:
                await connection_manager.safe_send_text(client_id, json.dumps({
                    "type": "error",
                    "message": "Failed to start continuous mode"
                }))
            
            return True
        
        elif cmd_type == "stop_continuous":
            await connection_manager.stop_continuous_session(client_id)
            
            await connection_manager.safe_send_text(client_id, json.dumps({
                "type": "continuous_stopped",
                "message": "Continuous mode stopped"
            }))
            
            return True
        
        elif cmd_type == "get_status":
            mode = connection_manager.connection_modes.get(
                client_id, ConnectionMode.SINGLE_SHOT
            )
            stats = connection_manager.connection_stats.get(client_id, {})
            user_id = connection_manager.get_user_id(client_id)
            
            if user_id != "no_user":
                user_exists, user_has_data, validation_msg = await validate_user_and_data(user_id)
                user_status = "ready" if (user_exists and user_has_data) else "not_trained"
                
                try:
                    emotion_stats = await asyncio.get_running_loop().run_in_executor(
                        None,
                        get_user_emotion_stats,
                        user_id
                    )
                except Exception:
                    emotion_stats = None
            else:
                user_status = "invalid_user"
                validation_msg = "No user_id"
                emotion_stats = None
            
            await connection_manager.safe_send_text(client_id, json.dumps({
                "type": "status_response",
                "mode": mode.value,
                "user_id": user_id,
                "user_status": user_status,
                "user_validation": validation_msg,
                "emotion_stats": emotion_stats,
                "stats": stats
            }))
            
            return True
        
        elif cmd_type == "ping":
            await connection_manager.safe_send_text(client_id, json.dumps({
                "type": "pong",
                "timestamp": time.time(),
                "user_id": connection_manager.get_user_id(client_id)
            }))
            
            return True
        
        return False
    
    except Exception as e:
        logger.error(f"Command handling error: {e}")
        return True


async def process_voice_request(
    websocket: WebSocket, 
    client_id: str, 
    audio_data: bytes, 
    metrics: RequestMetrics
) -> bool:
    """Process single voice request with full emotion tracking"""
    
    overall_start = time.perf_counter()
    
    try:
        logger.info(f"Processing request {metrics.request_id}: {len(audio_data)} bytes")
        
        # STAGE 1: User Validation
        metrics.stage = RequestStage.USER_VALIDATION
        validation_start = time.perf_counter()
        
        if metrics.user_id == "no_user":
            metrics.user_validation_duration = int((time.perf_counter() - validation_start) * 1000)
            metrics.error = "No user_id"
            metrics.stage = RequestStage.USER_NOT_FOUND
            
            await connection_manager.safe_send_text(client_id, json.dumps({
                "type": "error", 
                "message": "User ID not provided",
                "error_code": "NO_USER_ID"
            }))
            return False
        
        user_exists, user_has_data, validation_msg = await validate_user_and_data(metrics.user_id)
        metrics.user_exists = user_exists
        metrics.user_has_data = user_has_data
        metrics.user_validation_duration = int((time.perf_counter() - validation_start) * 1000)
        
        if not user_exists or not user_has_data:
            metrics.error = validation_msg
            metrics.stage = RequestStage.USER_NOT_FOUND
            
            error_audio = await synthesize_speech(f"Sorry, I can't help you right now. {validation_msg}")
            if error_audio:
                await connection_manager.safe_send_bytes(client_id, error_audio)
            
            await connection_manager.record_emotion_turn(
                client_id, "", validation_msg, False,
                int((time.perf_counter() - overall_start) * 1000)
            )
            return False
        
        # STAGE 2: STT Processing
        metrics.stage = RequestStage.STT_START
        stt_start = time.perf_counter()
        
        try:
            stt_result = await transcribe_wav_bytes(audio_data)
            metrics.stt_duration = int((time.perf_counter() - stt_start) * 1000)
            
            if not stt_result.is_success or not stt_result.text.strip():
                await connection_manager.safe_send_text(
                    client_id,
                    json.dumps({"error": "transcription_failed", "message": "Could not transcribe audio"})
                )
                metrics.error = stt_result.error or "Empty transcription"
                return False
            
            metrics.transcript = stt_result.text
            metrics.stt_provider = stt_result.provider.value
            metrics.stt_confidence = stt_result.confidence
            metrics.stage = RequestStage.STT_COMPLETE
            
            logger.info(f"STT: '{metrics.transcript}' (provider: {metrics.stt_provider}, confidence: {metrics.stt_confidence:.3f})")
            
        except Exception as stt_error:
            metrics.stt_duration = int((time.perf_counter() - stt_start) * 1000)
            metrics.error = str(stt_error)
            logger.error(f"STT failed: {stt_error}")
            return False
        
        # STAGE 3: Check for booking
        metrics.stage = RequestStage.BOOKING_CHECK
        booking_start = time.perf_counter()
        
        # Check if this is a greeting query (hi, hello, hey, etc.)
        is_greeting_query = any(word in metrics.transcript.lower() for word in [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'
        ]) and len(metrics.transcript.split()) <= 5
        
        if is_greeting_query:
            # Respond with adaptive greeting
            metrics.booking_check_duration = int((time.perf_counter() - booking_start) * 1000)
            
            metrics.stage = RequestStage.AI_AGENT_START
            ai_start = time.perf_counter()
            
            # Get adaptive greeting
            greeting = await asyncio.get_running_loop().run_in_executor(
                None,
                get_adaptive_greeting,
                metrics.user_id
            )
            
            response_text = greeting
            metrics.ai_agent_duration = int((time.perf_counter() - ai_start) * 1000)
            metrics.response_text = response_text
            metrics.stage = RequestStage.AI_AGENT_COMPLETE
            
            # Greeting responses don't get emotion enhancement
            metrics.emotion_enhanced = False
            metrics.detected_mood = "neutral"
            metrics.sentiment_score = 0.0
            
            logger.info(f"👋 Greeting response: '{response_text}'")
            
        elif await is_booking_query(metrics.transcript):
            metrics.is_booking = True
            metrics.booking_check_duration = int((time.perf_counter() - booking_start) * 1000)
            
            metrics.stage = RequestStage.AI_AGENT_START
            ai_start = time.perf_counter()
            
            response_text = await process_booking_query(metrics.transcript, metrics.user_id)
            metrics.ai_agent_duration = int((time.perf_counter() - ai_start) * 1000)
            metrics.response_text = response_text
            metrics.stage = RequestStage.AI_AGENT_COMPLETE
            
            # Booking responses don't get emotion enhancement
            metrics.emotion_enhanced = False
            metrics.detected_mood = "neutral"
            metrics.sentiment_score = 0.0
            
        else:
            metrics.booking_check_duration = int((time.perf_counter() - booking_start) * 1000)
            
            # STAGE 4: Regular AI agent query with emotion
            metrics.stage = RequestStage.AI_AGENT_START
            ai_start = time.perf_counter()
            
            response_text, emotion_enhanced, detected_mood, sentiment_score = await query_ai_agent_with_emotion(
                metrics.transcript, 
                metrics.user_id,
                client_id,
                connection_manager
            )
            
            metrics.ai_agent_duration = int((time.perf_counter() - ai_start) * 1000)
            metrics.response_text = response_text
            metrics.emotion_enhanced = emotion_enhanced
            metrics.detected_mood = detected_mood
            metrics.sentiment_score = sentiment_score
            metrics.stage = RequestStage.AI_AGENT_COMPLETE
        
        # STAGE 5: TTS Processing
        metrics.stage = RequestStage.TTS_START
        tts_start = time.perf_counter()
        
        try:
            audio_bytes = await synthesize_speech(response_text)
            metrics.tts_duration = int((time.perf_counter() - tts_start) * 1000)
            
            if not audio_bytes or len(audio_bytes) == 0:
                raise ValueError("TTS returned empty audio")
            
            metrics.stage = RequestStage.TTS_COMPLETE
            logger.info(f"TTS generated: {len(audio_bytes)} bytes")
            
        except Exception as tts_error:
            metrics.tts_duration = int((time.perf_counter() - tts_start) * 1000)
            metrics.error = str(tts_error)
            logger.error(f"TTS failed: {tts_error}")
            
            # Send text-only response
            await connection_manager.safe_send_text(
                client_id,
                json.dumps({
                    "type": "text_response",
                    "text": response_text,
                    "error": "TTS failed"
                })
            )
            return False
        
        # STAGE 6: Send Response
        response_start = time.perf_counter()
        
        try:
            # Send metadata first
            await connection_manager.safe_send_text(client_id, json.dumps({
                "type": "response_metadata",
                "transcript": metrics.transcript,
                "response_text": response_text,
                "confidence": metrics.stt_confidence,
                "audio_size": len(audio_bytes),
                "user_id": metrics.user_id,
                "is_booking": metrics.is_booking,
                "emotion_enhanced": metrics.emotion_enhanced,
                "detected_mood": metrics.detected_mood,
                "sentiment_score": metrics.sentiment_score
            }))
            
            # Send audio
            success = await connection_manager.safe_send_bytes(client_id, audio_bytes)
            
            metrics.response_duration = int((time.perf_counter() - response_start) * 1000)
            metrics.total_duration = int((time.perf_counter() - overall_start) * 1000)
            
            # Record emotional turn
            await connection_manager.record_emotion_turn(
                client_id,
                metrics.transcript,
                response_text,
                success,
                metrics.total_duration
            )
            
            if success:
                metrics.stage = RequestStage.RESPONSE_SENT
                
                logger.info(
                    f"Request completed: {metrics.request_id} | "
                    f"Duration: {metrics.total_duration}ms | "
                    f"Emotion enhanced: {metrics.emotion_enhanced} | "
                    f"Mood: {metrics.detected_mood}"
                )
                return True
            else:
                metrics.error = "Failed to send response"
                return False
                
        except Exception as send_error:
            metrics.response_duration = int((time.perf_counter() - response_start) * 1000)
            metrics.error = str(send_error)
            logger.error(f"Send failed: {send_error}")
            return False
    
    except Exception as e:
        metrics.stage = RequestStage.ERROR
        metrics.error = str(e)
        metrics.total_duration = int((time.perf_counter() - overall_start) * 1000)
        logger.error(f"Fatal error in request processing: {e}", exc_info=True)
        
        # Record failed turn
        await connection_manager.record_emotion_turn(
            client_id, metrics.transcript or "", f"Error: {str(e)}", False,
            metrics.total_duration
        )
        return False


# ============================================================================
# ADDITIONAL API ENDPOINTS
# ============================================================================

@app.get("/stats")
async def get_performance_stats():
    """Get comprehensive performance statistics"""
    return {
        "performance": perf_monitor.get_current_stats(),
        "connections": connection_manager.get_connection_stats(),
        "timestamp": time.time()
    }


@app.get("/debug/recent-requests")
async def get_recent_requests(limit: int = 10):
    """Get recent request metrics for debugging"""
    if limit > 100:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 100")
    
    return {
        "recent_requests": perf_monitor.get_recent_performance(limit),
        "summary": perf_monitor.get_current_stats(),
        "timestamp": time.time()
    }


@app.get("/users/{user_id}/validate")
async def validate_user_endpoint(user_id: str):
    """Validate user training status"""
    try:
        user_exists, user_has_data, validation_msg = await validate_user_and_data(user_id)
        
        return {
            "user_id": user_id,
            "exists": user_exists,
            "has_data": user_has_data,
            "message": validation_msg,
            "status": "ready" if (user_exists and user_has_data) else "not_ready",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Validation endpoint failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "user_id": user_id,
                "exists": False,
                "has_data": False,
                "message": str(e),
                "status": "error"
            }
        )


@app.get("/users/{user_id}/stats")
async def get_user_stats_endpoint(user_id: str):
    """Get user training statistics"""
    try:
        stats = await asyncio.get_running_loop().run_in_executor(
            None,
            get_agent_stats,
            user_id
        )
        
        if not stats or stats.get("qa_pairs", 0) == 0:
            return JSONResponse(
                status_code=404,
                content={
                    "user_id": user_id,
                    "error": "User not found or not trained",
                    "message": "Please train via POST /data endpoint first"
                }
            )
        
        return {
            "user_id": user_id,
            "stats": stats,
            "status": "trained",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"User stats endpoint failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "user_id": user_id,
                "error": str(e)
            }
        )


@app.get("/users/{user_id}/emotion-stats")
async def get_user_emotion_stats_endpoint(user_id: str):
    """Get user emotional intelligence statistics"""
    try:
        emotion_stats = await asyncio.get_running_loop().run_in_executor(
            None,
            get_user_emotion_stats,
            user_id
        )
        
        return {
            "user_id": user_id,
            "emotion_stats": emotion_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Emotion stats endpoint failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "user_id": user_id,
                "error": str(e),
                "message": "Failed to retrieve emotion statistics"
            }
        )


@app.get("/users/{user_id}/greeting")
async def get_user_greeting_endpoint(user_id: str):
    """Get adaptive greeting for user based on history"""
    try:
        greeting = await asyncio.get_running_loop().run_in_executor(
            None,
            get_adaptive_greeting,
            user_id
        )
        
        return {
            "user_id": user_id,
            "greeting": greeting,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Greeting endpoint failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "user_id": user_id,
                "error": str(e),
                "greeting": "Hello! How can I help you today?"
            }
        )


@app.post("/admin/warmup")
async def manual_warmup():
    """Manual warmup trigger for all components"""
    try:
        await warmup_components()
        return {
            "success": True, 
            "message": "All components warmed up successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Manual warmup failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False, 
                "error": str(e),
                "message": "Warmup failed"
            }
        )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("Production Voice Assistant with Emotional Intelligence v4.0.0")
    print("=" * 80)
    print("\nCORE FEATURES:")
    print("  ✓ User-specific AI agents with ChromaDB")
    print("  ✓ Real-time WebSocket communication")
    print("  ✓ Single-shot and continuous microphone modes")
    print("  ✓ Deepgram + Whisper STT with fallback")
    print("  ✓ Byte-based TTS synthesis")
    print("  ✓ Booking system integration")
    print("\nEMOTIONAL INTELLIGENCE:")
    print("  ✓ Real-time sentiment analysis with Groq")
    print("  ✓ Mood trajectory tracking across sessions")
    print("  ✓ Session outcome classification")
    print("  ✓ Adaptive greetings based on history")
    print("  ✓ Emotion-enhanced AI responses")
    print("  ✓ Automatic repair strategies")
    print("\nMONITORING:")
    print("  ✓ Comprehensive performance metrics")
    print("  ✓ Request-level tracking")
    print("  ✓ Error analysis by stage and type")
    print("  ✓ Connection statistics")
    print("\nENDPOINTS:")
    print("  • POST   /data                          - Train AI agent")
    print("  • WS     /ws/voice?user_id=USER_ID      - Voice interaction")
    print("  • GET    /stats                          - Performance stats")
    print("  • GET    /users/{user_id}/validate      - Validate user")
    print("  • GET    /users/{user_id}/stats         - Training stats")
    print("  • GET    /users/{user_id}/emotion-stats - Emotion stats")
    print("  • GET    /users/{user_id}/greeting      - Adaptive greeting")
    print("  • GET    /healthz                       - Health check")
    print("=" * 80)
    print("\nStarting server on http://0.0.0.0:5000")
    print("Documentation available at http://localhost:5000/docs")
    print("=" * 80 + "\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        access_log=False,
        workers=1,
        log_level="info"
    )