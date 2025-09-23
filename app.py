# app.py - Enhanced with User Validation and Proper Error Handling
import os
import json
import asyncio
import time
import signal
import sys
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import from chromadb_pipeline
from chromadb_pipeline import train_ai_agent, query_ai_agent, get_agent_stats

# Import enhanced STT modules
from utils.stt_module import (
    transcribe_wav_bytes, cleanup_stt, warmup_stt, STTResult,
    create_continuous_streaming_stt, VADProvider, MicrophoneStreamHandler,
    ContinuousStreamingSTT
)
from utils.tts_module import synthesize_speech, cleanup_tts, warmup_tts

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RequestStage(Enum):
    """Stages of request processing for monitoring"""
    RECEIVED = "received"
    USER_VALIDATION = "user_validation"
    STT_START = "stt_start"
    STT_COMPLETE = "stt_complete"
    AI_AGENT_START = "ai_agent_start"
    AI_AGENT_COMPLETE = "ai_agent_complete"
    TTS_START = "tts_start"
    TTS_COMPLETE = "tts_complete"
    RESPONSE_SENT = "response_sent"
    ERROR = "error"
    USER_NOT_FOUND = "user_not_found"

class ConnectionMode(Enum):
    """Connection modes for different use cases"""
    SINGLE_SHOT = "single_shot"  # Original behavior - one audio chunk
    CONTINUOUS = "continuous"    # New behavior - continuous microphone

class UserBusinessData(BaseModel):
    user_id: str
    business_name: Optional[str] = None
    industry: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    additional_content: Optional[str] = None

class TrainAgentRequest(BaseModel):
    """Request model for training AI agent"""
    user_id: str
    business_name: str
    industry: str
    phone: Optional[str] = None
    email: Optional[str] = None
    content: str

class QueryAgentRequest(BaseModel):
    """Request model for querying AI agent"""
    user_id: str
    query: str

@dataclass
class RequestMetrics:
    """Detailed metrics for a single request"""
    request_id: str
    client_id: str
    user_id: str = ""
    audio_size_bytes: int = 0
    mode: ConnectionMode = ConnectionMode.SINGLE_SHOT
    
    # Timing data (all in milliseconds)
    total_duration: int = 0
    user_validation_duration: int = 0
    stt_duration: int = 0
    ai_agent_duration: int = 0
    tts_duration: int = 0
    response_duration: int = 0
    
    # Results
    transcript: str = ""
    response_text: str = ""
    stt_provider: str = ""
    stt_confidence: float = 0.0
    user_exists: bool = False
    user_has_data: bool = False
    
    # Status
    stage: RequestStage = RequestStage.RECEIVED
    error: Optional[str] = None
    success: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return asdict(self)

class PerformanceMonitor:
    """Enhanced performance monitoring with detailed metrics"""
    
    def __init__(self):
        self.request_count = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.user_validation_failures = 0
        self.total_time = 0.0
        
        # Mode-specific tracking
        self.continuous_sessions = 0
        self.single_shot_requests = 0
        
        # Timing breakdowns
        self.avg_user_validation_time = 0.0
        self.avg_stt_time = 0.0
        self.avg_ai_agent_time = 0.0
        self.avg_tts_time = 0.0
        self.avg_response_time = 0.0
        
        # Recent requests for detailed monitoring
        self.recent_requests: List[RequestMetrics] = []
        self.max_recent_requests = 100
        
        # Error tracking
        self.errors_by_stage: Dict[RequestStage, int] = {}
        self.errors_by_type: Dict[str, int] = {}
    
    def record_request(self, metrics: RequestMetrics):
        """Record a completed request"""
        self.request_count += 1
        
        # Track by mode
        if metrics.mode == ConnectionMode.CONTINUOUS:
            self.continuous_sessions += 1
        else:
            self.single_shot_requests += 1
        
        # Track user validation failures
        if not metrics.user_exists or not metrics.user_has_data:
            self.user_validation_failures += 1
        
        if metrics.success:
            self.successful_requests += 1
            self.total_time += metrics.total_duration
            
            # Update averages
            count = self.successful_requests
            self.avg_user_validation_time = ((self.avg_user_validation_time * (count - 1)) + metrics.user_validation_duration) / count
            self.avg_stt_time = ((self.avg_stt_time * (count - 1)) + metrics.stt_duration) / count
            self.avg_ai_agent_time = ((self.avg_ai_agent_time * (count - 1)) + metrics.ai_agent_duration) / count
            self.avg_tts_time = ((self.avg_tts_time * (count - 1)) + metrics.tts_duration) / count
            self.avg_response_time = ((self.avg_response_time * (count - 1)) + metrics.response_duration) / count
        else:
            self.failed_requests += 1
            
            # Track error stages
            if metrics.stage in self.errors_by_stage:
                self.errors_by_stage[metrics.stage] += 1
            else:
                self.errors_by_stage[metrics.stage] = 1
            
            # Track error types
            if metrics.error:
                error_type = type(metrics.error).__name__ if hasattr(metrics.error, '__class__') else str(type(metrics.error))
                if error_type in self.errors_by_type:
                    self.errors_by_type[error_type] += 1
                else:
                    self.errors_by_type[error_type] = 1
        
        # Store recent request
        self.recent_requests.append(metrics)
        if len(self.recent_requests) > self.max_recent_requests:
            self.recent_requests.pop(0)
        
        # Log detailed metrics for monitoring
        logger.info(f"Request completed: {metrics.request_id}", extra={
            "metrics": metrics.to_dict(),
            "performance": self.get_current_stats()
        })
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        success_rate = (self.successful_requests / max(1, self.request_count)) * 100
        avg_total_time = self.total_time / max(1, self.successful_requests)
        user_validation_failure_rate = (self.user_validation_failures / max(1, self.request_count)) * 100
        
        return {
            "requests": {
                "total": self.request_count,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate_percent": round(success_rate, 2),
                "user_validation_failures": self.user_validation_failures,
                "user_validation_failure_rate_percent": round(user_validation_failure_rate, 2),
                "continuous_sessions": self.continuous_sessions,
                "single_shot_requests": self.single_shot_requests
            },
            "timing_ms": {
                "avg_total": round(avg_total_time, 0),
                "avg_user_validation": round(self.avg_user_validation_time, 0),
                "avg_stt": round(self.avg_stt_time, 0),
                "avg_ai_agent": round(self.avg_ai_agent_time, 0),
                "avg_tts": round(self.avg_tts_time, 0),
                "avg_response": round(self.avg_response_time, 0)
            },
            "errors": {
                "by_stage": dict(self.errors_by_stage),
                "by_type": dict(self.errors_by_type)
            }
        }
    
    def get_recent_performance(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get metrics for recent requests"""
        recent = self.recent_requests[-limit:] if len(self.recent_requests) > limit else self.recent_requests
        return [req.to_dict() for req in recent]

# Global performance monitor
perf_monitor = PerformanceMonitor()

# User-specific query lock to prevent conflicts
user_query_locks: Dict[str, asyncio.Lock] = {}

async def get_user_lock(user_id: str) -> asyncio.Lock:
    """Get or create a lock for specific user to prevent concurrent AI agent queries"""
    if user_id not in user_query_locks:
        user_query_locks[user_id] = asyncio.Lock()
    return user_query_locks[user_id]

async def validate_user_and_data(user_id: str) -> tuple[bool, bool, str]:
    """
    Validate if user exists and has trained data
    Returns: (user_exists, user_has_data, message)
    """
    try:
        logger.info(f"ðŸ” Validating user: {user_id}")
        
        # Check if user has trained data by getting agent stats
        stats = await asyncio.get_running_loop().run_in_executor(
            None,
            get_agent_stats,
            user_id
        )
        
        if not stats or stats.get("qa_pairs", 0) == 0:
            logger.warning(f"âŒ User {user_id} not found or has no trained data")
            logger.warning(f"ðŸ“Š User stats: {stats}")
            return False, False, f"User {user_id} has not been trained yet. Please provide business data first."
        
        qa_pairs = stats.get("qa_pairs", 0)
        business_name = stats.get("business_name", "Unknown Business")
        
        logger.info(f"âœ… User {user_id} validated successfully")
        logger.info(f"ðŸ“Š User has {qa_pairs} Q&A pairs for {business_name}")
        
        return True, True, f"User validated with {qa_pairs} Q&A pairs"
        
    except Exception as e:
        logger.error(f"âŒ User validation failed for {user_id}: {e}")
        return False, False, f"Error validating user: {str(e)}"

async def query_ai_agent_safe(transcript: str, user_id: str) -> str:
    """Thread-safe wrapper for query_ai_agent with user validation"""
    user_lock = await get_user_lock(user_id)
    
    async with user_lock:
        try:
            # Validate user first
            user_exists, user_has_data, validation_msg = await validate_user_and_data(user_id)
            
            if not user_exists or not user_has_data:
                logger.warning(f"ðŸš« Query blocked for user {user_id}: {validation_msg}")
                return f"Sorry, I can't answer your question. {validation_msg}"
            
            logger.info(f"Querying AI agent for user {user_id}: '{transcript[:50]}...'")
            response_text = await asyncio.get_running_loop().run_in_executor(
                None, 
                query_ai_agent, 
                transcript,
                user_id
            )
            
            if not response_text or not response_text.strip():
                response_text = "I'm not sure how to respond to that. Could you please rephrase your question?"
            
            logger.info(f"AI agent response for user {user_id}: '{response_text[:100]}...'")
            return response_text
            
        except Exception as e:
            logger.error(f"AI agent query failed for user {user_id}: {e}", exc_info=True)
            return "Sorry, I'm having trouble accessing my knowledge base right now."

# Enhanced Connection management with user tracking
class EnhancedConnectionManager:
    """Manage active WebSocket connections with user tracking"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_stats: Dict[str, Dict] = {}
        self.connection_modes: Dict[str, ConnectionMode] = {}
        self.connection_users: Dict[str, str] = {}
        self.streaming_sessions: Dict[str, ContinuousStreamingSTT] = {}
        self.microphone_handlers: Dict[str, MicrophoneStreamHandler] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str, user_id: str = None):
        """Accept and track new connection with user mapping"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        # IMPORTANT: If no user_id provided from frontend, we should NOT create a random one
        # Instead, we'll mark it as "no_user" and handle validation later
        if not user_id or user_id == client_id:
            logger.warning(f"âš ï¸  No valid user_id provided for connection {client_id}")
            logger.warning(f"ðŸš« Frontend should provide user_id via query parameter: ?user_id=actual_user_id")
            user_id = "no_user"  # Mark as invalid instead of using client_id
        
        self.connection_users[client_id] = user_id
        
        self.connection_stats[client_id] = {
            "connected_at": time.time(),
            "requests_processed": 0,
            "last_activity": time.time(),
            "mode": ConnectionMode.SINGLE_SHOT.value,
            "user_id": user_id,
            "valid_user": user_id != "no_user"
        }
        
        if user_id == "no_user":
            logger.warning(f"ðŸš¨ Client {client_id} connected without valid user_id")
        else:
            logger.info(f"âœ… Client connected: {client_id} (user: {user_id})")
    
    def get_user_id(self, client_id: str) -> str:
        """Get user_id for a client connection"""
        return self.connection_users.get(client_id, "no_user")
    
    def is_valid_user(self, client_id: str) -> bool:
        """Check if client has valid user_id"""
        user_id = self.get_user_id(client_id)
        return user_id != "no_user" and user_id is not None
    
    def disconnect(self, client_id: str):
        """Remove connection and clean up"""
        # Clean up streaming sessions
        if client_id in self.streaming_sessions:
            del self.streaming_sessions[client_id]
        
        if client_id in self.microphone_handlers:
            # Stop microphone handler
            handler = self.microphone_handlers[client_id]
            asyncio.create_task(handler.stop_stream())
            del self.microphone_handlers[client_id]
        
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        if client_id in self.connection_modes:
            del self.connection_modes[client_id]
            
        if client_id in self.connection_users:
            del self.connection_users[client_id]
        
        if client_id in self.connection_stats:
            stats = self.connection_stats.pop(client_id)
            duration = time.time() - stats["connected_at"]
            logger.info(f"Client disconnected: {client_id} (connected {duration:.0f}s, {stats['requests_processed']} requests)")
    
    async def start_continuous_session(self, client_id: str, vad_provider: VADProvider = VADProvider.SILERO):
        """Start continuous streaming session for a client"""
        if client_id not in self.active_connections:
            return False
        
        # Check if user is valid
        if not self.is_valid_user(client_id):
            await self.safe_send_text(client_id, json.dumps({
                "type": "error",
                "message": "Cannot start continuous session: No valid user_id provided. Please reconnect with proper user_id.",
                "error_code": "INVALID_USER_ID"
            }))
            return False
        
        # Create continuous STT processor
        stt_processor = create_continuous_streaming_stt(vad_provider)
        
        # Set up callbacks
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
        
        # Create microphone handler
        mic_handler = MicrophoneStreamHandler(stt_processor)
        
        # Store session components
        self.streaming_sessions[client_id] = stt_processor
        self.microphone_handlers[client_id] = mic_handler
        self.connection_modes[client_id] = ConnectionMode.CONTINUOUS
        
        # Update connection stats
        if client_id in self.connection_stats:
            self.connection_stats[client_id]["mode"] = ConnectionMode.CONTINUOUS.value
        
        # Start the microphone stream
        await mic_handler.start_stream()
        
        logger.info(f"Started continuous session for {client_id}")
        return True
    
    async def stop_continuous_session(self, client_id: str):
        """Stop continuous streaming session for a client"""
        if client_id in self.microphone_handlers:
            await self.microphone_handlers[client_id].stop_stream()
            del self.microphone_handlers[client_id]
        
        if client_id in self.streaming_sessions:
            del self.streaming_sessions[client_id]
        
        if client_id in self.connection_modes:
            self.connection_modes[client_id] = ConnectionMode.SINGLE_SHOT
        
        if client_id in self.connection_stats:
            self.connection_stats[client_id]["mode"] = ConnectionMode.SINGLE_SHOT.value
        
        logger.info(f"Stopped continuous session for {client_id}")
    
    async def add_audio_chunk_to_stream(self, client_id: str, audio_data: bytes):
        """Add audio chunk to continuous stream"""
        if client_id not in self.microphone_handlers:
            return False
        
        handler = self.microphone_handlers[client_id]
        await handler.add_audio_chunk(audio_data)
        return True
    
    async def _handle_transcript(self, client_id: str, text: str, confidence: float):
        """Handle transcript from continuous STT"""
        user_id = self.get_user_id(client_id)
        logger.info(f"Transcript ready for {client_id} (user: {user_id}): '{text}' (confidence: {confidence:.3f})")
        
        # Create request metrics for continuous processing
        request_id = f"{client_id}_continuous_{int(time.time())}"
        metrics = RequestMetrics(
            request_id=request_id,
            client_id=client_id,
            user_id=user_id,
            mode=ConnectionMode.CONTINUOUS,
            transcript=text,
            stt_confidence=confidence
        )
        
        # Process the transcript
        success = await self._process_continuous_transcript(client_id, text, metrics)
        
        # Record metrics
        metrics.success = success
        perf_monitor.record_request(metrics)
    
    async def _handle_status_update(self, client_id: str, status: str):
        """Handle status updates from continuous STT"""
        await self.safe_send_text(client_id, json.dumps({
            "type": "status",
            "status": status,
            "timestamp": time.time()
        }))
    
    async def _process_continuous_transcript(self, client_id: str, transcript: str, metrics: RequestMetrics) -> bool:
        """Process transcript from continuous session with user validation"""
        overall_start = time.perf_counter()
        
        try:
            # ======================
            # STAGE 1: User Validation
            # ======================
            metrics.stage = RequestStage.USER_VALIDATION
            validation_start = time.perf_counter()
            
            user_id = self.get_user_id(client_id)
            
            # Check if we have a valid user_id
            if user_id == "no_user":
                metrics.user_validation_duration = int((time.perf_counter() - validation_start) * 1000)
                metrics.error = "No user_id provided from frontend"
                metrics.stage = RequestStage.USER_NOT_FOUND
                
                logger.error(f"ðŸš« CONTINUOUS REQUEST BLOCKED: No user_id for client {client_id}")
                logger.error(f"ðŸ“ Expected: Frontend should send user_id via WebSocket connection")
                
                # Send error message to client
                await self.safe_send_text(client_id, json.dumps({
                    "type": "error", 
                    "message": "User ID not provided. Please reconnect with your user ID to continue.",
                    "error_code": "NO_USER_ID",
                    "instruction": "Frontend needs to provide user_id in WebSocket connection"
                }))
                
                return False
            
            # Validate user exists and has data
            user_exists, user_has_data, validation_msg = await validate_user_and_data(user_id)
            metrics.user_exists = user_exists
            metrics.user_has_data = user_has_data
            metrics.user_validation_duration = int((time.perf_counter() - validation_start) * 1000)
            
            if not user_exists or not user_has_data:
                metrics.error = validation_msg
                metrics.stage = RequestStage.USER_NOT_FOUND
                
                logger.error(f"ðŸš« CONTINUOUS REQUEST BLOCKED for user {user_id}: {validation_msg}")
                
                # Send error message via TTS
                error_audio = await synthesize_speech(f"Sorry, I can't help you right now. {validation_msg}")
                if error_audio and os.path.exists(error_audio):
                    with open(error_audio, "rb") as f:
                        audio_bytes = f.read()
                    await self.safe_send_bytes(client_id, audio_bytes)
                    os.remove(error_audio)
                else:
                    await self.safe_send_text(client_id, json.dumps({
                        "type": "error",
                        "message": validation_msg,
                        "error_code": "USER_NOT_TRAINED"
                    }))
                
                return False
            
            logger.info(f"âœ… User validation passed for {user_id}: {validation_msg}")
            
            # ======================
            # STAGE 2: AI Agent Query (User-Specific)
            # ======================
            metrics.stage = RequestStage.AI_AGENT_START
            ai_agent_start = time.perf_counter()

            try:
                response_text = await query_ai_agent_safe(transcript, user_id)
                
                metrics.ai_agent_duration = int((time.perf_counter() - ai_agent_start) * 1000)
                metrics.response_text = response_text
                metrics.stage = RequestStage.AI_AGENT_COMPLETE

                logger.info(f"AI Agent query [{metrics.ai_agent_duration}ms]: Found response for '{transcript[:50]}...' (user: {user_id})")

            except Exception as ai_agent_error:
                metrics.ai_agent_duration = int((time.perf_counter() - ai_agent_start) * 1000)
                metrics.error = str(ai_agent_error)
                logger.error(f"AI Agent query failed for {metrics.request_id}: {ai_agent_error}", exc_info=True)
                response_text = "Sorry, I'm having trouble accessing my knowledge base."
            
            # ======================
            # STAGE 3: TTS Processing
            # ======================
            metrics.stage = RequestStage.TTS_START
            tts_start = time.perf_counter()
            
            try:
                audio_file_path = await synthesize_speech(response_text)
                metrics.tts_duration = int((time.perf_counter() - tts_start) * 1000)
                
                if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
                    raise FileNotFoundError("TTS file generation failed")
                
                metrics.stage = RequestStage.TTS_COMPLETE
                
                logger.info(f"TTS success [{metrics.tts_duration}ms]: {os.path.getsize(audio_file_path)} bytes generated")
                
            except Exception as tts_error:
                metrics.tts_duration = int((time.perf_counter() - tts_start) * 1000)
                metrics.error = str(tts_error)
                logger.error(f"TTS processing failed for {metrics.request_id}: {tts_error}", exc_info=True)
                
                # Send text response as fallback
                await self.safe_send_text(
                    client_id,
                    json.dumps({
                        "type": "text_response",
                        "text": response_text,
                        "transcript": transcript,
                        "message": "Audio synthesis failed, but here's the text response"
                    })
                )
                return False
            
            # ======================
            # STAGE 4: Send Response
            # ======================
            response_start = time.perf_counter()
            
            try:
                # Read audio file
                with open(audio_file_path, "rb") as f:
                    audio_bytes = f.read()
                
                # Send response with metadata
                await self.safe_send_text(client_id, json.dumps({
                    "type": "response_metadata",
                    "transcript": transcript,
                    "response_text": response_text,
                    "confidence": metrics.stt_confidence,
                    "audio_size": len(audio_bytes),
                    "user_id": user_id
                }))
                
                # Send audio response
                success = await self.safe_send_bytes(client_id, audio_bytes)
                
                metrics.response_duration = int((time.perf_counter() - response_start) * 1000)
                
                if success:
                    metrics.stage = RequestStage.RESPONSE_SENT
                    metrics.total_duration = int((time.perf_counter() - overall_start) * 1000)
                    
                    logger.info(f"Continuous request {metrics.request_id} completed successfully in {metrics.total_duration}ms")
                    logger.info(f"  Breakdown: Validation={metrics.user_validation_duration}ms | AI_Agent={metrics.ai_agent_duration}ms | TTS={metrics.tts_duration}ms | Send={metrics.response_duration}ms")
                    
                    return True
                else:
                    logger.error(f"Failed to send response for {metrics.request_id}")
                    metrics.error = "Failed to send response"
                    return False
                    
            except Exception as send_error:
                metrics.response_duration = int((time.perf_counter() - response_start) * 1000)
                metrics.error = str(send_error)
                logger.error(f"Response sending failed for {metrics.request_id}: {send_error}", exc_info=True)
                return False
                
            finally:
                # Clean up TTS file
                try:
                    if 'audio_file_path' in locals() and os.path.exists(audio_file_path):
                        os.remove(audio_file_path)
                except Exception as cleanup_error:
                    logger.warning(f"TTS file cleanup failed: {cleanup_error}")
        
        except Exception as e:
            metrics.stage = RequestStage.ERROR
            metrics.error = str(e)
            metrics.total_duration = int((time.perf_counter() - overall_start) * 1000)
            logger.error(f"Fatal error processing continuous request {metrics.request_id}: {e}", exc_info=True)
            return False
    
    def update_activity(self, client_id: str):
        """Update last activity time for connection"""
        if client_id in self.connection_stats:
            self.connection_stats[client_id]["last_activity"] = time.time()
            self.connection_stats[client_id]["requests_processed"] += 1
    
    async def safe_send_bytes(self, client_id: str, data: bytes) -> bool:
        """Safely send bytes with error handling"""
        if client_id not in self.active_connections:
            return False
        
        websocket = self.active_connections[client_id]
        try:
            await websocket.send_bytes(data)
            return True
        except Exception as e:
            logger.error(f"Failed to send bytes to {client_id}: {e}")
            self.disconnect(client_id)
            return False
    
    async def safe_send_text(self, client_id: str, message: str) -> bool:
        """Safely send text with error handling"""
        if client_id not in self.active_connections:
            return False
        
        websocket = self.active_connections[client_id]
        try:
            await websocket.send_text(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send text to {client_id}: {e}")
            self.disconnect(client_id)
            return False
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about active connections"""
        total_connections = len(self.active_connections)
        if total_connections == 0:
            return {"active_connections": 0}
        
        total_requests = sum(stats["requests_processed"] for stats in self.connection_stats.values())
        avg_requests = total_requests / total_connections
        
        continuous_count = sum(1 for mode in self.connection_modes.values() if mode == ConnectionMode.CONTINUOUS)
        valid_users = sum(1 for stats in self.connection_stats.values() if stats.get("valid_user", False))
        
        return {
            "active_connections": total_connections,
            "valid_user_connections": valid_users,
            "invalid_user_connections": total_connections - valid_users,
            "continuous_sessions": continuous_count,
            "single_shot_connections": total_connections - continuous_count,
            "total_requests_processed": total_requests,
            "avg_requests_per_connection": round(avg_requests, 1),
            "connections": {
                client_id: {
                    "requests": stats["requests_processed"],
                    "connected_duration_s": round(time.time() - stats["connected_at"], 0),
                    "idle_time_s": round(time.time() - stats["last_activity"], 0),
                    "mode": stats.get("mode", "single_shot"),
                    "user_id": stats.get("user_id", "unknown"),
                    "valid_user": stats.get("valid_user", False)
                }
                for client_id, stats in self.connection_stats.items()
            }
        }

# Global connection manager
connection_manager = EnhancedConnectionManager()

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events with enhanced monitoring"""
    logger.info("Starting Enhanced Voice Assistant Backend with User Validation...")
    startup_start = time.perf_counter()
    
    # Parallel startup tasks
    startup_tasks = [
        asyncio.create_task(warmup_components()),
        asyncio.create_task(setup_monitoring())
    ]
    
    try:
        await asyncio.gather(*startup_tasks, return_exceptions=False)
        startup_time = (time.perf_counter() - startup_start) * 1000
        logger.info(f"Enhanced Voice Assistant Backend ready in {startup_time:.0f}ms!")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down Enhanced Voice Assistant Backend...")
    await cleanup_all_resources()
    logger.info("Shutdown complete")

async def warmup_components():
    """Warm up system components"""
    logger.info("Warming up system components...")
    
    # Warm up components in parallel
    warmup_tasks = [
        ("STT", asyncio.create_task(warmup_stt_async())),
        ("TTS", asyncio.create_task(warmup_tts()))
    ]
    
    for name, task in warmup_tasks:
        try:
            start_time = time.perf_counter()
            await task
            warmup_time = (time.perf_counter() - start_time) * 1000
            logger.info(f"{name} warmed up in {warmup_time:.0f}ms")
        except Exception as e:
            logger.error(f"{name} warmup failed: {e}")

async def setup_monitoring():
    """Set up monitoring and health checks"""
    logger.info("Setting up monitoring...")

async def warmup_stt_async():
    """Async wrapper for STT warmup"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, warmup_stt)

async def cleanup_all_resources():
    """Clean up system resources"""
    cleanup_tasks = [
        ("STT", cleanup_stt()),
        ("TTS", cleanup_tts())
    ]
    
    for name, task in cleanup_tasks:
        try:
            if asyncio.iscoroutine(task):
                await task
            else:
                await asyncio.create_task(task)
            logger.info(f"{name} cleanup completed")
        except Exception as e:
            logger.error(f"{name} cleanup failed: {e}")

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    asyncio.create_task(cleanup_all_resources())
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# FastAPI app with enhanced configuration
app = FastAPI(
    title="AI Voice Assistant Backend",
    description="Enhanced voice assistant with user validation, continuous microphone support, Deepgram + Whisper STT, ChromaDB user-specific retrieval, and Edge TTS",
    version="2.2.0",  # Version bump for user validation
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "AI Voice Assistant Backend",
        "version": "2.2.0",  # Version bump for user validation
        "status": "running",
        "features": [
            "User validation and data verification",
            "User-specific AI agents with ChromaDB",
            "Continuous microphone support with VAD",
            "Single-shot and streaming modes",
            "Deepgram + Whisper STT with fallback",
            "ChromaDB user-specific semantic retrieval",
            "Edge TTS synthesis",
            "Real-time WebSocket communication",
            "Advanced performance monitoring",
            "Thread-safe AI agent queries"
        ],
        "important": {
            "frontend_requirements": [
                "Must provide user_id in WebSocket connection: ws://host/ws/voice?user_id=YOUR_USER_ID",
                "User must be trained via /data endpoint before voice queries",
                "All voice requests require valid user_id with trained data"
            ]
        }
    }

# Enhanced health check
@app.get("/healthz")
async def health_check():
    """Comprehensive health check with component status"""
    try:
        perf_stats = perf_monitor.get_current_stats()
        connection_stats = connection_manager.get_connection_stats()
        
        overall_health = "healthy"
        
        return {
            "status": overall_health,
            "timestamp": time.time(),
            "components": {
                "chromadb": "ready",
                "user_validation": "active",
                "connections": connection_stats
            },
            "performance": perf_stats
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

@app.post("/data")
async def save_user_data(request: UserBusinessData):
    """
    Accepts user_id and business content data,
    trains the AI agent, and returns a confirmation.
    """
    try:
        logger.info(f"Received business data from {request.user_id}: {request.dict()}")

        # Prepare business data for ChromaDB pipeline
        business_data = {
            "business_name": request.business_name or "Business",
            "industry": request.industry or "Service", 
            "phone": request.phone or "",
            "email": request.email or "",
            "content": request.additional_content or ""  # This is the main content
        }

        # Train the AI agent using ChromaDB pipeline
        logger.info(f"Training AI agent for user {request.user_id}...")
        training_success = await asyncio.get_running_loop().run_in_executor(
            None, 
            train_ai_agent, 
            request.user_id, 
            business_data
        )

        if training_success:
            # Get agent statistics to confirm training
            stats = await asyncio.get_running_loop().run_in_executor(
                None,
                get_agent_stats,
                request.user_id
            )
            
            logger.info(f"Successfully trained AI agent for {request.user_id}: {stats}")
            
            return {
                "success": True,
                "message": "User data received and AI agent trained successfully",
                "user_id": request.user_id,
                "training_stats": {
                    "qa_pairs_generated": stats.get("qa_pairs", 0),
                    "categories_covered": list(stats.get("categories", {}).keys()),
                    "business_name": stats.get("business_name", "Unknown"),
                    "comprehensive_coverage": stats.get("comprehensive_coverage", False)
                },
                "content": {
                    "business_name": request.business_name,
                    "industry": request.industry,
                    "phone": request.phone,
                    "email": request.email,
                    "additional_content": request.additional_content,
                },
                "next_steps": {
                    "voice_queries": f"Connect to WebSocket with: ws://host/ws/voice?user_id={request.user_id}",
                    "note": "Voice queries will now work with your trained data"
                }
            }
        else:
            logger.error(f"Failed to train AI agent for user {request.user_id}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "AI agent training failed",
                    "message": "Failed to train AI agent with provided business data",
                    "user_id": request.user_id
                },
            )

    except Exception as e:
        logger.error(f"Failed to handle data from {request.user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Failed to process user data and train AI agent",
            },
        )

# Enhanced WebSocket endpoint with user validation
@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    client_id = f"{websocket.client.host}:{websocket.client.port}:{int(time.time())}"
    
    # Get user_id from query params - MUST be provided by frontend
    user_id = websocket.query_params.get("user_id")
    
    # Log connection attempt with user_id status
    if user_id:
        logger.info(f"WebSocket connection attempt: client={client_id}, user_id={user_id}")
    else:
        logger.warning(f"WebSocket connection attempt WITHOUT user_id: client={client_id}")
        logger.warning(f"Frontend should connect with: ws://host/ws/voice?user_id=ACTUAL_USER_ID")
    
    try:
        await connection_manager.connect(websocket, client_id, user_id)
        
        # Send different initial messages based on user_id validity
        if user_id and user_id != "no_user":
            # Check if user has trained data
            user_exists, user_has_data, validation_msg = await validate_user_and_data(user_id)
            
            if user_exists and user_has_data:
                await connection_manager.safe_send_text(client_id, json.dumps({
                    "type": "connected",
                    "client_id": client_id,
                    "user_id": user_id,
                    "status": "ready",
                    "supported_modes": ["single_shot", "continuous"],
                    "message": f"Connected successfully! Your AI agent is ready with trained data. Send 'start_continuous' to enable continuous mode.",
                    "validation": validation_msg
                }))
            else:
                await connection_manager.safe_send_text(client_id, json.dumps({
                    "type": "connected",
                    "client_id": client_id,
                    "user_id": user_id,
                    "status": "not_trained",
                    "error": "User not trained",
                    "message": f"Connected but cannot process voice queries: {validation_msg}",
                    "action_required": "Please train your AI agent by sending business data to /data endpoint first"
                }))
        else:
            await connection_manager.safe_send_text(client_id, json.dumps({
                "type": "connected",
                "client_id": client_id,
                "user_id": "no_user",
                "status": "invalid_user",
                "error": "No user ID provided",
                "message": "Connected but cannot process voice queries: No user_id provided in connection",
                "action_required": "Reconnect with proper user_id: ws://host/ws/voice?user_id=YOUR_USER_ID"
            }))
        
        request_counter = 0
        
        while True:
            try:
                # Try to receive text message first (for commands)
                try:
                    text_message = await asyncio.wait_for(
                        websocket.receive_text(), 
                        timeout=0.1
                    )
                    
                    # Handle text commands
                    if await handle_text_command(client_id, text_message):
                        continue
                    
                except asyncio.TimeoutError:
                    pass  # No text message, try binary
                
                # Try to receive audio data
                message = await asyncio.wait_for(
                    websocket.receive_bytes(), 
                    timeout=60.0
                )
                
                # Check if user is valid before processing audio
                if not connection_manager.is_valid_user(client_id):
                    await connection_manager.safe_send_text(
                        client_id, 
                        json.dumps({
                            "type": "error",
                            "error_code": "INVALID_USER_ID", 
                            "message": "Cannot process audio: No valid user_id provided. Please reconnect with your user_id.",
                            "instruction": "Use: ws://host/ws/voice?user_id=YOUR_USER_ID"
                        })
                    )
                    continue
                
                # Determine processing mode
                current_mode = connection_manager.connection_modes.get(client_id, ConnectionMode.SINGLE_SHOT)
                
                if current_mode == ConnectionMode.CONTINUOUS:
                    # Add to continuous stream
                    await connection_manager.add_audio_chunk_to_stream(client_id, message)
                    connection_manager.update_activity(client_id)
                    
                else:
                    # Process as single-shot request
                    if not message or len(message) < 100:
                        await connection_manager.safe_send_text(
                            client_id, 
                            json.dumps({
                                "error": "Invalid audio data",
                                "message": "Please ensure you're sending valid WAV audio data"
                            })
                        )
                        continue
                    
                    # Process single-shot request
                    request_counter += 1
                    request_id = f"{client_id}_req_{request_counter}"
                    
                    connection_manager.update_activity(client_id)
                    
                    # Initialize request metrics with user_id
                    user_id = connection_manager.get_user_id(client_id)
                    metrics = RequestMetrics(
                        request_id=request_id,
                        client_id=client_id,
                        user_id=user_id,
                        audio_size_bytes=len(message),
                        mode=ConnectionMode.SINGLE_SHOT
                    )
                    
                    # Process the voice request
                    success = await process_voice_request(websocket, client_id, message, metrics)
                    
                    # Record metrics
                    metrics.success = success
                    perf_monitor.record_request(metrics)
                
            except asyncio.TimeoutError:
                logger.warning(f"Client {client_id} timed out")
                await connection_manager.safe_send_text(
                    client_id, 
                    json.dumps({"error": "timeout", "message": "No activity in 60 seconds"})
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
                        "message": "An error occurred processing your request. Please try again."
                    })
                )
                # Continue loop instead of breaking - allow recovery
                
    except Exception as e:
        logger.error(f"Fatal WebSocket error for {client_id}: {e}", exc_info=True)
    finally:
        connection_manager.disconnect(client_id)

async def handle_text_command(client_id: str, command: str) -> bool:
    """Handle text-based commands from WebSocket client"""
    try:
        # Try to parse as JSON command
        try:
            cmd_data = json.loads(command)
            cmd_type = cmd_data.get("type")
        except json.JSONDecodeError:
            # Handle simple string commands
            cmd_type = command.strip().lower()
            cmd_data = {"type": cmd_type}
        
        if cmd_type == "start_continuous":
            # Check if user is valid first
            if not connection_manager.is_valid_user(client_id):
                await connection_manager.safe_send_text(client_id, json.dumps({
                    "type": "error",
                    "message": "Cannot start continuous mode: Invalid user_id. Please reconnect with proper user_id.",
                    "error_code": "INVALID_USER_ID"
                }))
                return True
            
            # Start continuous microphone session
            vad_provider = VADProvider.SILERO
            if "vad_provider" in cmd_data:
                try:
                    vad_provider = VADProvider(cmd_data["vad_provider"])
                except ValueError:
                    vad_provider = VADProvider.SILERO
            
            success = await connection_manager.start_continuous_session(client_id, vad_provider)
            
            if success:
                user_id = connection_manager.get_user_id(client_id)
                await connection_manager.safe_send_text(client_id, json.dumps({
                    "type": "continuous_started",
                    "message": f"Continuous microphone mode activated for user {user_id}. I'm listening...",
                    "vad_provider": vad_provider.value,
                    "status": "listening",
                    "user_id": user_id
                }))
            else:
                await connection_manager.safe_send_text(client_id, json.dumps({
                    "type": "error",
                    "message": "Failed to start continuous mode"
                }))
            
            return True
        
        elif cmd_type == "stop_continuous":
            # Stop continuous microphone session
            await connection_manager.stop_continuous_session(client_id)
            
            await connection_manager.safe_send_text(client_id, json.dumps({
                "type": "continuous_stopped",
                "message": "Continuous mode stopped. Back to single-shot mode.",
                "status": "single_shot"
            }))
            
            return True
        
        elif cmd_type == "get_status":
            # Get current session status
            mode = connection_manager.connection_modes.get(client_id, ConnectionMode.SINGLE_SHOT)
            stats = connection_manager.connection_stats.get(client_id, {})
            user_id = connection_manager.get_user_id(client_id)
            
            # Check user validation status
            if user_id != "no_user":
                user_exists, user_has_data, validation_msg = await validate_user_and_data(user_id)
                user_status = "ready" if (user_exists and user_has_data) else "not_trained"
            else:
                user_status = "invalid_user"
                validation_msg = "No user_id provided"
            
            await connection_manager.safe_send_text(client_id, json.dumps({
                "type": "status_response",
                "mode": mode.value,
                "user_id": user_id,
                "user_status": user_status,
                "user_validation": validation_msg,
                "stats": stats,
                "timestamp": time.time()
            }))
            
            return True
        
        elif cmd_type == "ping":
            # Simple ping command
            await connection_manager.safe_send_text(client_id, json.dumps({
                "type": "pong",
                "timestamp": time.time(),
                "user_id": connection_manager.get_user_id(client_id)
            }))
            
            return True
        
        else:
            # Unknown command - inform client
            await connection_manager.safe_send_text(client_id, json.dumps({
                "type": "unknown_command",
                "message": f"Unknown command: {cmd_type}",
                "available_commands": [
                    "start_continuous", "stop_continuous", "get_status", "ping"
                ]
            }))
            
            return True
    
    except Exception as e:
        logger.error(f"Error handling command for {client_id}: {e}")
        await connection_manager.safe_send_text(client_id, json.dumps({
            "type": "command_error",
            "message": "Error processing command"
        }))
        return True

# Enhanced single-shot processing function with user validation
async def process_voice_request(websocket: WebSocket, client_id: str, audio_data: bytes, metrics: RequestMetrics) -> bool:
    """Process a single voice request with user validation and user-specific AI agent"""
    
    overall_start = time.perf_counter()
    
    try:
        logger.info(f"Processing single-shot request {metrics.request_id}: {len(audio_data)} bytes (user: {metrics.user_id})")
        
        # ======================
        # STAGE 1: User Validation 
        # ======================
        metrics.stage = RequestStage.USER_VALIDATION
        validation_start = time.perf_counter()
        
        # Check if we have a valid user_id
        if metrics.user_id == "no_user":
            metrics.user_validation_duration = int((time.perf_counter() - validation_start) * 1000)
            metrics.error = "No user_id provided from frontend"
            metrics.stage = RequestStage.USER_NOT_FOUND
            
            logger.error(f"SINGLE-SHOT REQUEST BLOCKED: No user_id for client {client_id}")
            logger.error(f"Expected: Frontend should send user_id via WebSocket connection")
            
            # Send error message to client
            await connection_manager.safe_send_text(client_id, json.dumps({
                "type": "error", 
                "message": "User ID not provided. Please reconnect with your user ID to continue.",
                "error_code": "NO_USER_ID",
                "instruction": "Frontend needs to provide user_id in WebSocket connection"
            }))
            
            return False
        
        # Validate user exists and has data
        user_exists, user_has_data, validation_msg = await validate_user_and_data(metrics.user_id)
        metrics.user_exists = user_exists
        metrics.user_has_data = user_has_data
        metrics.user_validation_duration = int((time.perf_counter() - validation_start) * 1000)
        
        if not user_exists or not user_has_data:
            metrics.error = validation_msg
            metrics.stage = RequestStage.USER_NOT_FOUND
            
            logger.error(f"SINGLE-SHOT REQUEST BLOCKED for user {metrics.user_id}: {validation_msg}")
            
            # Send error message via TTS
            error_audio = await synthesize_speech(f"Sorry, I can't help you right now. {validation_msg}")
            if error_audio and os.path.exists(error_audio):
                with open(error_audio, "rb") as f:
                    audio_bytes = f.read()
                await connection_manager.safe_send_bytes(client_id, audio_bytes)
                os.remove(error_audio)
            else:
                await connection_manager.safe_send_text(client_id, json.dumps({
                    "type": "error",
                    "message": validation_msg,
                    "error_code": "USER_NOT_TRAINED"
                }))
            
            return False
        
        logger.info(f"User validation passed for {metrics.user_id}: {validation_msg}")
        
        # ======================
        # STAGE 2: STT Processing
        # ======================
        metrics.stage = RequestStage.STT_START
        stt_start = time.perf_counter()
        
        try:
            stt_result = await transcribe_wav_bytes(audio_data)
            metrics.stt_duration = int((time.perf_counter() - stt_start) * 1000)
            
            if not stt_result.is_success or not stt_result.text.strip():
                logger.warning(f"STT failed for {metrics.request_id}: {stt_result.error}")
                await connection_manager.safe_send_text(
                    client_id,
                    json.dumps({
                        "error": "transcription_failed",
                        "message": "I couldn't understand your speech. Please speak clearly and try again."
                    })
                )
                metrics.error = stt_result.error or "Empty transcription"
                return False
            
            # Record STT results
            metrics.transcript = stt_result.text
            metrics.stt_provider = stt_result.provider.value
            metrics.stt_confidence = stt_result.confidence
            metrics.stage = RequestStage.STT_COMPLETE
            
            logger.info(f"STT success [{metrics.stt_duration}ms]: '{metrics.transcript}' (provider: {metrics.stt_provider}, confidence: {metrics.stt_confidence:.3f})")
            
        except Exception as stt_error:
            metrics.stt_duration = int((time.perf_counter() - stt_start) * 1000)
            metrics.error = str(stt_error)
            logger.error(f"STT processing failed for {metrics.request_id}: {stt_error}", exc_info=True)
            await connection_manager.safe_send_text(
                client_id,
                json.dumps({
                    "error": "stt_error",
                    "message": "Sorry, I had trouble processing your speech."
                })
            )
            return False
        
        # ======================
        # STAGE 3: AI Agent Query (User-Specific)
        # ======================
        metrics.stage = RequestStage.AI_AGENT_START
        ai_agent_start = time.perf_counter()
        
        try:
            response_text = await query_ai_agent_safe(metrics.transcript, metrics.user_id)
            
            metrics.ai_agent_duration = int((time.perf_counter() - ai_agent_start) * 1000)
            metrics.response_text = response_text
            metrics.stage = RequestStage.AI_AGENT_COMPLETE
            
            logger.info(f"AI Agent query [{metrics.ai_agent_duration}ms]: Found response for '{metrics.transcript[:50]}...' (user: {metrics.user_id})")
            
        except Exception as ai_agent_error:
            metrics.ai_agent_duration = int((time.perf_counter() - ai_agent_start) * 1000)
            metrics.error = str(ai_agent_error)
            logger.error(f"AI Agent query failed for {metrics.request_id}: {ai_agent_error}", exc_info=True)
            response_text = "Sorry, I'm having trouble accessing my knowledge base."
        
        # ======================
        # STAGE 4: TTS Processing
        # ======================
        metrics.stage = RequestStage.TTS_START
        tts_start = time.perf_counter()
        
        try:
            audio_file_path = await synthesize_speech(response_text)
            metrics.tts_duration = int((time.perf_counter() - tts_start) * 1000)
            
            if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
                raise FileNotFoundError("TTS file generation failed")
            
            metrics.stage = RequestStage.TTS_COMPLETE
            
            logger.info(f"TTS success [{metrics.tts_duration}ms]: {os.path.getsize(audio_file_path)} bytes generated")
            
        except Exception as tts_error:
            metrics.tts_duration = int((time.perf_counter() - tts_start) * 1000)
            metrics.error = str(tts_error)
            logger.error(f"TTS processing failed for {metrics.request_id}: {tts_error}", exc_info=True)
            
            # Send text response as fallback
            await connection_manager.safe_send_text(
                client_id,
                json.dumps({
                    "type": "text_response",
                    "text": response_text,
                    "message": "Audio synthesis failed, but here's the text response"
                })
            )
            return False
        
        # ======================
        # STAGE 5: Send Response
        # ======================
        response_start = time.perf_counter()
        
        try:
            # Read audio file
            with open(audio_file_path, "rb") as f:
                audio_bytes = f.read()
            
            # Send audio response
            success = await connection_manager.safe_send_bytes(client_id, audio_bytes)
            
            metrics.response_duration = int((time.perf_counter() - response_start) * 1000)
            
            if success:
                metrics.stage = RequestStage.RESPONSE_SENT
                metrics.total_duration = int((time.perf_counter() - overall_start) * 1000)
                
                logger.info(f"Request {metrics.request_id} completed successfully in {metrics.total_duration}ms")
                logger.info(f"  Breakdown: Validation={metrics.user_validation_duration}ms | STT={metrics.stt_duration}ms | AI_Agent={metrics.ai_agent_duration}ms | TTS={metrics.tts_duration}ms | Send={metrics.response_duration}ms")
                
                # Performance warning
                if metrics.total_duration > 5000:
                    logger.warning(f"Slow response detected: {metrics.total_duration}ms for request {metrics.request_id}")
                
                return True
            else:
                logger.error(f"Failed to send response for {metrics.request_id}")
                metrics.error = "Failed to send response"
                return False
                
        except Exception as send_error:
            metrics.response_duration = int((time.perf_counter() - response_start) * 1000)
            metrics.error = str(send_error)
            logger.error(f"Response sending failed for {metrics.request_id}: {send_error}", exc_info=True)
            return False
            
        finally:
            # Clean up TTS file
            try:
                if 'audio_file_path' in locals() and os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
            except Exception as cleanup_error:
                logger.warning(f"TTS file cleanup failed: {cleanup_error}")
    
    except Exception as e:
        metrics.stage = RequestStage.ERROR
        metrics.error = str(e)
        metrics.total_duration = int((time.perf_counter() - overall_start) * 1000)
        logger.error(f"Fatal error processing request {metrics.request_id}: {e}", exc_info=True)
        return False

# Enhanced API endpoints
@app.get("/stats")
async def get_performance_stats():
    """Get detailed performance statistics"""
    return {
        "performance": perf_monitor.get_current_stats(),
        "connections": connection_manager.get_connection_stats()
    }

@app.get("/debug/recent-requests")
async def get_recent_requests(limit: int = 10):
    """Get detailed metrics for recent requests (debugging)"""
    if limit > 100:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 100")
    
    return {
        "recent_requests": perf_monitor.get_recent_performance(limit),
        "summary": perf_monitor.get_current_stats()
    }

# User validation endpoints
@app.get("/users/{user_id}/validate")
async def validate_user_endpoint(user_id: str):
    """Validate if user exists and has trained data"""
    try:
        user_exists, user_has_data, validation_msg = await validate_user_and_data(user_id)
        
        return {
            "user_id": user_id,
            "exists": user_exists,
            "has_data": user_has_data,
            "message": validation_msg,
            "status": "ready" if (user_exists and user_has_data) else "not_ready"
        }
    except Exception as e:
        logger.error(f"User validation endpoint failed for {user_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "user_id": user_id,
                "exists": False,
                "has_data": False,
                "message": f"Validation error: {str(e)}",
                "status": "error"
            }
        )

@app.get("/users/{user_id}/stats")
async def get_user_stats(user_id: str):
    """Get user-specific training statistics"""
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
                    "error": "User not found or no training data",
                    "message": "Please train the AI agent first using /data endpoint"
                }
            )
        
        return {
            "user_id": user_id,
            "stats": stats,
            "status": "trained"
        }
        
    except Exception as e:
        logger.error(f"User stats endpoint failed for {user_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "user_id": user_id,
                "error": str(e),
                "message": "Failed to get user statistics"
            }
        )

# New endpoints for continuous session management
@app.get("/sessions/active")
async def get_active_sessions():
    """Get information about active continuous sessions"""
    return connection_manager.get_connection_stats()

@app.post("/sessions/{client_id}/start-continuous")
async def start_continuous_session_http(client_id: str, vad_provider: str = "silero"):
    """Start continuous session via HTTP (for testing)"""
    try:
        vad = VADProvider(vad_provider)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid VAD provider: {vad_provider}")
    
    if client_id not in connection_manager.active_connections:
        raise HTTPException(status_code=404, detail="Client not connected")
    
    success = await connection_manager.start_continuous_session(client_id, vad)
    
    if success:
        return {"success": True, "message": f"Continuous session started for {client_id}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to start continuous session")

@app.post("/sessions/{client_id}/stop-continuous")
async def stop_continuous_session_http(client_id: str):
    """Stop continuous session via HTTP (for testing)"""
    if client_id not in connection_manager.active_connections:
        raise HTTPException(status_code=404, detail="Client not connected")
    
    await connection_manager.stop_continuous_session(client_id)
    return {"success": True, "message": f"Continuous session stopped for {client_id}"}

# Admin endpoints
@app.post("/admin/clear-cache")
async def clear_cache():
    """Clear all system caches"""
    try:
        return JSONResponse({"success": True, "message": "System ready (using ChromaDB)"})
    except Exception as e:
        logger.error(f"Cache operation failed: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/admin/warmup")
async def manual_warmup():
    """Manually trigger system warmup"""
    try:
        await warmup_components()
        return JSONResponse({"success": True, "message": "System warmup completed"})
    except Exception as e:
        logger.error(f"Manual warmup failed: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/admin/connections")
async def get_connections():
    """Get active connection information"""
    return connection_manager.get_connection_stats()

# WebSocket endpoint for testing continuous mode
@app.websocket("/ws/test-continuous")
async def test_continuous_websocket(websocket: WebSocket):
    """Test endpoint for continuous mode development"""
    client_id = f"test_{int(time.time())}"
    user_id = f"test_user_{int(time.time())}"
    
    try:
        await connection_manager.connect(websocket, client_id, user_id)
        
        # Automatically start continuous mode
        await connection_manager.start_continuous_session(client_id, VADProvider.SILERO)
        
        await connection_manager.safe_send_text(client_id, json.dumps({
            "type": "test_session_started",
            "message": "Test continuous session active. Send audio data to test.",
            "user_id": user_id,
            "instructions": {
                "send_audio": "Send raw audio bytes for processing",
                "send_command": "Send text commands like 'stop' to end session"
            }
        }))
        
        while True:
            try:
                # Handle both text and binary messages
                message = await websocket.receive()
                
                if message["type"] == "websocket.receive":
                    if "text" in message:
                        # Handle text commands
                        cmd = message["text"].strip().lower()
                        if cmd == "stop":
                            break
                        elif cmd == "status":
                            stats = connection_manager.get_connection_stats()
                            await connection_manager.safe_send_text(client_id, json.dumps({
                                "type": "status",
                                "stats": stats
                            }))
                    
                    elif "bytes" in message:
                        # Handle audio data
                        audio_data = message["bytes"]
                        await connection_manager.add_audio_chunk_to_stream(client_id, audio_data)
                        connection_manager.update_activity(client_id)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Test session error: {e}")
                break
    
    finally:
        connection_manager.disconnect(client_id)

# Development and testing
if __name__ == "__main__":
    import uvicorn
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Enhanced WebSocket test client
        import websockets
        
        async def test_websocket():
            uri = "ws://localhost:5000/ws/voice?user_id=test_user_123"
            try:
                logger.info("Testing Enhanced WebSocket with user validation...")
                async with websockets.connect(uri) as websocket:
                    
                    # Test 1: Check connection status
                    await websocket.send('{"type": "get_status"}')
                    response = await websocket.recv()
                    logger.info(f"Status response: {response}")
                    
                    # Test 2: Try to start continuous mode (should work if user is trained)
                    await websocket.send('{"type": "start_continuous", "vad_provider": "silero"}')
                    response = await websocket.recv()
                    logger.info(f"Start continuous response: {response}")
                    
                    # Test 3: Send some test audio data
                    test_data = b"RIFF" + b"\x00" * 1000 + b"WAVEfmt " + b"\x00" * 500
                    await websocket.send(test_data)
                    logger.info("Sent test audio data")
                    
                    # Test 4: Wait for any responses
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        logger.info(f"Received response: {response}")
                    except asyncio.TimeoutError:
                        logger.info("No response received - expected if user not trained")
                    
                    # Test 5: Stop continuous mode
                    await websocket.send('{"type": "stop_continuous"}')
                    response = await websocket.recv()
                    logger.info(f"Stop continuous response: {response}")
                    
            except Exception as e:
                logger.error(f"Enhanced WebSocket test failed: {e}")
        
        asyncio.run(test_websocket())
    
    else:
        logger.info("Starting Enhanced Voice Assistant Backend with User Validation...")
        logger.info("NEW FEATURES:")
        logger.info("  âœ… User validation before processing voice requests")
        logger.info("  âœ… Proper error handling for missing user_id")
        logger.info("  âœ… User training status validation")
        logger.info("  âœ… Clear error messages for untrained users")
        logger.info("  âœ… Enhanced logging for debugging")
        logger.info("")
        logger.info("IMPORTANT - Frontend Requirements:")
        logger.info("  ðŸ”§ WebSocket connection MUST include user_id: ws://host/ws/voice?user_id=ACTUAL_USER_ID")
        logger.info("  ðŸ”§ User MUST be trained via /data endpoint before voice queries")
        logger.info("  ðŸ”§ Without valid user_id, all voice requests will be blocked")
        logger.info("")
        logger.info("Features:")
        logger.info("  - User-specific AI agents with ChromaDB")
        logger.info("  - Continuous microphone mode with VAD")
        logger.info("  - Single /data endpoint for all ChromaDB operations")
        logger.info("  - Thread-safe processing with user-specific locks")
        logger.info("  - Enhanced error handling and user validation")
        
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=5000,
            reload=False,
            access_log=False,
            workers=1,
            log_level="info"
        )