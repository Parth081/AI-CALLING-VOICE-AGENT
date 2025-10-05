#!/usr/bin/env python3
"""
Emotional Intelligence Memory System
Tracks user mood, sentiment trajectory, and conversation outcomes
Enables empathetic, context-aware interactions across sessions
"""

import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MoodState(Enum):
    """User mood states"""
    HAPPY = "happy"
    SATISFIED = "satisfied"
    NEUTRAL = "neutral"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    RESIGNED = "resigned"
    SKEPTICAL = "skeptical"
    ENGAGED = "engaged"
    DISAPPOINTED = "disappointed"


class SessionOutcome(Enum):
    """Overall session outcome classifications"""
    HIGHLY_SATISFIED = "highly_satisfied"
    SATISFIED = "satisfied"
    NEUTRAL = "neutral"
    UNSATISFIED = "unsatisfied"
    FRUSTRATED = "frustrated"
    ABANDONED = "abandoned"


@dataclass
class ConversationTurn:
    """Single conversation turn with sentiment"""
    timestamp: float
    user_message: str
    assistant_response: str
    detected_mood: MoodState
    sentiment_score: float  # -1.0 to 1.0
    intent_fulfilled: bool
    response_time_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'detected_mood': self.detected_mood.value,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp).isoformat()
        }


@dataclass
class SessionSummary:
    """Complete session summary with mood trajectory"""
    session_id: str
    user_id: str
    start_time: float
    end_time: float
    duration_seconds: float
    total_turns: int
    
    # Mood trajectory
    mood_trajectory: List[MoodState]
    sentiment_trajectory: List[float]
    average_sentiment: float
    
    # Outcome
    session_outcome: SessionOutcome
    outcome_confidence: float
    
    # Performance
    intents_fulfilled: int
    intents_failed: int
    average_response_time_ms: float
    
    # Key insights
    primary_topics: List[str]
    pain_points: List[str]
    positive_moments: List[str]
    
    # Next session recommendations
    opening_strategy: str
    conversation_context: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'mood_trajectory': [m.value for m in self.mood_trajectory],
            'session_outcome': self.session_outcome.value,
            'start_time_iso': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time_iso': datetime.fromtimestamp(self.end_time).isoformat()
        }


class EmotionalIntelligenceEngine:
    """Core engine for emotional intelligence and memory"""
    
    def __init__(self, chroma_path: str = "./chroma_emotion_memory"):
        self.chroma_path = chroma_path
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY required for sentiment analysis")
        
        self.groq_client = Groq(api_key=self.groq_api_key)
        
        # Initialize ChromaDB
        os.makedirs(chroma_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        # Collections for different data types
        self.conversations_collection = self._get_or_create_collection("conversations")
        self.sessions_collection = self._get_or_create_collection("sessions")
        
        # In-memory active sessions
        self.active_sessions: Dict[str, List[ConversationTurn]] = {}
        
        logger.info(f"Emotional Intelligence Engine initialized at {chroma_path}")
    
    def _get_or_create_collection(self, name: str):
        """Get or create ChromaDB collection"""
        try:
            return self.client.get_or_create_collection(
                name=name,
                embedding_function=DefaultEmbeddingFunction()
            )
        except Exception as e:
            logger.error(f"Collection creation failed: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Tuple[MoodState, float]:
        """
        Analyze sentiment and mood from text using Groq
        Returns: (MoodState, sentiment_score)
        """
        try:
            prompt = f"""Analyze the sentiment and emotional tone of this text:

"{text}"

Respond in JSON format:
{{
    "mood": "one of: happy, satisfied, neutral, confused, frustrated, angry, resigned, skeptical, engaged, disappointed",
    "sentiment_score": float between -1.0 (very negative) and 1.0 (very positive),
    "reasoning": "brief explanation"
}}"""

            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert at emotional intelligence and sentiment analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                mood_str = data.get("mood", "neutral").lower()
                sentiment_score = float(data.get("sentiment_score", 0.0))
                
                # Map to MoodState
                mood_mapping = {
                    "happy": MoodState.HAPPY,
                    "satisfied": MoodState.SATISFIED,
                    "neutral": MoodState.NEUTRAL,
                    "confused": MoodState.CONFUSED,
                    "frustrated": MoodState.FRUSTRATED,
                    "angry": MoodState.ANGRY,
                    "resigned": MoodState.RESIGNED,
                    "skeptical": MoodState.SKEPTICAL,
                    "engaged": MoodState.ENGAGED,
                    "disappointed": MoodState.DISAPPOINTED
                }
                
                mood = mood_mapping.get(mood_str, MoodState.NEUTRAL)
                
                logger.debug(f"Sentiment: {mood.value} ({sentiment_score:.2f}) - {data.get('reasoning', '')}")
                return mood, sentiment_score
            
            return MoodState.NEUTRAL, 0.0
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return MoodState.NEUTRAL, 0.0
    
    def start_session(self, user_id: str) -> str:
        """Start a new conversation session"""
        session_id = f"{user_id}_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        self.active_sessions[session_id] = []
        
        logger.info(f"Started session: {session_id} for user {user_id}")
        return session_id
    
    def add_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        intent_fulfilled: bool,
        response_time_ms: int
    ):
        """Add a conversation turn with sentiment analysis"""
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found, creating new")
            self.active_sessions[session_id] = []
        
        # Analyze user sentiment
        mood, sentiment = self.analyze_sentiment(user_message)
        
        turn = ConversationTurn(
            timestamp=time.time(),
            user_message=user_message,
            assistant_response=assistant_response,
            detected_mood=mood,
            sentiment_score=sentiment,
            intent_fulfilled=intent_fulfilled,
            response_time_ms=response_time_ms
        )
        
        self.active_sessions[session_id].append(turn)
        
        # Store turn in ChromaDB
        self._store_conversation_turn(session_id, turn)
        
        logger.info(f"Turn added: {session_id} - Mood: {mood.value}, Sentiment: {sentiment:.2f}")
    
    def _store_conversation_turn(self, session_id: str, turn: ConversationTurn):
        """Store conversation turn in ChromaDB"""
        try:
            turn_id = f"{session_id}_{int(turn.timestamp)}"
            
            self.conversations_collection.add(
                documents=[turn.user_message],
                metadatas=[{
                    'session_id': session_id,
                    'timestamp': turn.timestamp,
                    'mood': turn.detected_mood.value,
                    'sentiment': turn.sentiment_score,
                    'assistant_response': turn.assistant_response,
                    'intent_fulfilled': turn.intent_fulfilled,
                    'response_time_ms': turn.response_time_ms
                }],
                ids=[turn_id]
            )
        except Exception as e:
            logger.error(f"Failed to store turn: {e}")
    
    def end_session(self, session_id: str, user_id: str) -> SessionSummary:
        """End session and generate summary with outcome analysis"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        turns = self.active_sessions[session_id]
        
        if not turns:
            logger.warning(f"Session {session_id} has no turns")
            return self._create_empty_session_summary(session_id, user_id)
        
        # Calculate session metrics
        start_time = turns[0].timestamp
        end_time = turns[-1].timestamp
        duration = end_time - start_time
        
        mood_trajectory = [turn.detected_mood for turn in turns]
        sentiment_trajectory = [turn.sentiment_score for turn in turns]
        avg_sentiment = sum(sentiment_trajectory) / len(sentiment_trajectory)
        
        intents_fulfilled = sum(1 for turn in turns if turn.intent_fulfilled)
        intents_failed = len(turns) - intents_fulfilled
        avg_response_time = sum(turn.response_time_ms for turn in turns) / len(turns)
        
        # Determine session outcome
        outcome, confidence = self._determine_session_outcome(
            sentiment_trajectory,
            mood_trajectory,
            intents_fulfilled,
            intents_failed
        )
        
        # Extract insights
        primary_topics = self._extract_topics([turn.user_message for turn in turns])
        pain_points = self._identify_pain_points(turns)
        positive_moments = self._identify_positive_moments(turns)
        
        # Generate next session strategy
        opening_strategy, context = self._generate_opening_strategy(
            outcome,
            mood_trajectory,
            pain_points,
            positive_moments
        )
        
        summary = SessionSummary(
            session_id=session_id,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            total_turns=len(turns),
            mood_trajectory=mood_trajectory,
            sentiment_trajectory=sentiment_trajectory,
            average_sentiment=avg_sentiment,
            session_outcome=outcome,
            outcome_confidence=confidence,
            intents_fulfilled=intents_fulfilled,
            intents_failed=intents_failed,
            average_response_time_ms=avg_response_time,
            primary_topics=primary_topics,
            pain_points=pain_points,
            positive_moments=positive_moments,
            opening_strategy=opening_strategy,
            conversation_context=context
        )
        
        # Store session summary
        self._store_session_summary(summary)
        
        # Clean up active session
        del self.active_sessions[session_id]
        
        logger.info(f"Session ended: {session_id} - Outcome: {outcome.value} ({confidence:.2f})")
        return summary
    
    def _determine_session_outcome(
        self,
        sentiment_trajectory: List[float],
        mood_trajectory: List[MoodState],
        intents_fulfilled: int,
        intents_failed: int
    ) -> Tuple[SessionOutcome, float]:
        """Determine overall session outcome with confidence score"""
        
        # Calculate trend
        if len(sentiment_trajectory) >= 2:
            sentiment_trend = sentiment_trajectory[-1] - sentiment_trajectory[0]
        else:
            sentiment_trend = 0
        
        avg_sentiment = sum(sentiment_trajectory) / len(sentiment_trajectory)
        final_sentiment = sentiment_trajectory[-1]
        
        # Check for negative moods
        negative_moods = {MoodState.FRUSTRATED, MoodState.ANGRY, MoodState.DISAPPOINTED, MoodState.RESIGNED}
        negative_count = sum(1 for mood in mood_trajectory if mood in negative_moods)
        negative_ratio = negative_count / len(mood_trajectory)
        
        # Intent success rate
        success_rate = intents_fulfilled / max(1, intents_fulfilled + intents_failed)
        
        # Scoring system
        outcome_score = (
            avg_sentiment * 0.3 +
            final_sentiment * 0.3 +
            sentiment_trend * 0.2 +
            success_rate * 0.2
        )
        
        # Penalize for negative moods
        outcome_score -= negative_ratio * 0.5
        
        # Determine outcome
        if outcome_score >= 0.6:
            outcome = SessionOutcome.HIGHLY_SATISFIED
            confidence = min(0.95, 0.7 + (outcome_score - 0.6) * 0.5)
        elif outcome_score >= 0.3:
            outcome = SessionOutcome.SATISFIED
            confidence = 0.75
        elif outcome_score >= -0.1:
            outcome = SessionOutcome.NEUTRAL
            confidence = 0.70
        elif outcome_score >= -0.4:
            outcome = SessionOutcome.UNSATISFIED
            confidence = 0.75
        else:
            outcome = SessionOutcome.FRUSTRATED
            confidence = min(0.95, 0.7 + abs(outcome_score + 0.4) * 0.5)
        
        # Check for abandonment (very short session with negative trend)
        if len(mood_trajectory) <= 2 and sentiment_trend < -0.3:
            outcome = SessionOutcome.ABANDONED
            confidence = 0.80
        
        return outcome, confidence
    
    def _extract_topics(self, messages: List[str]) -> List[str]:
        """Extract primary topics from conversation"""
        try:
            combined = " ".join(messages[:5])  # First 5 messages
            
            prompt = f"""Extract the 3-5 main topics from this conversation:

{combined}

Return as JSON array: ["topic1", "topic2", ...]"""

            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            content = response.choices[0].message.content.strip()
            
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                topics = json.loads(json_match.group())
                return topics[:5]
            
            return ["general_inquiry"]
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            return ["general_inquiry"]
    
    def _identify_pain_points(self, turns: List[ConversationTurn]) -> List[str]:
        """Identify pain points from conversation"""
        pain_points = []
        
        negative_moods = {MoodState.FRUSTRATED, MoodState.ANGRY, MoodState.CONFUSED, MoodState.DISAPPOINTED}
        
        for turn in turns:
            if turn.detected_mood in negative_moods or not turn.intent_fulfilled:
                # Extract key phrases
                words = turn.user_message.lower().split()
                if any(word in words for word in ["not", "can't", "won't", "don't", "unable", "failed"]):
                    pain_points.append(f"Issue with: {turn.user_message[:50]}...")
        
        return pain_points[:3]  # Top 3 pain points
    
    def _identify_positive_moments(self, turns: List[ConversationTurn]) -> List[str]:
        """Identify positive moments"""
        positive = []
        
        positive_moods = {MoodState.HAPPY, MoodState.SATISFIED, MoodState.ENGAGED}
        
        for turn in turns:
            if turn.detected_mood in positive_moods and turn.sentiment_score > 0.4:
                positive.append(f"Success: {turn.user_message[:50]}...")
        
        return positive[:3]
    
    def _generate_opening_strategy(
        self,
        outcome: SessionOutcome,
        mood_trajectory: List[MoodState],
        pain_points: List[str],
        positive_moments: List[str]
    ) -> Tuple[str, str]:
        """Generate adaptive opening strategy for next session"""
        
        if outcome in [SessionOutcome.FRUSTRATED, SessionOutcome.UNSATISFIED, SessionOutcome.ABANDONED]:
            strategy = "repair"
            context = (
                "User had difficulties last session. "
                "Acknowledge challenges, offer improved support, be extra attentive. "
                f"Pain points: {', '.join(pain_points) if pain_points else 'response quality'}"
            )
            
        elif outcome == SessionOutcome.HIGHLY_SATISFIED:
            strategy = "continuity_positive"
            context = (
                "User was highly satisfied. "
                "Maintain quality, build on success, show consistency. "
                f"Strengths: {', '.join(positive_moments[:2]) if positive_moments else 'helpful responses'}"
            )
            
        elif outcome == SessionOutcome.SATISFIED:
            strategy = "continuity_neutral"
            context = (
                "User was satisfied but room for improvement. "
                "Continue good practices, slightly increase attentiveness."
            )
            
        else:  # NEUTRAL
            strategy = "standard"
            context = "Standard greeting, no special context needed."
        
        return strategy, context
    
    def _store_session_summary(self, summary: SessionSummary):
        """Store session summary in ChromaDB"""
        try:
            # Create searchable document from session
            doc_text = f"Session for {summary.user_id}: {', '.join(summary.primary_topics)}"
            
            self.sessions_collection.add(
                documents=[doc_text],
                metadatas=[{
                    'session_id': summary.session_id,
                    'user_id': summary.user_id,
                    'outcome': summary.session_outcome.value,
                    'outcome_confidence': summary.outcome_confidence,
                    'average_sentiment': summary.average_sentiment,
                    'total_turns': summary.total_turns,
                    'duration_seconds': summary.duration_seconds,
                    'opening_strategy': summary.opening_strategy,
                    'conversation_context': summary.conversation_context,
                    'pain_points': json.dumps(summary.pain_points),
                    'positive_moments': json.dumps(summary.positive_moments),
                    'timestamp': summary.end_time
                }],
                ids=[summary.session_id]
            )
            
            logger.info(f"Session summary stored: {summary.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store session summary: {e}")
    
    def _create_empty_session_summary(self, session_id: str, user_id: str) -> SessionSummary:
        """Create empty summary for sessions with no turns"""
        return SessionSummary(
            session_id=session_id,
            user_id=user_id,
            start_time=time.time(),
            end_time=time.time(),
            duration_seconds=0,
            total_turns=0,
            mood_trajectory=[],
            sentiment_trajectory=[],
            average_sentiment=0,
            session_outcome=SessionOutcome.ABANDONED,
            outcome_confidence=0.9,
            intents_fulfilled=0,
            intents_failed=0,
            average_response_time_ms=0,
            primary_topics=[],
            pain_points=["Session abandoned immediately"],
            positive_moments=[],
            opening_strategy="repair",
            conversation_context="User abandoned session without interaction"
        )
    
    def get_last_session(self, user_id: str) -> Optional[SessionSummary]:
        """Retrieve last session summary for user"""
        try:
            results = self.sessions_collection.query(
                query_texts=[user_id],
                n_results=20,
                where={"user_id": user_id}
            )
            
            if not results['metadatas'] or not results['metadatas'][0]:
                return None
            
            # Find most recent
            sessions = results['metadatas'][0]
            most_recent = max(sessions, key=lambda x: x.get('timestamp', 0))
            
            # Reconstruct SessionSummary
            return SessionSummary(
                session_id=most_recent['session_id'],
                user_id=most_recent['user_id'],
                start_time=most_recent['timestamp'] - most_recent.get('duration_seconds', 0),
                end_time=most_recent['timestamp'],
                duration_seconds=most_recent.get('duration_seconds', 0),
                total_turns=most_recent.get('total_turns', 0),
                mood_trajectory=[],  # Not stored in retrieval
                sentiment_trajectory=[],
                average_sentiment=most_recent.get('average_sentiment', 0),
                session_outcome=SessionOutcome(most_recent['outcome']),
                outcome_confidence=most_recent.get('outcome_confidence', 0.5),
                intents_fulfilled=0,
                intents_failed=0,
                average_response_time_ms=0,
                primary_topics=[],
                pain_points=json.loads(most_recent.get('pain_points', '[]')),
                positive_moments=json.loads(most_recent.get('positive_moments', '[]')),
                opening_strategy=most_recent['opening_strategy'],
                conversation_context=most_recent['conversation_context']
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve last session: {e}")
            return None
    
    def generate_greeting(self, user_id: str) -> str:
        """Generate adaptive greeting based on last session"""
        last_session = self.get_last_session(user_id)
        
        if not last_session:
            return "Hello! How can I help you today?"
        
        # Check if last session was recent (within 24 hours)
        hours_since = (time.time() - last_session.end_time) / 3600
        
        if hours_since > 72:  # More than 3 days
            return f"Welcome back! It's been a while. How can I assist you today?"
        
        strategy = last_session.opening_strategy
        outcome = last_session.session_outcome
        
        if strategy == "repair":
            if outcome == SessionOutcome.ABANDONED:
                return "Hello again. I noticed our last conversation was brief. I'm here to help—what can I do for you today?"
            else:
                return "Welcome back. Last time I may not have fully met your expectations. I'll make sure to be extra attentive today. How can I help?"
        
        elif strategy == "continuity_positive":
            return "Great to see you again! Let's keep the momentum going. What would you like to discuss?"
        
        elif strategy == "continuity_neutral":
            return "Hello! Ready to help you again. What's on your mind today?"
        
        else:  # standard
            return "Hello! How can I assist you today?"
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user statistics"""
        try:
            results = self.sessions_collection.query(
                query_texts=[user_id],
                n_results=100,
                where={"user_id": user_id}
            )
            
            if not results['metadatas'] or not results['metadatas'][0]:
                return {
                    'user_id': user_id,
                    'total_sessions': 0,
                    'has_history': False
                }
            
            sessions = results['metadatas'][0]
            
            total_sessions = len(sessions)
            outcomes = [s.get('outcome', 'neutral') for s in sessions]
            avg_sentiment = sum(s.get('average_sentiment', 0) for s in sessions) / total_sessions
            
            outcome_distribution = {}
            for outcome in SessionOutcome:
                outcome_distribution[outcome.value] = outcomes.count(outcome.value)
            
            return {
                'user_id': user_id,
                'total_sessions': total_sessions,
                'has_history': True,
                'average_sentiment': round(avg_sentiment, 3),
                'outcome_distribution': outcome_distribution,
                'last_session_outcome': sessions[0].get('outcome', 'neutral'),
                'recommended_greeting': self.generate_greeting(user_id)
            }
            
        except Exception as e:
            logger.error(f"Failed to get user stats: {e}")
            return {
                'user_id': user_id,
                'total_sessions': 0,
                'has_history': False,
                'error': str(e)
            }


# Global instance
_emotion_engine: Optional[EmotionalIntelligenceEngine] = None


def get_emotion_engine() -> EmotionalIntelligenceEngine:
    """Get or create global emotion engine instance"""
    global _emotion_engine
    if _emotion_engine is None:
        _emotion_engine = EmotionalIntelligenceEngine()
    return _emotion_engine


# External API functions
def start_emotional_session(user_id: str) -> str:
    """Start emotion-tracked session"""
    engine = get_emotion_engine()
    return engine.start_session(user_id)


def record_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_response: str,
    intent_fulfilled: bool,
    response_time_ms: int
):
    """Record conversation turn with emotion tracking"""
    engine = get_emotion_engine()
    engine.add_conversation_turn(
        session_id,
        user_message,
        assistant_response,
        intent_fulfilled,
        response_time_ms
    )


def end_emotional_session(session_id: str, user_id: str) -> Dict[str, Any]:
    """End session and get summary"""
    engine = get_emotion_engine()
    summary = engine.end_session(session_id, user_id)
    return summary.to_dict()


def get_adaptive_greeting(user_id: str) -> str:
    """Get adaptive greeting based on history"""
    engine = get_emotion_engine()
    return engine.generate_greeting(user_id)


def get_user_emotion_stats(user_id: str) -> Dict[str, Any]:
    """Get user emotional intelligence stats"""
    engine = get_emotion_engine()
    return engine.get_user_stats(user_id)


if __name__ == "__main__":
    print("=" * 80)
    print("EMOTIONAL INTELLIGENCE MEMORY SYSTEM")
    print("=" * 80)
    print("\nFeatures:")
    print("  ✓ Real-time sentiment analysis with Groq")
    print("  ✓ Mood trajectory tracking")
    print("  ✓ Session outcome classification")
    print("  ✓ Pain point identification")
    print("  ✓ Adaptive greeting generation")
    print("  ✓ ChromaDB-based memory storage")
    print("\nThis creates empathetic AI with conversation memory")
    print("=" * 80)