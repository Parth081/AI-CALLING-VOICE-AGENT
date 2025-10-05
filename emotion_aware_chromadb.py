#!/usr/bin/env python3
"""
Emotion-Aware AI Agent Extension
Modifies responses based on detected user mood
"""

import logging
from typing import Optional, Dict, Any
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Mood-based response strategies
MOOD_STRATEGIES = {
    "frustrated": {
        "tone": "apologetic and extra helpful",
        "prefix": "I apologize for any confusion. Let me provide you with clear, specific information: ",
        "style": "Be direct, specific, avoid vague answers. Provide step-by-step details."
    },
    "angry": {
        "tone": "calm, understanding, and solution-focused",
        "prefix": "I understand your frustration. Let me help resolve this: ",
        "style": "Acknowledge the issue, provide immediate solutions, be concise."
    },
    "confused": {
        "tone": "patient and explanatory",
        "prefix": "Let me clarify that for you: ",
        "style": "Break down complex information, use examples, check understanding."
    },
    "disappointed": {
        "tone": "empathetic and reassuring",
        "prefix": "I'm sorry I didn't meet your expectations. Here's what I can do: ",
        "style": "Acknowledge disappointment, provide better information, show improvement."
    },
    "resigned": {
        "tone": "encouraging and proactive",
        "prefix": "I want to help you properly. ",
        "style": "Be more proactive, offer alternatives, show care."
    },
    "skeptical": {
        "tone": "confident and evidence-based",
        "prefix": "",
        "style": "Provide specific facts, be transparent, build credibility."
    },
    "happy": {
        "tone": "warm and engaging",
        "prefix": "",
        "style": "Match positive energy, be conversational, maintain quality."
    },
    "satisfied": {
        "tone": "friendly and consistent",
        "prefix": "",
        "style": "Maintain current quality, be helpful, stay professional."
    },
    "engaged": {
        "tone": "enthusiastic and informative",
        "prefix": "",
        "style": "Provide detailed information, encourage questions, be thorough."
    },
    "neutral": {
        "tone": "professional and helpful",
        "prefix": "",
        "style": "Standard helpful response, clear and informative."
    }
}


class EmotionAwareAgent:
    """Wraps AI agent responses with emotional intelligence"""
    
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY required")
        self.groq_client = Groq(api_key=self.groq_api_key)
    
    def enhance_response_with_emotion(
        self, 
        base_response: str,
        detected_mood: str,
        sentiment_score: float,
        user_query: str,
        conversation_context: Optional[str] = None
    ) -> str:
        """
        Enhance AI response based on user's emotional state
        
        Args:
            base_response: Original response from ChromaDB
            detected_mood: User's detected mood (frustrated, happy, etc.)
            sentiment_score: Sentiment score (-1.0 to 1.0)
            user_query: What the user asked
            conversation_context: Optional context from session history
        
        Returns:
            Emotionally-adapted response
        """
        
        # Get strategy for this mood
        strategy = MOOD_STRATEGIES.get(detected_mood, MOOD_STRATEGIES["neutral"])
        
        # If mood is neutral or positive, return base response
        if detected_mood in ["neutral", "happy", "satisfied", "engaged"] and sentiment_score >= 0:
            return base_response
        
        # For negative moods, enhance the response
        try:
            logger.info(f"Enhancing response for mood: {detected_mood} (sentiment: {sentiment_score:.2f})")
            
            prompt = f"""The user is feeling {detected_mood} (sentiment: {sentiment_score:.2f}).

Their question: "{user_query}"

Base answer from knowledge base: "{base_response}"

{f"Recent conversation context: {conversation_context}" if conversation_context else ""}

TASK: Rewrite this answer with the following emotional adaptation:
- Tone: {strategy['tone']}
- Style: {strategy['style']}
{f"- Start with: {strategy['prefix']}" if strategy['prefix'] else ""}

CRITICAL RULES:
1. Keep ALL factual information from the base answer
2. Don't add fake information or make promises we can't keep
3. Adapt the TONE and DELIVERY, not the facts
4. Be natural - don't sound robotic
5. Keep it concise (2-3 sentences max unless needed)
6. {f"Start naturally with acknowledgment like: {strategy['prefix']}" if strategy['prefix'] else "Be direct and helpful"}

Emotionally-adapted answer:"""

            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an empathetic AI assistant that adapts responses based on user emotions while maintaining factual accuracy."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            enhanced = response.choices[0].message.content.strip()
            
            # Remove quotes if wrapped
            enhanced = enhanced.strip('"').strip("'")
            
            logger.info(f"Response enhanced from {len(base_response)} to {len(enhanced)} chars")
            logger.debug(f"Original: {base_response[:100]}...")
            logger.debug(f"Enhanced: {enhanced[:100]}...")
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Response enhancement failed: {e}")
            # Fallback: add simple prefix
            if strategy["prefix"]:
                return f"{strategy['prefix']}{base_response}"
            return base_response
    
    def should_enhance_response(self, mood: str, sentiment: float) -> bool:
        """Determine if response needs emotional enhancement"""
        # Enhance for negative moods
        negative_moods = ["frustrated", "angry", "confused", "disappointed", "resigned", "skeptical"]
        return mood in negative_moods or sentiment < -0.2


# Global instance
_emotion_aware_agent: Optional[EmotionAwareAgent] = None


def get_emotion_aware_agent() -> EmotionAwareAgent:
    """Get or create global emotion-aware agent"""
    global _emotion_aware_agent
    if _emotion_aware_agent is None:
        _emotion_aware_agent = EmotionAwareAgent()
    return _emotion_aware_agent


def enhance_ai_response(
    base_response: str,
    user_query: str,
    detected_mood: str = "neutral",
    sentiment_score: float = 0.0,
    conversation_context: Optional[str] = None
) -> str:
    """
    Main function to enhance AI responses based on emotion
    
    Example:
        base = "We're open 9-5 Monday-Friday."
        enhanced = enhance_ai_response(
            base, 
            "What are your hours?",
            detected_mood="frustrated",
            sentiment_score=-0.65
        )
        # Returns: "I apologize for any confusion. Let me be specific: 
        #           We're open 9 AM to 5 PM, Monday through Friday."
    """
    try:
        agent = get_emotion_aware_agent()
        
        # Check if enhancement needed
        if not agent.should_enhance_response(detected_mood, sentiment_score):
            return base_response
        
        return agent.enhance_response_with_emotion(
            base_response,
            detected_mood,
            sentiment_score,
            user_query,
            conversation_context
        )
    except Exception as e:
        logger.error(f"Emotion enhancement failed: {e}")
        return base_response


if __name__ == "__main__":
    # Test examples
    print("=" * 70)
    print("EMOTION-AWARE RESPONSE ENHANCEMENT TEST")
    print("=" * 70)
    
    test_cases = [
        {
            "query": "What are your hours?",
            "base": "We're open Monday to Friday, 9 AM to 5 PM.",
            "mood": "frustrated",
            "sentiment": -0.65
        },
        {
            "query": "Where are you located?",
            "base": "We're in the downtown area.",
            "mood": "disappointed",
            "sentiment": -0.80
        },
        {
            "query": "Do you deliver?",
            "base": "Yes, we offer delivery services.",
            "mood": "confused",
            "sentiment": -0.30
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Query: {test['query']}")
        print(f"Mood: {test['mood']} (sentiment: {test['sentiment']})")
        print(f"Base: {test['base']}")
        
        enhanced = enhance_ai_response(
            test['base'],
            test['query'],
            test['mood'],
            test['sentiment']
        )
        
        print(f"Enhanced: {enhanced}")