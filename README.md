# AI Voice Assistant with Emotional Intelligence Memory

## 🎯 Version 4.0.0 - Revolutionary Update

This system now has **emotional memory**—it remembers not just what users said, but *how they felt* during conversations. It adapts greetings, repairs poor experiences, and builds genuine rapport across sessions.

---

## 🧠 What's New: Emotional Intelligence Engine

### Core Capabilities

1. **Real-Time Sentiment Analysis**
   - Analyzes user mood in every message using Groq's LLaMA 3.3 70B
   - Detects: happy, satisfied, neutral, confused, frustrated, angry, resigned, skeptical, engaged, disappointed
   - Scores sentiment from -1.0 (very negative) to +1.0 (very positive)

2. **Mood Trajectory Tracking**
   - Tracks emotional journey throughout conversation
   - Example: `frustrated → skeptical → satisfied → engaged`
   - Identifies turning points and pain points

3. **Session Outcome Analysis**
   - Automatically classifies sessions:
     - **Highly Satisfied**: User had excellent experience
     - **Satisfied**: Good experience, room for improvement
     - **Neutral**: Standard interaction
     - **Unsatisfied**: User needs weren't fully met
     - **Frustrated**: Significant issues encountered
     - **Abandoned**: User left quickly
   - Confidence scores for each classification

4. **Adaptive Greetings**
   - Next session greetings based on last experience:
     - **After poor experience**: "Last time I may not have met your expectations. I'll make sure to be extra attentive today."
     - **After great experience**: "Great to see you again! Let's keep the momentum going."
     - **After abandonment**: "Hello again. I noticed our last conversation was brief. I'm here to help—what can I do for you today?"
     - **Standard**: "Hello! How can I assist you today?"

5. **Pain Point Identification**
   - Automatically identifies what frustrated users
   - Tracks failed intents and negative moments
   - Uses this data for improvement

6. **Positive Moment Recognition**
   - Captures what worked well
   - Reinforces successful interaction patterns

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     WebSocket Connection                     │
│                    (User connects with ID)                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Start Emotional Intelligence Session            │
│           • Generate adaptive greeting from history          │
│           • Initialize mood tracking for session             │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   For Each User Message                      │
├─────────────────────────────────────────────────────────────┤
│  1. STT (Speech-to-Text)                                    │
│  2. User Validation & ChromaDB Query                        │
│  3. Booking Detection (if applicable)                       │
│  4. AI Agent Response Generation                            │
│  5. TTS (Text-to-Speech)                                    │
│  6. ✨ Emotion Analysis ✨                                  │
│     • Analyze user sentiment in message                     │
│     • Detect mood state                                     │
│     • Record conversation turn                              │
│     • Track response time & intent success                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              End Session (On Disconnect)                     │
├─────────────────────────────────────────────────────────────┤
│  • Calculate mood trajectory                                │
│  • Determine session outcome (satisfied/frustrated/etc)     │
│  • Extract pain points & positive moments                   │
│  • Generate opening strategy for next session               │
│  • Store summary in ChromaDB for future reference           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation & Setup

### 1. Prerequisites

```bash
Python 3.9+
pip
```

### 2. Install Dependencies

```bash
pip install fastapi uvicorn websockets
pip install chromadb groq openai edge-tts
pip install python-dotenv pydantic
pip install google-api-python-client google-auth
```

### 3. Environment Variables

Create `.env` file:

```env
# Required APIs
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Google Sheets (for booking)
SPREADSHEET_ID=your_spreadsheet_id
SERVICE_ACCOUNT_FILE=service_account.json

# Email (for booking confirmations)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
ADMIN_EMAIL=business_owner@example.com
BUSINESS_NAME=Your Business Name
```

### 4. File Structure

```
project/
├── app.py                      # Main application (v4.0.0)
├── emotion_memory.py           # NEW: Emotional intelligence engine
├── chromadb_pipeline.py        # AI agent with ChromaDB
├── utils/
│   ├── booking_module.py       # Booking system
│   ├── stt_module.py           # Speech-to-text
│   └── tts_module.py           # Text-to-speech
├── service_account.json        # Google Sheets credentials
├── .env                        # Environment variables
└── README.md                   # This file
```

### 5. Run the Server

```bash
python app.py
```

Server starts at `http://localhost:5000`

---

## 📡 API Endpoints

### Core Endpoints

#### `GET /`
System information and features

#### `GET /healthz`
Health check with emotion engine status

#### `POST /data`
Train AI agent with business data
```json
{
  "user_id": "user123",
  "business_name": "Tech Solutions",
  "industry": "Technology",
  "phone": "+1234567890",
  "email": "contact@example.com",
  "additional_content": "Full business details..."
}
```

#### `WebSocket /ws/voice?user_id=USER_ID`
Main voice interaction endpoint with emotion tracking

### Emotional Intelligence Endpoints

#### `GET /users/{user_id}/emotion-stats`
Get comprehensive emotional intelligence statistics
```json
{
  "user_id": "user123",
  "emotion_stats": {
    "total_sessions": 15,
    "has_history": true,
    "average_sentiment": 0.65,
    "outcome_distribution": {
      "highly_satisfied": 8,
      "satisfied": 5,
      "neutral": 1,
      "unsatisfied": 1,
      "frustrated": 0,
      "abandoned": 0
    },
    "last_session_outcome": "satisfied",
    "recommended_greeting": "Welcome back! Ready to help you again."
  }
}
```

#### `GET /users/{user_id}/greeting`
Get adaptive greeting based on conversation history
```json
{
  "user_id": "user123",
  "greeting": "Last time I may not have fully met your expectations. I'll make sure to be extra attentive today. How can I help?"
}
```

#### `GET /users/{user_id}/stats`
Get AI agent training statistics

#### `GET /users/{user_id}/validate`
Validate user has trained data

---

## 💬 WebSocket Protocol

### Connection Message (Sent on Connect)

```json
{
  "type": "connected",
  "client_id": "192.168.1.100:54321:1696253400",
  "user_id": "user123",
  "status": "ready",
  "greeting": "Welcome back! Last time was great—let's continue.",
  "emotion_tracking": true,
  "emotion_stats": {
    "total_sessions": 10,
    "average_sentiment": 0.72
  }
}
```

### Send Audio (Client → Server)

Binary WebSocket message with WAV audio bytes

### Response Metadata (Server → Client)

```json
{
  "type": "response_metadata",
  "transcript": "What are your hours?",
  "response_text": "We're open Monday-Friday, 9 AM to 5 PM.",
  "confidence": 0.95,
  "audio_size": 45600,
  "user_id": "user123",
  "is_booking": false
}
```

### Audio Response (Server → Client)

Binary WebSocket message with MP3 audio bytes

### Commands

#### Start Continuous Mode
```json
{
  "type": "start_continuous",
  "vad_provider": "silero"
}
```

#### Stop Continuous Mode
```json
{
  "type": "stop_continuous"
}
```

#### Get Status
```json
{
  "type": "get_status"
}
```

Response includes emotion stats:
```json
{
  "type": "status_response",
  "mode": "single_shot",
  "user_id": "user123",
  "user_status": "ready",
  "emotion_stats": {...}
}
```

---

## 🎭 How Emotional Intelligence Works

### Example: User Has Poor Experience

**Session 1 (Friday 3 PM)**
```
User: "What's your address?"
AI: [Provides generic response]
User: "That doesn't help. Where exactly are you located?" (frustrated)
AI: [Better response but user already frustrated]
User: "Whatever..." (resigned)
[User disconnects]

Emotion Engine Analysis:
- Mood trajectory: neutral → frustrated → resigned
- Session outcome: UNSATISFIED (confidence: 0.85)
- Pain points: ["Failed to provide specific location"]
- Opening strategy: "repair"
```

**Session 2 (Monday 10 AM)**
```
[User reconnects]
AI: "Welcome back. Last time I may not have fully met your expectations. 
     I'll make sure to be extra attentive today. How can I help?"
User: "I need your exact address again"
AI: [Provides detailed, specific location with landmarks]
User: "Perfect, thank you!" (satisfied)

Emotion Engine Analysis:
- Mood trajectory: skeptical → satisfied → happy
- Session outcome: SATISFIED (confidence: 0.80)
- Positive moments: ["Successfully provided detailed location"]
- Opening strategy for next session: "continuity_positive"
```

### Example: Consistent Positive Experience

**Session 1**
```
User: "Book me for tomorrow at 2 PM"
AI: [Books successfully]
User: "Great, thanks!" (happy)

Outcome: HIGHLY_SATISFIED
```

**Session 2 (Next day)**
```
AI: "Great to see you again! Let's keep the momentum going. 
     What would you like to discuss?"
```

---

## 📈 Monitoring & Analytics

### Performance Dashboard

Access `/stats` for real-time metrics:

```json
{
  "performance": {
    "requests": {
      "total": 1247,
      "successful": 1198,
      "failed": 49,
      "success_rate_percent": 96.07
    },
    "timing_ms": {
      "avg_total": 2341,
      "avg_stt": 892,
      "avg_ai_agent": 654,
      "avg_tts": 432
    }
  },
  "connections": {
    "active_connections": 8,
    "emotion_tracked_sessions": 8,
    "continuous_sessions": 3
  }
}
```

### User Emotional Journey

Query `/users/{user_id}/emotion-stats` to see:
- Total sessions over time
- Average sentiment trend
- Outcome distribution (how many satisfied vs frustrated)
- Recommended greeting strategy

---

## 🧪 Testing Emotional Intelligence

### Test Script

```python
import asyncio
import websockets
import json

async def test_emotion_tracking():
    uri = "ws://localhost:5000/ws/voice?user_id=test_user"
    
    async with websockets.connect(uri) as ws:
        # Get connection message with adaptive greeting
        msg = await ws.recv()
        print("Connection:", json.loads(msg))
        
        # Send audio (or simulate)
        # ... send WAV audio bytes
        
        # Get emotion-tracked response
        metadata = await ws.recv()
        audio = await ws.recv()
        
        print("Response metadata:", json.loads(metadata))
    
    # After disconnect, check emotion stats
    import requests
    stats = requests.get("http://localhost:5000/users/test_user/emotion-stats")
    print("Emotion stats:", stats.json())

asyncio.run(test_emotion_tracking())
```

---

## 🔧 Configuration

### Emotion Analysis Settings

Edit `emotion_memory.py`:

```python
# Mood states (add custom states)
class MoodState(Enum):
    HAPPY = "happy"
    CUSTOM_STATE = "custom_state"  # Add your own

# Session outcome thresholds
def _determine_session_outcome(...):
    if outcome_score >= 0.6:  # Adjust threshold
        outcome = SessionOutcome.HIGHLY_SATISFIED
```

### ChromaDB Storage

Emotion data stored in: `./chroma_emotion_memory/`

Two collections:
1. **conversations**: Individual turns with sentiment
2. **sessions**: Complete session summaries

---

## 🎯 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

### Environment Variables for Production

```env
# Use production API keys
GROQ_API_KEY=prod_key
OPENAI_API_KEY=prod_key

# Secure email credentials
SENDER_PASSWORD=app_specific_password

# Production database path
CHROMA_DB_PATH=/var/data/chroma_emotion_memory
```

### Load Balancing Considerations

⚠️ **Important**: Emotional intelligence sessions are stateful!

- Use sticky sessions (session affinity) in load balancer
- Or use Redis for shared session state
- WebSocket connections must stay on same server

---

## 🛡️ Privacy & Ethics

### Data Stored

1. **Conversation turns**: User messages, sentiment scores, mood states
2. **Session summaries**: Outcome classifications, pain points
3. **No PII by default**: Only user_id is required

### GDPR Compliance

```python
# Delete all user data
from emotion_memory import get_emotion_engine

engine = get_emotion_engine()
engine.delete_user_data(user_id="user123")
```

### Ethical Considerations

- Sentiment analysis is **not perfect**—use as guidance, not absolute truth
- Always allow users to provide feedback on accuracy
- Don't make critical decisions based solely on mood detection
- Be transparent that emotion tracking is happening

---

## 🐛 Troubleshooting

### Issue: Emotion sessions not starting

**Solution**: Check GROQ_API_KEY is valid
```bash
# Test Groq connection
python -c "from emotion_memory import get_emotion_engine; get_emotion_engine()"
```

### Issue: Greetings always default

**Solution**: User has no prior sessions
- New users get standard greeting
- After first session completes, adaptive greetings activate

### Issue: ChromaDB errors

**Solution**: Check permissions on `./chroma_emotion_memory/`
```bash
chmod -R 755 ./chroma_emotion_memory/
```

---

## 📚 Advanced Features

### Custom Sentiment Models

Replace Groq with custom model:

```python
def analyze_sentiment(self, text: str) -> Tuple[MoodState, float]:
    # Use your own model
    result = your_custom_sentiment_model(text)
    return MoodState.HAPPY, result.score
```

### Webhook Notifications

Get notified of poor experiences:

```python
def end_session(self, session_id: str, user_id: str):
    summary = self._generate_summary(...)
    
    if summary.session_outcome == SessionOutcome.FRUSTRATED:
        # Send webhook
        requests.post("https://your-webhook.com/alert", json={
            "user_id": user_id,
            "outcome": "frustrated",
            "pain_points": summary.pain_points
        })
```

---

## 🎓 Learning Resources

- [Sentiment Analysis Basics](https://en.wikipedia.org/wiki/Sentiment_analysis)
- [Groq API Documentation](https://console.groq.com/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Empathetic AI Design Principles](https://www.interaction-design.org/literature/topics/empathy)

---

## 📝 Changelog

### v4.0.0 (Current)
- ✨ **NEW**: Emotional Intelligence Memory System
- ✨ Real-time sentiment analysis with Groq
- ✨ Mood trajectory tracking across sessions
- ✨ Adaptive greetings based on history
- ✨ Session outcome classification
- ✨ Pain point identification
- ✨ Automatic repair strategies

### v3.0.0
- User validation system
- Booking integration
- Continuous mode
- Performance monitoring

---

## 🤝 Contributing

To add new emotion states:

1. Edit `MoodState` enum in `emotion_memory.py`
2. Update sentiment analysis prompt
3. Test with various user inputs

---

## 📄 License

MIT License - Use freely in production

---

## 💡 Future Enhancements

- [ ] Multi-language emotion detection
- [ ] Voice tone analysis (prosody)
- [ ] Long-term user personality profiles
- [ ] Predictive mood forecasting
- [ ] A/B testing different greeting strategies
- [ ] Integration with CRM systems
- [ ] Real-time emotion dashboards

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review logs in `./logs/`
3. Test with `/healthz` endpoint

---

**Built with ❤️ for creating genuinely empathetic AI assistants**