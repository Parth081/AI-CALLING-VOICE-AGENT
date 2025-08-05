# AI-CALLING-VOICE-AGENT
This project is a fully open-source, cost-effective voice AI assistant designed to handle inbound phone calls using a toll-free number. It transcribes speech, understands the caller’s intent using a locally hosted Large Language Model (LLM), and responds naturally using text-to-speech — all without relying on expensive APIs or cloud services. 
# Real-Time Voice Agent AI

A real-time voice assistant using Flask, Twilio, websockets, speech-to-text, LLM, and gTTS.

## Features

- Receives calls via Twilio
- Real-time audio streaming and transcription
- AI-powered responses (LLM)
- Text-to-speech with gTTS

## Setup

1. Clone the repo:
   ```
   git clone https://github.com/yourusername/voice-agent.git
   cd voice-agent
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Add a `.env` file with your Twilio credentials and agent prompt.

4. Run the app:
   ```
   python app.py
   ```

## File Structure

- `app.py` — Main server and websocket logic
- `stt_handler.py` — Speech-to-text
- `llm_handler.py` — LLM query logic
- `tts_handler.py` — Text-to-speech (gTTS)
- `requirements.txt` — Python dependencies

## License

MIT
