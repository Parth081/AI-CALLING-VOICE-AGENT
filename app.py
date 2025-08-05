import asyncio
import base64
import wave
import tempfile
from flask import Flask, request, Response
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Start, Stream
from threading import Thread
import websockets
from stt_handler import transcribe_audio
from llm_handler import query_llama3
from tts_handler import text_to_speech

load_dotenv()
app = Flask(__name__)

# Twilio webhook: Answer call and start media stream
@app.route("/voice", methods=["POST"])
def voice():
    resp = VoiceResponse()
    # Use ws:// for local testing, wss:// for production with SSL
    stream = Stream(url=f"ws://{request.host}/media")
    start = Start()
    start.append(stream)
    resp.append(start)
    resp.say("Hello, this is the AI real estate agent. How can I help you?")
    return Response(str(resp), mimetype="text/xml")

# Real-time WebSocket handler for Twilio media
async def handle_media(websocket, path):
    print("WebSocket connected.")
    audio_chunks = []
    while True:
        try:
            msg = await websocket.recv()
            import json
            data = json.loads(msg)
            if data.get("event") == "media":
                audio_b64 = data["media"]["payload"]
                audio_bytes = base64.b64decode(audio_b64)
                audio_chunks.append(audio_bytes)
            elif data.get("event") == "stop":
                break
        except Exception as e:
            print("Error in WebSocket:", e)
            break

    # Save audio to .wav
    if audio_chunks:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
            wf = wave.open(wav_path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            for chunk in audio_chunks:
                wf.writeframes(chunk)
            wf.close()

        # Transcribe
        user_utterance = transcribe_audio(wav_path)
        print("Caller:", user_utterance)
        if user_utterance.strip():
            ai_reply = query_llama3(user_utterance)
            print("AI:", ai_reply)
            # TTS
            tts_path = text_to_speech(ai_reply)
            # Respond by sending <Play> TwiML with the generated audio (if using Twilio <Play> out-of-band)
        else:
            print("No audio received or transcribed.")
    else:
        print("No audio received.")

def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

def run_websocket():
    asyncio.set_event_loop(asyncio.new_event_loop())
    start_server = websockets.serve(handle_media, "0.0.0.0", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    Thread(target=run_flask, daemon=True).start()
    Thread(target=run_websocket, daemon=True).start()
    # Keep main thread alive
    while True:
        pass