import requests
import os
from flask import Flask, request, jsonify
import asyncio
import websockets
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
from gtts import gTTS
import speech_recognition as sr

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

app = Flask(__name__)

def query_llama3(prompt):
    # Dummy response for testing
    return "This is a sample AI response."

@app.route("/")
def home():
    return "Welcome to the AI chatbot server!"

@app.route("/webhook", methods=["POST"])
def webhook():
    # Get the incoming message from WhatsApp
    incoming_msg = request.values.get("Body", "").strip()
    resp = MessagingResponse()

    if incoming_msg:
        # Query the AI model with the incoming message
        answer = query_llama3(incoming_msg)
        resp.message(answer)

    return str(resp)

async def handle_connection(websocket, path):
    async for message in websocket:
        # Here you would handle incoming messages from the WebSocket
        print(f"Received message: {message}")
        # For now, let's just echo the message back
        await websocket.send(f"Echo: {message}")

start_server = websockets.serve(handle_connection, "localhost", 8765)

if __name__ == "__main__":
    # Run the WebSocket server in the background
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    loop.run_forever()

    app.run(port=5000)