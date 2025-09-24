from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import requests
import os
import time
import wave
import subprocess
import shutil
import contextlib

try:
    import speech_recognition as sr
    _sr_available = True
except Exception:
    _sr_available = False

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    _vosk_available = True
except Exception:
    _vosk_available = False

# Optional Vosk model path (download a small en model and set this path)
VOSK_MODEL_PATH = os.environ.get('VOSK_MODEL_PATH', os.path.join('models', 'vosk-model-small-en-us-0.15'))
_vosk_model = None

app = Flask(__name__)
CORS(app)

# ---------------- Paths ----------------
MODEL_PATH = "medibot_model.keras"
TOKENIZER_PATH = "tokenizer.json"
LABEL_ENCODER_PATH = "labels.npy"
INTENTS_PATH = "intents.json"

# ---------------- Load Model & Artifacts ----------------
print("Loading model and artifacts...")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Load tokenizer
with open(TOKENIZER_PATH, "r") as f:
    tokenizer_json = f.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

# Load label encoder
lbl_encoder = LabelEncoder()
lbl_encoder.classes_ = np.load(LABEL_ENCODER_PATH, allow_pickle=True)

# Load intents
with open(INTENTS_PATH, "r") as f:
    data = json.load(f)

responses = {item["tag"]: item["responses"] for item in data["intents"]}

print("Model and artifacts loaded successfully!")
print(f"Available intents: {lbl_encoder.classes_}")

# ---------------- Gemini API ----------------
GEMINI_API_KEY = "AIzaSyCPulfljNVHbBn5VzqCO0Py_y3zHDwmSxg"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def call_gemini(user_input):
    """Call Gemini API as fallback when model confidence is low"""
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [
            {"parts": [{"text": f"""You are MediBot, a medical assistant specializing in heart health. Please provide helpful, accurate information about heart health, heart attacks, and related medical topics. 

IMPORTANT FORMATTING RULES:
- For lists, use bullet points with "•" symbol
- Keep responses concise but informative
- Focus on heart health topics
- If the question is not about heart health, politely redirect to heart health topics
- Use clear, simple language

User question: {user_input}"""}]}
        ]
    }
    params = {"key": GEMINI_API_KEY}
    try:
        response = requests.post(GEMINI_URL, headers=headers, params=params, json=body, timeout=10)
        resp_json = response.json()
        if "candidates" in resp_json and len(resp_json["candidates"]) > 0:
            text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
            return text
        else:
            return "I'm sorry, I couldn't process that request. Please try rephrasing your question."
    except Exception as e:
        print("Gemini API error:", e)
        return "I'm sorry, I'm having trouble connecting to my knowledge base. Please try again later."

def format_response(response_text):
    """Format the response with bullet points for lists"""
    # Check if the response contains list-like content
    if any(keyword in response_text.lower() for keyword in ['symptoms', 'signs', 'causes', 'risk factors', 'prevention', 'steps', 'tips', 'include', 'are:', 'include:']):
        # Split by common list indicators and format with bullet points
        lines = response_text.split('. ')
        formatted_lines = []
        for line in lines:
            if line.strip():
                # If line looks like a list item, add bullet point
                if any(indicator in line.lower() for indicator in ['chest pain', 'shortness', 'nausea', 'dizziness', 'sweating', 'fatigue', 'pressure', 'discomfort', 'pain in', 'feeling of']):
                    formatted_lines.append(f"• {line.strip()}")
                else:
                    formatted_lines.append(line.strip())
        return '. '.join(formatted_lines)
    return response_text

def chatbot(user_input, threshold=0.95):
    """Main chatbot function that uses ML model or Gemini fallback - based on working implementation"""
    try:
        # Tokenize and pad input
        seq = tokenizer.texts_to_sequences([user_input])
        padded_seq = pad_sequences(seq, maxlen=model.input_shape[1], padding="post")
        
        # Get prediction
        pred = model.predict(padded_seq, verbose=0)
        tag_index = np.argmax(pred)
        confidence = pred[0][tag_index]

        # Debug logs like in the working version
        print("DEBUG:", {lbl_encoder.inverse_transform([i])[0]: round(float(p), 3)
                        for i, p in enumerate(pred[0])})
        print(f"→ Predicted: {lbl_encoder.inverse_transform([tag_index])[0]} "
              f"(conf={confidence:.2f})")

        # Use model prediction if confidence is high enough
        if confidence >= threshold:
            tag = lbl_encoder.inverse_transform([tag_index])[0]
            response = random.choice(responses[tag])
            formatted_response = format_response(response)
            return {
                "response": formatted_response,
                "confidence": float(confidence),
                "source": "model",
                "tag": tag
            }
        else:
            # Use Gemini as fallback
            print("Using Gemini fallback due to low confidence")
            response = call_gemini(user_input)
            return {
                "response": response,
                "confidence": float(confidence),
                "source": "gemini",
                "tag": "fallback"
            }
    except Exception as e:
        print(f"Chatbot error: {e}")
        return {
            "response": "I'm sorry, I encountered an error processing your request. Please try again.",
            "confidence": 0.0,
            "source": "error",
            "tag": "error"
        }

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({
                'success': False,
                'error': 'Message is required'
            }), 400
        
        # Get chatbot response
        result = chatbot(user_input)
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'confidence': result['confidence'],
            'source': result['source'],
            'tag': result['tag']
        })
        
    except Exception as e:
        print(f"API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/voice/stt', methods=['POST'])
def voice_stt():
    """Simple STT upload endpoint placeholder.
    Accepts multipart form field 'file' (or 'audio'), saves temp, returns dummy text.
    """
    try:
        uploaded = request.files.get('file') or request.files.get('audio')
        if not uploaded:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        os.makedirs('tmp', exist_ok=True)
        ext = os.path.splitext(uploaded.filename or '')[1] or '.m4a'
        temp_path = os.path.join('tmp', f'stt_{int(time.time()*1000)}{ext}')
        uploaded.save(temp_path)

        recognized_text = ''

        # First try SpeechRecognition with Google's free web API (requires internet)
        if _sr_available:
            try:
                # Ensure mono 16k WAV for best results
                wav_path_sr = os.path.join('tmp', f'stt_{int(time.time()*1000)}_sr.wav')
                ffmpeg = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
                if ffmpeg:
                    subprocess.run([ffmpeg, '-y', '-i', temp_path, '-ac', '1', '-ar', '16000', wav_path_sr],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    use_path = wav_path_sr
                else:
                    use_path = temp_path

                recognizer = sr.Recognizer()
                with contextlib.ExitStack() as stack:
                    audio_file = stack.enter_context(sr.AudioFile(use_path))
                    audio_data = recognizer.record(audio_file)
                google_text = recognizer.recognize_google(audio_data)
                if google_text and google_text.strip():
                    recognized_text = google_text.strip()
                try:
                    if os.path.exists(wav_path_sr):
                        os.remove(wav_path_sr)
                except Exception:
                    pass
            except Exception as e:
                print(f"SpeechRecognition (Google) error: {e}")

        # If SR not available or failed, try Vosk if present
        if (not recognized_text) and _vosk_available and os.path.isdir(VOSK_MODEL_PATH):
            global _vosk_model
            if _vosk_model is None:
                print(f"Loading Vosk model from: {VOSK_MODEL_PATH}")
                _vosk_model = VoskModel(VOSK_MODEL_PATH)

            wav_path = os.path.join('tmp', f'stt_{int(time.time()*1000)}.wav')
            ffmpeg = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
            try:
                if ffmpeg:
                    # Convert to 16kHz mono WAV PCM
                    subprocess.run([ffmpeg, '-y', '-i', temp_path, '-ac', '1', '-ar', '16000', wav_path],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    with wave.open(wav_path, 'rb') as wf:
                        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                            raise RuntimeError('Invalid WAV format after conversion')
                        rec = KaldiRecognizer(_vosk_model, wf.getframerate())
                        rec.SetWords(False)
                        text_parts = []
                        while True:
                            data = wf.readframes(4000)
                            if len(data) == 0:
                                break
                            if rec.AcceptWaveform(data):
                                res = json.loads(rec.Result())
                                if res.get('text'):
                                    text_parts.append(res['text'])
                        final = json.loads(rec.FinalResult())
                        if final.get('text'):
                            text_parts.append(final['text'])
                        joined = ' '.join(t for t in text_parts if t)
                        if joined.strip():
                            recognized_text = joined.strip()
                else:
                    print('ffmpeg not found; skipping Vosk transcription')
            except Exception as e:
                print(f"Vosk transcription error: {e}")
            finally:
                try:
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                except Exception:
                    pass

        try:
            os.remove(temp_path)
        except Exception:
            pass

        return jsonify({'success': True, 'text': recognized_text}), 200
    except Exception as e:
        print(f"STT endpoint error: {e}")
        return jsonify({'success': False, 'error': 'STT failed'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'message': 'Chatbot API is running'
    })

@app.route('/intents', methods=['GET'])
def get_intents():
    """Get available intents for debugging"""
    return jsonify({
        'success': True,
        'intents': lbl_encoder.classes_.tolist()
    })

@app.route("/", methods=["GET"])
def root():
    """Root endpoint for Render check"""
    return jsonify({
        "success": True,
        "message": "MediBot API is live and running!",
        "endpoints": ["/chat", "/voice/stt", "/health", "/intents"]
    })

application = app
