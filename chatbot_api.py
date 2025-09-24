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

app = Flask(__name__)
CORS(app)

# ---------------- Paths ----------------
MODEL_PATH = "medibot_model.keras"
TOKENIZER_PATH = "tokenizer.json"
LABEL_ENCODER_PATH = "labels.npy"
INTENTS_PATH = "intents.json"

# ---------------- Load Model & Artifacts ----------------
print("Loading model and artifacts...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "r") as f:
    tokenizer_json = f.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

lbl_encoder = LabelEncoder()
lbl_encoder.classes_ = np.load(LABEL_ENCODER_PATH, allow_pickle=True)

with open(INTENTS_PATH, "r") as f:
    data = json.load(f)
responses = {item["tag"]: item["responses"] for item in data["intents"]}

print("Model and artifacts loaded successfully!")

# ---------------- Gemini API ----------------
GEMINI_API_KEY = "AIzaSyCPulfljNVHbBn5VzqCO0Py_y3zHDwmSxg"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def call_gemini(user_input):
    """Call Gemini API as fallback when model confidence is low"""
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [
            {"parts": [{"text": f"You are a medical assistant specializing in heart health. Please provide helpful, accurate information about heart health, heart attacks, and related medical topics. Keep responses concise and informative. User question: {user_input}"}]}
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

def chatbot(user_input, threshold=0.95):
    """Main chatbot function that uses ML model or Gemini fallback"""
    try:
        # Tokenize and pad input
        seq = tokenizer.texts_to_sequences([user_input])
        padded_seq = pad_sequences(seq, maxlen=model.input_shape[1], padding="post")
        
        # Get prediction
        pred = model.predict(padded_seq, verbose=0)
        tag_index = np.argmax(pred)
        confidence = pred[0][tag_index]

        print(f"DEBUG: Confidence = {confidence:.3f}")
        print(f"Predicted tag: {lbl_encoder.inverse_transform([tag_index])[0]}")

        # Use model prediction if confidence is high enough
        if confidence >= threshold:
            tag = lbl_encoder.inverse_transform([tag_index])[0]
            response = random.choice(responses[tag])
            return {
                "response": response,
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
        'intents': list(responses.keys())
    })

if __name__ == '__main__':
    print("Starting MediBot Chatbot API...")
    print("Available intents:", list(responses.keys()))
    app.run(host='0.0.0.0', port=5000, debug=True)
