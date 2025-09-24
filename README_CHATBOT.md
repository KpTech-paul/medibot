# MediBot Chatbot Setup Guide

This guide will help you set up the MediBot chatbot feature with ML model integration, voice input, and text-to-speech capabilities.

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Flask and required packages** (will be installed via requirements.txt)
3. **Flutter app** with the updated dependencies
4. **XAMPP** running with the backend

## Setup Instructions

### 1. Install Python Dependencies

Navigate to the modal directory and install the required packages:

```bash
cd C:\xampp\htdocs\backend\modal
pip install -r requirements.txt
```

### 2. Start the Flask Chatbot API

You can start the chatbot API in two ways:

**Option A: Using the batch file (Windows)**
```bash
# Double-click start_chatbot.bat or run:
start_chatbot.bat
```

**Option B: Using Python directly**
```bash
cd C:\xampp\htdocs\backend\modal
python chatbot_api.py
```

The API will start on `http://localhost:5000`

### 3. Update Flutter Dependencies

In your Flutter project root, run:

```bash
flutter pub get
```

### 4. Test the System

1. **Start XAMPP** and ensure Apache is running
2. **Start the Flask chatbot API** (step 2 above)
3. **Run your Flutter app**:
   ```bash
   flutter run
   ```

### 5. Using the Chatbot

1. **Open the patient dashboard** in your Flutter app
2. **Tap the chat icon** in the top-right corner of the AppBar
3. **Chat with MediBot** using either:
   - **Text input**: Type your questions about heart health
   - **Voice input**: Tap the microphone button to speak your questions

## Features

### 🤖 **Intelligent Responses**
- **ML Model**: Uses your trained Keras model for heart health questions
- **Gemini Fallback**: When confidence < 95%, uses Gemini API for general questions
- **Confidence Scoring**: Shows prediction confidence for transparency

### 🎤 **Voice Input**
- **Speech-to-Text**: Convert spoken questions to text
- **Real-time Processing**: Listen for up to 30 seconds
- **Visual Feedback**: Button changes to show listening state

### 🔊 **Text-to-Speech**
- **Audio Responses**: Bot can speak responses aloud
- **Volume Control**: Adjustable speech rate and volume
- **Playback Control**: Stop/start speaking

### 💬 **Chat Interface**
- **Modern UI**: Clean, intuitive chat interface
- **Message History**: Scrollable conversation history
- **Loading States**: Visual feedback during processing
- **Error Handling**: Graceful error messages

## API Endpoints

### Flask API (Port 5000)
- `POST /chat` - Send message to chatbot
- `GET /health` - Health check
- `GET /intents` - List available intents

### PHP Backend (XAMPP)
- `POST /backend/chatbot/chat` - Proxy to Flask API
- `GET /backend/chatbot/health` - Health check

## Troubleshooting

### Flask API Won't Start
```bash
# Check if port 5000 is in use
netstat -an | findstr :5000

# If in use, kill the process or change port in chatbot_api.py
```

### Model Loading Errors
- Ensure all model files are in the modal directory:
  - `medibot_model.keras`
  - `tokenizer.json`
  - `labels.npy`
  - `intents.json`

### Voice Recognition Issues
- **Android**: Grant microphone permission in app settings
- **iOS**: Grant microphone permission when prompted
- **Desktop**: Ensure microphone is connected and working

### TTS Issues
- **Android**: Install Google TTS or Samsung TTS
- **iOS**: Uses system TTS (should work out of the box)
- **Desktop**: May need additional TTS engine

## Configuration

### Changing the Gemini API Key
Edit `C:\xampp\htdocs\backend\modal\chatbot_api.py`:
```python
GEMINI_API_KEY = "your-new-api-key-here"
```

### Adjusting Confidence Threshold
Edit the threshold in `chatbot_api.py`:
```python
def chatbot(user_input, threshold=0.95):  # Change 0.95 to your desired threshold
```

### Modifying Intents
Edit `intents.json` to add new intents or modify existing ones.

## File Structure

```
C:\xampp\htdocs\backend\modal\
├── chatbot_api.py          # Flask API server
├── requirements.txt        # Python dependencies
├── start_chatbot.bat      # Windows startup script
├── medibot_model.keras    # Trained ML model
├── tokenizer.json         # Text tokenizer
├── labels.npy            # Label encoder
├── intents.json          # Intent definitions
└── README_CHATBOT.md     # This file

lib/src/screens/
└── chatbot_screen.dart   # Flutter chat interface
```

## Support

If you encounter any issues:

1. **Check logs** in the Flask API console
2. **Verify permissions** for microphone and network access
3. **Test API endpoints** using Postman or curl
4. **Check Flutter console** for error messages

## Example Usage

**User**: "What are the symptoms of a heart attack?"
**Bot**: "Common symptoms include chest pain, shortness of breath, and nausea. Heart attack symptoms typically include chest pain or pressure, shortness of breath, nausea, and pain radiating to the arm, jaw, or back."

**User**: "How can I prevent heart disease?"
**Bot**: "Maintain a healthy diet, exercise regularly, and avoid smoking. Eating a balanced diet, exercising regularly, and managing stress can help. To reduce heart attack risk: eat healthy, exercise regularly, avoid smoking, manage stress, and control blood pressure."
