# 🇴🇲 Omani Voice Therapist

A culturally-sensitive, voice-only therapeutic chatbot tailored for Omani Arabic mental health support.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)

## 🌟 Overview

The Omani Voice Therapist is a specialized mental health support tool designed to provide culturally-appropriate therapeutic conversations in Omani Arabic. This voice-only interface ensures accessibility and comfort for users seeking mental health support in their native language and cultural context.

## ✨ Features

- **Voice-Only Interface**: Natural conversation flow without text barriers
- **Cultural Sensitivity**: Tailored for Omani cultural context and values
- **Arabic Language Support**: Specialized for Omani Arabic dialect
- **Privacy-Focused**: Secure and confidential therapeutic conversations
- **Local Testing**: Complete local development environment

## 🧰 Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python 3.11+**
- **Git**
- **Virtual environment tool** (`venv` or `conda`)
- **pip** (Python package manager)

## 🚀 Installation

### Quick Setup (Copy and paste these commands):

```bash
# Create and activate virtual environment
python -m venv omani_env
source omani_env/bin/activate    # On Linux/macOS
# omani_env\Scripts\activate     # Uncomment for Windows

# Clone the repository
git clone -b ForLocalTesting https://github.com/AhmadHazem/omani-voice-therapist.git
cd omani-voice-therapist

# Install dependencies
pip install -r requirements.txt

# Run the application
python ./app.py
```

### Alternative with Conda:

```bash
# Create and activate conda environment
conda create -n omani_env python=3.11
conda activate omani_env

# Clone the repository
git clone -b ForLocalTesting https://github.com/yourusername/omani-voice-therapist.git
cd omani-voice-therapist

# Install dependencies
pip install -r requirements.txt

# Run the application
python ./app.py
```

## 🎯 Usage

1. **Start the Application**: Run `python ./app.py`
2. **Voice Interaction**: The system will activate voice input/output
3. **Cultural Context**: Speak naturally in Omani Arabic
4. **Therapeutic Session**: Engage in supportive conversation

## 📁 Project Structure

```
omani-voice-therapist/
├── app.py                 # Main application entry point
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── dockerfile            # dockerfile
├── docker-compose        # docker compose file

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# API Keys (if required)
OPENAI_API_KEY=your_api_key_here
SPEECH_API_KEY=your_speech_key_here

# Language Settings
DEFAULT_LANGUAGE=ar-OM
CULTURAL_CONTEXT=omani

# Audio Settings
SAMPLE_RATE=16000
AUDIO_FORMAT=wav
```

### Audio Configuration

Ensure your microphone and speakers are properly configured for optimal voice interaction.




**Note**: This tool is designed to supplement, not replace, professional mental health care. Please consult qualified mental health professionals for serious concerns.
