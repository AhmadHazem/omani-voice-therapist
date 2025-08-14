# 🇴🇲 Omani Voice Therapist

A culturally-sensitive, voice-only therapeutic chatbot tailored for Omani Arabic mental health support.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)

## 🌟 Overview

The Omani Voice Therapist is a specialized mental health support tool designed to provide culturally-appropriate therapeutic conversations in Omani Arabic. This voice-only interface ensures accessibility and comfort for users seeking mental health support in their native language and cultural context.

## ✨ Features

- **Voice-Only Interface**: Natural conversation flow without text barriers
- **Cultural Sensitivity**: Tailored for Omani cultural context and values
- **Arabic Language Support**: Specialized for Omani Arabic dialect
- **Privacy-Focused**: Secure and confidential therapeutic conversations
- **Local Testing**: Complete local development environment
- **Memory**: Persistent conversational memory for context-aware responses
- **Tavily Tools**: Integrated for enhanced information retrieval and workflow automation
- **LangChain API**: Orchestrates multi-agent and tool-based workflows
- **Multi-Model Support**: GPT-4o, Claude Sonnet 3.7, Whisper-1, and Azure TTS Omani voice
- **Gradio & FastAPI**: Interactive web UI and robust backend API
- **CI/CD**: Automated deployment on Azure App Service
- **Azure Container Registry**: Containerized builds and secure image storage

## 🛠️ Tech Stack

- **Memory**: Custom or LangChain memory for session context
- **Tavily Tools**: Automated tools for enhanced agent capabilities
- **LangChain API**: Agent orchestration and tool integration
- **GPT-4o**: Advanced conversational AI
- **Claude Sonnet 3.7**: Alternative LLM for nuanced responses
- **Whisper-1**: Speech-to-text for voice input
- **Azure TTS (Omani Voice)**: Natural Omani Arabic voice output
- **Gradio**: User-friendly web interface
- **FastAPI**: High-performance backend API
- **CI/CD**: Azure App Service for automated deployment
- **Azure Container Registry**: Container image management

## 🧰 Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python 3.11+**
- **Git**
- **Virtual environment tool** (`venv` or `conda`)
- **pip** (Python package manager)
- **Docker** (for containerization)
- **Azure CLI** (for deployment)

## 🚀 Installation

### Quick Setup (Copy and paste these commands):

```bash
# Create and activate virtual environment
python -m venv omani_env
source omani_env/bin/activate    # On Linux/macOS
# omani_env\Scripts\activate     # Uncomment for Windows

# Clone the repository
git clone https://github.com/AhmadHazem/omani-voice-therapist.git
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
2. **Voice Interaction**: The system will activate voice input/output using Whisper-1 and Azure TTS Omani voice
3. **Cultural Context**: Speak naturally in Omani Arabic
4. **Therapeutic Session**: Engage in supportive conversation
5. **Web Interface**: Access via Gradio for interactive sessions
6. **API Access**: Integrate with FastAPI endpoints for advanced workflows

## 📁 Project Structure

```
omani-voice-therapist/
├── app.py                 # Main application entry point
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── dockerfile             # Dockerfile for containerization
├── docker-compose.yml     # Docker Compose configuration
├── .github/workflows/     # CI/CD pipelines for Azure App Service
├── src/                   # Source code (agents, tools, API, UI)
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# API Keys (if required)
OPENAI_API_KEY=your_api_key_here
CLAUDE_API_KEY=your_claude_key_here
SPEECH_API_KEY=your_speech_key_here
AZURE_TTS_KEY=your_azure_tts_key_here

# Language Settings
DEFAULT_LANGUAGE=ar-OM
CULTURAL_CONTEXT=omani

# Audio Settings
SAMPLE_RATE=16000
AUDIO_FORMAT=wav

# LangChain & Tavily
LANGCHAIN_API_KEY=your_langchain_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### Audio Configuration

Ensure your microphone and speakers are properly configured for optimal voice interaction.

## 🚢 CI/CD & Deployment

- **Azure App Service**: Automated deployment via GitHub Actions
- **Azure Container Registry**: Store and manage Docker images
- **Docker**: Build and run containers locally or in the cloud

**Note**: This tool is designed to supplement, not replace, professional mental health care. Please consult qualified mental health professionals for serious concerns.

