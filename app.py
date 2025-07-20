import gradio as gr
import numpy as np
import os
import tempfile
from scipy.io.wavfile import write
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from therapist import EnhancedOmaniTherapyApp, Config

app = FastAPI()
omaniTherapyApp = EnhancedOmaniTherapyApp(Config())

class AppState:
    stream: np.ndarray | None = None
    sampling_rate: int = 0
    conversation: list = []
    lock = False  # To prevent concurrent access issues

def process_audio(audio, state):
    sr, chunk = audio
    if state.stream is None:
        state.stream = chunk
        state.sampling_rate = sr
    else:
        state.stream = np.concatenate((state.stream, chunk))

    return None, state  # No need to stop based on length now

def response(state):
    if state.stream is None or len(state.stream) == 0:
        return None, state, state.conversation

    # audio = (state.sampling_rate, state.stream)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        write(temp_audio.name, state.sampling_rate, state.stream)
        temp_audio_path = temp_audio.name


    transcript, ai_reply = omaniTherapyApp._process_audio_pipeline(temp_audio_path)
    audio = omaniTherapyApp.tts.speak(ai_reply)


    state.conversation.append([f"المستخدم: {transcript}", None])  # Left bubble
    state.conversation.append([None, f"المختص الآلي: {ai_reply}"])  # Right bubble
    state.stream = None

    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    return audio, state, state.conversation

with gr.Blocks() as demo:
    state = gr.State(AppState())
    input_audio = gr.Audio(type="numpy", streaming=True, label="Record your voice")
    output_audio = gr.Audio(label="AI Response", autoplay=True)
    chatbot = gr.Chatbot()

    input_audio.stream(process_audio, [input_audio, state], [input_audio, state], stream_every= 1, queue=False)
    input_audio.stop_recording(response, [state], [output_audio, state, chatbot])
    #output_audio.stop(lambda s: gr.Audio(recording=True), [state], [input_audio])

app = gr.mount_gradio_app(app, demo, path="/gradio")

# Redirect root '/' to '/gradio'
@app.get("/")
def redirect_to_gradio():
    return RedirectResponse(url="/gradio")

