from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from Speaker import Speaker
from WhisperTranscriber import WhisperTranscriber

from Agent import chat_map
from main import therapist_chat

import numpy as np
import uuid
import asyncio
import gradio as gr

app = FastAPI()
transcriber = WhisperTranscriber()
speaker = Speaker()


class AppState:
    stream: np.ndarray | None = None
    session_id: str = str(uuid.uuid4()).replace('-', '')[:10]
    sampling_rate: int = 0
    conversation: list = []
    emergency_contacts: list = []

def add_contact(new_contact, state):
    if new_contact.strip():
        state.emergency_contacts.append(new_contact.strip())
    return '\n'.join(state.emergency_contacts), state

def delete_contacts(state):
    state.emergency_contacts = []
    return None, state

def estimate_audio_duration(audio_bytes: bytes, sampling_rate: int = 16000) -> float:
    # Assuming 16-bit mono PCM: 2 bytes per sample
    byte_count = len(audio_bytes)
    return byte_count / (sampling_rate * 2)

def clear_conversation(state):
    state.conversation = []
    state.stream = None
    chat_map[state.session_id].clear()
    return None, state.conversation, state

def process_audio(audio, state):
    sr, chunk = audio
    if state.stream is None:
        state.stream = chunk
        state.sampling_rate = sr
    else:
        state.stream = np.concatenate((state.stream, chunk))
    return None, state  # Just update the stream

# 🔊 Run blocking speaker.speak() in async thread
async def speak_chunk(text):
    return await asyncio.to_thread(speaker.speak, text)

# 🚀 Response loop with async streaming playback
async def response(audio, state):
    if state.stream is None or len(state.stream) == 0:
        yield None, state, state.conversation

    user_prompt = transcriber.TranscribeStream((state.sampling_rate, state.stream))
    state.conversation.append([user_prompt, "....."])
    yield None, state, state.conversation

    state.conversation[-1][1] = ""  # Clear placeholder

    buffer = ""
    buffer_threshold = 1000  # Batching threshold
    reply = ""

    async for token in therapist_chat(user_prompt, timeout=12, session_id= state.session_id, emergency_contacts= state.emergency_contacts):
        buffer += token.content
        state.conversation[-1][1] += token.content  # Update chatbot text live

        if len(buffer) > buffer_threshold or buffer.endswith((".", "!", "?", "،",'\n')):
            reply += buffer
            audio_chunk = await speak_chunk(buffer)
            buffer = ""
            yield audio_chunk, state, state.conversation
            duration = estimate_audio_duration(audio_chunk, state.sampling_rate)
            await asyncio.sleep(duration * 2.5)  # Slightly less for smoother UX


    # Catch leftover buffer
    if buffer:
        reply += buffer
        audio_chunk = await speak_chunk(buffer)
        state.conversation[-1][1] += buffer
        yield audio_chunk, state, state.conversation

    state.stream = None  # Reset stream for next round


with gr.Blocks() as demo:

    state = gr.State(AppState())
    input_audio = gr.Audio(type="numpy", streaming=True, label="Record your voice")
    output_audio = gr.Audio(label="AI Response", autoplay=True)
    chatbot = gr.Chatbot()
    clear_btn = gr.Button("🔄 Clear Conversation")
    gr.Markdown("### 🆘 Emergency Contacts")
    with gr.Row():
        contact_input = gr.Textbox(placeholder="Enter emergency contact (e.g., name & number)", label="Add Contact")
        add_btn = gr.Button("➕ Add Contact")
        delete_btn = gr.Button("Clear Contacts")
    contacts_display = gr.Textbox(label="Saved Contacts", interactive=False)

    add_btn.click(fn=add_contact, inputs=[contact_input, state], outputs=[contacts_display, state])
    delete_btn.click(fn=delete_contacts, inputs = [state], outputs= [contacts_display, state])
    input_audio.stream(process_audio, [input_audio, state], [input_audio, state], stream_every = 0.5, queue=False)
    input_audio.stop_recording(response, [input_audio,state], [output_audio, state, chatbot])
    clear_btn.click(fn=clear_conversation, inputs=[state], outputs=[output_audio, chatbot, state])

# demo.launch()
app = gr.mount_gradio_app(app, demo, path="/gradio")

# Redirect root '/' to '/gradio'
@app.get("/")
def redirect_to_gradio():
    return RedirectResponse(url="/gradio")