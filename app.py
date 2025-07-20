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
    
    state.conversation.append([f"👤 المستخدم: {transcript}", None])  # Left bubble with emoji
    state.conversation.append([None, f"🤖 المختص الآلي: {ai_reply}"])  # Right bubble with emoji
    state.stream = None
    
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
    
    return audio, state, state.conversation

# Custom CSS for better Arabic support and styling
custom_css = """
.gradio-container {
    direction: rtl;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.main-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.section-header {
    background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 10px 15px;
    border-radius: 10px;
    text-align: center;
    margin: 10px 0;
    font-weight: bold;
}

.instructions {
    background: #f8f9ff;
    border: 2px dashed #667eea;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    text-align: center;
}

.footer-note {
    background: #e8f4fd;
    border-left: 4px solid #2196F3;
    padding: 15px;
    border-radius: 5px;
    margin-top: 20px;
    font-style: italic;
}

/* Enhanced chatbot styling */
.chatbot .message {
    border-radius: 15px !important;
    padding: 12px !important;
    margin: 5px 0 !important;
}

/* Audio component styling */
.audio-container {
    border: 2px solid #e3f2fd;
    border-radius: 15px;
    padding: 15px;
    background: linear-gradient(145deg, #f5f7fa, #c3cfe2);
}
"""

with gr.Blocks(css=custom_css, title="🧠 المعالج النفسي الذكي | AI Therapist", theme=gr.themes.Soft()) as demo:
    state = gr.State(AppState())
    
    # Main header
    gr.HTML("""
    <div class="main-header">
        <h1>🧠 المعالج النفسي الذكي</h1>
        <h2>AI Therapy Assistant</h2>
        <p>🌟 مساعدك الشخصي للصحة النفسية والاستشارة | Your Personal Mental Health Companion</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Instructions section
            gr.HTML("""
            <div class="section-header">
                📋 تعليمات الاستخدام | Instructions
            </div>
            <div class="instructions">
                <p><strong>🎤 للتسجيل:</strong> اضغط على زر التسجيل وتحدث بوضوح</p>
                <p><strong>⏹️ للإنهاء:</strong> اضغط على إيقاف التسجيل</p>
                <p><strong>🔊 للاستماع:</strong> سيتم تشغيل الرد تلقائياً</p>
                <hr style="margin: 10px 0;">
                <p><strong>🎤 To Record:</strong> Click record and speak clearly</p>
                <p><strong>⏹️ To Stop:</strong> Click stop recording</p>
                <p><strong>🔊 To Listen:</strong> Response will play automatically</p>
            </div>
            """)
            
            # Audio input section
            gr.HTML("""<div class="section-header">🎙️ التسجيل الصوتي | Voice Recording</div>""")
            input_audio = gr.Audio(
                type="numpy", 
                streaming=True, 
                label="🎤 سجل صوتك هنا | Record your voice here",
                elem_classes=["audio-container"]
            )
            
            # Audio output section
            gr.HTML("""<div class="section-header">🔊 رد المعالج | Therapist Response</div>""")
            output_audio = gr.Audio(
                label="🤖 استمع للرد | Listen to AI Response", 
                autoplay=True,
                elem_classes=["audio-container"]
            )
    
        with gr.Column(scale=2):
            # Conversation section
            gr.HTML("""<div class="section-header">💬 سجل المحادثة | Conversation History</div>""")
            chatbot = gr.Chatbot(
                label="📝 المحادثة | Conversation",
                height=400,
                show_label=False,
                elem_id="chatbot",
                rtl=True
            )
    
    # Footer with additional info
    gr.HTML("""
    <div class="footer-note">
        <p><strong>🔒 الخصوصية:</strong> جميع المحادثات سرية ومحمية</p>
        <p><strong>🔒 Privacy:</strong> All conversations are confidential and secure</p>
        <p><strong>⚠️ تنبيه:</strong> هذا مساعد ذكي وليس بديلاً عن العلاج النفسي المتخصص</p>
        <p><strong>⚠️ Disclaimer:</strong> This is an AI assistant and not a replacement for professional therapy</p>
    </div>
    """)
    
    # Event handlers (unchanged)
    input_audio.stream(
        process_audio, 
        [input_audio, state], 
        [input_audio, state], 
        stream_every=1, 
        queue=False
    )
    input_audio.stop_recording(
        response, 
        [state], 
        [output_audio, state, chatbot]
    )

app = gr.mount_gradio_app(app, demo, path="/gradio")

# Redirect root '/' to '/gradio'
@app.get("/")
def redirect_to_gradio():
    return RedirectResponse(url="/gradio")