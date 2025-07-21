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
    emergency_contacts: list = []
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
    # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
    #     write(temp_audio.name, state.sampling_rate, state.stream)
    #     temp_audio_path = temp_audio.name
    
    transcript, ai_reply = omaniTherapyApp._process_audio_pipeline((state.sampling_rate, state.stream))
    audio = omaniTherapyApp.tts.speak(ai_reply)
    
    state.conversation.append([f"👤 المستخدم: {transcript}", None])  # Left bubble with emoji
    state.conversation.append([None, f"🤖 المختص الآلي: {ai_reply}"])  # Right bubble with emoji
    state.stream = None
    
    # if os.path.exists(temp_audio_path):
    #     os.remove(temp_audio_path)
    
    return audio, state, state.conversation

def add_emergency_contact(email, state):
    """Add an emergency contact email to the state"""
    if email and email.strip():
        email = email.strip()
        if email not in state.emergency_contacts:
            state.emergency_contacts.append(email)
            omaniTherapyApp.add_emergency_contact(state.emergency_contacts)
            return "", state, format_contact_list(state.emergency_contacts), gr.update(visible=True)
    return email, state, format_contact_list(state.emergency_contacts), gr.update(visible=len(state.emergency_contacts) > 0)

def remove_emergency_contact(contact_to_remove, state):
    """Remove an emergency contact email from the state"""
    if contact_to_remove in state.emergency_contacts:
        state.emergency_contacts.remove(contact_to_remove)
        omaniTherapyApp.remove_emergency_contact(state.emergency_contacts)
    return state, format_contact_list(state.emergency_contacts), gr.update(visible=len(state.emergency_contacts) > 0)

def format_contact_list(contacts):
    """Format the emergency contacts list for display"""
    if not contacts:
        return "لم يتم إضافة أي جهات اتصال طوارئ بعد | No emergency contacts added yet"
    
    formatted = "📧 جهات الاتصال للطوارئ | Emergency Contacts:\n"
    for i, contact in enumerate(contacts, 1):
        formatted += f"{i}. {contact}\n"
    return formatted

def clear_conversation(state):
    """Clear the conversation history"""
    state.conversation = []
    omaniTherapyApp.clear_conversation()
    return state, []

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

.emergency-section {
    background: linear-gradient(90deg, #ff6b6b 0%, #ee5a24 100%);
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

.emergency-info {
    background: #fff3cd;
    border: 2px solid #ffc107;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    text-align: center;
}

.contact-list {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 10px;
    margin: 5px 0;
    font-family: monospace;
    white-space: pre-line;
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

/* Emergency contact styling */
.emergency-container {
    border: 2px solid #ff6b6b;
    border-radius: 15px;
    padding: 15px;
    background: linear-gradient(145deg, #fff5f5, #ffe8e8);
}

/* Clear button styling */
.clear-button {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
}

.clear-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4) !important;
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
    
    # Emergency contacts section (at the top for visibility)
    gr.HTML("""<div class="emergency-section">🚨 جهات الاتصال للطوارئ | Emergency Contacts</div>""")
    
    with gr.Row():
        with gr.Column():
            gr.HTML("""
            <div class="emergency-info">
                <p><strong>⚠️ مهم:</strong> أضف جهات اتصال يمكن التواصل معها في حالات الطوارئ</p>
                <p><strong>⚠️ Important:</strong> Add contacts that can be reached in emergency situations</p>
            </div>
            """)
            
            with gr.Row():
                emergency_email_input = gr.Textbox(
                    label="📧 إضافة بريد إلكتروني للطوارئ | Add Emergency Email",
                    placeholder="example@email.com",
                    elem_classes=["emergency-container"]
                )
                add_contact_btn = gr.Button("➕ إضافة | Add", variant="primary")
    
    with gr.Row():
        with gr.Column():
            contact_list_display = gr.Textbox(
                label="📋 قائمة جهات الاتصال | Contact List",
                value="لم يتم إضافة أي جهات اتصال طوارئ بعد | No emergency contacts added yet",
                interactive=False,
                lines=3,
                elem_classes=["contact-list"]
            )
            
            remove_contact_dropdown = gr.Dropdown(
                label="🗑️ حذف جهة اتصال | Remove Contact",
                choices=[],
                visible=False,
                elem_classes=["emergency-container"]
            )
            remove_contact_btn = gr.Button("🗑️ حذف | Remove", variant="secondary", visible=False)
    
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
            # Conversation section with clear button
            gr.HTML("""<div class="section-header">💬 سجل المحادثة | Conversation History</div>""")
            
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        label="📝 المحادثة | Conversation",
                        height=350,
                        show_label=False,
                        elem_id="chatbot",
                        rtl=True
                    )
            with gr.Row(scale=1):
                clear_chat_btn = gr.Button(
                    "🗑️ مسح المحادثة\nClear Chat",
                    variant="stop",
                    elem_classes=["clear-button"],
                    size="lg"
                )
    
    # Footer with additional info
    gr.HTML("""
    <div class="footer-note">
        <p><strong>🔒 الخصوصية:</strong> جميع المحادثات سرية ومحمية</p>
        <p><strong>🔒 Privacy:</strong> All conversations are confidential and secure</p>
        <p><strong>⚠️ تنبيه:</strong> هذا مساعد ذكي وليس بديلاً عن العلاج النفسي المتخصص</p>
        <p><strong>⚠️ Disclaimer:</strong> This is an AI assistant and not a replacement for professional therapy</p>
        <p><strong>🚨 في حالات الطوارئ:</strong> إذا كنت تواجه أفكار إيذاء النفس، اتصل فوراً بخدمات الطوارئ المحلية</p>
        <p><strong>🚨 Emergency:</strong> If you're having thoughts of self-harm, contact local emergency services immediately</p>
    </div>
    """)
    
    # Event handlers for emergency contacts
    add_contact_btn.click(
        fn=add_emergency_contact,
        inputs=[emergency_email_input, state],
        outputs=[emergency_email_input, state, contact_list_display, remove_contact_dropdown]
    ).then(
        fn=lambda state: gr.update(choices=state.emergency_contacts, visible=len(state.emergency_contacts) > 0),
        inputs=[state],
        outputs=[remove_contact_dropdown]
    ).then(
        fn=lambda state: gr.update(visible=len(state.emergency_contacts) > 0),
        inputs=[state],
        outputs=[remove_contact_btn]
    )
    
    remove_contact_btn.click(
        fn=remove_emergency_contact,
        inputs=[remove_contact_dropdown, state],
        outputs=[state, contact_list_display, remove_contact_dropdown]
    ).then(
        fn=lambda state: gr.update(choices=state.emergency_contacts, visible=len(state.emergency_contacts) > 0),
        inputs=[state],
        outputs=[remove_contact_dropdown]
    ).then(
        fn=lambda state: gr.update(visible=len(state.emergency_contacts) > 0),
        inputs=[state],
        outputs=[remove_contact_btn]
    )
    
    # Clear conversation event handler
    clear_chat_btn.click(
        fn=clear_conversation,
        inputs=[state],
        outputs=[state, chatbot]
    )
    
    # Original event handlers (unchanged)
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

demo.launch()

# app = gr.mount_gradio_app(app, demo, path="/gradio")

# # Redirect root '/' to '/gradio'
# @app.get("/")
# def redirect_to_gradio():
#     return RedirectResponse(url="/gradio")