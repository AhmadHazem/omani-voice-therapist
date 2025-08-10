from logger import logger
from Agent import SPEECH_KEY
from Agent import SPEECH_ENDPOINT

import azure.cognitiveservices.speech as speechsdk

class Speaker:

    def __init__(self):
        speech_config = speechsdk.SpeechConfig(subscription= SPEECH_KEY ,endpoint= SPEECH_ENDPOINT)
        speech_config.speech_synthesis_voice_name = "ar-OM-AbdullahNeural"
        self.synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    
    def speak(self, text: str):
        """Synthesize and play text"""
        try:
            logger.info("🔊 Speaking with Azure TTS...")
            result = self.synthesizer.speak_text_async(text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info("✅ Speech synthesized.")
                return result.audio_data
            else:
                logger.warning(f"❌ Speech synthesis failed: {result.reason}")
                return None
                
        except Exception as e:
            logger.error(f"Speaker Error: {e}")
            return None

# speaker = Speaker()
# speaker.speak("السلام عليكم ورحمه الله و بركاته")