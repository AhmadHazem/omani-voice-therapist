from logger import logger
from openai import OpenAI
from Agent import OPENAI_API_KEY

import soundfile as sf
import io

class WhisperTranscriber:

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def TranscribeStream(self, audio_tuple) -> str:

        sample_rate, audio_np = audio_tuple

        # Convert NumPy array to WAV in-memory
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, samplerate=sample_rate, format='WAV')
        buffer.seek(0)

        # Wrap the buffer with filename and MIME type
        buffer.name = "audio.wav"  # Important for format recognition

        try:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=buffer,
                prompt= "You may recieve some english word with the arabic speech, "
                "and you will have to transcribe them as they are. Do not transcribe an English word into arabic",
                language="ar",
                response_format="text",
                timeout=10
            )

            buffer.close()  # Close the buffer after use
            return transcript
        except Exception as e:
            logger.warning(f"⚠️ Whisper transcription failed: {e}")
            return "فشل في التسجيل"