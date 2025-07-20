# refactored_omani_therapy.py

import os
import tempfile
import json
import logging
import time
import concurrent.futures
from typing import Dict, List, Optional, Tuple
from enum import Enum
from functools import lru_cache
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import keyboard
from scipy.io.wavfile import write
from openai import OpenAI
import anthropic
import azure.cognitiveservices.speech as speechsdk


# LangChain imports
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.memory import ConversationBufferWindowMemory


# 🧭 Logger Setup
def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


logger = setup_logging()

@dataclass
class AudioConfig:
    """Audio-related configuration for sounddevice"""
    chunk: int = 1024
    rate: int = 44100
    channels: int = 1
    record_key: str = "r"
    whisper_model: str = "whisper-1"
    
    # sounddevice specific settings
    dtype: type = np.float32  # sounddevice default input dtype
    output_dtype: type = np.int16  # Output format for WAV files
    
    # Device settings (None = use default)
    input_device: Optional[int] = None
    latency: str = 'low'  # 'low', 'high', or specific latency in seconds


@dataclass
class VoiceConfig:
    """Voice and TTS configuration"""
    voice_name: str = "Abdullah"
    tts_voice: str = "ar-OM-AbdullahNeural"
    
    def __post_init__(self):
        self.tts_voice = f"ar-OM-{self.voice_name}Neural"


@dataclass
class ModelConfig:
    """AI model configuration"""
    claude_model: str = "claude-3-7-sonnet-latest"
    claude_temp: float = 0.7
    gpt_model: str = "gpt-4o"
    gpt_temp: float = 0.7


@dataclass
class PerformanceConfig:
    """Performance and optimization settings"""
    max_workers: int = 4
    timeout_seconds: int = 15
    memory_window: int = 3
    max_tokens: int = 300
    use_gpt_validation: bool = False


@dataclass
class PathConfig:
    """File paths and directories"""
    cbt_knowledge_base_path: str = "cbt_knowledge_base"
    vector_store_path: str = "vector_store"


class Config:
    """Main configuration class"""
    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.claude_api_key = os.environ.get("CLAUDE_API_KEY")
        self.speech_key = os.environ.get("SPEECH_KEY")
        self.endpoint = os.environ.get("ENDPOINT")
        self.tavily_api_key = os.environ.get("TAVILY_API_KEY")
        
        self.audio = AudioConfig()
        self.voice = VoiceConfig()
        self.model = ModelConfig()
        self.performance = PerformanceConfig()
        self.paths = PathConfig()
        
        # System prompts - kept unchanged as requested
        self.system_prompt =(
        "أنت معالج نفسي عماني متخصص في تقديم الدعم النفسي باللهجة العمانية الأصيلة.\n"
        "لديك خبرة واسعة في العلاج المعرفي السلوكي (CBT) والتقنيات النفسية المختلفة.\n"
        "هدفك هو تقديم الدعم العاطفي والنفسي مع مراعاة:\n"
        "- دقة اللهجة العُمانية الأصيلة\n"
        "- الحساسية الثقافية والقيم الإسلامية\n"
        "- احترام التقاليد العائلية والعادات الاجتماعية\n"
        "- استخدام تقنيات العلاج المعرفي السلوكي عند الحاجة\n"
        "- التعامل مع التداخل اللغوي بين العربية والإنجليزية\n"
        "- الاستجابة للإشارات العاطفية بتعاطف واحترافية\n\n"
        "يتم تزويدك بمعلومات عن:\n"
        "- مستوى الخطورة النفسية للمستخدم\n"
        "- العاطفة السائدة\n"
        "- محتوى CBT المناسب (إن وجد)\n"
        "- توصيات حول استخدام تقنيات CBT\n\n"
        "استخدم هذه المعلومات لتقديم رد مخصص وفعال. لا تستخدم التنسيق أو النقاط، فقط نص خام.\n"
        "حاول أن تكون موجزًا ومختصرًا في ردك.\n"
        "و استخدم الآيات القرآنية والأحاديث النبوية عند الحاجة.\n"
        "ولكن لا تستعملها بشكل مفرط أو في كل رد، فقط عند الحاجة.\n"
    )
        
        self.cbt_decision_prompt = (
            "أنت خبير في العلاج المعرفي السلوكي (CBT). مهمتك هي تحليل النص والعاطفة "
            "وتحديد ما إذا كان استخدام تقنيات CBT مناسباً أم لا.\n"
            "Strictly Respond with a JSON object containing:\n"
            "Follow strictly this format without any additional symbols or texts:\n"
            "{\"use_cbt\": true/false, \"cbt_technique\": \"اسم التقنية المناسبة أو null\", \"reasoning\": \"سبب القرار باللغة العربية\", \"severity\": \"low/medium/high\"}\n"
        )
        
        self.whisper_prompt = (
            "نسخ الصوت بالعربية والإنجليزية كما هو منطوق."
            "Transcribe the audio in Arabic and English as spoken."
            "You may recieve English Words or Phrases in the Audio, so you should transcribe them in English."
        )
        
        self.risk_assessor_system_prompt = (
            "You are an Omani culturally sensitive mental health assistant trained to assess emotional risk levels "
            "based on user expressions. Your job is to produce a JSON response with the following keys:\n"
            "- risk_level: one of LOW, MEDIUM, HIGH, CRITICAL\n"
            "- emotion: the dominant feeling inferred\n"
            "- emotional_intensity: scale 1-10\n"
            "- needs_immediate_attention: boolean\n"
            "- cultural_context: brief note about cultural considerations\n"
            "Consider linguistic cues, Islamic cultural context, and psychological indicators when analyzing input.\n"
            "Respond only with valid JSON."
            "Follow Strictly this format without any additional symbols or texts:\n"
            "{\"risk_level\": \"LOW\",  \"emotion\": \"neutral\",  \"emotional_intensity\": 1,  \"needs_immediate_attention\": false,  \"cultural_context\": \"The greeting 'السلام عليكم' is a common and polite way to say hello in Islamic cultures, indicating a peaceful and respectful interaction.\"}"
        )


# 🔄 Enums
class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CBTTechnique(Enum):
    COGNITIVE_RESTRUCTURING = "cognitive_restructuring"
    BEHAVIORAL_ACTIVATION = "behavioral_activation"
    EXPOSURE_THERAPY = "exposure_therapy"
    MINDFULNESS = "mindfulness"
    PROBLEM_SOLVING = "problem_solving"
    RELAXATION = "relaxation"
    THOUGHT_CHALLENGING = "thought_challenging"
    ACTIVITY_SCHEDULING = "activity_scheduling"


# 🧠 CBT Knowledge Base
class CBTKnowledgeBase:
    """Enhanced CBT Knowledge Base with caching and optimization"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
        self.vector_store = None
        self.cbt_data = self._load_cbt_data()
        self._build_vector_store()
        self._cache = {}
        
    @lru_cache(maxsize=128)
    def _load_cbt_data(self) -> Dict:
        """Load CBT knowledge base with Arabic content"""
        return {
            "cognitive_restructuring": {
                "name": "إعادة البناء المعرفي",
                "description": "تقنية لتحديد وتعديل الأفكار السلبية",
                "techniques": [
                    "تحديد الأفكار التلقائية السلبية",
                    "فحص الأدلة المؤيدة والمعارضة",
                    "إيجاد أفكار بديلة أكثر واقعية"
                ],
                "suitable_for": ["depression", "anxiety", "anger", "low_self_esteem"],
                "arabic_content": "إعادة البناء المعرفي تقنية أساسية لمساعدة الشخص على تحديد الأفكار السلبية وتعديلها بطريقة منطقية.",
                "cultural_adaptation": "مراعاة القيم الإسلامية والثقافة العمانية مع التركيز على التوكل والدعاء."
            },
            "behavioral_activation": {
                "name": "التفعيل السلوكي",
                "description": "زيادة الأنشطة الممتعة لتحسين المزاج",
                "techniques": [
                    "جدولة الأنشطة الممتعة",
                    "تحديد الأهداف الصغيرة",
                    "مراقبة المزاج والأنشطة"
                ],
                "suitable_for": ["depression", "apathy", "isolation"],
                "arabic_content": "التفعيل السلوكي يساعد على استعادة الاهتمام بالحياة من خلال الأنشطة الممتعة والمعنوية.",
                "cultural_adaptation": "تضمين الأنشطة الدينية والاجتماعية المناسبة للثقافة العمانية."
            },
            "mindfulness": {
                "name": "الوعي التام",
                "description": "التركيز على اللحظة الحالية وقبول المشاعر",
                "techniques": [
                    "التنفس العميق",
                    "التأمل الإسلامي",
                    "مراقبة الأفكار والمشاعر"
                ],
                "suitable_for": ["anxiety", "stress", "overthinking"],
                "arabic_content": "الوعي التام يعني التركيز على اللحظة الحالية وقبول المشاعر دون أحكام.",
                "cultural_adaptation": "دمج الذكر والتسبيح كأشكال من الوعي التام المناسبة للثقافة الإسلامية."
            },
            "problem_solving": {
                "name": "حل المشكلات",
                "description": "منهج منظم لتحليل المشكلات وإيجاد حلول",
                "techniques": [
                    "تحديد المشكلة بوضوح",
                    "عصف ذهني للحلول",
                    "تقييم الخيارات المتاحة"
                ],
                "suitable_for": ["stress", "relationship_issues", "work_problems"],
                "arabic_content": "حل المشكلات منهج منظم يساعد على مواجهة التحديات بطريقة منطقية وعملية.",
                "cultural_adaptation": "مراعاة القيم الأسرية والاجتماعية العمانية مع التأكيد على الاستشارة والشورى."
            }
        }
    
    def _build_vector_store(self):
        """Build FAISS vector store from CBT knowledge"""
        documents = []
        for technique_id, technique_data in self.cbt_data.items():
            content = f"{technique_data['name']}: {technique_data['description']} - {technique_data['arabic_content']}"
            doc = Document(
                page_content=content,
                metadata={
                    "technique_id": technique_id,
                    "name": technique_data['name'],
                    "suitable_for": technique_data['suitable_for']
                }
            )
            documents.append(doc)
        
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        logger.info("✅ CBT knowledge base vector store created")
    
    def retrieve_relevant_cbt(self, emotion: str, query: str, k: int = 1) -> List[Document]:
        """Retrieve relevant CBT techniques with caching"""
        cache_key = f"{emotion}_{hash(query)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        if not self.vector_store:
            return []
        
        search_query = f"{emotion} {query[:100]}"
        relevant_docs = self.vector_store.similarity_search(search_query, k=k)
        
        self._cache[cache_key] = relevant_docs
        return relevant_docs


# 🎯 CBT Decision Maker
class CBTDecisionMaker:
    """Determines when to use CBT techniques"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self._cache = {}
        
    def should_use_cbt(self, transcript: str, emotion: str, risk_level: str) -> Dict:
        """Determine if CBT should be used with caching"""
        cache_key = f"{emotion}_{risk_level}_{hash(transcript[:50])}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        prompt = f"النص: {transcript}\nالعاطفة: {emotion}\nمستوى الخطورة: {risk_level}\n\nحدد استخدام CBT:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model.gpt_model,
                messages=[
                    {"role": "system", "content": self.config.cbt_decision_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100,
                timeout=5
            )
            
            result = json.loads(response.choices[0].message.content)
            self._cache[cache_key] = result
            logger.info(f"📥 CBT decision made: {result}")
            return result
        except Exception as e:
            logger.warning(f"⚠️ CBT decision making failed: {e}")
            default_result = {
                "use_cbt": False,
                "cbt_technique": None,
                "reasoning": "فشل في تحليل الحاجة للعلاج المعرفي السلوكي",
                "severity": "low"
            }
            self._cache[cache_key] = default_result
            return default_result


# 🔍 Risk Assessor
class EnhancedRiskAssessor:
    """Assesses emotional risk levels from user input"""
    
    def __init__(self, config: Config):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.config = config
        self._cache = {}
        
    def assess(self, input_text: str) -> Dict:
        """Enhanced risk assessment with caching"""
        cache_key = hash(input_text[:100])
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        try:
            response = self.client.chat.completions.create(
                model=self.config.model.gpt_model,
                messages=[
                    {"role": "system", "content": self.config.risk_assessor_system_prompt},
                    {"role": "user", "content": input_text[:300]}
                ],
                temperature=0.0,
                max_tokens=100,
                timeout=5
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate and fill missing fields
            required_fields = ["risk_level", "emotion", "emotional_intensity", "needs_immediate_attention"]
            for field in required_fields:
                if field not in result:
                    result[field] = self._get_default_value(field)
                    
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Enhanced risk assessment failed: {e}")
            default_result = {
                "risk_level": "low",
                "emotion": "neutral",
                "emotional_intensity": 1,
                "needs_immediate_attention": False,
                "cultural_context": "غير محدد"
            }
            self._cache[cache_key] = default_result
            return default_result
    
    def _get_default_value(self, field: str):
        """Get default values for missing fields"""
        defaults = {
            "risk_level": "low",
            "emotion": "neutral",
            "emotional_intensity": 1,
            "needs_immediate_attention": False,
            "cultural_context": "غير محدد"
        }
        return defaults.get(field, None)


# 🎙️ Audio Recording Module - Updated to use sounddevice
class AudioRecorder:
    """Handles audio recording functionality using sounddevice"""
    
    def __init__(self, config: Config):
        self.config = config
        self.sample_rate = config.audio.rate
        self.channels = config.audio.channels
        # sounddevice uses float32 by default, but we'll convert to int16 for compatibility
        self.dtype = np.float32
        
        # Check available devices
        try:
            devices = sd.query_devices()
            logger.info(f"🎤 Available audio devices: {len(devices)}")
            
            # Get default input device
            default_device = sd.default.device[0]  # input device
            logger.info(f"🎤 Using default input device: {default_device}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not query audio devices: {e}")

    def record_while_key_pressed(self) -> str:
        """Record audio while key is pressed using sounddevice"""
        logger.info("🔴 Recording started...")
        
        # List to store audio chunks
        audio_chunks = []
        
        def audio_callback(indata, frames, time, status):
            """Callback function to collect audio data"""
            if status:
                logger.warning(f"Audio callback status: {status}")
            # Convert float32 to int16 and store
            audio_chunks.append((indata[:, 0] * 32767).astype(np.int16))
        
        try:
            # Start recording stream
            with sd.InputStream(
                callback=audio_callback,
                channels=1,  # Mono recording
                samplerate=self.sample_rate,
                dtype=self.dtype,
                blocksize=self.config.audio.chunk
            ):
                # Record while key is pressed
                while keyboard.is_pressed(self.config.audio.record_key):
                    sd.sleep(10)  # Small sleep to prevent busy waiting
                    
        except Exception as e:
            logger.error(f"❌ Recording error: {e}")
            return None
            
        logger.info("⏹️ Recording stopped.")
        
        if not audio_chunks:
            logger.warning("⚠️ No audio data recorded")
            return None
        
        # Concatenate all audio chunks
        audio_data = np.concatenate(audio_chunks)
        
        # Save to temporary file
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            # Use scipy.io.wavfile to write the audio file
            from scipy.io.wavfile import write
            write(temp_audio.name, self.sample_rate, audio_data)
            logger.info(f"💾 Audio saved to: {temp_audio.name}")
            return temp_audio.name
            
        except Exception as e:
            logger.error(f"❌ Error saving audio file: {e}")
            return None

    def record_fixed_duration(self, duration: float = 5.0) -> str:
        """Record audio for a fixed duration (useful for testing)"""
        logger.info(f"🔴 Recording for {duration} seconds...")
        
        try:
            # Record audio for specified duration
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=self.dtype
            )
            
            # Wait for recording to complete
            sd.wait()
            
            # Convert to int16
            audio_data = (audio_data[:, 0] * 32767).astype(np.int16)
            
            # Save to temporary file
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            from scipy.io.wavfile import write
            write(temp_audio.name, self.sample_rate, audio_data)
            
            logger.info(f"💾 Fixed duration audio saved to: {temp_audio.name}")
            return temp_audio.name
            
        except Exception as e:
            logger.error(f"❌ Fixed duration recording error: {e}")
            return None

    def test_audio_devices(self):
        """Test and display available audio devices"""
        try:
            logger.info("🔍 Testing audio devices...")
            devices = sd.query_devices()
            
            for i, device in enumerate(devices):
                device_info = f"Device {i}: {device['name']}"
                if device['max_input_channels'] > 0:
                    device_info += f" (Input: {device['max_input_channels']} channels)"
                if device['max_output_channels'] > 0:
                    device_info += f" (Output: {device['max_output_channels']} channels)"
                logger.info(device_info)
                
            # Test default devices
            default_in, default_out = sd.default.device
            logger.info(f"🎤 Default input device: {default_in}")
            logger.info(f"🔊 Default output device: {default_out}")
            
        except Exception as e:
            logger.error(f"❌ Device test error: {e}")

    def close(self):
        """Clean up audio resources (sounddevice handles cleanup automatically)"""
        logger.info("🔇 Audio recorder cleanup completed")


# 📝 Whisper Transcription
class WhisperTranscriber:
    """Handles audio transcription using Whisper"""
    
    def __init__(self, config: Config):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.config = config

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text"""
        if not audio_path or not os.path.exists(audio_path):
            logger.warning("⚠️ Invalid audio path for transcription")
            return "فشل في التسجيل"
            
        logger.info("📤 Sending audio to Whisper...")
        
        try:
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self.config.audio.whisper_model,
                    file=audio_file,
                    prompt=self.config.whisper_prompt,
                    language="ar",
                    response_format="text",
                    timeout=10
                )
            
            logger.info("📝 Transcript received.")
            return response
            
        except Exception as e:
            logger.warning(f"⚠️ Whisper transcription failed: {e}")
            return "فشل في التسجيل"


# 🧠 Enhanced Therapist
class EnhancedTherapist:
    """Main therapist class with AI integration"""
    
    def __init__(self, config: Config):
        self.config = config
        self.claude = anthropic.Anthropic(api_key=config.claude_api_key)
        self.openai = OpenAI(api_key=config.openai_api_key)
        self.cbt_knowledge = CBTKnowledgeBase(config)
        self.cbt_decision_maker = CBTDecisionMaker(config)
        self.memory = ConversationBufferWindowMemory(k=config.performance.memory_window)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.performance.max_workers
        )
        
    def _get_cbt_context_parallel(self, emotion: str, transcript: str, cbt_decision: Dict) -> str:
        """Get CBT context in parallel"""
        if not cbt_decision["use_cbt"]:
            return ""
            
        relevant_cbt = self.cbt_knowledge.retrieve_relevant_cbt(emotion, transcript)
        return relevant_cbt[0].page_content if relevant_cbt else ""
        
    def respond(self, transcript: str, risk_assessment: Dict) -> str:
        """Generate optimized response with parallel processing"""
        start_time = time.time()
        
        emotion = risk_assessment["emotion"]
        risk_level = risk_assessment["risk_level"]
        intensity = risk_assessment.get("emotional_intensity", 1)
        
        # Parallel processing for CBT decision and context
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                cbt_future = executor.submit(
                    self.cbt_decision_maker.should_use_cbt, 
                    transcript, emotion, risk_level
                )
                
                cbt_decision = cbt_future.result(timeout=5)
                
                cbt_context = ""
                if cbt_decision["use_cbt"]:
                    cbt_context = self._get_cbt_context_parallel(emotion, transcript, cbt_decision)
        
        except concurrent.futures.TimeoutError:
            logger.warning("⚠️ Parallel processing timeout")
            cbt_decision = {"use_cbt": False, "cbt_technique": None, "reasoning": "timeout", "severity": "low"}
            cbt_context = ""
        
        # Prepare prompt
        enhanced_prompt = f"""
        المستخدم: {transcript}
        
        الحالة: {emotion} ({intensity}/10) - {risk_level}
        CBT: {cbt_decision['use_cbt']} - {cbt_decision.get('cbt_technique', 'لا يوجد')}
        
        {cbt_context}
        """
        
        # Try GPT first, fallback to Claude
        try:
            response = self._get_gpt_response(enhanced_prompt)
            logger.info(f"📥 GPT response received in {time.time() - start_time:.2f}s")
            result = response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"⚠️ GPT failed: {e}")
            result = self._get_claude_response(enhanced_prompt)
            logger.info(f"✅ Claude fallback complete in {time.time() - start_time:.2f}s")
        
        # Update memory
        self.memory.save_context(
            {"input": transcript[:100]}, 
            {"output": result[:100]}
        )
        
        return result
    
    def _get_gpt_response(self, prompt: str):
        """Get response from GPT model"""
        return self.openai.chat.completions.create(
            model=self.config.model.gpt_model,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.model.gpt_temp,
            max_tokens=self.config.performance.max_tokens,
            timeout=self.config.performance.timeout_seconds
        )
    
    def _get_claude_response(self, prompt: str) -> str:
        """Get response from Claude model"""
        try:
            response = self.claude.messages.create(
                model=self.config.model.claude_model,
                max_tokens=self.config.performance.max_tokens,
                temperature=self.config.model.claude_temp,
                system=self.config.system_prompt,
                messages=[{"role": "user", "content": prompt}],
                timeout=self.config.performance.timeout_seconds
            )
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"❌ Both models failed: {e}")
            return "عذراً، حدث خطأ تقني. يرجى المحاولة مرة أخرى."


# 🔈 TTS Player
class TTSPlayer:
    """Handles text-to-speech functionality"""
    
    def __init__(self, config: Config):
        self.config = config
        self.synthesizer = self._initialize_synthesizer()
        self.audio_queue = []
        self.is_playing = False

    def _initialize_synthesizer(self):
        """Initialize Azure TTS synthesizer"""
        speech_config = speechsdk.SpeechConfig(subscription=self.config.speech_key,endpoint=self.config.endpoint)
        speech_config.speech_synthesis_voice_name = self.config.voice.tts_voice
        return speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

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
            logger.error(f"TTS Error: {e}")
            return None


# 🚀 Main Application
class EnhancedOmaniTherapyApp:
    """Main application controller"""
    
    def __init__(self, config: Config):
        self.config = config
        self.recorder = AudioRecorder(config)
        self.transcriber = WhisperTranscriber(config)
        self.therapist = EnhancedTherapist(config)
        self.tts = TTSPlayer(config)
        self.risk_assessor = EnhancedRiskAssessor(config)
        
    def _process_audio_pipeline(self, audio_path: str) -> Tuple[str, str]:
        """Optimized audio processing pipeline"""
        start_time = time.time()
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Transcribe audio
                transcript_future = executor.submit(self.transcriber.transcribe, audio_path)
                transcript = transcript_future.result(timeout=15)
                
                if not transcript or transcript == "فشل في التسجيل":
                    return transcript, "عذراً، لم أتمكن من فهم التسجيل. يرجى المحاولة مرة أخرى."
                
                # Assess risk and generate response
                risk_future = executor.submit(self.risk_assessor.assess, transcript)
                risk_assessment = risk_future.result(timeout=8)
                
                logger.info(f"🎯 Risk Assessment: {risk_assessment}")
                
                response_future = executor.submit(self.therapist.respond, transcript, risk_assessment)
                response = response_future.result(timeout=15)
                
                logger.info(f"💬 Total processing time: {time.time() - start_time:.2f}s")
                return transcript, response
                
        except concurrent.futures.TimeoutError:
            logger.warning("⚠️ Processing timeout")
            return transcript, "عذراً، استغرق الأمر وقتاً أطول من المتوقع. يرجى المحاولة مرة أخرى."
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            return transcript, "عذراً، حدث خطأ في المعالجة. يرجى المحاولة مرة أخرى."
        
    def test_audio_setup(self):
        """Test the audio setup"""
        logger.info("🧪 Testing audio setup...")
        self.recorder.test_audio_devices()
        
        # Test short recording
        logger.info("🎤 Testing 3-second recording...")
        test_audio = self.recorder.record_fixed_duration(3.0)
        
        if test_audio:
            logger.info("✅ Audio test completed successfully")
            # Clean up test file
            if os.path.exists(test_audio):
                os.remove(test_audio)
        else:
            logger.error("❌ Audio test failed")
        
    def run(self):
        """Main application loop"""
        logger.info("🎧 Optimized Omani Therapy Assistant Ready!")
        logger.info("🚀 Performance optimizations: Parallel processing, caching, faster models")
        logger.info("🎤 Now using sounddevice for better audio handling")
        logger.info(f"🎯 Hold [{self.config.audio.record_key.upper()}] to speak, release to get fast response")
        
        # Test audio setup
        self.test_audio_setup()
        
        try:
            while True:
                keyboard.wait(self.config.audio.record_key)
                audio_path = self.recorder.record_while_key_pressed()
                
                if not audio_path:
                    logger.warning("⚠️ No audio recorded, please try again")
                    continue
                
                try:
                    _, response = self._process_audio_pipeline(audio_path)
                    self.tts.speak(response)
                    
                finally:
                    if audio_path and os.path.exists(audio_path):
                        os.remove(audio_path)
                        
        except KeyboardInterrupt:
            logger.info("👋 Optimized therapy session ended.")
            self.recorder.close()


# # 🟢 Entry Point
# def main():
#     """Application entry point"""
#     config = Config()
#     app = EnhancedOmaniTherapyApp(config)
#     app.run()


# if __name__ == "__main__":
#     main()