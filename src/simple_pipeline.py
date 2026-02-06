import httpx
import json
import io
import wave
import numpy as np
from typing import Optional, Dict, Any
from groq import Groq


class SimplePipeline:
    """Simple conversation pipeline using Groq STT/TTS and Ollama LLM."""
    
    def __init__(
        self,
        groq_api_key: str,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "mistral:7b",
        voice: str = "autumn",
        system_prompt: str = "You are a friendly AI assistant. Keep responses concise and conversational."
    ):
        self.groq_client = Groq(api_key=groq_api_key)
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self.voice = voice
        self.system_prompt = system_prompt
        self.conversation_history = [
            {"role": "system", "content": system_prompt}
        ]
        self.http_client = httpx.AsyncClient(timeout=60.0)
        
        # Audio buffer for accumulating chunks
        self.audio_buffer = bytearray()
        self.sample_rate = 16000
        self.channels = 1
        self.sample_width = 2  # 16-bit audio
        
    async def process_audio(self, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """Process incoming audio data."""
        # Accumulate audio data
        self.audio_buffer.extend(audio_data)
        
        # Check if we have enough audio (e.g., 1 second of silence or VAD trigger)
        # For simplicity, process when we have at least 0.5 seconds of audio
        min_samples = int(self.sample_rate * 0.5) * self.sample_width
        
        if len(self.audio_buffer) < min_samples:
            return None
            
        # Convert buffer to WAV format for Groq
        try:
            audio_bytes = bytes(self.audio_buffer)
            self.audio_buffer.clear()
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.sample_width)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_bytes)
            
            wav_buffer.seek(0)
            
            # Transcribe with Groq Whisper
            transcript = await self._transcribe(wav_buffer)
            
            if not transcript or len(transcript.strip()) < 2:
                return None
                
            # Get LLM response
            response_text = await self._get_llm_response(transcript)
            
            # Generate TTS
            audio_response = await self._generate_speech(response_text)
            
            return {
                "transcript": transcript,
                "response": response_text,
                "audio": audio_response
            }
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    async def process_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Process text input directly."""
        try:
            # Get LLM response
            response_text = await self._get_llm_response(text)
            
            # Generate TTS
            audio_response = await self._generate_speech(response_text)
            
            return {
                "response": response_text,
                "audio": audio_response
            }
            
        except Exception as e:
            print(f"Error processing text: {e}")
            return None
            
    async def _transcribe(self, audio_file: io.BytesIO) -> str:
        """Transcribe audio using Groq Whisper."""
        try:
            # Use synchronous Groq client (it handles async internally)
            transcription = self.groq_client.audio.transcriptions.create(
                file=("audio.wav", audio_file),
                model="whisper-large-v3",
                response_format="text"
            )
            return transcription.strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
            
    async def _get_llm_response(self, user_message: str) -> str:
        """Get response from Ollama LLM."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            response = await self.http_client.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": self.conversation_history,
                    "stream": False
                }
            )
            
            data = response.json()
            assistant_message = data.get("message", {}).get("content", "")
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            # Keep history manageable (last 10 exchanges)
            if len(self.conversation_history) > 21:
                self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-20:]
                
            return assistant_message
            
        except Exception as e:
            print(f"LLM error: {e}")
            return "I'm having trouble responding right now."
            
    async def _generate_speech(self, text: str) -> Optional[bytes]:
        """Generate speech using Groq TTS."""
        try:
            # Use Groq's TTS
            response = self.groq_client.audio.speech.create(
                model="playai/playht-tts-v3",
                voice=f"Celeste-PlayAI",  # Available voices vary
                input=text,
                response_format="wav"
            )
            
            # Get audio bytes
            audio_bytes = response.read()
            return audio_bytes
            
        except Exception as e:
            print(f"TTS error: {e}")
            return None
