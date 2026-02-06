import os
import base64
from typing import Optional
import numpy as np
import cv2
from PIL import Image
import io


class AvatarManager:
    """Manages avatar images and state."""
    
    def __init__(self, avatars_dir: str = "avatars"):
        self._avatars_dir = avatars_dir
        self._current_avatar: Optional[np.ndarray] = None
        self._current_avatar_path: Optional[str] = None
        self._current_voice: str = "af_heart"  # Default to Kokoro's top quality voice
        self._voice_change_callback = None  # Callback to notify TTS service of voice changes
        
        os.makedirs(avatars_dir, exist_ok=True)
        
    @property
    def current_avatar(self) -> Optional[np.ndarray]:
        return self._current_avatar
    
    @property
    def current_voice(self) -> str:
        return self._current_voice
    
    def set_voice_change_callback(self, callback):
        """Set a callback to be called when voice changes."""
        self._voice_change_callback = callback
    
    def get_avatar_path(self) -> Optional[str]:
        """Get the file path of the current avatar."""
        return self._current_avatar_path
    
    def set_voice(self, voice: str):
        """Set the TTS voice. Accepts any Kokoro voice ID."""
        self._current_voice = voice
        # Notify TTS service of voice change
        if self._voice_change_callback:
            self._voice_change_callback(voice)
            
    def load_avatar_from_path(self, path: str) -> bool:
        """Load avatar image from file path."""
        try:
            image = cv2.imread(path)
            if image is not None:
                self._current_avatar = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self._current_avatar_path = os.path.abspath(path)
                return True
        except Exception as e:
            print(f"Error loading avatar: {e}")
        return False
    
    def load_avatar_from_base64(self, data: str) -> bool:
        """Load avatar image from base64 string."""
        try:
            # Remove data URL prefix if present
            if "," in data:
                data = data.split(",")[1]
                
            image_bytes = base64.b64decode(data)
            image = Image.open(io.BytesIO(image_bytes))
            self._current_avatar = np.array(image.convert("RGB"))
            
            # Save to temp file for Ditto
            temp_path = os.path.join(self._avatars_dir, "current_avatar.png")
            image.save(temp_path)
            self._current_avatar_path = os.path.abspath(temp_path)
            
            return True
        except Exception as e:
            print(f"Error loading avatar from base64: {e}")
        return False
        return False
    
    def get_avatar_base64(self) -> Optional[str]:
        """Get current avatar as base64 string."""
        if self._current_avatar is None:
            return None
            
        try:
            image = Image.fromarray(self._current_avatar)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception:
            return None
