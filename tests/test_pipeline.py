import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import OllamaLLMService
from services.avatar_manager import AvatarManager


def test_ollama_service_creation():
    """Test Ollama LLM service creation."""
    service = OllamaLLMService(
        base_url="http://localhost:11434",
        model="mistral:7b"
    )
    
    assert service._model == "mistral:7b"
    assert service._base_url == "http://localhost:11434"
    assert service._context == []


def test_ollama_service_base_url_normalization():
    """Test that base URL trailing slash is removed."""
    service = OllamaLLMService(
        base_url="http://localhost:11434/",
        model="llama2"
    )
    
    assert service._base_url == "http://localhost:11434"


def test_avatar_manager_creation():
    """Test AvatarManager creation."""
    manager = AvatarManager()
    
    assert manager.current_avatar is None
    assert manager.current_voice == "autumn"


def test_avatar_manager_set_voice():
    """Test setting voice."""
    manager = AvatarManager()
    
    manager.set_voice("ember")
    assert manager.current_voice == "ember"
    
    # Invalid voice should be ignored
    manager.set_voice("invalid_voice")
    assert manager.current_voice == "ember"


def test_avatar_manager_valid_voices():
    """Test all valid voices."""
    manager = AvatarManager()
    valid_voices = ["autumn", "breeze", "ember", "juniper"]
    
    for voice in valid_voices:
        manager.set_voice(voice)
        assert manager.current_voice == voice
