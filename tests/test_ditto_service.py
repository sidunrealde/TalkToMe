import pytest
import asyncio
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.ditto_video import DittoVideoService
from config import config


@pytest.fixture
def ditto_service():
    return DittoVideoService(
        config_path=config.ditto_config_path,
        checkpoint_path=config.ditto_checkpoint_path
    )


def test_service_creation(ditto_service):
    """Test that Ditto service can be created."""
    assert ditto_service is not None
    assert ditto_service._fps == 25
    assert ditto_service._source_image is None


def test_source_image_setting(ditto_service):
    """Test setting a source image."""
    # Create a dummy image
    test_image = np.zeros((512, 512, 3), dtype=np.uint8)
    ditto_service.set_source_image(test_image)
    
    assert ditto_service._source_image is not None
    assert ditto_service._source_image.shape == (512, 512, 3)


def test_audio_buffer_initialization(ditto_service):
    """Test that audio buffer is properly initialized."""
    assert ditto_service._audio_buffer == []
    assert ditto_service._is_speaking is False


@pytest.mark.asyncio
async def test_service_start_without_checkpoints(ditto_service):
    """Test that service handles missing checkpoints gracefully."""
    from pipecat.frames.frames import StartFrame
    
    # This should not raise even if checkpoints are missing
    try:
        await ditto_service.start(StartFrame())
    except Exception as e:
        # Expected if Ditto SDK is not available
        assert "ditto" in str(e).lower() or "import" in str(e).lower()
