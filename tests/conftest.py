import os
import pytest
import numpy as np
from PIL import Image

@pytest.fixture
def sample_plant_image():
    """Returns a synthetic textured image that passes sharpness and exposure screening."""
    np.random.seed(42)
    # Generate green textured gradient
    base = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        for j in range(224):
            base[i, j, 0] = int(40 + 30 * np.sin(i / 10.0))
            base[i, j, 1] = int(120 + 40 * np.cos(j / 10.0))
            base[i, j, 2] = int(50 + 20 * np.sin((i + j) / 15.0))
    # Add high-frequency texture noise
    noise = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
    textured = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(textured)

@pytest.fixture
def checkpoint_dir():
    return "./checkpoints"
