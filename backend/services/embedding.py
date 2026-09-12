"""
============================================================
JEWELLERY AI MATCHING
Image Embedding Module
============================================================

Version 1:
- No segmentation
- No model training
- Uses pretrained DINOv2
- Model is loaded directly from E: drive
- Prevents Hugging Face from using the full C: drive
============================================================
"""

import os

import torch
import numpy as np

from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModel
)


# ============================================================
# MODEL LOCATION
# ============================================================

MODEL_NAME = "facebook/dinov2-base"

# IMPORTANT:
# The model has already been downloaded to E:.
# We will explicitly use the local Hugging Face snapshot.

MODEL_CACHE_DIR = (
    r"E:\huggingface_cache"
    r"\hub\models--facebook--dinov2-base"
)


# ============================================================
# FIND MODEL SNAPSHOT
# ============================================================

SNAPSHOT_DIR = os.path.join(
    MODEL_CACHE_DIR,
    "snapshots"
)


def find_model_snapshot():
    """
    Find the downloaded DINOv2 model snapshot.
    """

    if not os.path.exists(SNAPSHOT_DIR):

        raise FileNotFoundError(
            "DINOv2 model cache was not found.\n\n"
            f"Expected location:\n{SNAPSHOT_DIR}\n\n"
            "Please check the E:\\huggingface_cache folder."
        )

    snapshot_folders = [
        folder
        for folder in os.listdir(
            SNAPSHOT_DIR
        )
        if os.path.isdir(
            os.path.join(
                SNAPSHOT_DIR,
                folder
            )
        )
    ]

    if not snapshot_folders:

        raise FileNotFoundError(
            "No DINOv2 snapshot was found in:\n"
            f"{SNAPSHOT_DIR}"
        )

    # Use the first available snapshot.
    snapshot_path = os.path.join(
        SNAPSHOT_DIR,
        snapshot_folders[0]
    )

    return snapshot_path


MODEL_PATH = find_model_snapshot()

print()
print("=" * 60)
print("DINOv2 LOCAL MODEL")
print("=" * 60)

print(
    "Model:",
    MODEL_NAME
)

print(
    "Local model path:"
)

print(
    MODEL_PATH
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Embedding device:",
    DEVICE
)


# ============================================================
# LOAD PROCESSOR
# ============================================================

print()
print(
    "Loading image processor..."
)

processor = AutoImageProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print(
    "Image processor loaded."
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print(
    "Loading DINOv2 model..."
)

model = AutoModel.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.to(
    DEVICE
)

model.eval()

print(
    "DINOv2 model loaded successfully."
)

print(
    "=" * 60
)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):
    """
    Load an image and convert it to RGB.
    """

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    return image


# ============================================================
# CREATE EMBEDDING
# ============================================================

def get_embedding(image_path):
    """
    Convert a jewellery image into a normalized
    DINOv2 embedding.

    Parameters
    ----------
    image_path : str
        Path to the jewellery image.

    Returns
    -------
    numpy.ndarray
        Normalized 768-dimensional embedding.
    """

    image = load_image(
        image_path
    )

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        # CLS token embedding
        embedding = (
            outputs
            .last_hidden_state[:, 0, :]
        )

    # Normalize embedding
    embedding = torch.nn.functional.normalize(
        embedding,
        p=2,
        dim=1
    )

    embedding = (
        embedding
        .cpu()
        .numpy()[0]
    )

    return embedding