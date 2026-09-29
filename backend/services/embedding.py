"""
JewelMatch AI - DINOv2-Base Embedding Service

Purpose:
- DINOv2-Base visual embeddings
- 768-dimensional normalized embeddings
- Reduce colour/material influence
- Use multiple jewellery-focused views
- Use shared segmentation.py
- Batch all views in a single DINO inference call
- Compatible with matcher.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from PIL import Image, ImageEnhance
from transformers import AutoImageProcessor, AutoModel

from .segmentation import get_segmented_crop

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "facebook/dinov2-base"
DEVICE = "cpu"

EMBEDDING_SIZE = 768
MAX_IMAGE_SIZE = 768

# Number of views:
# 1. Original
# 2. Grayscale
# 3. Contrast grayscale
# 4. Jewellery-focused crop

VIEW_WEIGHTS = np.array(
    [
        0.15,  # Original
        0.30,  # Grayscale
        0.20,  # Contrast grayscale
        0.35,  # Jewellery crop
    ],
    dtype=np.float32,
)


# ============================================================
# CPU SETTINGS
# ============================================================

try:
    torch.set_num_threads(min(max(torch.get_num_threads(), 1), 4))
except Exception:
    pass


# ============================================================
# MODEL CACHE
# ============================================================

_processor = None
_model = None


def _load_model():
    """Load DINOv2-Base lazily."""

    global _processor
    global _model

    if _processor is not None and _model is not None:
        return _processor, _model

    print("=" * 70)
    print("[EMBEDDING] Loading DINOv2-Base")
    print("=" * 70)
    print(f"[EMBEDDING] Model: {MODEL_NAME}")
    print(f"[EMBEDDING] Device: {DEVICE}")
    print(f"[EMBEDDING] Dimension: {EMBEDDING_SIZE}")

    _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    _model = AutoModel.from_pretrained(MODEL_NAME)

    _model.to(DEVICE)
    _model.eval()

    print("[EMBEDDING] DINOv2-Base loaded successfully")

    return _processor, _model


# ============================================================
# IMAGE LOADING
# ============================================================


def load_image(image_path):
    """Load image as RGB PIL image."""

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(path).convert("RGB")

    width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {image_path}")

    maximum = max(width, height)

    if maximum > MAX_IMAGE_SIZE:

        scale = MAX_IMAGE_SIZE / float(maximum)

        new_width = max(
            1,
            int(width * scale),
        )

        new_height = max(
            1,
            int(height * scale),
        )

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )

    return image


# ============================================================
# IMAGE VIEWS
# ============================================================


def create_grayscale_view(image):
    """Create grayscale RGB image."""

    gray = image.convert("L")

    return gray.convert("RGB")


def create_contrast_view(image):
    """Create high-contrast grayscale RGB image."""

    gray = image.convert("L")

    contrast = ImageEnhance.Contrast(gray).enhance(1.8)

    return contrast.convert("RGB")


def create_center_crop(
    image,
    crop_ratio=0.82,
):
    """Create centre crop."""

    width, height = image.size

    crop_width = max(
        1,
        int(width * crop_ratio),
    )

    crop_height = max(
        1,
        int(height * crop_ratio),
    )

    left = max(
        0,
        (width - crop_width) // 2,
    )

    top = max(
        0,
        (height - crop_height) // 2,
    )

    right = min(
        width,
        left + crop_width,
    )

    bottom = min(
        height,
        top + crop_height,
    )

    return image.crop(
        (
            left,
            top,
            right,
            bottom,
        )
    )


def create_zoom_view(
    image,
    crop_ratio=0.70,
):
    """Compatibility helper for older matcher code."""

    return create_center_crop(
        image,
        crop_ratio=crop_ratio,
    )


# ============================================================
# JEWELLERY FOREGROUND VIEW
# ============================================================


def create_foreground_pil_view(
    image_path,
):
    """
    Use the shared segmentation.py pipeline.

    Returns a PIL RGB jewellery-focused crop.
    """

    image = load_image(image_path)

    try:

        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            return create_center_crop(image)

        segmented_crop, mask = get_segmented_crop(image_bgr)

        if segmented_crop is None or segmented_crop.size == 0:
            return create_center_crop(image)

        crop_rgb = cv2.cvtColor(
            segmented_crop,
            cv2.COLOR_BGR2RGB,
        )

        crop = Image.fromarray(crop_rgb).convert("RGB")

        return crop

    except Exception as exc:

        print("[SEGMENTATION] " f"Foreground view failed: {exc}")

        return create_center_crop(image)


# ============================================================
# CREATE ALL VIEWS
# ============================================================


def create_views(
    image,
    image_path=None,
):
    """
    Create four DINO views.

    1. Original
    2. Grayscale
    3. Contrast grayscale
    4. Jewellery-focused crop
    """

    views = [
        image,
        create_grayscale_view(image),
        create_contrast_view(image),
    ]

    if image_path is not None:

        views.append(create_foreground_pil_view(image_path))

    else:

        views.append(create_center_crop(image))

    return views


# ============================================================
# BATCH DINO EXTRACTION
# ============================================================


@torch.no_grad()
def _extract_embeddings(images):
    """
    Extract embeddings for all views in ONE DINO call.

    Returns:
        numpy array with shape:
        (number_of_views, 768)
    """

    processor, model = _load_model()

    inputs = processor(
        images=images,
        return_tensors="pt",
    )

    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[:, 0, :]

    embeddings = embeddings.detach().cpu().numpy()

    embeddings = embeddings.astype(np.float32)

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    embeddings = embeddings / norms

    return embeddings.astype(np.float32)


# ============================================================
# MAIN EMBEDDING FUNCTION
# ============================================================


def create_embedding(
    image_path,
):
    """
    Create one 768-dimensional DINOv2-Base embedding.

    Four views are processed in one model call.
    """

    image_path = str(image_path)

    print(f"[DINO-BASE] Creating embedding: " f"{Path(image_path).name}")

    image = load_image(image_path)

    views = create_views(
        image,
        image_path=image_path,
    )

    embeddings = _extract_embeddings(views)

    if embeddings.shape != (
        len(views),
        EMBEDDING_SIZE,
    ):
        raise ValueError("Unexpected DINO embedding shape: " f"{embeddings.shape}")

    weights = VIEW_WEIGHTS.reshape(
        -1,
        1,
    )

    combined = np.sum(
        embeddings * weights,
        axis=0,
    )

    norm = np.linalg.norm(combined)

    if norm <= 1e-12:
        raise ValueError("Combined DINO embedding is zero.")

    combined = combined / norm

    return combined.astype(np.float32)


# ============================================================
# VALIDATION
# ============================================================


def validate_embedding(
    embedding,
):
    """Validate a 768-dimensional embedding."""

    if embedding is None:
        return False

    array = np.asarray(
        embedding,
        dtype=np.float32,
    )

    if array.ndim != 1:
        return False

    if array.shape[0] != EMBEDDING_SIZE:
        return False

    norm = np.linalg.norm(array)

    if norm <= 1e-12:
        return False

    return True


# ============================================================
# ORB COMPATIBILITY
# ============================================================


def create_orb_descriptors(
    image_path,
    nfeatures=700,
):
    """
    ORB compatibility helper.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    orb = cv2.ORB_create(nfeatures=nfeatures)

    keypoints, descriptors = orb.detectAndCompute(
        gray,
        None,
    )

    return {
        "keypoints": keypoints,
        "descriptors": descriptors,
    }
