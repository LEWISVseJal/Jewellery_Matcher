"""
JewelMatch AI
DINOv2-Small embedding service

Design goals:
- CPU friendly
- Render friendly
- Load DINO only once
- 384-dimensional embeddings
- Multiple views for jewellery
- Reduced dependence on colour
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import torch
from PIL import Image

from transformers import AutoImageProcessor, AutoModel

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv("DINO_MODEL", "facebook/dinov2-small")

MAX_IMAGE_SIZE = 768

# DINOv2-Small output dimension
EMBEDDING_SIZE = 384

DEVICE = torch.device("cpu")

# Prevent excessive CPU threading on Render
try:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
except Exception:
    pass


# ============================================================
# GLOBAL MODEL
# ============================================================

_processor = None
_model = None


def get_model():
    """
    Load DINOv2 only once.

    Important:
    Never call AutoModel.from_pretrained() for every image.
    """

    global _processor, _model

    if _model is not None:
        return _processor, _model

    print("=" * 70)
    print("[DINO] Loading DINOv2-Small")
    print(f"[DINO] Model: {MODEL_NAME}")
    print(f"[DINO] Device: {DEVICE}")
    print("=" * 70)

    _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    _model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

    _model.to(DEVICE)
    _model.eval()

    print("[DINO] Model loaded successfully")

    return _processor, _model


# ============================================================
# IMAGE LOADING
# ============================================================


def load_image(image_path: str) -> Optional[Image.Image]:
    """
    Load image and resize it before sending to DINO.
    """

    try:
        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        largest = max(width, height)

        if largest > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / float(largest)

            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))

            image = image.resize(new_size, Image.Resampling.LANCZOS)

        return image

    except Exception as exc:
        print(f"[DINO] Image loading failed: {exc}")
        return None


# ============================================================
# IMAGE PREPARATION
# ============================================================


def create_grayscale_view(image: Image.Image) -> Image.Image:
    """
    Convert image to grayscale and return RGB.

    This reduces dependence on:
    gold / green / silver / black colour differences.
    """

    gray = image.convert("L")

    return gray.convert("RGB")


def create_center_crop(image: Image.Image) -> Image.Image:
    """
    Crop the central jewellery region.

    This helps when catalogue/query images have different
    amounts of background.
    """

    width, height = image.size

    crop_ratio = 0.82

    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio)

    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)

    right = min(width, left + crop_width)
    bottom = min(height, top + crop_height)

    cropped = image.crop((left, top, right, bottom))

    return cropped


def create_zoom_view(image: Image.Image) -> Image.Image:
    """
    Slight zoom into the jewellery.

    Helps DINO focus more on the design rather than the
    surrounding background.
    """

    width, height = image.size

    ratio = 0.70

    crop_width = int(width * ratio)
    crop_height = int(height * ratio)

    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)

    right = min(width, left + crop_width)
    bottom = min(height, top + crop_height)

    cropped = image.crop((left, top, right, bottom))

    return cropped


def create_views(image: Image.Image) -> List[Image.Image]:
    """
    Generate views used for the final embedding.

    View 1:
        Original

    View 2:
        Grayscale

    View 3:
        Center crop

    View 4:
        Zoom crop
    """

    return [
        image,
        create_grayscale_view(image),
        create_center_crop(image),
        create_zoom_view(image),
    ]


# ============================================================
# DINO FEATURE EXTRACTION
# ============================================================


@torch.inference_mode()
def extract_dino_embedding(image: Image.Image) -> np.ndarray:

    processor, model = get_model()

    inputs = processor(images=image, return_tensors="pt")

    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    outputs = model(**inputs)

    # CLS token
    embedding = outputs.last_hidden_state[:, 0, :]

    embedding = embedding.float()

    # L2 normalize
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    result = embedding[0].cpu().numpy()

    return result.astype(np.float32)


# ============================================================
# FINAL EMBEDDING
# ============================================================


def create_embedding(image_path: str) -> np.ndarray:
    """
    Generate a robust jewellery embedding.

    We combine several views:

        Original       -> 20%
        Grayscale      -> 40%
        Center crop    -> 25%
        Zoom crop      -> 15%

    Grayscale receives higher weight because the application
    needs gold/green/prototype cross-material matching.
    """

    image = load_image(image_path)

    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    print(f"[DINO] Creating embedding: " f"{Path(image_path).name}")

    views = create_views(image)

    weights = np.array([0.20, 0.40, 0.25, 0.15], dtype=np.float32)

    embeddings = []

    for index, view in enumerate(views):

        print(f"[DINO] Processing view " f"{index + 1}/{len(views)}")

        emb = extract_dino_embedding(view)

        embeddings.append(emb)

    embeddings = np.stack(embeddings, axis=0)

    combined = np.sum(embeddings * weights[:, None], axis=0)

    # Final normalization
    norm = np.linalg.norm(combined)

    if norm > 1e-8:
        combined = combined / norm

    return combined.astype(np.float32)


# ============================================================
# COMPATIBILITY FUNCTIONS
# ============================================================


def get_embedding(image_path: str) -> np.ndarray:
    """
    Backwards-compatible alias.
    """

    return create_embedding(image_path)


def extract_orb_descriptors(image_path: str):
    """
    Optional lightweight local descriptor.

    Kept for compatibility with older code.
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    orb = cv2.ORB_create(nfeatures=500)

    keypoints, descriptors = orb.detectAndCompute(image, None)

    return descriptors


# ============================================================
# LIGHTWEIGHT IMAGE VALIDATION
# ============================================================


def calculate_image_metrics(image_path: str) -> dict:

    image = cv2.imread(str(image_path))

    if image is None:
        return {"valid": False, "reason": "image_load_failed"}

    height, width = image.shape[:2]

    if min(height, width) < 80:
        return {"valid": False, "reason": "image_too_small"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_std = float(np.std(gray))

    edges = cv2.Canny(gray, 60, 150)

    edge_density = float(np.mean(edges > 0))

    return {
        "valid": True,
        "width": width,
        "height": height,
        "gray_std": gray_std,
        "edge_density": edge_density,
    }


def validate_query_image(image_path: str) -> tuple[bool, str]:

    metrics = calculate_image_metrics(image_path)

    if not metrics.get("valid"):
        return False, metrics.get("reason", "invalid_image")

    if metrics["gray_std"] < 8:
        return False, "image_has_too_little_detail"

    if metrics["edge_density"] < 0.002:
        return False, "image_has_too_little_structure"

    return True, "ok"


# ============================================================
# FOREGROUND COMPATIBILITY
# ============================================================


def create_foreground_mask(image: np.ndarray) -> np.ndarray:

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def crop_foreground(image: np.ndarray, mask: np.ndarray):

    ys, xs = np.where(mask > 0)

    if len(xs) < 50:
        return image, mask

    x1 = max(0, int(xs.min()))
    x2 = min(image.shape[1], int(xs.max()) + 1)

    y1 = max(0, int(ys.min()))
    y2 = min(image.shape[0], int(ys.max()) + 1)

    return (image[y1:y2, x1:x2], mask[y1:y2, x1:x2])


def prepare_view(image: np.ndarray, mask: Optional[np.ndarray] = None, size: int = 224):

    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    return resized
