import os
import gc
import numpy as np

# ============================================================
# JEWELMATCH AI
# Lightweight DINOv2-Small Embedding Service
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)

HF_CACHE = os.path.join(PROJECT_DIR, "model_cache", "huggingface")

os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")

# IMPORTANT:
# DINOv2-small = 384-dimensional embedding
MODEL_NAME = "facebook/dinov2-small"

_model = None
_processor = None


# ============================================================
# LOAD MODEL ONCE
# ============================================================


def _load_model():
    global _model, _processor

    if _model is not None and _processor is not None:
        return _processor, _model

    print("=" * 70)
    print("[EMBEDDING] Loading DINOv2-Small...")
    print("=" * 70)

    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device("cpu")

    _processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME, cache_dir=HF_CACHE, local_files_only=False
    )

    _model = AutoModel.from_pretrained(
        MODEL_NAME, cache_dir=HF_CACHE, local_files_only=False
    )

    _model.to(device)
    _model.eval()

    print("[EMBEDDING] DINOv2-Small loaded.")
    print("[EMBEDDING] Embedding dimension: 384")

    return _processor, _model


# ============================================================
# IMAGE PREPARATION
# ============================================================


def prepare_design_image(image_path):

    from PIL import Image, ImageOps, ImageFilter

    image = Image.open(image_path).convert("RGB")

    # Remove most colour influence.
    # This helps Gold <-> Prototype matching.
    image = ImageOps.grayscale(image)

    image = ImageOps.autocontrast(image)

    image = image.filter(ImageFilter.SHARPEN)

    image = Image.merge("RGB", (image, image, image))

    return image


# ============================================================
# CREATE EMBEDDING
# ============================================================


def create_embedding(image_path):

    processor, model = _load_model()

    import torch

    device = torch.device("cpu")

    image = prepare_design_image(image_path)

    inputs = processor(images=image, return_tensors="pt")

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():

        outputs = model(**inputs)

        # CLS token
        embedding = outputs.last_hidden_state[:, 0, :]

        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        embedding = embedding.detach().cpu().numpy()[0].astype(np.float32)

    # Cleanup temporary objects
    del inputs
    del outputs
    del image

    gc.collect()

    print(f"[EMBEDDING] {os.path.basename(image_path)} " f"-> {embedding.shape}")

    return embedding


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================


def get_embedding(image_path):
    return create_embedding(image_path)
