import os
import gc
import numpy as np

# ============================================================
# LOCAL MODEL CACHE
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_DIR = os.path.dirname(BASE_DIR)

HF_CACHE = os.path.join(PROJECT_DIR, "model_cache", "huggingface")

os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "facebook/dinov2-base"

_model = None
_processor = None


# ============================================================
# LAZY MODEL LOADING
# ============================================================


def _load_model():

    global _model
    global _processor

    if _model is not None and _processor is not None:
        return _processor, _model

    print()
    print("=" * 60)
    print("[EMBEDDING] Loading DINOv2...")
    print("=" * 60)

    # Import heavy libraries ONLY when the model is required.
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoModel,
    )

    # Force CPU on Render.
    device = torch.device("cpu")

    print(f"[EMBEDDING] Device: {device}")
    print(f"[EMBEDDING] Model: {MODEL_NAME}")

    _processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME, cache_dir=HF_CACHE, local_files_only=False
    )

    _model = AutoModel.from_pretrained(
        MODEL_NAME, cache_dir=HF_CACHE, local_files_only=False
    )

    _model.to(device)

    _model.eval()

    print("[EMBEDDING] DINOv2 loaded.")

    return _processor, _model


# ============================================================
# COLOUR-INVARIANT IMAGE PREPARATION
# ============================================================


def prepare_design_image(image_path):

    from PIL import (
        Image,
        ImageOps,
        ImageFilter,
    )

    image = Image.open(image_path).convert("RGB")

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    image = ImageOps.grayscale(image)

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    image = ImageOps.autocontrast(image)

    # --------------------------------------------------------
    # Slight sharpening
    # --------------------------------------------------------

    image = image.filter(ImageFilter.SHARPEN)

    # --------------------------------------------------------
    # Convert back to RGB
    # DINOv2 expects 3 channels.
    # --------------------------------------------------------

    image = Image.merge("RGB", (image, image, image))

    return image


# ============================================================
# DINO EMBEDDING
# ============================================================


def create_embedding(image_path):

    print()
    print("[EMBEDDING] Creating colour-invariant embedding:")

    print(image_path)

    # --------------------------------------------------------
    # Lazy-load model
    # --------------------------------------------------------

    processor, model = _load_model()

    # Import torch only when needed.
    import torch

    device = torch.device("cpu")

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    image = prepare_design_image(image_path)

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    inputs = processor(images=image, return_tensors="pt")

    inputs = {key: value.to(device) for key, value in inputs.items()}

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.inference_mode():

        outputs = model(**inputs)

        # CLS token
        embedding = outputs.last_hidden_state[:, 0, :]

        # L2 normalization
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        # Copy to CPU immediately
        embedding = embedding.detach().cpu().numpy()[0].astype(np.float32)

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del inputs
    del outputs
    del image

    gc.collect()

    print("[EMBEDDING] Embedding created.")

    print(f"[EMBEDDING] Shape: {embedding.shape}")

    return embedding


# ============================================================
# COMPATIBILITY ALIAS
# ============================================================


def get_embedding(image_path):

    return create_embedding(image_path)
