import os
import gc
import numpy as np
import torch

from PIL import Image, ImageOps, ImageFilter
from transformers import AutoImageProcessor, AutoModel


# ============================================================
# LOCAL MODEL CACHE
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)

HF_CACHE = os.path.join(
    PROJECT_DIR,
    "model_cache",
    "huggingface"
)

os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "facebook/dinov2-base"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"[EMBEDDING] Device: {DEVICE}")
print(f"[EMBEDDING] Model: {MODEL_NAME}")


_processor = None
_model = None


def _load_model():
    global _processor
    global _model

    if _processor is None or _model is None:

        print("[EMBEDDING] Loading DINOv2...")

        _processor = AutoImageProcessor.from_pretrained(
            MODEL_NAME,
            cache_dir=HF_CACHE,
            local_files_only=False
        )

        _model = AutoModel.from_pretrained(
            MODEL_NAME,
            cache_dir=HF_CACHE,
            local_files_only=False
        )

        _model.to(DEVICE)
        _model.eval()

        print("[EMBEDDING] DINOv2 loaded.")

    return _processor, _model


# ============================================================
# COLOUR-INVARIANT IMAGE PREPARATION
# ============================================================

def prepare_design_image(image_path):
    """
    Converts jewellery image into a colour-invariant representation.

    The image is:
        1. Loaded as RGB
        2. Converted to grayscale
        3. Contrast enhanced
        4. Converted back to RGB

    DINOv2 therefore sees design/structure much more strongly
    than the original material colour.
    """

    image = Image.open(image_path).convert("RGB")

    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # Improve contrast
    image = ImageOps.autocontrast(image)

    # Slight sharpening
    image = image.filter(ImageFilter.SHARPEN)

    # DINO expects 3 channels
    image = Image.merge(
        "RGB",
        (image, image, image)
    )

    return image


# ============================================================
# DINO EMBEDDING
# ============================================================

def create_embedding(image_path):
    """
    Creates a normalized 768-dimensional DINOv2 embedding.

    IMPORTANT:
    The image is converted to grayscale before embedding.
    """

    print()
    print("Creating colour-invariant embedding:")
    print(image_path)

    processor, model = _load_model()

    image = prepare_design_image(image_path)

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        # CLS token
        embedding = outputs.last_hidden_state[:, 0, :]

        # L2 normalize
        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )

    embedding = embedding.cpu().numpy()[0].astype(
        np.float32
    )

    print("Colour-invariant embedding created.")
    print(f"Shape: {embedding.shape}")

    # Cleanup
    del inputs
    del outputs

    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return embedding


# ============================================================
# COMPATIBILITY ALIAS
# ============================================================

def get_embedding(image_path):
    return create_embedding(image_path)