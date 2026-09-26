"""
JewelMatch AI
Lightweight Jewellery Image Feature Extraction

This module intentionally does NOT use:
- PyTorch
- Transformers
- DINOv2
- rembg
- U2-Net

The feature pipeline is designed for:
- jewellery design
- shape
- structure
- edges
- silhouette
- local visual patterns
- colour-independent comparison

It also contains lightweight image validation used to reject
obvious non-jewellery inputs such as logos, documents and flat graphics.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

MAX_IMAGE_SIZE = 640

FEATURE_IMAGE_SIZE = 128

GRAY_FEATURE_SIZE = (16, 16)
EDGE_FEATURE_SIZE = (16, 16)
GRADIENT_FEATURE_SIZE = (16, 16)
MASK_FEATURE_SIZE = (16, 16)

FEATURE_SIZE = (16 * 16 + 16 * 16 + 16 * 16 + 16 * 16) * 2

ORB_FEATURES = 700

MIN_IMAGE_WIDTH = 80
MIN_IMAGE_HEIGHT = 80


# ============================================================
# BASIC HELPERS
# ============================================================


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """
    L2 normalize a feature vector.
    """

    vector = np.asarray(
        vector,
        dtype=np.float32,
    ).reshape(-1)

    norm = float(np.linalg.norm(vector))

    if norm < 1e-8:
        return vector

    return vector / norm


def normalize_block(block: np.ndarray) -> np.ndarray:
    """
    Normalize one feature block independently.

    This prevents one block from dominating the entire vector.
    """

    block = np.asarray(
        block,
        dtype=np.float32,
    )

    mean = float(block.mean())
    std = float(block.std())

    if std < 1e-6:
        return np.zeros_like(
            block,
            dtype=np.float32,
        )

    block = (block - mean) / std

    return block.astype(np.float32)


# ============================================================
# IMAGE LOADING
# ============================================================


def load_color_image(image_path: str | Path) -> np.ndarray:
    """
    Load image as BGR.
    """

    image_path = Path(image_path)

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    if height < MIN_IMAGE_HEIGHT or width < MIN_IMAGE_WIDTH:
        raise ValueError("Image resolution is too small for visual matching.")

    largest_side = max(
        height,
        width,
    )

    if largest_side > MAX_IMAGE_SIZE:

        scale = MAX_IMAGE_SIZE / float(largest_side)

        new_width = max(
            1,
            int(width * scale),
        )

        new_height = max(
            1,
            int(height * scale),
        )

        image = cv2.resize(
            image,
            (
                new_width,
                new_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    return image


# ============================================================
# LETTERBOX
# ============================================================


def letterbox(
    image: np.ndarray,
    size: int = FEATURE_IMAGE_SIZE,
) -> np.ndarray:
    """
    Resize without destroying aspect ratio.
    """

    if image is None or image.size == 0:
        raise ValueError("Empty image received.")

    height, width = image.shape[:2]

    scale = min(
        size / float(width),
        size / float(height),
    )

    new_width = max(
        1,
        int(width * scale),
    )

    new_height = max(
        1,
        int(height * scale),
    )

    resized = cv2.resize(
        image,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    if len(image.shape) == 2:

        canvas = np.zeros(
            (
                size,
                size,
            ),
            dtype=image.dtype,
        )

    else:

        canvas = np.zeros(
            (
                size,
                size,
                image.shape[2],
            ),
            dtype=image.dtype,
        )

    x = (size - new_width) // 2
    y = (size - new_height) // 2

    canvas[
        y : y + new_height,
        x : x + new_width,
    ] = resized

    return canvas


# ============================================================
# FOREGROUND MASK
# ============================================================


def create_foreground_mask(
    image: np.ndarray,
) -> np.ndarray:
    """
    Estimate the jewellery/object foreground without using
    a heavy neural segmentation model.

    The method compares pixels against the image border
    background and combines colour-distance and grayscale
    differences.

    This is deliberately colour-agnostic for matching.
    """

    if image is None or image.size == 0:
        raise ValueError("Cannot create foreground mask from empty image.")

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    height, width = gray.shape[:2]

    border_size = max(
        2,
        int(min(height, width) * 0.05),
    )

    border_pixels = np.concatenate(
        [
            lab[:border_size, :, :].reshape(-1, 3),
            lab[-border_size:, :, :].reshape(-1, 3),
            lab[:, :border_size, :].reshape(-1, 3),
            lab[:, -border_size:, :].reshape(-1, 3),
        ],
        axis=0,
    )

    background_color = np.median(
        border_pixels,
        axis=0,
    ).astype(np.float32)

    color_distance = np.linalg.norm(
        lab.astype(np.float32) - background_color.reshape(1, 1, 3),
        axis=2,
    )

    color_distance = cv2.GaussianBlur(
        color_distance,
        (5, 5),
        0,
    )

    color_distance_uint8 = cv2.normalize(
        color_distance,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    _, color_mask = cv2.threshold(
        color_distance_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    background_gray = np.median(
        np.concatenate(
            [
                gray[:border_size, :].reshape(-1),
                gray[-border_size:, :].reshape(-1),
                gray[:, :border_size].reshape(-1),
                gray[:, -border_size:].reshape(-1),
            ]
        )
    )

    gray_distance = np.abs(gray.astype(np.float32) - float(background_gray))

    gray_distance = cv2.GaussianBlur(
        gray_distance,
        (5, 5),
        0,
    )

    gray_distance_uint8 = np.clip(
        gray_distance,
        0,
        255,
    ).astype(np.uint8)

    _, gray_mask = cv2.threshold(
        gray_distance_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    mask = cv2.bitwise_or(
        color_mask,
        gray_mask,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    # Remove very small components.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cleaned = np.zeros_like(mask)

    if contours:

        image_area = float(height * width)

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True,
        )

        for contour in contours[:8]:

            area = cv2.contourArea(contour)

            if area >= image_area * 0.002:

                cv2.drawContours(
                    cleaned,
                    [contour],
                    -1,
                    255,
                    thickness=cv2.FILLED,
                )

    return cleaned


# ============================================================
# FOREGROUND CROP
# ============================================================


def crop_foreground(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop the main foreground object.

    Falls back to the original image when a reliable
    foreground region cannot be identified.
    """

    height, width = image.shape[:2]

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return (
            image.copy(),
            np.ones(
                (
                    height,
                    width,
                ),
                dtype=np.uint8,
            )
            * 255,
        )

    image_area = float(height * width)

    valid_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area >= image_area * 0.01 and area <= image_area * 0.92:
            valid_contours.append(contour)

    if not valid_contours:
        return (
            image.copy(),
            np.ones(
                (
                    height,
                    width,
                ),
                dtype=np.uint8,
            )
            * 255,
        )

    contour = max(
        valid_contours,
        key=cv2.contourArea,
    )

    x, y, w, h = cv2.boundingRect(contour)

    padding = int(max(w, h) * 0.10)

    x1 = max(
        0,
        x - padding,
    )

    y1 = max(
        0,
        y - padding,
    )

    x2 = min(
        width,
        x + w + padding,
    )

    y2 = min(
        height,
        y + h + padding,
    )

    cropped_image = image[
        y1:y2,
        x1:x2,
    ]

    cropped_mask = mask[
        y1:y2,
        x1:x2,
    ]

    if cropped_image.size == 0 or cropped_mask.size == 0:
        return (
            image.copy(),
            np.ones(
                (
                    height,
                    width,
                ),
                dtype=np.uint8,
            )
            * 255,
        )

    return (
        cropped_image,
        cropped_mask,
    )


# ============================================================
# PREPROCESS VIEW
# ============================================================


def prepare_view(
    image: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create normalized grayscale image + silhouette mask.
    """

    if mask is None:
        mask = create_foreground_mask(image)

    image = letterbox(
        image,
        FEATURE_IMAGE_SIZE,
    )

    mask = letterbox(
        mask,
        FEATURE_IMAGE_SIZE,
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # CLAHE improves structure while remaining mostly
    # colour-independent.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    gray = clahe.apply(gray)

    return (
        gray,
        mask,
    )


# ============================================================
# FEATURE BLOCK
# ============================================================


def extract_feature_block(
    gray: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Extract:
    - intensity structure
    - edges
    - gradients
    - silhouette
    """

    gray_small = cv2.resize(
        gray,
        GRAY_FEATURE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    gray_small = gray_small.astype(np.float32) / 255.0

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    edges_small = cv2.resize(
        edges,
        EDGE_FEATURE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    edges_small = edges_small.astype(np.float32) / 255.0

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        gx,
        gy,
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0.0,
        1.0,
        cv2.NORM_MINMAX,
    )

    magnitude_small = cv2.resize(
        magnitude,
        GRADIENT_FEATURE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    silhouette = mask.astype(np.float32) / 255.0

    silhouette_small = cv2.resize(
        silhouette,
        MASK_FEATURE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    blocks = [
        normalize_block(gray_small.flatten()),
        normalize_block(edges_small.flatten()),
        normalize_block(magnitude_small.flatten()),
        normalize_block(silhouette_small.flatten()),
    ]

    return np.concatenate(blocks).astype(np.float32)


# ============================================================
# EMBEDDING
# ============================================================


def create_embedding(
    image_path: str | Path,
) -> np.ndarray:
    """
    Create the final lightweight visual embedding.

    Two views are used:

    1. Original image
    2. Foreground/object crop

    The object crop receives more weight so background does
    not dominate the search.
    """

    image = load_color_image(image_path)

    foreground_mask = create_foreground_mask(image)

    cropped_image, cropped_mask = crop_foreground(
        image,
        foreground_mask,
    )

    full_gray, full_mask = prepare_view(
        image,
        foreground_mask,
    )

    crop_gray, crop_mask = prepare_view(
        cropped_image,
        cropped_mask,
    )

    full_features = extract_feature_block(
        full_gray,
        full_mask,
    )

    crop_features = extract_feature_block(
        crop_gray,
        crop_mask,
    )

    # Object crop is intentionally more important.
    combined = np.concatenate(
        [
            full_features * 0.35,
            crop_features * 0.65,
        ]
    )

    return l2_normalize(combined).astype(np.float32)


# ============================================================
# ORB LOCAL FEATURES
# ============================================================


def extract_orb_descriptors(
    image_path: str | Path,
) -> dict:
    """
    Extract lightweight local visual descriptors.

    ORB is used instead of SIFT because ORB descriptors are
    compact binary descriptors and require very little memory.
    """

    image = load_color_image(image_path)

    mask = create_foreground_mask(image)

    cropped_image, cropped_mask = crop_foreground(
        image,
        mask,
    )

    gray_full = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray_crop = cv2.cvtColor(
        cropped_image,
        cv2.COLOR_BGR2GRAY,
    )

    orb = cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31,
        fastThreshold=15,
    )

    keypoints_full, descriptors_full = orb.detectAndCompute(
        gray_full,
        None,
    )

    keypoints_crop, descriptors_crop = orb.detectAndCompute(
        gray_crop,
        cropped_mask,
    )

    return {
        "full": descriptors_full,
        "crop": descriptors_crop,
        "keypoints_full": (len(keypoints_full) if keypoints_full else 0),
        "keypoints_crop": (len(keypoints_crop) if keypoints_crop else 0),
    }


# ============================================================
# IMAGE METRICS
# ============================================================


def calculate_image_metrics(
    image_path: str | Path,
) -> dict:
    """
    Calculate lightweight metrics used by the no-match gate.
    """

    image = load_color_image(image_path)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    mask = create_foreground_mask(image)

    image_area = float(image.shape[0] * image.shape[1])

    foreground_pixels = float(np.count_nonzero(mask))

    foreground_ratio = foreground_pixels / image_area if image_area > 0 else 0.0

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    edge_density = (
        float(np.count_nonzero(edges)) / image_area if image_area > 0 else 0.0
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    rectangularity = 0.0
    bbox_aspect = 0.0
    contour_count = len(contours)

    if contours:

        largest = max(
            contours,
            key=cv2.contourArea,
        )

        area = cv2.contourArea(largest)

        x, y, w, h = cv2.boundingRect(largest)

        bbox_area = float(
            max(
                1,
                w * h,
            )
        )

        rectangularity = area / bbox_area

        bbox_aspect = w / float(h) if h > 0 else 0.0

    gray_std = float(gray.std())

    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "foreground_ratio": float(foreground_ratio),
        "edge_density": float(edge_density),
        "rectangularity": float(rectangularity),
        "bbox_aspect": float(bbox_aspect),
        "contour_count": int(contour_count),
        "gray_std": float(gray_std),
    }


# ============================================================
# QUERY VALIDATION
# ============================================================


def validate_query_image(
    image_path: str | Path,
) -> dict:
    """
    Reject obvious non-jewellery images.

    This is intentionally conservative.

    It is NOT claiming to be a universal jewellery classifier.
    It is a search safety gate preventing obviously irrelevant
    graphics/logos/documents from being forced into the catalogue.
    """

    metrics = calculate_image_metrics(image_path)

    reasons = []

    width = metrics["width"]
    height = metrics["height"]

    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        reasons.append("Image resolution is too small.")

    if metrics["gray_std"] < 7.0:
        reasons.append("Image has very little visual information.")

    if metrics["edge_density"] < 0.003:
        reasons.append("Image contains very little structural detail.")

    # A large, nearly perfect rectangle is commonly:
    # - logo
    # - screenshot
    # - document
    # - product card
    #
    # Jewellery itself normally has a more irregular silhouette.
    if metrics["rectangularity"] >= 0.94 and metrics["foreground_ratio"] >= 0.30:
        reasons.append("Image appears to contain a large rectangular graphic/object.")

    # Very large foreground occupying almost the whole frame
    # with a highly rectangular shape is another strong
    # non-jewellery signal.
    if metrics["foreground_ratio"] >= 0.82 and metrics["rectangularity"] >= 0.90:
        reasons.append("Foreground has a document/logo-like rectangular structure.")

    # A completely empty mask is suspicious.
    if metrics["foreground_ratio"] < 0.005:
        reasons.append("No clear foreground object was detected.")

    return {
        "valid": len(reasons) == 0,
        "reasons": reasons,
        "metrics": metrics,
    }


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================


def get_embedding(
    image_path: str | Path,
) -> np.ndarray:
    """
    Backward-compatible alias.
    """

    return create_embedding(image_path)
