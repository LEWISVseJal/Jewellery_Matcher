"""
JewelMatch AI - Lightweight Jewellery Segmentation

No:
- rembg
- U2-Net
- pymatting
- numba
- ONNX

Used by:
- embedding.py
- add/update catalogue
- query processing
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

# ============================================================
# MASK CREATION
# ============================================================


def get_mask(
    image: np.ndarray,
) -> np.ndarray:
    """
    Create a lightweight foreground mask.

    Combines:
    - border background estimation
    - global difference
    - adaptive threshold
    - edge information
    - morphology
    """

    if image is None or image.size == 0:
        raise ValueError("Invalid image supplied to segmentation.")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    height, width = gray.shape[:2]

    if height < 20 or width < 20:
        return np.zeros_like(gray)

    # --------------------------------------------------------
    # Border background estimate
    # --------------------------------------------------------

    border_pixels = np.concatenate(
        [
            gray[0, :],
            gray[-1, :],
            gray[:, 0],
            gray[:, -1],
        ]
    )

    background_value = float(np.median(border_pixels))

    difference = cv2.absdiff(
        gray,
        np.full_like(
            gray,
            int(background_value),
        ),
    )

    _, global_mask = cv2.threshold(
        difference,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # --------------------------------------------------------
    # Adaptive mask
    # --------------------------------------------------------

    adaptive_mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )

    # --------------------------------------------------------
    # Edge mask
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        40,
        140,
    )

    edge_kernel = np.ones(
        (5, 5),
        np.uint8,
    )

    edges = cv2.dilate(
        edges,
        edge_kernel,
        iterations=1,
    )

    # --------------------------------------------------------
    # Combine masks
    # --------------------------------------------------------

    mask = cv2.bitwise_or(
        global_mask,
        adaptive_mask,
    )

    mask = cv2.bitwise_or(
        mask,
        edges,
    )

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    kernel_medium = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_medium,
        iterations=2,
    )

    return mask


# ============================================================
# COMPONENT FILTERING
# ============================================================


def _clean_components(
    mask,
    minimum_area_ratio=0.005,
):
    """
    Remove very small disconnected noise.
    """

    height, width = mask.shape[:2]

    image_area = height * width

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if count <= 1:
        return mask

    cleaned = np.zeros_like(mask)

    components = []

    for label in range(
        1,
        count,
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA,
        ]

        if area <= 0:
            continue

        ratio = area / float(image_area)

        if ratio >= minimum_area_ratio:
            components.append(
                (
                    area,
                    label,
                )
            )

    components.sort(reverse=True)

    # Keep strongest components.
    for _, label in components[:5]:

        cleaned[labels == label] = 255

    return cleaned


# ============================================================
# SEGMENTED CROP
# ============================================================


def get_segmented_crop(
    image: np.ndarray,
):
    """
    Return:

        cropped_image,
        cropped_mask
    """

    if image is None or image.size == 0:
        raise ValueError("Invalid image.")

    mask = get_mask(image)

    mask = _clean_components(mask)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return image, mask

    height, width = image.shape[:2]

    image_area = height * width

    valid_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area <= 0:
            continue

        ratio = area / float(image_area)

        if ratio >= 0.005 and ratio <= 0.98:
            valid_contours.append(
                (
                    area,
                    contour,
                )
            )

    if not valid_contours:
        return image, mask

    valid_contours.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    largest = valid_contours[0][1]

    x, y, w, h = cv2.boundingRect(largest)

    if w <= 0 or h <= 0:
        return image, mask

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    padding = max(
        8,
        int(max(w, h) * 0.12),
    )

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

    crop = image[
        y1:y2,
        x1:x2,
    ]

    crop_mask = mask[
        y1:y2,
        x1:x2,
    ]

    if crop is None or crop.size == 0:
        return image, mask

    return crop, crop_mask


# ============================================================
# FILE SEGMENTATION
# ============================================================


def segment_jewellery(
    input_path: str,
    output_path: str,
) -> str:
    """
    Segment and save jewellery image.
    """

    print(f"[SEGMENTATION] Processing: {input_path}")

    image = cv2.imread(str(input_path))

    if image is None:
        raise ValueError(f"Unable to read image: {input_path}")

    crop, _ = get_segmented_crop(image)

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output),
        crop,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            92,
        ],
    )

    print(f"[SEGMENTATION] Saved: {output}")

    return str(output)


# ============================================================
# COMPATIBILITY
# ============================================================


def create_segmented_image(
    input_path: str,
    output_path: str,
) -> str:
    return segment_jewellery(
        input_path,
        output_path,
    )


def get_segmentation_session():
    """
    Compatibility function.

    No ONNX segmentation session is used.
    """

    return None
