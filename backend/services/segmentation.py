"""
Lightweight jewellery segmentation.

No:
- rembg
- U2-Net
- pymatting
- numba
- ONNX

This keeps Add Jewellery safe for Render.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def get_mask(image: np.ndarray) -> np.ndarray:
    """
    Lightweight foreground estimation.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Estimate background from border
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])

    background_value = float(np.median(border))

    difference = cv2.absdiff(gray, np.full_like(gray, int(background_value)))

    # Adaptive threshold
    _, mask = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def get_segmented_crop(image: np.ndarray):

    mask = get_mask(image)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image, mask

    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)

    image_area = image.shape[0] * image.shape[1]

    if area < image_area * 0.005:
        return image, mask

    x, y, w, h = cv2.boundingRect(largest)

    padding = int(max(w, h) * 0.10)

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(image.shape[1], x + w + padding)

    y2 = min(image.shape[0], y + h + padding)

    return (image[y1:y2, x1:x2], mask[y1:y2, x1:x2])


def segment_jewellery(input_path: str, output_path: str) -> str:

    print(f"[SEGMENTATION] Processing: " f"{input_path}")

    image = cv2.imread(str(input_path))

    if image is None:
        raise ValueError(f"Unable to read image: {input_path}")

    crop, mask = get_segmented_crop(image)

    # Save lightweight cropped version
    output = Path(output_path)

    output.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"[SEGMENTATION] Saved: {output}")

    return str(output)


def create_segmented_image(input_path: str, output_path: str) -> str:

    return segment_jewellery(input_path, output_path)


def get_segmentation_session():
    """
    Compatibility function.

    Heavy segmentation model no longer exists.
    """

    return None
