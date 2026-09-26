"""
JewelMatch AI - Lightweight image feature extraction.

CPU-only OpenCV implementation.
No torch, transformers, DINOv2, rembg or ONNX model required.
"""

from pathlib import Path

import cv2
import numpy as np

FEATURE_SIZE = 256
MAX_IMAGE_SIZE = 512


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float32).flatten()

    norm = float(np.linalg.norm(vector))

    if norm < 1e-8:
        return np.zeros_like(vector, dtype=np.float32)

    return (vector / norm).astype(np.float32)


def load_grayscale_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    largest_side = max(height, width)

    if largest_side > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / float(largest_side)

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    return image


def crop_to_design(gray):
    """
    Attempts to remove large empty borders/background.

    This is intentionally lightweight and does not use rembg.
    """

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(
        blurred,
        40,
        120,
    )

    kernel = np.ones((5, 5), np.uint8)

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return gray

    height, width = gray.shape

    image_area = float(max(1, height * width))

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < image_area * 0.002:
            continue

        x, y, contour_width, contour_height = cv2.boundingRect(contour)

        box_area = contour_width * contour_height

        if box_area < image_area * 0.01:
            continue

        candidates.append(
            (
                box_area,
                x,
                y,
                contour_width,
                contour_height,
            )
        )

    if not candidates:
        return gray

    _, x, y, crop_width, crop_height = max(
        candidates,
        key=lambda item: item[0],
    )

    padding = int(max(crop_width, crop_height) * 0.08)

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(
        width,
        x + crop_width + padding,
    )

    y2 = min(
        height,
        y + crop_height + padding,
    )

    cropped = gray[y1:y2, x1:x2]

    if cropped.size < 100:
        return gray

    return cropped


def resize_flat(image, width, height):
    resized = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )

    return (resized.astype(np.float32) / 255.0).flatten()


def extract_intensity_features(gray):
    """
    64 values.
    """

    return resize_flat(
        gray,
        8,
        8,
    )


def extract_edge_features(gray):
    """
    64 values.
    """

    edges = cv2.Canny(
        gray,
        40,
        120,
    )

    return resize_flat(
        edges,
        8,
        8,
    )


def extract_gradient_features(gray):
    """
    64 values.
    """

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

    minimum, maximum, _, _ = cv2.minMaxLoc(magnitude)

    if maximum > minimum:

        magnitude = (magnitude - minimum) / (maximum - minimum)

    else:

        magnitude = np.zeros_like(magnitude)

    magnitude = (magnitude * 255.0).astype(np.uint8)

    return resize_flat(
        magnitude,
        8,
        8,
    )


def extract_shape_features(gray):
    """
    64 values.

    Uses contour geometry and Hu moments.
    """

    features = []

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    contour_sets = []

    contour_sets.append(
        cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0]
    )

    inverted = cv2.bitwise_not(threshold)

    contour_sets.append(
        cv2.findContours(
            inverted,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0]
    )

    contours = []

    for contour_set in contour_sets:
        contours.extend(contour_set)

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    height, width = gray.shape

    image_area = float(max(1, height * width))

    for contour in contours[:5]:

        area = float(cv2.contourArea(contour))

        perimeter = float(
            cv2.arcLength(
                contour,
                True,
            )
        )

        x, y, contour_width, contour_height = cv2.boundingRect(contour)

        area_ratio = area / image_area

        aspect_ratio = contour_width / float(max(1, contour_height))

        circularity = (
            (4.0 * np.pi * area) / (perimeter * perimeter) if perimeter > 1e-8 else 0.0
        )

        extent = area / float(
            max(
                1,
                contour_width * contour_height,
            )
        )

        features.extend(
            [
                area_ratio,
                min(
                    aspect_ratio,
                    5.0,
                )
                / 5.0,
                min(
                    circularity,
                    1.0,
                ),
                min(
                    extent,
                    1.0,
                ),
            ]
        )

    moments = cv2.moments(threshold)

    hu = cv2.HuMoments(moments).flatten()

    for value in hu:

        value = -np.sign(value) * np.log10(abs(value) + 1e-12)

        features.append(
            float(
                np.clip(
                    value / 20.0,
                    -1.0,
                    1.0,
                )
            )
        )

    features = np.asarray(
        features[:27],
        dtype=np.float32,
    )

    output = np.zeros(
        64,
        dtype=np.float32,
    )

    output[: len(features)] = features

    return output


def create_embedding(image_path):

    gray = load_grayscale_image(image_path)

    gray = crop_to_design(gray)

    intensity = extract_intensity_features(gray)

    edges = extract_edge_features(gray)

    gradient = extract_gradient_features(gray)

    shape = extract_shape_features(gray)

    vector = np.concatenate(
        [
            intensity,
            edges,
            gradient,
            shape,
        ]
    ).astype(np.float32)

    if vector.shape != (256,):
        raise ValueError(f"Invalid embedding size: {vector.shape}")

    return normalize_vector(vector)


def get_embedding(image_path):
    return create_embedding(image_path)
