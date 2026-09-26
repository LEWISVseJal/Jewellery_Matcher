import os
import cv2
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_SIZE = 256
MAX_IMAGE_SIZE = 512


# ============================================================
# IMAGE LOADING
# ============================================================


def load_grayscale_image(image_path):
    """
    Load an image as grayscale and resize it to a manageable size.
    """

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape

    largest_side = max(
        height,
        width,
    )

    if largest_side > MAX_IMAGE_SIZE:

        scale = MAX_IMAGE_SIZE / largest_side

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
# VECTOR NORMALIZATION
# ============================================================


def normalize_vector(vector):
    """
    L2-normalize a feature vector.
    """

    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    norm = np.linalg.norm(vector)

    if norm < 1e-8:
        return vector

    return vector / norm


# ============================================================
# FIXED FEATURE SIZE
# ============================================================


def resize_feature_vector(
    feature,
    size=FEATURE_SIZE,
):
    """
    Make sure every image produces exactly
    the same feature-vector length.
    """

    feature = np.asarray(
        feature,
        dtype=np.float32,
    ).flatten()

    if len(feature) == size:
        return feature

    if len(feature) > size:
        return feature[:size]

    output = np.zeros(
        size,
        dtype=np.float32,
    )

    output[: len(feature)] = feature

    return output


# ============================================================
# SHAPE FEATURES
# ============================================================


def extract_shape_features(
    gray,
):
    """
    Extract basic shape and contour information.

    These features are intentionally independent
    of jewellery colour.
    """

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

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    features = []

    image_area = gray.shape[0] * gray.shape[1]

    for contour in contours[:5]:

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        x, y, width, height = cv2.boundingRect(contour)

        area_ratio = area / image_area if image_area > 0 else 0.0

        aspect_ratio = width / height if height > 0 else 0.0

        circularity = (
            (4.0 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
        )

        extent = area / (width * height) if width > 0 and height > 0 else 0.0

        features.extend(
            [
                area_ratio,
                aspect_ratio,
                circularity,
                extent,
            ]
        )

    while len(features) < 20:
        features.append(0.0)

    return np.asarray(
        features[:20],
        dtype=np.float32,
    )


# ============================================================
# EDGE FEATURES
# ============================================================


def extract_edge_features(
    gray,
):
    """
    Extract edge structure at multiple scales.
    """

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    features = []

    for size in [
        (16, 16),
        (32, 32),
    ]:

        resized = cv2.resize(
            edges,
            size,
            interpolation=cv2.INTER_AREA,
        )

        resized = resized.astype(np.float32) / 255.0

        features.append(resized.flatten())

    return np.concatenate(features)


# ============================================================
# GRAYSCALE STRUCTURE
# ============================================================


def extract_intensity_features(
    gray,
):
    """
    Extract grayscale structure.

    This deliberately avoids RGB/color features so
    gold and prototype versions can match.
    """

    features = []

    for size in [
        (16, 16),
        (32, 32),
    ]:

        resized = cv2.resize(
            gray,
            size,
            interpolation=cv2.INTER_AREA,
        )

        resized = resized.astype(np.float32) / 255.0

        features.append(resized.flatten())

    return np.concatenate(features)


# ============================================================
# GRADIENT FEATURES
# ============================================================


def extract_gradient_features(
    gray,
):
    """
    Extract gradient/texture structure.
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

    magnitude = cv2.resize(
        magnitude,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0.0,
        1.0,
        cv2.NORM_MINMAX,
    )

    return magnitude.flatten().astype(np.float32)


# ============================================================
# MAIN EMBEDDING
# ============================================================


def create_embedding(
    image_path,
):
    """
    Create a lightweight visual feature vector.

    IMPORTANT:
    - No PyTorch
    - No Transformers
    - No DINOv2
    - CPU only
    - Low memory
    """

    gray = load_grayscale_image(image_path)

    intensity_features = extract_intensity_features(gray)

    edge_features = extract_edge_features(gray)

    gradient_features = extract_gradient_features(gray)

    shape_features = extract_shape_features(gray)

    combined = np.concatenate(
        [
            intensity_features,
            edge_features,
            gradient_features,
            shape_features,
        ]
    )

    combined = resize_feature_vector(
        combined,
        FEATURE_SIZE,
    )

    combined = normalize_vector(combined)

    return combined.astype(np.float32)


# ============================================================
# COMPATIBILITY FUNCTION
# ============================================================


def get_embedding(
    image_path,
):
    return create_embedding(image_path)
