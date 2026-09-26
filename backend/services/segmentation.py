import os
import io
import gc

# ============================================================
# U2-NET CACHE
# ============================================================

PROJECT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

U2NET_HOME = os.path.join(
    PROJECT_DIR,
    "model_cache",
    "rembg"
)

os.environ["U2NET_HOME"] = U2NET_HOME

os.makedirs(
    U2NET_HOME,
    exist_ok=True
)


# ============================================================
# IMPORTS
# ============================================================

from PIL import Image
from rembg import remove, new_session


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "u2net"

# Keep memory usage manageable on CPU.
MAX_IMAGE_SIZE = 1024


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("JEWELLERY SEGMENTATION")
print("=" * 60)

print("Model:", MODEL_NAME)
print("U2-Net cache:", U2NET_HOME)

print()
print("Loading U2-Net segmentation model...")

SEGMENTATION_SESSION = new_session(
    MODEL_NAME
)

print("U2-Net loaded successfully.")
print("=" * 60)


# ============================================================
# RESIZE IMAGE
# ============================================================

def resize_for_segmentation(image):

    width, height = image.size

    largest_dimension = max(
        width,
        height
    )

    if largest_dimension <= MAX_IMAGE_SIZE:
        return image

    scale = (
        MAX_IMAGE_SIZE /
        largest_dimension
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    print(
        "Resizing:",
        f"{width}x{height}",
        "->",
        f"{new_width}x{new_height}"
    )

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )


# ============================================================
# SEGMENT JEWELLERY
# ============================================================

def segment_jewellery(
    image_path,
    output_path
):

    print()
    print("Segmenting image:")
    print(image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Input image not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    input_image = Image.open(
        image_path
    ).convert("RGB")

    print(
        "Original size:",
        input_image.size
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized_image = resize_for_segmentation(
        input_image
    )

    print(
        "Segmentation size:",
        resized_image.size
    )

    # We no longer need the original image.
    if resized_image is not input_image:
        input_image.close()

    # --------------------------------------------------------
    # Convert to JPEG
    # --------------------------------------------------------

    input_buffer = io.BytesIO()

    resized_image.save(
        input_buffer,
        format="JPEG",
        quality=85
    )

    resized_image.close()

    input_bytes = input_buffer.getvalue()

    input_buffer.close()

    gc.collect()

    # --------------------------------------------------------
    # Background removal
    # --------------------------------------------------------

    print(
        "Running U2-Net..."
    )

    output_bytes = remove(
        input_bytes,
        session=SEGMENTATION_SESSION
    )

    del input_bytes

    gc.collect()

    # --------------------------------------------------------
    # Read segmentation result
    # --------------------------------------------------------

    result_buffer = io.BytesIO(
        output_bytes
    )

    segmented_image = Image.open(
        result_buffer
    ).convert("RGBA")

    result_buffer.close()

    del output_bytes

    gc.collect()

    # --------------------------------------------------------
    # White background
    # --------------------------------------------------------

    white_background = Image.new(
        "RGBA",
        segmented_image.size,
        (255, 255, 255, 255)
    )

    final_image = Image.alpha_composite(
        white_background,
        segmented_image
    ).convert("RGB")

    segmented_image.close()
    white_background.close()

    gc.collect()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    final_image.save(
        output_path,
        format="JPEG",
        quality=85
    )

    final_image.close()

    gc.collect()

    print(
        "Segmented image saved:"
    )

    print(
        output_path
    )

    return output_path