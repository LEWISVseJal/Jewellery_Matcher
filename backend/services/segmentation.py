import os
import io
import gc

# ============================================================
# U2-NET CACHE
# ============================================================

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

U2NET_HOME = os.path.join(PROJECT_DIR, "model_cache", "rembg")

os.environ["U2NET_HOME"] = U2NET_HOME

os.makedirs(U2NET_HOME, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "u2net"

# Limit large uploaded images.
MAX_IMAGE_SIZE = 1024


# ============================================================
# LAZY SESSION
# ============================================================

SEGMENTATION_SESSION = None


def get_segmentation_session():
    """
    Load U2-Net only when segmentation is actually required.

    This prevents U2-Net from being loaded when Flask starts.
    """

    global SEGMENTATION_SESSION

    if SEGMENTATION_SESSION is not None:
        return SEGMENTATION_SESSION

    print()
    print("=" * 60)
    print("JEWELLERY SEGMENTATION")
    print("=" * 60)

    print("Model:", MODEL_NAME)

    print("U2-Net cache:", U2NET_HOME)

    print()
    print("Loading U2-Net segmentation model...")

    # Import rembg only when required.
    from rembg import new_session

    SEGMENTATION_SESSION = new_session(MODEL_NAME)

    print("U2-Net loaded successfully.")

    print("=" * 60)

    return SEGMENTATION_SESSION


# ============================================================
# RESIZE IMAGE
# ============================================================


def resize_for_segmentation(image):

    from PIL import Image

    width, height = image.size

    largest_dimension = max(width, height)

    if largest_dimension <= MAX_IMAGE_SIZE:

        return image

    scale = MAX_IMAGE_SIZE / largest_dimension

    new_width = max(1, int(width * scale))

    new_height = max(1, int(height * scale))

    print("Resizing:", f"{width}x{height}", "->", f"{new_width}x{new_height}")

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return resized_image


# ============================================================
# SEGMENT JEWELLERY
# ============================================================


def segment_jewellery(image_path, output_path):

    print()
    print("Segmenting image:")

    print(image_path)

    if not os.path.exists(image_path):

        raise FileNotFoundError(f"Input image not found:\n" f"{image_path}")

    # --------------------------------------------------------
    # Lazy imports
    # --------------------------------------------------------

    from PIL import Image

    from rembg import remove

    # --------------------------------------------------------
    # Get U2-Net session
    # --------------------------------------------------------

    session = get_segmentation_session()

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    input_image = Image.open(image_path).convert("RGB")

    print("Original size:", input_image.size)

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized_image = resize_for_segmentation(input_image)

    print("Segmentation size:", resized_image.size)

    # --------------------------------------------------------
    # Convert to JPEG bytes
    # --------------------------------------------------------

    input_buffer = io.BytesIO()

    resized_image.save(input_buffer, format="JPEG", quality=85)

    # We no longer need PIL images here.
    resized_image.close()

    if resized_image is not input_image:

        input_image.close()

    else:

        input_image.close()

    input_bytes = input_buffer.getvalue()

    input_buffer.close()

    del input_buffer

    gc.collect()

    # --------------------------------------------------------
    # Background removal
    # --------------------------------------------------------

    print("Running U2-Net...")

    output_bytes = remove(input_bytes, session=session)

    del input_bytes

    gc.collect()

    # --------------------------------------------------------
    # Read segmentation result
    # --------------------------------------------------------

    result_buffer = io.BytesIO(output_bytes)

    segmented_image = Image.open(result_buffer).convert("RGBA")

    result_buffer.close()

    del result_buffer
    del output_bytes

    gc.collect()

    # --------------------------------------------------------
    # White background
    # --------------------------------------------------------

    white_background = Image.new("RGBA", segmented_image.size, (255, 255, 255, 255))

    final_image = Image.alpha_composite(white_background, segmented_image).convert(
        "RGB"
    )

    segmented_image.close()
    white_background.close()

    del segmented_image
    del white_background

    gc.collect()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_directory = os.path.dirname(output_path)

    if output_directory:

        os.makedirs(output_directory, exist_ok=True)

    # --------------------------------------------------------
    # Save final image
    # --------------------------------------------------------

    final_image.save(output_path, format="JPEG", quality=85)

    final_image.close()

    del final_image

    gc.collect()

    print()
    print("Segmented image saved:")

    print(output_path)

    return output_path
