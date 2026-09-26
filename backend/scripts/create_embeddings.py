import os
import sys
import json
import gc
import numpy as np

# ============================================================
# PROJECT ROOT
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(BACKEND_DIR)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from backend.config import (
    JEWELLERY_JSON,
    GOLD_CATALOGUE_DIR,
    PROTOTYPE_CATALOGUE_DIR,
    GOLD_EMBEDDINGS_FILE,
    PROTOTYPE_EMBEDDINGS_FILE,
    GOLD_SEGMENTED_DIR,
    PROTOTYPE_SEGMENTED_DIR,
)

from backend.services.embedding import create_embedding
from backend.services.segmentation import segment_jewellery

# ============================================================
# HELPERS
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_image_files(folder):
    if not os.path.exists(folder):
        return []

    files = []

    for filename in sorted(os.listdir(folder)):

        full_path = os.path.join(folder, filename)

        if not os.path.isfile(full_path):
            continue

        extension = os.path.splitext(filename)[1].lower()

        if extension in IMAGE_EXTENSIONS:
            files.append(filename)

    return files


def load_catalogue():
    if not os.path.exists(JEWELLERY_JSON):
        return []

    with open(JEWELLERY_JSON, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("jewellery.json must contain a JSON list.")

    return data


def save_embeddings(path, embeddings):
    if embeddings:

        matrix = np.vstack(embeddings).astype(np.float32)

    else:

        matrix = np.empty((0, 384), dtype=np.float32)

    np.save(path, matrix)

    print()
    print("Saved embeddings:")
    print(path)
    print("Shape:", matrix.shape)


# ============================================================
# PROCESS COLLECTION
# ============================================================


def process_collection(collection_name, catalogue_dir, segmented_dir, output_file):

    print()
    print("=" * 70)
    print(f"PROCESSING {collection_name.upper()} COLLECTION")
    print("=" * 70)

    os.makedirs(segmented_dir, exist_ok=True)

    image_files = get_image_files(catalogue_dir)

    print(f"Images found: {len(image_files)}")

    embeddings = []

    for index, filename in enumerate(image_files, start=1):

        source_path = os.path.join(catalogue_dir, filename)

        segmented_filename = os.path.splitext(filename)[0] + ".jpg"

        segmented_path = os.path.join(segmented_dir, segmented_filename)

        print()
        print(f"[{index}/{len(image_files)}] " f"{filename}")

        # ----------------------------------------------------
        # SEGMENT
        # ----------------------------------------------------

        try:

            segment_jewellery(source_path, segmented_path)

        except Exception as error:

            print("[ERROR] Segmentation failed:")
            print(error)

            # Fallback to original image
            segmented_path = source_path

        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

        try:

            embedding = create_embedding(segmented_path)

            embeddings.append(embedding)

        except Exception as error:

            print("[ERROR] Embedding failed:")
            print(error)

            continue

        gc.collect()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_embeddings(output_file, embeddings)

    return len(embeddings)


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 70)
    print("JEWELLERY COLOUR-INVARIANT EMBEDDING GENERATION")
    print("=" * 70)

    catalogue = load_catalogue()

    print(f"Catalogue records: {len(catalogue)}")

    gold_count = process_collection(
        collection_name="gold",
        catalogue_dir=GOLD_CATALOGUE_DIR,
        segmented_dir=GOLD_SEGMENTED_DIR,
        output_file=GOLD_EMBEDDINGS_FILE,
    )

    prototype_count = process_collection(
        collection_name="prototype",
        catalogue_dir=PROTOTYPE_CATALOGUE_DIR,
        segmented_dir=PROTOTYPE_SEGMENTED_DIR,
        output_file=PROTOTYPE_EMBEDDINGS_FILE,
    )

    print()
    print("=" * 70)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 70)

    print(f"Gold embeddings:      {gold_count}")

    print(f"Prototype embeddings:  {prototype_count}")

    print()
    print("IMPORTANT:")
    print(
        "The embeddings were generated from grayscale "
        "colour-invariant representations."
    )


if __name__ == "__main__":
    main()
