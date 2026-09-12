"""
============================================================
JEWELLERY AI MATCHING
Create Catalogue Embeddings
============================================================

Purpose:
    This script reads all jewellery records from jewellery.json,
    creates an embedding for each catalogue image, and saves
    the embeddings into embeddings.npy.

Run this script from the backend folder:

    python scripts/create_embeddings.py
============================================================
"""

import os
import sys
import json

import numpy as np


# ============================================================
# STEP 1: FIND THE BACKEND FOLDER
# ============================================================

# __file__ points to:
#
# backend/scripts/create_embeddings.py
#
# dirname(__file__) gives:
#
# backend/scripts
#
# Going one level up gives:
#
# backend

BACKEND_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# STEP 2: ADD BACKEND TO PYTHON PATH
# ============================================================

# This allows Python to find folders such as:
#
# backend/services
# backend/config.py

if BACKEND_DIR not in sys.path:

    sys.path.insert(
        0,
        BACKEND_DIR
    )


# ============================================================
# STEP 3: IMPORT OUR PROJECT FILES
# ============================================================

from services.embedding import get_embedding

from config import (
    CATALOGUE_DIR,
    JEWELLERY_JSON,
    EMBEDDINGS_FILE
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print()
    print("=" * 60)
    print("JEWELLERY AI - CREATE EMBEDDINGS")
    print("=" * 60)


    # ========================================================
    # STEP 4: CHECK JEWELLERY JSON
    # ========================================================

    if not os.path.exists(JEWELLERY_JSON):

        raise FileNotFoundError(
            f"Jewellery JSON file not found:\n"
            f"{JEWELLERY_JSON}"
        )


    # ========================================================
    # STEP 5: LOAD JEWELLERY JSON
    # ========================================================

    with open(
        JEWELLERY_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        jewellery = json.load(file)


    print()
    print(
        "Jewellery records found:",
        len(jewellery)
    )


    # ========================================================
    # STEP 6: PREPARE EMBEDDING LIST
    # ========================================================

    embeddings = []

    valid_items = []


    # ========================================================
    # STEP 7: PROCESS EACH JEWELLERY IMAGE
    # ========================================================

    for item in jewellery:

        jewellery_id = item["id"]

        image_name = item["image"]


        print()
        print("-" * 60)

        print(
            "Processing:",
            jewellery_id
        )

        print(
            "Image:",
            image_name
        )


        # ----------------------------------------------------
        # Create complete image path
        # ----------------------------------------------------

        image_path = os.path.join(
            CATALOGUE_DIR,
            image_name
        )


        print(
            "Path:",
            image_path
        )


        # ----------------------------------------------------
        # Check whether image exists
        # ----------------------------------------------------

        if not os.path.exists(image_path):

            print(
                "WARNING: Image not found."
            )

            print(
                "Skipping:",
                image_name
            )

            continue


        # ----------------------------------------------------
        # Create embedding
        # ----------------------------------------------------

        try:

            embedding = get_embedding(
                image_path
            )


            # Add embedding to list
            embeddings.append(
                embedding
            )


            # Keep the metadata in exactly the same
            # order as the embeddings.
            valid_items.append(
                item
            )


            print(
                "Embedding created successfully."
            )

            print(
                "Embedding shape:",
                embedding.shape
            )


        except Exception as error:

            print()
            print(
                "ERROR processing:",
                image_name
            )

            print(
                "Error:",
                error
            )

            print(
                "Skipping this image."
            )


    # ========================================================
    # STEP 8: MAKE SURE WE HAVE EMBEDDINGS
    # ========================================================

    if len(embeddings) == 0:

        raise RuntimeError(
            "No embeddings were created.\n"
            "Please check your catalogue images and "
            "jewellery.json."
        )


    # ========================================================
    # STEP 9: CONVERT TO NUMPY ARRAY
    # ========================================================

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )


    # ========================================================
    # STEP 10: SAVE EMBEDDINGS
    # ========================================================

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )


    # ========================================================
    # STEP 11: UPDATE JSON
    # ========================================================

    # Only keep records for images that actually
    # received an embedding.

    with open(
        JEWELLERY_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            valid_items,
            file,
            indent=4
        )


    # ========================================================
    # STEP 12: FINAL RESULT
    # ========================================================

    print()
    print("=" * 60)
    print("EMBEDDING DATABASE CREATED SUCCESSFULLY")
    print("=" * 60)

    print(
        "Jewellery processed:",
        len(embeddings)
    )

    print(
        "Embedding matrix shape:",
        embeddings.shape
    )

    print()
    print(
        "Embeddings saved to:"
    )

    print(
        EMBEDDINGS_FILE
    )

    print()
    print("=" * 60)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()