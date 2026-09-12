"""
============================================================
JEWELLERY AI MATCHING
Matcher + Jewellery Validation
============================================================

Purpose:
    1. Check whether the uploaded image is likely to be
       jewellery.
    2. If not jewellery -> return no match.
    3. If jewellery -> compare against catalogue embeddings.
============================================================
"""

import json

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from services.embedding import get_embedding

from config import (
    JEWELLERY_JSON,
    EMBEDDINGS_FILE,
    TOP_K,
    MATCH_THRESHOLD
)


# ============================================================
# LOAD DATABASE
# ============================================================

def load_database():

    with open(
        JEWELLERY_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        jewellery = json.load(file)


    embeddings = np.load(
        EMBEDDINGS_FILE
    )


    return jewellery, embeddings


# ============================================================
# JEWELLERY VALIDATION
# ============================================================

def is_jewellery_image(image_path):
    """
    Basic jewellery validation.

    Version 1 uses the image embedding and compares it against
    the catalogue as a whole.

    If the image is extremely different from every catalogue
    image, it is treated as non-jewellery.

    This is a first-stage validation. A dedicated jewellery
    classifier can be added in a later version.
    """

    jewellery, catalogue_embeddings = load_database()


    # --------------------------------------------------------
    # Create embedding
    # --------------------------------------------------------

    query_embedding = get_embedding(
        image_path
    )


    query_embedding = query_embedding.reshape(
        1,
        -1
    )


    # --------------------------------------------------------
    # Compare against all jewellery
    # --------------------------------------------------------

    similarities = cosine_similarity(
        query_embedding,
        catalogue_embeddings
    )[0]


    highest_similarity = float(
        np.max(similarities)
    )


    # --------------------------------------------------------
    # Jewellery validation threshold
    # --------------------------------------------------------

    # This is intentionally lower than the final match
    # threshold because we are only checking whether the
    # image looks sufficiently related to the catalogue.
    JEWELLERY_VALIDATION_THRESHOLD = 0.45


    is_jewellery = (
        highest_similarity >=
        JEWELLERY_VALIDATION_THRESHOLD
    )


    return (
        is_jewellery,
        highest_similarity
    )


# ============================================================
# FIND MATCHES
# ============================================================

def find_matches(
    image_path,
    top_k=TOP_K
):
    """
    Find the most visually similar jewellery.
    """

    jewellery, catalogue_embeddings = (
        load_database()
    )


    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = get_embedding(
        image_path
    )


    query_embedding = query_embedding.reshape(
        1,
        -1
    )


    # --------------------------------------------------------
    # Calculate similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(
        query_embedding,
        catalogue_embeddings
    )[0]


    # --------------------------------------------------------
    # Sort highest similarity first
    # --------------------------------------------------------

    sorted_indices = np.argsort(
        similarities
    )[::-1]


    sorted_indices = sorted_indices[:top_k]


    results = []


    # --------------------------------------------------------
    # Create results
    # --------------------------------------------------------

    for index in sorted_indices:

        similarity = float(
            similarities[index]
        )


        item = jewellery[index]


        similarity_percent = round(
            similarity * 100,
            2
        )


        is_match = (
            similarity >=
            MATCH_THRESHOLD
        )


        results.append({

            "id":
                item["id"],

            "name":
                item["name"],

            "category":
                item["category"],

            "image":
                item["image"],

            "similarity":
                similarity_percent,

            "match":
                is_match

        })


    return results