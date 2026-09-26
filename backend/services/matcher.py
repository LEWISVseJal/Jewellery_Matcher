import os
import json

import cv2
import numpy as np

from .embedding import (
    create_embedding,
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_DIR = os.path.dirname(BASE_DIR)

DATABASE_DIR = os.path.join(
    BASE_DIR,
    "database",
)

CATALOGUE_FILE = os.path.join(
    DATABASE_DIR,
    "jewellery.json",
)

INDEX_FILE = os.path.join(
    DATABASE_DIR,
    "lightweight_index.npz",
)


# ============================================================
# MEMORY CACHE
# ============================================================

_INDEX_CACHE = None


# ============================================================
# CATALOGUE
# ============================================================


def load_catalogue():

    if not os.path.exists(CATALOGUE_FILE):
        return []

    try:

        with open(
            CATALOGUE_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

    except Exception as exc:

        print(f"[MATCHER] Catalogue load error: {exc}")

        return []

    if isinstance(
        data,
        dict,
    ):

        if "jewellery" in data:

            return data["jewellery"]

        if "items" in data:

            return data["items"]

    if isinstance(
        data,
        list,
    ):

        return data

    return []


# ============================================================
# ITEM ID
# ============================================================


def get_item_id(
    item,
):

    return str(
        item.get(
            "id",
            item.get(
                "design_id",
                "",
            ),
        )
    )


# ============================================================
# COLLECTION
# ============================================================


def get_collection(
    item,
):

    return (
        str(
            item.get(
                "collection",
                item.get(
                    "category",
                    "",
                ),
            )
        )
        .strip()
        .lower()
    )


# ============================================================
# IMAGE PATH
# ============================================================


def get_image_path(
    item,
):

    image_path = item.get("image_path")

    if not image_path:

        image_path = item.get("image")

    if not image_path:

        image_path = item.get("path")

    if not image_path:

        return None

    image_path = str(image_path).replace(
        "\\",
        "/",
    )

    candidates = []

    if os.path.isabs(image_path):

        candidates.append(image_path)

    candidates.extend(
        [
            os.path.join(
                PROJECT_DIR,
                image_path,
            ),
            os.path.join(
                BASE_DIR,
                image_path,
            ),
        ]
    )

    for candidate in candidates:

        if os.path.exists(candidate):

            return candidate

    filename = os.path.basename(image_path)

    for root, _, files in os.walk(PROJECT_DIR):

        if filename in files:

            return os.path.join(
                root,
                filename,
            )

    return None


# ============================================================
# LOAD INDEX
# ============================================================


def load_index():

    global _INDEX_CACHE

    if _INDEX_CACHE is not None:

        return _INDEX_CACHE

    if not os.path.exists(INDEX_FILE):

        print("[MATCHER] Lightweight index not found.")

        return None

    try:

        data = np.load(
            INDEX_FILE,
            allow_pickle=False,
        )

        _INDEX_CACHE = {
            "features": data["features"],
            "ids": data["ids"],
            "collections": data["collections"],
        }

        print("[MATCHER] Lightweight index loaded.")

        print(
            "[MATCHER] Feature matrix:",
            _INDEX_CACHE["features"].shape,
        )

        return _INDEX_CACHE

    except Exception as exc:

        print(f"[MATCHER] Index load failed: {exc}")

        return None


# ============================================================
# CLEAR INDEX
# ============================================================


def clear_index_cache():

    global _INDEX_CACHE

    _INDEX_CACHE = None

    print("[MATCHER] Index cache cleared.")


# ============================================================
# COSINE SIMILARITY
# ============================================================


def cosine_similarity_vector(
    query,
    matrix,
):

    query = np.asarray(
        query,
        dtype=np.float32,
    )

    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    )

    if matrix.ndim == 1:

        matrix = matrix.reshape(
            1,
            -1,
        )

    query_norm = np.linalg.norm(query)

    if query_norm < 1e-8:

        return np.zeros(
            len(matrix),
            dtype=np.float32,
        )

    matrix_norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    matrix_norms[matrix_norms < 1e-8] = 1.0

    scores = (matrix @ query) / (matrix_norms * query_norm)

    return scores


# ============================================================
# STRUCTURAL SIMILARITY
# ============================================================


def structural_similarity(
    query_path,
    candidate_path,
):

    try:

        query = cv2.imread(
            query_path,
            cv2.IMREAD_GRAYSCALE,
        )

        candidate = cv2.imread(
            candidate_path,
            cv2.IMREAD_GRAYSCALE,
        )

        if query is None:
            return 0.0

        if candidate is None:
            return 0.0

        query = cv2.resize(
            query,
            (128, 128),
            interpolation=cv2.INTER_AREA,
        )

        candidate = cv2.resize(
            candidate,
            (128, 128),
            interpolation=cv2.INTER_AREA,
        )

        query_edges = cv2.Canny(
            query,
            50,
            150,
        )

        candidate_edges = cv2.Canny(
            candidate,
            50,
            150,
        )

        query_vector = query_edges.astype(np.float32).flatten() / 255.0

        candidate_vector = candidate_edges.astype(np.float32).flatten() / 255.0

        query_norm = np.linalg.norm(query_vector)

        candidate_norm = np.linalg.norm(candidate_vector)

        if query_norm < 1e-8 or candidate_norm < 1e-8:

            return 0.0

        score = np.dot(
            query_vector,
            candidate_vector,
        ) / (query_norm * candidate_norm)

        return float(
            max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            )
        )

    except Exception as exc:

        print(f"[MATCHER] Structural error: {exc}")

        return 0.0


# ============================================================
# SOURCE COLLECTION
# ============================================================


def identify_source_collection(
    image_path,
):

    index = load_index()

    if index is None:

        print("[MATCHER] Cannot identify source: " "index unavailable.")

        return "unknown"

    query_embedding = create_embedding(image_path)

    similarities = cosine_similarity_vector(
        query_embedding,
        index["features"],
    )

    gold_mask = index["collections"] == "gold"

    prototype_mask = index["collections"] == "prototype"

    gold_scores = similarities[gold_mask]

    prototype_scores = similarities[prototype_mask]

    gold_score = float(np.max(gold_scores)) if len(gold_scores) else 0.0

    prototype_score = float(np.max(prototype_scores)) if len(prototype_scores) else 0.0

    print(f"[MATCHER] Gold similarity: " f"{gold_score:.4f}")

    print(f"[MATCHER] Prototype similarity: " f"{prototype_score:.4f}")

    if gold_score == 0.0 and prototype_score == 0.0:

        return "unknown"

    if gold_score >= prototype_score:

        return "gold"

    return "prototype"


# ============================================================
# MAIN MATCH FUNCTION
# ============================================================


def match_jewellery(
    image_path,
    target_collection="auto",
    top_k=8,
):

    print("=" * 70)

    print("JEWELMATCH AI - LIGHTWEIGHT CROSS-COLLECTION SEARCH")

    print("=" * 70)

    print(f"Query image: " f"{os.path.basename(image_path)}")

    # --------------------------------------------------------
    # Catalogue
    # --------------------------------------------------------

    catalogue = load_catalogue()

    if not catalogue:

        return {
            "success": False,
            "message": "Catalogue is empty.",
            "results": [],
        }

    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------

    index = load_index()

    if index is None:

        return {
            "success": False,
            "message": (
                "Search index is not available. " "Please rebuild the catalogue index."
            ),
            "results": [],
        }

    # --------------------------------------------------------
    # Source collection
    # --------------------------------------------------------

    print("[MATCHER] Identifying source collection...")

    source_collection = identify_source_collection(image_path)

    print(f"[MATCHER] Source collection: " f"{source_collection}")

    # --------------------------------------------------------
    # Target collection
    # --------------------------------------------------------

    if target_collection == "auto":

        if source_collection == "gold":

            target_collection = "prototype"

        elif source_collection == "prototype":

            target_collection = "gold"

        else:

            target_collection = "gold"

    print(f"[MATCHER] Target collection: " f"{target_collection}")

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = create_embedding(image_path)

    # --------------------------------------------------------
    # Target candidates
    # --------------------------------------------------------

    collection_mask = index["collections"] == target_collection

    candidate_features = index["features"][collection_mask]

    candidate_ids = index["ids"][collection_mask]

    if len(candidate_features) == 0:

        return {
            "success": True,
            "source_collection": source_collection,
            "target_collection": target_collection,
            "results": [],
            "message": (f"No {target_collection} " "jewellery found."),
        }

    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    similarities = cosine_similarity_vector(
        query_embedding,
        candidate_features,
    )

    ranking = np.argsort(similarities)[::-1]

    # Examine more candidates before
    # applying structural comparison.

    candidate_limit = min(
        len(ranking),
        max(
            top_k * 3,
            20,
        ),
    )

    ranking = ranking[:candidate_limit]

    catalogue_by_id = {get_item_id(item): item for item in catalogue}

    results = []

    # --------------------------------------------------------
    # Detailed comparison
    # --------------------------------------------------------

    for position in ranking:

        item_id = str(candidate_ids[position])

        item = catalogue_by_id.get(item_id)

        if item is None:
            continue

        embedding_score = float(similarities[position])

        candidate_path = get_image_path(item)

        structure_score = 0.0

        if candidate_path and os.path.exists(candidate_path):

            structure_score = structural_similarity(
                image_path,
                candidate_path,
            )

        # ----------------------------------------------------
        # Final design score
        # ----------------------------------------------------

        final_score = 0.75 * embedding_score + 0.25 * structure_score

        results.append(
            {
                "id": item_id,
                "design_id": item.get(
                    "design_id",
                    item_id,
                ),
                "name": item.get(
                    "name",
                    "Unnamed Jewellery",
                ),
                "collection": get_collection(item),
                "gender": item.get(
                    "gender",
                    "",
                ),
                "type": item.get(
                    "type",
                    "",
                ),
                "subtype": item.get(
                    "subtype",
                    "",
                ),
                "description": item.get(
                    "description",
                    "",
                ),
                "image_path": item.get(
                    "image_path",
                    item.get(
                        "image",
                        "",
                    ),
                ),
                "similarity": round(
                    final_score,
                    4,
                ),
                "score": round(
                    final_score * 100,
                    2,
                ),
                "embedding_score": round(
                    embedding_score,
                    4,
                ),
                "structure_score": round(
                    structure_score,
                    4,
                ),
            }
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results.sort(
        key=lambda result: result["similarity"],
        reverse=True,
    )

    results = results[:top_k]

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    print(f"[MATCHER] Returning " f"{len(results)} results")

    if results:

        print(f"[MATCHER] Best similarity: " f"{results[0]['similarity']:.4f}")

        print(f"[MATCHER] Best match: " f"{results[0]['name']}")

    print("=" * 70)

    return {
        "success": True,
        "source_collection": source_collection,
        "target_collection": target_collection,
        "query_image": os.path.basename(image_path),
        "results": results,
    }
