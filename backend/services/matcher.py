import os
import json
import cv2
import numpy as np

from backend.config import (
    JEWELLERY_JSON,
    GOLD_EMBEDDINGS_FILE,
    PROTOTYPE_EMBEDDINGS_FILE,
    GOLD_CATALOGUE_DIR,
    PROTOTYPE_CATALOGUE_DIR,
    GOLD_SEGMENTED_DIR,
    PROTOTYPE_SEGMENTED_DIR,
    TOP_K,
)

from backend.services.embedding import create_embedding
from backend.services.segmentation import segment_jewellery

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_DIR = os.path.join(BASE_DIR, "database")

QUERY_SEGMENTED_DIR = os.path.join(DATABASE_DIR, "segmented_catalogue")

QUERY_SEGMENTED_PATH = os.path.join(QUERY_SEGMENTED_DIR, "query_hybrid.jpg")


# ============================================================
# SUPPORTED COLLECTIONS
# ============================================================

VALID_COLLECTIONS = {"gold", "prototype"}


# ============================================================
# LOAD CATALOGUE
# ============================================================


def load_catalogue():

    if not os.path.exists(JEWELLERY_JSON):

        print("[MATCHER] jewellery.json not found.")

        return []

    try:

        with open(JEWELLERY_JSON, "r", encoding="utf-8") as file:

            data = json.load(file)

    except Exception as error:

        print("[MATCHER] Failed to read jewellery.json:")

        print(error)

        return []

    if not isinstance(data, list):

        print("[MATCHER] jewellery.json must contain a list.")

        return []

    return data


# ============================================================
# LOAD EMBEDDINGS
# ============================================================


def load_embeddings(collection):

    collection = collection.lower()

    if collection == "gold":

        embedding_file = GOLD_EMBEDDINGS_FILE

    elif collection == "prototype":

        embedding_file = PROTOTYPE_EMBEDDINGS_FILE

    else:

        print(f"[MATCHER] Invalid collection: " f"{collection}")

        return np.empty((0, 768), dtype=np.float32)

    if not os.path.exists(embedding_file):

        print("[MATCHER] Embedding file not found:")

        print(embedding_file)

        return np.empty((0, 768), dtype=np.float32)

    try:

        embeddings = np.load(embedding_file)

    except Exception as error:

        print("[MATCHER] Failed to load embeddings:")

        print(error)

        return np.empty((0, 768), dtype=np.float32)

    if embeddings.ndim == 1:

        embeddings = embeddings.reshape(1, -1)

    return embeddings.astype(np.float32)


# ============================================================
# COSINE SIMILARITY
# ============================================================


def cosine_similarity(query_embedding, embeddings):

    if embeddings is None or len(embeddings) == 0:

        return np.array([], dtype=np.float32)

    query = np.asarray(query_embedding, dtype=np.float32)

    matrix = np.asarray(embeddings, dtype=np.float32)

    query_norm = np.linalg.norm(query)

    if query_norm == 0:

        return np.zeros(len(matrix), dtype=np.float32)

    query = query / query_norm

    matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)

    matrix_norm[matrix_norm == 0] = 1.0

    matrix = matrix / matrix_norm

    return np.dot(matrix, query)


# ============================================================
# GET COLLECTION ITEMS
# ============================================================


def get_collection_items(collection):

    collection = collection.lower()

    catalogue = load_catalogue()

    items = []

    for item in catalogue:

        item_collection = str(item.get("collection", "")).lower().strip()

        if item_collection == collection:

            items.append(item)

    return items


# ============================================================
# GET ORIGINAL IMAGE PATH
# ============================================================


def get_original_image_path(collection, filename):

    collection = collection.lower()

    if collection == "gold":

        return os.path.join(GOLD_CATALOGUE_DIR, filename)

    if collection == "prototype":

        return os.path.join(PROTOTYPE_CATALOGUE_DIR, filename)

    return ""


# ============================================================
# GET SEGMENTED IMAGE PATH
# ============================================================


def get_segmented_image_path(collection, filename):

    collection = collection.lower()

    base_name = os.path.splitext(filename)[0]

    segmented_filename = base_name + ".jpg"

    if collection == "gold":

        return os.path.join(GOLD_SEGMENTED_DIR, segmented_filename)

    if collection == "prototype":

        return os.path.join(PROTOTYPE_SEGMENTED_DIR, segmented_filename)

    return ""


# ============================================================
# GET BEST AVAILABLE IMAGE
# ============================================================


def get_best_catalogue_image(collection, filename):

    segmented_path = get_segmented_image_path(collection, filename)

    if os.path.exists(segmented_path):

        return segmented_path

    original_path = get_original_image_path(collection, filename)

    if os.path.exists(original_path):

        return original_path

    return ""


# ============================================================
# SHAPE DESCRIPTOR
# ============================================================


def create_shape_descriptor(image_path):

    if not image_path:

        return None

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:

        return None

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    # --------------------------------------------------------
    # Improve contrast
    # --------------------------------------------------------

    image = cv2.equalizeHist(image)

    # --------------------------------------------------------
    # Reduce noise
    # --------------------------------------------------------

    image = cv2.GaussianBlur(image, (5, 5), 0)

    # --------------------------------------------------------
    # Edge detection
    # --------------------------------------------------------

    edges = cv2.Canny(image, 30, 100)

    # --------------------------------------------------------
    # Close small gaps
    # --------------------------------------------------------

    kernel = np.ones((3, 3), np.uint8)

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # --------------------------------------------------------
    # Resize descriptor
    # --------------------------------------------------------

    edges = cv2.resize(edges, (64, 64), interpolation=cv2.INTER_AREA)

    descriptor = edges.astype(np.float32) / 255.0

    descriptor = descriptor.flatten()

    norm = np.linalg.norm(descriptor)

    if norm > 0:

        descriptor = descriptor / norm

    return descriptor


# ============================================================
# SHAPE SIMILARITY
# ============================================================


def calculate_shape_similarity(query_descriptor, target_descriptor):

    if query_descriptor is None or target_descriptor is None:

        return 0.0

    score = np.dot(query_descriptor, target_descriptor)

    score = max(0.0, min(float(score), 1.0))

    return score


# ============================================================
# ORB SIMILARITY
# ============================================================


def calculate_orb_similarity(query_image_path, target_image_path):

    if not query_image_path or not target_image_path:

        return 0.0

    query = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

    target = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)

    if query is None or target is None:

        return 0.0

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    query = cv2.resize(query, (512, 512), interpolation=cv2.INTER_AREA)

    target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_AREA)

    # --------------------------------------------------------
    # ORB
    # --------------------------------------------------------

    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)

    keypoints1, descriptors1 = orb.detectAndCompute(query, None)

    keypoints2, descriptors2 = orb.detectAndCompute(target, None)

    if descriptors1 is None or descriptors2 is None:

        return 0.0

    if len(descriptors1) < 2 or len(descriptors2) < 2:

        return 0.0

    # --------------------------------------------------------
    # Feature matching
    # --------------------------------------------------------

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    try:

        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    except Exception:

        return 0.0

    good_matches = []

    for pair in matches:

        if len(pair) < 2:

            continue

        first, second = pair

        if first.distance < 0.75 * second.distance:

            good_matches.append(first)

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    denominator = max(1, min(len(keypoints1), len(keypoints2)))

    score = len(good_matches) / denominator

    return float(min(score, 1.0))


# ============================================================
# HYBRID SCORE
# ============================================================


def calculate_hybrid_score(dino_score, shape_score, orb_score):
    """
    Design-focused score.

    DINO  = 75%
    Shape = 20%
    ORB   = 5%

    The current DINO embedding is intended
    to reduce colour influence.
    """

    score = (0.75 * dino_score) + (0.20 * shape_score) + (0.05 * orb_score)

    return float(score)


# ============================================================
# GET USER-FACING NAME
# ============================================================


def get_display_name(item):
    """
    Returns the jewellery name that should be
    shown to the user.

    Priority:

        name
        display_name
        jewellery_name
        title
        id
        image filename
    """

    name = (
        item.get("name")
        or item.get("display_name")
        or item.get("jewellery_name")
        or item.get("title")
        or item.get("id")
        or item.get("image")
        or "Jewellery"
    )

    return str(name).strip()


# ============================================================
# SEARCH ONE COLLECTION
# ============================================================


def search_collection(query_image_path, collection, query_embedding, top_k=TOP_K):

    collection = collection.lower()

    if collection not in VALID_COLLECTIONS:

        print(f"[MATCHER] Invalid target collection: " f"{collection}")

        return []

    print()

    print(f"[MATCHER] Searching ONLY: " f"{collection.upper()}")

    # --------------------------------------------------------
    # Load target collection
    # --------------------------------------------------------

    catalogue_items = get_collection_items(collection)

    embeddings = load_embeddings(collection)

    print(f"[MATCHER] Catalogue items: " f"{len(catalogue_items)}")

    print(f"[MATCHER] Embeddings: " f"{len(embeddings)}")

    if len(catalogue_items) == 0 or len(embeddings) == 0:

        print("[MATCHER] No searchable items.")

        return []

    # --------------------------------------------------------
    # Calculate DINO similarities
    # --------------------------------------------------------

    dino_scores = cosine_similarity(query_embedding, embeddings)

    # --------------------------------------------------------
    # Keep catalogue and embeddings aligned
    # --------------------------------------------------------

    count = min(len(catalogue_items), len(dino_scores))

    if count == 0:

        return []

    # --------------------------------------------------------
    # Query shape
    # --------------------------------------------------------

    query_shape = create_shape_descriptor(query_image_path)

    results = []

    # --------------------------------------------------------
    # Compare each target item
    # --------------------------------------------------------

    for index in range(count):

        item = catalogue_items[index]

        filename = str(item.get("image", "")).strip()

        if not filename:

            continue

        # ----------------------------------------------------
        # Target image
        # ----------------------------------------------------

        target_image = get_best_catalogue_image(collection, filename)

        if not target_image:

            print("[MATCHER] Image not found:")

            print(filename)

            continue

        # ----------------------------------------------------
        # DINO score
        # ----------------------------------------------------

        dino_score = float(dino_scores[index])

        # ----------------------------------------------------
        # Shape score
        # ----------------------------------------------------

        target_shape = create_shape_descriptor(target_image)

        shape_score = calculate_shape_similarity(query_shape, target_shape)

        # ----------------------------------------------------
        # ORB score
        # ----------------------------------------------------

        orb_score = calculate_orb_similarity(query_image_path, target_image)

        # ----------------------------------------------------
        # Hybrid score
        # ----------------------------------------------------

        hybrid_score = calculate_hybrid_score(dino_score, shape_score, orb_score)

        # ----------------------------------------------------
        # Build result
        # ----------------------------------------------------

        result = dict(item)

        # ----------------------------------------------------
        # Force collection
        # ----------------------------------------------------

        result["collection"] = collection

        # ----------------------------------------------------
        # USER-FACING NAME
        #
        # This is the important part.
        #
        # Example:
        #
        # J018
        # Gold Infinity Ring
        #
        # User sees:
        #
        # Gold Infinity Ring
        # ----------------------------------------------------

        result["display_name"] = get_display_name(item)

        # ----------------------------------------------------
        # Preserve internal ID
        # ----------------------------------------------------

        result["jewellery_id"] = str(item.get("id", ""))

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        result["dino_score"] = round(dino_score, 4)

        result["shape_score"] = round(shape_score, 4)

        result["orb_score"] = round(orb_score, 4)

        result["similarity"] = round(hybrid_score, 4)

        # ----------------------------------------------------
        # Image URL
        # ----------------------------------------------------

        result["image_url"] = f"/catalogue/" f"{collection}/" f"{filename}"

        results.append(result)

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results.sort(key=lambda item: item["similarity"], reverse=True)

    # --------------------------------------------------------
    # Debug output
    #
    # IMPORTANT:
    # Print the jewellery NAME, not J018.jpeg
    # --------------------------------------------------------

    print()

    print("[MATCHER] Top candidate scores:")

    for rank, result in enumerate(results[:5], start=1):

        display_name = (
            result.get("display_name")
            or result.get("name")
            or result.get("id")
            or result.get("image", "Unknown Jewellery")
        )

        print(
            f"  #{rank} "
            f"{display_name} | "
            f"DINO="
            f"{result['dino_score']:.4f} | "
            f"SHAPE="
            f"{result['shape_score']:.4f} | "
            f"ORB="
            f"{result['orb_score']:.4f} | "
            f"HYBRID="
            f"{result['similarity']:.4f}"
        )

    return results[:top_k]


# ============================================================
# IDENTIFY SOURCE COLLECTION
# ============================================================


def identify_source_collection(query_image_path):

    print()

    print("[MATCHER] Identifying source collection...")

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    gold_embeddings = load_embeddings("gold")

    prototype_embeddings = load_embeddings("prototype")

    if len(gold_embeddings) == 0 and len(prototype_embeddings) == 0:

        print("[MATCHER] No catalogue embeddings found.")

        return {
            "source_collection": None,
            "target_collection": None,
            "gold_similarity": 0.0,
            "prototype_similarity": 0.0,
        }

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = create_embedding(query_image_path)

    # --------------------------------------------------------
    # Gold similarity
    # --------------------------------------------------------

    gold_scores = cosine_similarity(query_embedding, gold_embeddings)

    gold_best = float(np.max(gold_scores)) if len(gold_scores) > 0 else 0.0

    # --------------------------------------------------------
    # Prototype similarity
    # --------------------------------------------------------

    prototype_scores = cosine_similarity(query_embedding, prototype_embeddings)

    prototype_best = (
        float(np.max(prototype_scores)) if len(prototype_scores) > 0 else 0.0
    )

    print(f"[MATCHER] Gold similarity: " f"{gold_best:.4f}")

    print(f"[MATCHER] Prototype similarity: " f"{prototype_best:.4f}")

    # --------------------------------------------------------
    # Decide source
    # --------------------------------------------------------

    if gold_best >= prototype_best:

        source_collection = "gold"

        target_collection = "prototype"

    else:

        source_collection = "prototype"

        target_collection = "gold"

    print(f"[MATCHER] Source collection: " f"{source_collection}")

    print(f"[MATCHER] Target collection: " f"{target_collection}")

    return {
        "source_collection": source_collection,
        "target_collection": target_collection,
        "gold_similarity": gold_best,
        "prototype_similarity": prototype_best,
    }


# ============================================================
# MAIN BIDIRECTIONAL MATCH
# ============================================================


def match_jewellery(image_path, target_collection="auto", top_k=None):
    """
    STRICT BIDIRECTIONAL SEARCH.

    Gold upload:
        Gold → Prototype

    Prototype upload:
        Prototype → Gold

    The source collection is NEVER returned
    in the search results.
    """

    if top_k is None:

        top_k = TOP_K

    print()

    print("=" * 70)

    print("JEWELLERY BIDIRECTIONAL SEARCH")

    print("=" * 70)

    print("Query image:", os.path.basename(image_path))

    # ========================================================
    # STEP 1
    # IDENTIFY SOURCE
    # ========================================================

    identification = identify_source_collection(image_path)

    source_collection = identification["source_collection"]

    if source_collection is None:

        return {
            "source_collection": None,
            "target_collection": None,
            "source_similarity": 0.0,
            "gold_similarity": 0.0,
            "prototype_similarity": 0.0,
            "best_similarity": 0.0,
            "results": [],
        }

    # ========================================================
    # STEP 2
    # FORCE OPPOSITE COLLECTION
    # ========================================================

    if source_collection == "gold":

        target_collection = "prototype"

    else:

        target_collection = "gold"

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if target_collection == source_collection:

        raise RuntimeError("Source and target collection " "cannot be the same.")

    # ========================================================
    # STEP 3
    # PRINT SEARCH DIRECTION
    # ========================================================

    print()

    print("[MATCHER] STRICT CROSS-COLLECTION SEARCH")

    print(
        f"[MATCHER] "
        f"{source_collection.upper()} "
        f"→ "
        f"{target_collection.upper()}"
    )

    print("[MATCHER] Searching ONLY the opposite collection.")

    # ========================================================
    # STEP 4
    # SEGMENT QUERY ONCE
    # ========================================================

    os.makedirs(QUERY_SEGMENTED_DIR, exist_ok=True)

    segment_jewellery(image_path, QUERY_SEGMENTED_PATH)

    # ========================================================
    # STEP 5
    # CREATE QUERY EMBEDDING
    # ========================================================

    query_embedding = create_embedding(QUERY_SEGMENTED_PATH)

    # ========================================================
    # STEP 6
    # SEARCH ONLY TARGET COLLECTION
    # ========================================================

    results = search_collection(
        QUERY_SEGMENTED_PATH, target_collection, query_embedding, top_k
    )

    # ========================================================
    # STEP 7
    # FINAL SAFETY FILTER
    # ========================================================

    safe_results = []

    for result in results:

        result_collection = str(result.get("collection", "")).lower().strip()

        if result_collection != target_collection:

            continue

        if result_collection == source_collection:

            continue

        safe_results.append(result)

    results = safe_results

    # ========================================================
    # STEP 8
    # BEST RESULT
    # ========================================================

    if results:

        best_similarity = float(results[0].get("similarity", 0.0))

    else:

        best_similarity = 0.0

    # ========================================================
    # SOURCE SIMILARITY
    # ========================================================

    if source_collection == "gold":

        source_similarity = identification["gold_similarity"]

    else:

        source_similarity = identification["prototype_similarity"]

    # ========================================================
    # FINAL LOG
    # ========================================================

    print()

    print("=" * 70)

    print("FINAL SEARCH RESULT")

    print("=" * 70)

    print(f"Source collection : " f"{source_collection}")

    print(f"Target collection : " f"{target_collection}")

    print(f"Results returned  : " f"{len(results)}")

    print(f"Best similarity   : " f"{best_similarity:.4f}")

    # --------------------------------------------------------
    # Print result names
    # --------------------------------------------------------

    if results:

        print()
        print("Returned jewellery:")

        for index, result in enumerate(results, start=1):

            print(f"  {index}. " f"{result.get('display_name', 'Jewellery')}")

    print("=" * 70)

    # ========================================================
    # API RESPONSE
    # ========================================================

    return {
        "source_collection": (source_collection),
        "target_collection": (target_collection),
        "source_similarity": round(source_similarity, 4),
        "gold_similarity": round(identification["gold_similarity"], 4),
        "prototype_similarity": round(identification["prototype_similarity"], 4),
        "best_similarity": round(best_similarity, 4),
        "results": results,
    }


# ============================================================
# DIRECT COLLECTION SEARCH
# ============================================================


def match_collection(image_path, collection, top_k=TOP_K):
    """
    Explicit collection search.

    Kept for compatibility.

    Normal UI search should use
    match_jewellery().
    """

    collection = collection.lower()

    if collection not in VALID_COLLECTIONS:

        return []

    query_segmented = os.path.join(QUERY_SEGMENTED_DIR, "query_collection.jpg")

    os.makedirs(QUERY_SEGMENTED_DIR, exist_ok=True)

    segment_jewellery(image_path, query_segmented)

    query_embedding = create_embedding(query_segmented)

    return search_collection(query_segmented, collection, query_embedding, top_k)


# ============================================================
# COMPATIBILITY WRAPPER
# ============================================================


def find_matches(image_path, collection="auto", top_k=TOP_K):

    return match_jewellery(image_path, target_collection="auto", top_k=top_k)
