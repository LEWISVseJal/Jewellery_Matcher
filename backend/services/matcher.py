# import os
# import json
# import cv2
# import numpy as np

# from backend.config import (
#     JEWELLERY_JSON,
#     GOLD_EMBEDDINGS_FILE,
#     PROTOTYPE_EMBEDDINGS_FILE,
#     GOLD_CATALOGUE_DIR,
#     PROTOTYPE_CATALOGUE_DIR,
#     GOLD_SEGMENTED_DIR,
#     PROTOTYPE_SEGMENTED_DIR,
#     TOP_K,
# )

# from backend.services.embedding import create_embedding
# from backend.services.segmentation import segment_jewellery

# # ============================================================
# # CONFIGURATION
# # ============================================================

# VALID_COLLECTIONS = {"gold", "prototype"}


# # Number of candidates retrieved by DINO before
# # detailed verification.
# CANDIDATE_COUNT = 20


# # Final matching thresholds.
# #
# # These are intentionally conservative starting values.
# # We will calibrate them using your actual jewellery images.
# #
# STRONG_MATCH_THRESHOLD = 0.70
# POSSIBLE_MATCH_THRESHOLD = 0.50


# # Score weights.
# #
# # DINO:
# # Overall visual/design representation
# #
# # SHAPE:
# # Overall jewellery silhouette and structure
# #
# # SIFT:
# # Local design details and repeated visual patterns
# #
# DINO_WEIGHT = 0.55
# SHAPE_WEIGHT = 0.25
# SIFT_WEIGHT = 0.20


# # ============================================================
# # PATHS
# # ============================================================

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DATABASE_DIR = os.path.join(BASE_DIR, "database")

# QUERY_SEGMENTED_DIR = os.path.join(DATABASE_DIR, "segmented_catalogue")

# QUERY_SEGMENTED_PATH = os.path.join(QUERY_SEGMENTED_DIR, "query_hybrid.jpg")


# # ============================================================
# # LOAD CATALOGUE
# # ============================================================


# def load_catalogue():

#     if not os.path.exists(JEWELLERY_JSON):

#         print("[MATCHER] jewellery.json not found.")

#         return []

#     try:

#         with open(JEWELLERY_JSON, "r", encoding="utf-8") as file:

#             data = json.load(file)

#     except Exception as error:

#         print("[MATCHER] Failed to read jewellery.json:")

#         print(error)

#         return []

#     if not isinstance(data, list):

#         print("[MATCHER] jewellery.json must contain a list.")

#         return []

#     return data


# # ============================================================
# # LOAD EMBEDDINGS
# # ============================================================


# def load_embeddings(collection):

#     collection = str(collection).lower().strip()

#     if collection == "gold":

#         embedding_file = GOLD_EMBEDDINGS_FILE

#     elif collection == "prototype":

#         embedding_file = PROTOTYPE_EMBEDDINGS_FILE

#     else:

#         print(f"[MATCHER] Invalid collection: {collection}")

#         return np.empty((0, 768), dtype=np.float32)

#     if not os.path.exists(embedding_file):

#         print("[MATCHER] Embedding file not found:")

#         print(embedding_file)

#         return np.empty((0, 768), dtype=np.float32)

#     try:

#         embeddings = np.load(embedding_file)

#     except Exception as error:

#         print("[MATCHER] Failed to load embeddings:")

#         print(error)

#         return np.empty((0, 768), dtype=np.float32)

#     if embeddings.ndim == 1:

#         embeddings = embeddings.reshape(1, -1)

#     return embeddings.astype(np.float32)


# # ============================================================
# # COSINE SIMILARITY
# # ============================================================


# def cosine_similarity(query_embedding, embeddings):

#     if embeddings is None or len(embeddings) == 0:

#         return np.array([], dtype=np.float32)

#     query = np.asarray(query_embedding, dtype=np.float32)

#     matrix = np.asarray(embeddings, dtype=np.float32)

#     query_norm = np.linalg.norm(query)

#     if query_norm == 0:

#         return np.zeros(len(matrix), dtype=np.float32)

#     query = query / query_norm

#     matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)

#     matrix_norm[matrix_norm == 0] = 1.0

#     matrix = matrix / matrix_norm

#     return np.dot(matrix, query)


# # ============================================================
# # GET COLLECTION ITEMS
# # ============================================================


# def get_collection_items(collection):

#     collection = str(collection).lower().strip()

#     catalogue = load_catalogue()

#     items = []

#     for item in catalogue:

#         item_collection = str(item.get("collection", "")).lower().strip()

#         if item_collection == collection:

#             items.append(item)

#     return items


# # ============================================================
# # GET ORIGINAL IMAGE
# # ============================================================


# def get_original_image_path(collection, filename):

#     collection = str(collection).lower().strip()

#     if collection == "gold":

#         return os.path.join(GOLD_CATALOGUE_DIR, filename)

#     if collection == "prototype":

#         return os.path.join(PROTOTYPE_CATALOGUE_DIR, filename)

#     return ""


# # ============================================================
# # GET SEGMENTED IMAGE
# # ============================================================


# def get_segmented_image_path(collection, filename):

#     collection = str(collection).lower().strip()

#     base_name = os.path.splitext(filename)[0]

#     segmented_filename = base_name + ".jpg"

#     if collection == "gold":

#         return os.path.join(GOLD_SEGMENTED_DIR, segmented_filename)

#     if collection == "prototype":

#         return os.path.join(PROTOTYPE_SEGMENTED_DIR, segmented_filename)

#     return ""


# # ============================================================
# # BEST CATALOGUE IMAGE
# # ============================================================


# def get_best_catalogue_image(collection, filename):

#     segmented_path = get_segmented_image_path(collection, filename)

#     if os.path.exists(segmented_path):

#         return segmented_path

#     original_path = get_original_image_path(collection, filename)

#     if os.path.exists(original_path):

#         return original_path

#     return ""


# # ============================================================
# # SHAPE DESCRIPTOR
# # ============================================================


# def create_shape_descriptor(image_path):

#     if not image_path:

#         return None

#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     if image is None:

#         return None

#     image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

#     # Improve contrast
#     image = cv2.equalizeHist(image)

#     # Reduce noise
#     image = cv2.GaussianBlur(image, (5, 5), 0)

#     # Edge detection
#     edges = cv2.Canny(image, 30, 100)

#     # Close gaps
#     kernel = np.ones((3, 3), np.uint8)

#     edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

#     # Normalize size
#     edges = cv2.resize(edges, (64, 64), interpolation=cv2.INTER_AREA)

#     descriptor = edges.astype(np.float32) / 255.0

#     descriptor = descriptor.flatten()

#     norm = np.linalg.norm(descriptor)

#     if norm > 0:

#         descriptor = descriptor / norm

#     return descriptor


# # ============================================================
# # SHAPE SIMILARITY
# # ============================================================


# def calculate_shape_similarity(query_descriptor, target_descriptor):

#     if query_descriptor is None or target_descriptor is None:

#         return 0.0

#     score = np.dot(query_descriptor, target_descriptor)

#     score = max(0.0, min(float(score), 1.0))

#     return score


# # ============================================================
# # SIFT SIMILARITY
# # ============================================================


# def calculate_sift_similarity(query_image_path, target_image_path):

#     if not query_image_path or not target_image_path:

#         return 0.0

#     query = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

#     target = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)

#     if query is None or target is None:

#         return 0.0

#     # --------------------------------------------------------
#     # Resize
#     # --------------------------------------------------------

#     query = cv2.resize(query, (512, 512), interpolation=cv2.INTER_AREA)

#     target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_AREA)

#     # --------------------------------------------------------
#     # SIFT
#     # --------------------------------------------------------

#     try:

#         sift = cv2.SIFT_create(nfeatures=800)

#     except Exception as error:

#         print("[MATCHER] SIFT unavailable:")

#         print(error)

#         return 0.0

#     keypoints1, descriptors1 = sift.detectAndCompute(query, None)

#     keypoints2, descriptors2 = sift.detectAndCompute(target, None)

#     if descriptors1 is None or descriptors2 is None:

#         return 0.0

#     if len(descriptors1) < 2 or len(descriptors2) < 2:

#         return 0.0

#     # --------------------------------------------------------
#     # FLANN matching
#     # --------------------------------------------------------

#     index_params = dict(algorithm=1, trees=5)

#     search_params = dict(checks=50)

#     try:

#         matcher = cv2.FlannBasedMatcher(index_params, search_params)

#         matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

#     except Exception:

#         return 0.0

#     # --------------------------------------------------------
#     # Lowe ratio test
#     # --------------------------------------------------------

#     good_matches = []

#     for pair in matches:

#         if len(pair) < 2:

#             continue

#         first, second = pair

#         if first.distance < 0.72 * second.distance:

#             good_matches.append(first)

#     if not good_matches:

#         return 0.0

#     # --------------------------------------------------------
#     # Normalize
#     # --------------------------------------------------------

#     denominator = max(1, min(len(keypoints1), len(keypoints2)))

#     score = len(good_matches) / denominator

#     return float(min(score, 1.0))


# # ============================================================
# # FINAL MULTI-FEATURE SCORE
# # ============================================================


# def calculate_final_score(dino_score, shape_score, sift_score):

#     dino_score = max(0.0, min(float(dino_score), 1.0))

#     shape_score = max(0.0, min(float(shape_score), 1.0))

#     sift_score = max(0.0, min(float(sift_score), 1.0))

#     score = (
#         (DINO_WEIGHT * dino_score)
#         + (SHAPE_WEIGHT * shape_score)
#         + (SIFT_WEIGHT * sift_score)
#     )

#     return float(max(0.0, min(score, 1.0)))


# # ============================================================
# # MATCH TYPE
# # ============================================================


# def determine_match_type(score):

#     score = float(score)

#     if score >= STRONG_MATCH_THRESHOLD:

#         return "strong_match"

#     if score >= POSSIBLE_MATCH_THRESHOLD:

#         return "possible_match"

#     return "low_confidence"


# # ============================================================
# # SEARCH COLLECTION
# # ============================================================


# def search_collection(query_image_path, collection, query_embedding, top_k=TOP_K):

#     collection = str(collection).lower().strip()

#     if collection not in VALID_COLLECTIONS:

#         print(f"[MATCHER] Invalid collection: " f"{collection}")

#         return []

#     print()
#     print(f"[MATCHER] Searching ONLY: " f"{collection.upper()}")

#     # ========================================================
#     # LOAD CATALOGUE
#     # ========================================================

#     catalogue_items = get_collection_items(collection)

#     embeddings = load_embeddings(collection)

#     print(f"[MATCHER] Catalogue items: " f"{len(catalogue_items)}")

#     print(f"[MATCHER] Embeddings: " f"{len(embeddings)}")

#     if len(catalogue_items) == 0 or len(embeddings) == 0:

#         return []

#     # ========================================================
#     # DINO RETRIEVAL
#     # ========================================================

#     dino_scores = cosine_similarity(query_embedding, embeddings)

#     count = min(len(catalogue_items), len(dino_scores))

#     if count == 0:

#         return []

#     # ========================================================
#     # SELECT TOP CANDIDATES
#     # ========================================================

#     candidate_count = min(CANDIDATE_COUNT, count)

#     candidate_indices = np.argsort(dino_scores[:count])[::-1][:candidate_count]

#     print()
#     print(f"[MATCHER] DINO candidates: " f"{candidate_count}")

#     # ========================================================
#     # QUERY SHAPE
#     # ========================================================

#     query_shape = create_shape_descriptor(query_image_path)

#     # ========================================================
#     # DETAILED VERIFICATION
#     # ========================================================

#     results = []

#     for index in candidate_indices:

#         index = int(index)

#         item = dict(catalogue_items[index])

#         filename = str(item.get("image", "")).strip()

#         if not filename:

#             continue

#         target_image = get_best_catalogue_image(collection, filename)

#         if not target_image:

#             print(f"[MATCHER] Image not found: " f"{filename}")

#             continue

#         # ----------------------------------------------------
#         # DINO
#         # ----------------------------------------------------

#         dino_score = float(dino_scores[index])

#         # Cosine can theoretically be negative.
#         # Convert it into a safe 0-1 range.
#         dino_score = (dino_score + 1.0) / 2.0

#         dino_score = max(0.0, min(dino_score, 1.0))

#         # ----------------------------------------------------
#         # SHAPE
#         # ----------------------------------------------------

#         target_shape = create_shape_descriptor(target_image)

#         shape_score = calculate_shape_similarity(query_shape, target_shape)

#         # ----------------------------------------------------
#         # SIFT
#         # ----------------------------------------------------

#         sift_score = calculate_sift_similarity(query_image_path, target_image)

#         # ----------------------------------------------------
#         # FINAL SCORE
#         # ----------------------------------------------------

#         final_score = calculate_final_score(dino_score, shape_score, sift_score)

#         match_type = determine_match_type(final_score)

#         # ----------------------------------------------------
#         # DISPLAY NAME
#         # ----------------------------------------------------

#         display_name = (
#             item.get("name")
#             or item.get("display_name")
#             or item.get("design_name")
#             or item.get("design_id")
#             or item.get("id")
#             or filename
#             or "Jewellery"
#         )

#         # ----------------------------------------------------
#         # RESULT
#         # ----------------------------------------------------

#         result = dict(item)

#         result["collection"] = collection

#         result["display_name"] = str(display_name)

#         result["dino_score"] = round(dino_score, 4)

#         result["shape_score"] = round(shape_score, 4)

#         result["sift_score"] = round(sift_score, 4)

#         result["similarity"] = round(final_score, 4)

#         result["match_type"] = match_type

#         result["image_url"] = f"/catalogue/" f"{collection}/" f"{filename}"

#         results.append(result)

#     # ========================================================
#     # SORT
#     # ========================================================

#     results.sort(key=lambda item: item["similarity"], reverse=True)

#     # ========================================================
#     # DEBUG
#     # ========================================================

#     print()
#     print("[MATCHER] Detailed candidate scores:")

#     for rank, result in enumerate(results[:10], start=1):

#         print(
#             f"  #{rank} "
#             f"{result.get('name', result.get('image', 'unknown'))} | "
#             f"DINO={result['dino_score']:.4f} | "
#             f"SHAPE={result['shape_score']:.4f} | "
#             f"SIFT={result['sift_score']:.4f} | "
#             f"FINAL={result['similarity']:.4f} | "
#             f"{result['match_type']}"
#         )

#     return results[:top_k]


# # ============================================================
# # IDENTIFY SOURCE COLLECTION
# # ============================================================


# def identify_source_collection(query_image_path):

#     print()
#     print("[MATCHER] Identifying source collection...")

#     gold_embeddings = load_embeddings("gold")

#     prototype_embeddings = load_embeddings("prototype")

#     if len(gold_embeddings) == 0 and len(prototype_embeddings) == 0:

#         return {
#             "source_collection": None,
#             "target_collection": None,
#             "gold_similarity": 0.0,
#             "prototype_similarity": 0.0,
#         }

#     # --------------------------------------------------------
#     # Query embedding
#     # --------------------------------------------------------

#     query_embedding = create_embedding(query_image_path)

#     # --------------------------------------------------------
#     # Gold
#     # --------------------------------------------------------

#     gold_scores = cosine_similarity(query_embedding, gold_embeddings)

#     gold_best = float(np.max(gold_scores)) if len(gold_scores) > 0 else 0.0

#     # --------------------------------------------------------
#     # Prototype
#     # --------------------------------------------------------

#     prototype_scores = cosine_similarity(query_embedding, prototype_embeddings)

#     prototype_best = (
#         float(np.max(prototype_scores)) if len(prototype_scores) > 0 else 0.0
#     )

#     print(f"[MATCHER] Gold similarity: " f"{gold_best:.4f}")

#     print(f"[MATCHER] Prototype similarity: " f"{prototype_best:.4f}")

#     if gold_best >= prototype_best:

#         source_collection = "gold"

#         target_collection = "prototype"

#     else:

#         source_collection = "prototype"

#         target_collection = "gold"

#     print(f"[MATCHER] Source collection: " f"{source_collection}")

#     print(f"[MATCHER] Target collection: " f"{target_collection}")

#     return {
#         "source_collection": source_collection,
#         "target_collection": target_collection,
#         "gold_similarity": gold_best,
#         "prototype_similarity": prototype_best,
#     }


# # ============================================================
# # MAIN BIDIRECTIONAL MATCH
# # ============================================================


# def match_jewellery(image_path, target_collection="auto", top_k=None):

#     if top_k is None:

#         top_k = TOP_K

#     print()
#     print("=" * 70)

#     print("JEWELMATCH AI - MULTI-STAGE VISUAL SEARCH")

#     print("=" * 70)

#     print("Query image:", os.path.basename(image_path))

#     # ========================================================
#     # STEP 1
#     # IDENTIFY SOURCE
#     # ========================================================

#     identification = identify_source_collection(image_path)

#     source_collection = identification["source_collection"]

#     if source_collection is None:

#         return {
#             "source_collection": None,
#             "target_collection": None,
#             "source_similarity": 0.0,
#             "gold_similarity": 0.0,
#             "prototype_similarity": 0.0,
#             "best_similarity": 0.0,
#             "best_match_type": "no_match",
#             "results": [],
#         }

#     # ========================================================
#     # STEP 2
#     # FORCE OPPOSITE COLLECTION
#     # ========================================================

#     if source_collection == "gold":

#         target_collection = "prototype"

#     else:

#         target_collection = "gold"

#     print()
#     print("[MATCHER] STRICT CROSS-COLLECTION SEARCH")

#     print(
#         f"[MATCHER] "
#         f"{source_collection.upper()} "
#         f"→ "
#         f"{target_collection.upper()}"
#     )

#     # ========================================================
#     # STEP 3
#     # SEGMENT QUERY
#     # ========================================================

#     os.makedirs(QUERY_SEGMENTED_DIR, exist_ok=True)

#     segment_jewellery(image_path, QUERY_SEGMENTED_PATH)

#     # ========================================================
#     # STEP 4
#     # CREATE EMBEDDING
#     # ========================================================

#     query_embedding = create_embedding(QUERY_SEGMENTED_PATH)

#     # ========================================================
#     # STEP 5
#     # SEARCH TARGET
#     # ========================================================

#     results = search_collection(
#         QUERY_SEGMENTED_PATH, target_collection, query_embedding, top_k
#     )

#     # ========================================================
#     # STEP 6
#     # SAFETY FILTER
#     # ========================================================

#     safe_results = []

#     for result in results:

#         result_collection = str(result.get("collection", "")).lower().strip()

#         if result_collection != target_collection:

#             continue

#         if result_collection == source_collection:

#             continue

#         safe_results.append(result)

#     results = safe_results

#     # ========================================================
#     # STEP 7
#     # BEST RESULT
#     # ========================================================

#     if results:

#         best = results[0]

#         best_similarity = float(best.get("similarity", 0.0))

#         best_match_type = best.get("match_type", "low_confidence")

#     else:

#         best_similarity = 0.0

#         best_match_type = "no_match"

#     # ========================================================
#     # SOURCE SIMILARITY
#     # ========================================================

#     if source_collection == "gold":

#         source_similarity = identification["gold_similarity"]

#     else:

#         source_similarity = identification["prototype_similarity"]

#     # ========================================================
#     # FINAL LOG
#     # ========================================================

#     print()
#     print("=" * 70)

#     print("FINAL SEARCH RESULT")

#     print("=" * 70)

#     print(f"Source collection : " f"{source_collection}")

#     print(f"Target collection : " f"{target_collection}")

#     print(f"Results returned  : " f"{len(results)}")

#     print(f"Best similarity   : " f"{best_similarity:.4f}")

#     print(f"Best match type   : " f"{best_match_type}")

#     print("=" * 70)

#     # ========================================================
#     # API RESPONSE
#     # ========================================================

#     return {
#         "source_collection": source_collection,
#         "target_collection": target_collection,
#         "source_similarity": round(source_similarity, 4),
#         "gold_similarity": round(identification["gold_similarity"], 4),
#         "prototype_similarity": round(identification["prototype_similarity"], 4),
#         "best_similarity": round(best_similarity, 4),
#         "best_match_type": (best_match_type),
#         "results": results,
#     }


# # ============================================================
# # DIRECT COLLECTION SEARCH
# # ============================================================


# def match_collection(image_path, collection, top_k=TOP_K):

#     collection = str(collection).lower().strip()

#     if collection not in VALID_COLLECTIONS:

#         return []

#     query_segmented = os.path.join(QUERY_SEGMENTED_DIR, "query_collection.jpg")

#     os.makedirs(QUERY_SEGMENTED_DIR, exist_ok=True)

#     segment_jewellery(image_path, query_segmented)

#     query_embedding = create_embedding(query_segmented)

#     return search_collection(query_segmented, collection, query_embedding, top_k)


# # ============================================================
# # COMPATIBILITY WRAPPER
# # ============================================================


# def find_matches(image_path, collection="auto", top_k=TOP_K):

#     return match_jewellery(image_path, target_collection="auto", top_k=top_k)


#################################################
####################################################
##################################################
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
# COLLECTIONS
# ============================================================

VALID_COLLECTIONS = {
    "gold",
    "prototype",
}


# ============================================================
# MATCHING SETTINGS
# ============================================================

# Number of visual candidates considered before
# selecting the source catalogue item.
SOURCE_CANDIDATES = 5

# Minimum visual similarity required to identify
# a source catalogue item confidently.
#
# This is NOT the final match score.
SOURCE_CONFIDENCE_THRESHOLD = 0.35

# If a manually paired design_id exists,
# return that opposite-collection item.
USE_DESIGN_ID_MATCHING = True


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

    collection = str(collection).lower().strip()

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

    collection = str(collection).lower().strip()

    catalogue = load_catalogue()

    items = []

    for item in catalogue:

        item_collection = str(item.get("collection", "")).lower().strip()

        if item_collection == collection:

            items.append(item)

    return items


# ============================================================
# ORIGINAL IMAGE PATH
# ============================================================


def get_original_image_path(collection, filename):

    collection = str(collection).lower().strip()

    if collection == "gold":

        return os.path.join(GOLD_CATALOGUE_DIR, filename)

    if collection == "prototype":

        return os.path.join(PROTOTYPE_CATALOGUE_DIR, filename)

    return ""


# ============================================================
# SEGMENTED IMAGE PATH
# ============================================================


def get_segmented_image_path(collection, filename):

    collection = str(collection).lower().strip()

    base_name = os.path.splitext(filename)[0]

    segmented_filename = base_name + ".jpg"

    if collection == "gold":

        return os.path.join(GOLD_SEGMENTED_DIR, segmented_filename)

    if collection == "prototype":

        return os.path.join(PROTOTYPE_SEGMENTED_DIR, segmented_filename)

    return ""


# ============================================================
# BEST CATALOGUE IMAGE
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

    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    image = cv2.equalizeHist(image)

    image = cv2.GaussianBlur(image, (5, 5), 0)

    edges = cv2.Canny(image, 30, 100)

    kernel = np.ones((3, 3), np.uint8)

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

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
# SIFT SIMILARITY
# ============================================================


def calculate_sift_similarity(query_image_path, target_image_path):

    if not query_image_path or not target_image_path:

        return 0.0

    query = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

    target = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)

    if query is None or target is None:

        return 0.0

    query = cv2.resize(query, (512, 512), interpolation=cv2.INTER_AREA)

    target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_AREA)

    try:

        sift = cv2.SIFT_create(nfeatures=800)

        keypoints1, descriptors1 = sift.detectAndCompute(query, None)

        keypoints2, descriptors2 = sift.detectAndCompute(target, None)

    except Exception:

        return 0.0

    if descriptors1 is None or descriptors2 is None:

        return 0.0

    if len(descriptors1) < 2 or len(descriptors2) < 2:

        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

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

    denominator = max(1, min(len(keypoints1), len(keypoints2)))

    score = len(good_matches) / denominator

    return float(min(score, 1.0))


# ============================================================
# HYBRID VISUAL SCORE
# ============================================================


def calculate_hybrid_score(dino_score, shape_score, sift_score):

    score = (0.75 * dino_score) + (0.20 * shape_score) + (0.05 * sift_score)

    return float(score)


# ============================================================
# VISUAL SEARCH
#
# This function ONLY finds candidates inside ONE collection.
# It does not decide the final cross-collection match.
# ============================================================


def search_collection(query_image_path, collection, query_embedding, top_k=TOP_K):

    collection = str(collection).lower().strip()

    if collection not in VALID_COLLECTIONS:

        print(f"[MATCHER] Invalid collection: " f"{collection}")

        return []

    print()
    print(f"[MATCHER] Visual search: " f"{collection.upper()}")

    catalogue_items = get_collection_items(collection)

    embeddings = load_embeddings(collection)

    print(f"[MATCHER] Catalogue items: " f"{len(catalogue_items)}")

    print(f"[MATCHER] Embeddings: " f"{len(embeddings)}")

    if len(catalogue_items) == 0 or len(embeddings) == 0:

        return []

    dino_scores = cosine_similarity(query_embedding, embeddings)

    count = min(len(catalogue_items), len(dino_scores))

    if count == 0:

        return []

    # --------------------------------------------------------
    # Only calculate expensive visual descriptors for
    # the strongest DINO candidates.
    # --------------------------------------------------------

    candidate_count = min(SOURCE_CANDIDATES, count)

    candidate_indexes = np.argsort(dino_scores[:count])[::-1][:candidate_count]

    print(f"[MATCHER] DINO candidates: " f"{len(candidate_indexes)}")

    query_shape = create_shape_descriptor(query_image_path)

    results = []

    for index in candidate_indexes:

        index = int(index)

        item = catalogue_items[index]

        filename = str(item.get("image", "")).strip()

        if not filename:

            continue

        target_image = get_best_catalogue_image(collection, filename)

        if not target_image:

            continue

        dino_score = float(dino_scores[index])

        target_shape = create_shape_descriptor(target_image)

        shape_score = calculate_shape_similarity(query_shape, target_shape)

        sift_score = calculate_sift_similarity(query_image_path, target_image)

        hybrid_score = calculate_hybrid_score(dino_score, shape_score, sift_score)

        result = dict(item)

        result["collection"] = collection

        result["dino_score"] = round(dino_score, 4)

        result["shape_score"] = round(shape_score, 4)

        result["sift_score"] = round(sift_score, 4)

        result["similarity"] = round(hybrid_score, 4)

        result["image_url"] = f"/catalogue/" f"{collection}/" f"{filename}"

        results.append(result)

    results.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)

    print()
    print("[MATCHER] Detailed candidate scores:")

    for rank, result in enumerate(results, start=1):

        print(
            f"  #{rank} "
            f"{result.get('design_id', 'NO_ID')} | "
            f"{result.get('image', 'unknown')} | "
            f"DINO="
            f"{result['dino_score']:.4f} | "
            f"SHAPE="
            f"{result['shape_score']:.4f} | "
            f"SIFT="
            f"{result['sift_score']:.4f} | "
            f"FINAL="
            f"{result['similarity']:.4f}"
        )

    return results[:top_k]


# ============================================================
# FIND MANUALLY PAIRED DESIGN
# ============================================================


def find_paired_design(source_item, target_collection):

    if not source_item:

        return None

    source_design_id = str(source_item.get("design_id", "")).strip()

    if not source_design_id:

        print("[MATCHER] Source item has no design_id.")

        return None

    catalogue = load_catalogue()

    target_collection = str(target_collection).lower().strip()

    print()
    print("[MATCHER] Looking for manual design pairing...")

    print(f"[MATCHER] Design ID: " f"{source_design_id}")

    print(f"[MATCHER] Target collection: " f"{target_collection}")

    matches = []

    for item in catalogue:

        item_collection = str(item.get("collection", "")).lower().strip()

        item_design_id = str(item.get("design_id", "")).strip()

        if item_collection == target_collection and item_design_id == source_design_id:

            matches.append(item)

    if not matches:

        print("[MATCHER] No paired design found.")

        return None

    print(f"[MATCHER] Paired designs found: " f"{len(matches)}")

    for item in matches:

        print(f"  → " f"{item.get('image', 'unknown')}")

    return matches[0]


# ============================================================
# IDENTIFY SOURCE COLLECTION
#
# IMPORTANT:
# We still need to determine whether the uploaded image
# belongs to Gold or Prototype.
#
# We use the strongest visual candidate from each side,
# but DO NOT use same-collection results as final output.
# ============================================================


def identify_source_collection(query_image_path):

    print()
    print("[MATCHER] Identifying source collection...")

    gold_embeddings = load_embeddings("gold")

    prototype_embeddings = load_embeddings("prototype")

    if len(gold_embeddings) == 0 and len(prototype_embeddings) == 0:

        return {
            "source_collection": None,
            "target_collection": None,
            "gold_similarity": 0.0,
            "prototype_similarity": 0.0,
        }

    query_embedding = create_embedding(query_image_path)

    gold_scores = cosine_similarity(query_embedding, gold_embeddings)

    prototype_scores = cosine_similarity(query_embedding, prototype_embeddings)

    gold_best = float(np.max(gold_scores)) if len(gold_scores) > 0 else 0.0

    prototype_best = (
        float(np.max(prototype_scores)) if len(prototype_scores) > 0 else 0.0
    )

    print(f"[MATCHER] Gold similarity: " f"{gold_best:.4f}")

    print(f"[MATCHER] Prototype similarity: " f"{prototype_best:.4f}")

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
# CREATE EXACT DESIGN RESULT
# ============================================================


def build_exact_result(source_item, target_item, source_collection, target_collection):

    result = dict(target_item)

    result["collection"] = target_collection

    result["source_collection"] = source_collection

    result["target_collection"] = target_collection

    result["match_type"] = "exact_design"

    result["match_method"] = "manual_design_id"

    result["design_id"] = target_item.get("design_id", source_item.get("design_id", ""))

    result["source_design_id"] = source_item.get("design_id", "")

    filename = str(target_item.get("image", "")).strip()

    result["image_url"] = f"/catalogue/" f"{target_collection}/" f"{filename}"

    # Exact design is deliberately not represented
    # as a fake percentage.
    result["similarity"] = 1.0

    return result


# ============================================================
# MAIN MATCH
# ============================================================


def match_jewellery(image_path, target_collection="auto", top_k=None):

    if top_k is None:

        top_k = TOP_K

    print()
    print("=" * 70)

    print("JEWELMATCH AI - MANUAL DESIGN PAIRING SEARCH")

    print("=" * 70)

    print("Query image:", os.path.basename(image_path))

    # ========================================================
    # STEP 1
    # IDENTIFY SOURCE COLLECTION
    # ========================================================

    identification = identify_source_collection(image_path)

    source_collection = identification["source_collection"]

    if source_collection is None:

        return {
            "source_collection": None,
            "target_collection": None,
            "match_type": "no_match",
            "best_similarity": 0.0,
            "results": [],
        }

    if source_collection == "gold":

        target_collection = "prototype"

    else:

        target_collection = "gold"

    # ========================================================
    # STEP 2
    # SEGMENT QUERY
    # ========================================================

    os.makedirs(QUERY_SEGMENTED_DIR, exist_ok=True)

    segment_jewellery(image_path, QUERY_SEGMENTED_PATH)

    # ========================================================
    # STEP 3
    # CREATE QUERY EMBEDDING
    # ========================================================

    query_embedding = create_embedding(QUERY_SEGMENTED_PATH)

    # ========================================================
    # STEP 4
    # FIND SOURCE ITEM
    #
    # This is the important change.
    #
    # We first find which catalogue item the user uploaded.
    # We DO NOT immediately search the opposite collection
    # and pretend the highest visual similarity is exact.
    # ========================================================

    source_candidates = search_collection(
        QUERY_SEGMENTED_PATH, source_collection, query_embedding, SOURCE_CANDIDATES
    )

    if not source_candidates:

        print("[MATCHER] No source candidates found.")

        return {
            "source_collection": source_collection,
            "target_collection": target_collection,
            "match_type": "no_source_match",
            "best_similarity": 0.0,
            "results": [],
        }

    # ========================================================
    # STEP 5
    # BEST SOURCE ITEM
    # ========================================================

    source_item = source_candidates[0]

    source_similarity = float(source_item.get("similarity", 0.0))

    print()
    print("[MATCHER] Best source item:")

    print(f"  Design ID: " f"{source_item.get('design_id', 'NONE')}")

    print(f"  Image: " f"{source_item.get('image', 'NONE')}")

    print(f"  Similarity: " f"{source_similarity:.4f}")

    # ========================================================
    # STEP 6
    # SOURCE CONFIDENCE
    # ========================================================

    if source_similarity < SOURCE_CONFIDENCE_THRESHOLD:

        print()
        print("[MATCHER] Source confidence too low.")

        print("[MATCHER] Returning visual candidates instead.")

        fallback_results = []

        # Search the opposite collection only for
        # displaying possible designs.
        target_candidates = search_collection(
            QUERY_SEGMENTED_PATH, target_collection, query_embedding, top_k
        )

        for result in target_candidates:

            result["match_type"] = "similar_design"

            result["match_method"] = "visual_similarity"

        fallback_results = target_candidates

        return {
            "source_collection": source_collection,
            "target_collection": target_collection,
            "match_type": "low_confidence",
            "source_similarity": round(source_similarity, 4),
            "best_similarity": round(
                (
                    float(fallback_results[0].get("similarity", 0.0))
                    if fallback_results
                    else 0.0
                ),
                4,
            ),
            "results": fallback_results,
        }

    # ========================================================
    # STEP 7
    # FIND MANUALLY PAIRED DESIGN
    # ========================================================

    target_item = None

    if USE_DESIGN_ID_MATCHING:

        target_item = find_paired_design(source_item, target_collection)

    # ========================================================
    # STEP 8
    # EXACT DESIGN FOUND
    # ========================================================

    if target_item is not None:

        exact_result = build_exact_result(
            source_item, target_item, source_collection, target_collection
        )

        print()
        print("=" * 70)

        print("EXACT DESIGN MATCH FOUND")

        print("=" * 70)

        print(f"Source collection : " f"{source_collection}")

        print(f"Source image      : " f"{source_item.get('image', '')}")

        print(f"Design ID         : " f"{source_item.get('design_id', '')}")

        print(f"Target collection : " f"{target_collection}")

        print(f"Target image      : " f"{target_item.get('image', '')}")

        print("Match method      : " "manual_design_id")

        print("=" * 70)

        return {
            "source_collection": source_collection,
            "target_collection": target_collection,
            "source_similarity": round(source_similarity, 4),
            "best_similarity": 1.0,
            "match_type": "exact_design",
            "match_method": "manual_design_id",
            "design_id": source_item.get("design_id", ""),
            "source_item": source_item,
            "results": [exact_result],
        }

    # ========================================================
    # STEP 9
    # NO MANUAL PAIRING YET
    #
    # We do NOT falsely claim exact match.
    # Return visual suggestions instead.
    # ========================================================

    print()
    print("[MATCHER] No manual design pairing exists.")

    print("[MATCHER] Returning visual suggestions.")

    target_candidates = search_collection(
        QUERY_SEGMENTED_PATH, target_collection, query_embedding, top_k
    )

    for result in target_candidates:

        result["match_type"] = "similar_design"

        result["match_method"] = "visual_similarity"

    best_similarity = (
        float(target_candidates[0].get("similarity", 0.0)) if target_candidates else 0.0
    )

    print()
    print("=" * 70)

    print("FINAL SEARCH RESULT")

    print("=" * 70)

    print(f"Source collection : " f"{source_collection}")

    print(f"Target collection : " f"{target_collection}")

    print(f"Match type        : " f"similar_design")

    print(f"Results returned  : " f"{len(target_candidates)}")

    print(f"Best similarity   : " f"{best_similarity:.4f}")

    print("=" * 70)

    return {
        "source_collection": source_collection,
        "target_collection": target_collection,
        "source_similarity": round(source_similarity, 4),
        "best_similarity": round(best_similarity, 4),
        "match_type": "similar_design",
        "match_method": "visual_similarity",
        "results": target_candidates,
    }


# ============================================================
# DIRECT COLLECTION SEARCH
# ============================================================


def match_collection(image_path, collection, top_k=TOP_K):

    collection = str(collection).lower().strip()

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
