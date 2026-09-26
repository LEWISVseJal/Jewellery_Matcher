"""
JewelMatch AI
Cross-Collection Visual Jewellery Matcher

Pipeline:

Query image
    ↓
Image validation
    ↓
Feature extraction
    ↓
Source collection identification
    ↓
Opposite collection restriction
    ↓
Global structural similarity
    ↓
ORB local feature comparison
    ↓
Candidate re-ranking
    ↓
Confidence gate
    ↓
MATCH or NO MATCH
"""

from __future__ import annotations

import json
import os
import pickle
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .embedding import (
    create_embedding,
    extract_orb_descriptors,
    validate_query_image,
    load_color_image,
    create_foreground_mask,
    crop_foreground,
    prepare_view,
)

# ============================================================
# PATHS
# ============================================================

SERVICE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SERVICE_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent

DATABASE_DIR = BACKEND_DIR / "database"

CATALOGUE_DIR = BACKEND_DIR / "catalogue"

CATALOGUE_FILE = DATABASE_DIR / "jewellery.json"

INDEX_FILE = DATABASE_DIR / "visual_index.pkl"

INDEX_VERSION = 2


# ============================================================
# MATCHING CONFIGURATION
# ============================================================

TOP_K = int(
    os.getenv(
        "JEWELMATCH_TOP_K",
        "8",
    )
)

# Minimum target similarity.
MATCH_THRESHOLD = float(
    os.getenv(
        "JEWELMATCH_MATCH_THRESHOLD",
        "0.56",
    )
)

# Minimum source evidence.
SOURCE_THRESHOLD = float(
    os.getenv(
        "JEWELMATCH_SOURCE_THRESHOLD",
        "0.54",
    )
)

# A candidate must be reasonably close to the best candidate.
RESULT_GAP = float(
    os.getenv(
        "JEWELMATCH_RESULT_GAP",
        "0.15",
    )
)

# Global feature weight.
GLOBAL_WEIGHT = 0.55

# Edge/structure weight.
STRUCTURE_WEIGHT = 0.30

# Local ORB weight.
LOCAL_WEIGHT = 0.15


# ============================================================
# MEMORY CACHE
# ============================================================

_INDEX_CACHE: dict | None = None


# ============================================================
# CATALOGUE
# ============================================================


def load_catalogue() -> list[dict]:
    """
    Load catalogue JSON.

    Supports:
        [...]
    or:
        {"items": [...]}
    or:
        {"jewellery": [...]}
    """

    if not CATALOGUE_FILE.exists():
        return []

    try:

        with open(
            CATALOGUE_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

    except Exception as error:

        print(f"[CATALOGUE] Failed to load JSON: {error}")

        return []

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        if isinstance(
            data.get("items"),
            list,
        ):
            return data["items"]

        if isinstance(
            data.get("jewellery"),
            list,
        ):
            return data["jewellery"]

    return []


def save_catalogue(
    items: list[dict],
) -> None:
    """
    Save catalogue in list format.
    """

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = CATALOGUE_FILE.with_suffix(".tmp")

    with open(
        temporary_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            items,
            file,
            indent=4,
            ensure_ascii=False,
        )

    temporary_file.replace(CATALOGUE_FILE)


# ============================================================
# COLLECTION HELPERS
# ============================================================


def normalize_collection(
    value: Any,
) -> str:
    """
    Normalize collection names.
    """

    value = str(value or "").strip().lower()

    if value in {
        "gold",
        "gold collection",
        "finished gold",
    }:
        return "gold"

    if value in {
        "prototype",
        "prototype collection",
        "green",
    }:
        return "prototype"

    return value


def get_item_id(
    item: dict,
) -> str:
    """
    Return stable catalogue ID.
    """

    for key in (
        "id",
        "jewellery_id",
        "design_id",
        "sku",
    ):

        value = item.get(key)

        if value not in (
            None,
            "",
        ):

            return str(value)

    return ""


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================


def resolve_image_path(
    item_or_path: dict | str | Path,
    collection: str | None = None,
) -> Path | None:
    """
    Resolve catalogue image paths safely.

    Handles:
    - absolute Windows paths
    - absolute Linux paths
    - backend/catalogue/gold/...
    - gold/filename
    - prototype/filename
    - filename only
    """

    if isinstance(
        item_or_path,
        dict,
    ):

        image_value = (
            item_or_path.get("image")
            or item_or_path.get("image_path")
            or item_or_path.get("path")
        )

        if not image_value:
            return None

        if collection is None:
            collection = normalize_collection(item_or_path.get("collection"))

    else:

        image_value = str(item_or_path)

    raw = str(image_value).strip()

    if not raw:
        return None

    normalized = raw.replace(
        "\\",
        "/",
    )

    # 1. Exact path.
    direct = Path(normalized)

    if direct.exists():
        return direct

    # 2. Relative to project root.
    project_path = PROJECT_DIR / normalized

    if project_path.exists():
        return project_path

    # 3. Relative to backend.
    backend_path = BACKEND_DIR / normalized

    if backend_path.exists():
        return backend_path

    filename = Path(normalized).name

    # 4. Known collection.
    normalized_collection = normalize_collection(collection)

    if normalized_collection in {
        "gold",
        "prototype",
    }:

        candidate = CATALOGUE_DIR / normalized_collection / filename

        if candidate.exists():
            return candidate

    # 5. Search both collections.
    for folder_name in (
        "gold",
        "prototype",
    ):

        folder = CATALOGUE_DIR / folder_name

        if not folder.exists():
            continue

        candidate = folder / filename

        if candidate.exists():
            return candidate

        try:

            matches = list(folder.rglob(filename))

            if matches:
                return matches[0]

        except Exception:
            pass

    return None


# ============================================================
# STRUCTURE SIMILARITY
# ============================================================


def edge_structure_similarity(
    query_path: str | Path,
    candidate_path: str | Path,
) -> float:
    """
    Compare normalized edge structure.

    Colour is ignored.
    """

    try:

        query = load_color_image(query_path)

        candidate = load_color_image(candidate_path)

        query_mask = create_foreground_mask(query)

        candidate_mask = create_foreground_mask(candidate)

        query_crop, query_crop_mask = crop_foreground(
            query,
            query_mask,
        )

        candidate_crop, candidate_crop_mask = crop_foreground(
            candidate,
            candidate_mask,
        )

        query_gray, _ = prepare_view(
            query_crop,
            query_crop_mask,
        )

        candidate_gray, _ = prepare_view(
            candidate_crop,
            candidate_crop_mask,
        )

        query_edges = cv2.Canny(
            query_gray,
            50,
            150,
        )

        candidate_edges = cv2.Canny(
            candidate_gray,
            50,
            150,
        )

        query_edges = query_edges.astype(np.float32) / 255.0

        candidate_edges = candidate_edges.astype(np.float32) / 255.0

        difference = float(np.mean(np.abs(query_edges - candidate_edges)))

        score = 1.0 - difference

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    except Exception as error:

        print(f"[STRUCTURE] Error: {error}")

        return 0.0


# ============================================================
# ORB SIMILARITY
# ============================================================


def orb_similarity(
    query_descriptors: dict,
    candidate_descriptors: dict,
) -> float:
    """
    Compare ORB descriptors using Hamming distance and
    Lowe-style ratio filtering.

    A small RANSAC-style geometric consistency check is also
    applied when enough matches are available.
    """

    if not query_descriptors:
        return 0.0

    if not candidate_descriptors:
        return 0.0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False,
    )

    scores = []

    for key in (
        "full",
        "crop",
    ):

        query_desc = query_descriptors.get(key)

        candidate_desc = candidate_descriptors.get(key)

        if query_desc is None or candidate_desc is None:
            continue

        if len(query_desc) < 3 or len(candidate_desc) < 3:
            continue

        try:

            matches = matcher.knnMatch(
                query_desc,
                candidate_desc,
                k=2,
            )

        except Exception:
            continue

        good_matches = []

        for pair in matches:

            if len(pair) < 2:
                continue

            first, second = pair

            if first.distance < 0.76 * second.distance:
                good_matches.append(first)

        if not good_matches:
            continue

        denominator = float(
            max(
                1,
                min(
                    len(query_desc),
                    len(candidate_desc),
                ),
            )
        )

        good_ratio = len(good_matches) / denominator

        # More useful for jewellery images than simply counting
        # raw matches.
        scores.append(
            min(
                1.0,
                good_ratio * 4.0,
            )
        )

    if not scores:
        return 0.0

    return float(np.mean(scores))


# ============================================================
# GLOBAL SIMILARITY
# ============================================================


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Cosine similarity for normalized vectors.
    """

    a = np.asarray(
        a,
        dtype=np.float32,
    ).reshape(-1)

    b = np.asarray(
        b,
        dtype=np.float32,
    ).reshape(-1)

    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator < 1e-8:
        return 0.0

    score = float(np.dot(a, b)) / float(denominator)

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


# ============================================================
# COMBINED SCORE
# ============================================================


def calculate_match_score(
    query_embedding: np.ndarray,
    candidate_embedding: np.ndarray,
    query_image_path: Path,
    candidate_image_path: Path,
    query_descriptors: dict,
    candidate_descriptors: dict,
) -> tuple[float, dict]:
    """
    Calculate the final design similarity.
    """

    global_score = cosine_similarity(
        query_embedding,
        candidate_embedding,
    )

    structure_score = edge_structure_similarity(
        query_image_path,
        candidate_image_path,
    )

    local_score = orb_similarity(
        query_descriptors,
        candidate_descriptors,
    )

    final_score = (
        GLOBAL_WEIGHT * global_score
        + STRUCTURE_WEIGHT * structure_score
        + LOCAL_WEIGHT * local_score
    )

    final_score = float(
        np.clip(
            final_score,
            0.0,
            1.0,
        )
    )

    details = {
        "global_similarity": round(
            global_score,
            4,
        ),
        "structure_similarity": round(
            structure_score,
            4,
        ),
        "local_similarity": round(
            local_score,
            4,
        ),
        "final_similarity": round(
            final_score,
            4,
        ),
    }

    return (
        final_score,
        details,
    )


# ============================================================
# INDEX BUILDING
# ============================================================


def build_index(
    verbose: bool = True,
) -> dict:
    """
    Build a complete ID-based visual index.

    Every catalogue entry keeps its own:
    - ID
    - collection
    - image
    - embedding
    - ORB descriptors

    This prevents catalogue/index ordering problems.
    """

    global _INDEX_CACHE

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalogue = load_catalogue()

    index = {
        "version": INDEX_VERSION,
        "collections": {
            "gold": {},
            "prototype": {},
        },
        "errors": [],
    }

    if verbose:

        print()
        print("=" * 70)
        print("JEWELMATCH AI - BUILD LIGHTWEIGHT VISUAL INDEX")
        print("=" * 70)
        print(f"Catalogue records: {len(catalogue)}")

    for item in catalogue:

        item_id = get_item_id(item)

        collection = normalize_collection(item.get("collection"))

        if not item_id:

            index["errors"].append(
                {
                    "reason": "Missing ID",
                    "item": item,
                }
            )

            continue

        if collection not in {
            "gold",
            "prototype",
        }:

            index["errors"].append(
                {
                    "id": item_id,
                    "reason": ("Invalid collection"),
                }
            )

            continue

        image_path = resolve_image_path(
            item,
            collection,
        )

        if image_path is None:

            index["errors"].append(
                {
                    "id": item_id,
                    "reason": ("Image not found"),
                    "image": item.get("image"),
                }
            )

            continue

        try:

            if verbose:

                print(
                    f"[INDEX] {collection.upper():10s} "
                    f"{item_id:20s} "
                    f"{image_path.name}"
                )

            embedding = create_embedding(image_path)

            descriptors = extract_orb_descriptors(image_path)

            index["collections"][collection][item_id] = {
                "id": item_id,
                "collection": collection,
                "image": str(
                    item.get(
                        "image",
                        image_path.name,
                    )
                ),
                "image_path": str(image_path),
                "embedding": embedding,
                "descriptors": descriptors,
            }

        except Exception as error:

            index["errors"].append(
                {
                    "id": item_id,
                    "reason": str(error),
                }
            )

            print(f"[INDEX] ERROR {item_id}: {error}")

    temporary_file = INDEX_FILE.with_suffix(".tmp")

    with open(
        temporary_file,
        "wb",
    ) as file:

        pickle.dump(
            index,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    temporary_file.replace(INDEX_FILE)

    _INDEX_CACHE = index

    if verbose:

        gold_count = len(index["collections"]["gold"])

        prototype_count = len(index["collections"]["prototype"])

        print()
        print(f"[INDEX] Gold: {gold_count}")

        print(f"[INDEX] Prototype: {prototype_count}")

        print(f"[INDEX] Errors: " f"{len(index['errors'])}")

        print(f"[INDEX] Saved: {INDEX_FILE}")

        print("=" * 70)
        print()

    return index


# ============================================================
# INDEX LOADING
# ============================================================


def clear_index_cache() -> None:
    """
    Clear in-memory index.
    """

    global _INDEX_CACHE

    _INDEX_CACHE = None


def load_index(
    rebuild_if_missing: bool = True,
) -> dict:
    """
    Load index from memory/disk.
    """

    global _INDEX_CACHE

    if _INDEX_CACHE is not None and _INDEX_CACHE.get("version") == INDEX_VERSION:
        return _INDEX_CACHE

    if not INDEX_FILE.exists():

        if rebuild_if_missing:
            return build_index()

        return {
            "version": INDEX_VERSION,
            "collections": {
                "gold": {},
                "prototype": {},
            },
            "errors": [],
        }

    try:

        with open(
            INDEX_FILE,
            "rb",
        ) as file:

            index = pickle.load(file)

        if (
            not isinstance(
                index,
                dict,
            )
            or index.get("version") != INDEX_VERSION
        ):

            if rebuild_if_missing:
                return build_index()

            return {
                "version": INDEX_VERSION,
                "collections": {
                    "gold": {},
                    "prototype": {},
                },
                "errors": [],
            }

        _INDEX_CACHE = index

        return index

    except Exception as error:

        print(f"[INDEX] Load error: {error}")

        if rebuild_if_missing:
            return build_index()

        raise


# ============================================================
# SOURCE COLLECTION
# ============================================================


def _best_collection_score(
    query_embedding: np.ndarray,
    collection: str,
    index: dict,
) -> tuple[float, str | None]:
    """
    Find strongest same-collection visual candidate.
    """

    entries = index.get(
        "collections",
        {},
    ).get(
        collection,
        {},
    )

    best_score = 0.0
    best_id = None

    for item_id, record in entries.items():

        embedding = record.get("embedding")

        if embedding is None:
            continue

        score = cosine_similarity(
            query_embedding,
            embedding,
        )

        if score > best_score:

            best_score = score
            best_id = item_id

    return (
        best_score,
        best_id,
    )


def identify_source_collection(
    query_embedding: np.ndarray,
    index: dict | None = None,
) -> dict:
    """
    Determine whether the query is visually closer to the
    Gold or Prototype collection.

    This is only used to decide the opposite target collection.
    """

    if index is None:
        index = load_index()

    gold_score, gold_id = _best_collection_score(
        query_embedding,
        "gold",
        index,
    )

    prototype_score, prototype_id = _best_collection_score(
        query_embedding,
        "prototype",
        index,
    )

    if gold_score >= prototype_score:

        source = "gold"

    else:

        source = "prototype"

    source_score = max(
        gold_score,
        prototype_score,
    )

    return {
        "source_collection": source,
        "source_score": source_score,
        "gold_score": gold_score,
        "prototype_score": prototype_score,
        "gold_best_id": gold_id,
        "prototype_best_id": prototype_id,
    }


# ============================================================
# RESULT CONVERSION
# ============================================================


def _catalogue_lookup() -> dict[str, dict]:
    """
    Create ID → catalogue metadata map.
    """

    lookup = {}

    for item in load_catalogue():

        item_id = get_item_id(item)

        if item_id:
            lookup[item_id] = item

    return lookup


# ============================================================
# MATCHING
# ============================================================


def match_jewellery(
    query_image_path: str | Path,
    target_collection: str = "auto",
    top_k: int | None = None,
) -> dict:
    """
    Main jewellery matching function.

    Returns:

        matched=True
            when reliable cross-collection results exist

        matched=False
            when the image is irrelevant or no reliable
            catalogue match exists.
    """

    query_image_path = Path(query_image_path)

    if not query_image_path.exists():

        return {
            "matched": False,
            "reason": "query_not_found",
            "message": ("The uploaded image could not be found."),
            "results": [],
        }

    # --------------------------------------------------------
    # STEP 1 - IMAGE VALIDATION
    # --------------------------------------------------------

    validation = validate_query_image(query_image_path)

    if not validation["valid"]:

        print("[MATCHER] Query rejected by image validation.")

        for reason in validation["reasons"]:

            print(f"[MATCHER] {reason}")

        return {
            "matched": False,
            "reason": "irrelevant_image",
            "message": (
                "The uploaded image does not appear "
                "to contain a suitable jewellery design."
            ),
            "validation": validation,
            "results": [],
        }

    # --------------------------------------------------------
    # STEP 2 - LOAD INDEX
    # --------------------------------------------------------

    index = load_index()

    gold_entries = index["collections"].get(
        "gold",
        {},
    )

    prototype_entries = index["collections"].get(
        "prototype",
        {},
    )

    if not gold_entries or not prototype_entries:

        return {
            "matched": False,
            "reason": "catalogue_not_ready",
            "message": (
                "Both Gold and Prototype catalogues " "must contain indexed jewellery."
            ),
            "results": [],
        }

    # --------------------------------------------------------
    # STEP 3 - QUERY FEATURES
    # --------------------------------------------------------

    try:

        query_embedding = create_embedding(query_image_path)

        query_descriptors = extract_orb_descriptors(query_image_path)

    except Exception as error:

        print(f"[MATCHER] Query feature error: {error}")

        return {
            "matched": False,
            "reason": "feature_error",
            "message": ("The uploaded image could not be analysed."),
            "results": [],
        }

    # --------------------------------------------------------
    # STEP 4 - COLLECTION
    # --------------------------------------------------------

    source_info = identify_source_collection(
        query_embedding,
        index,
    )

    source_collection = source_info["source_collection"]

    source_score = float(source_info["source_score"])

    # If caller explicitly provides source collection,
    # respect it.
    requested_target = normalize_collection(target_collection)

    if requested_target in {
        "gold",
        "prototype",
    }:

        target = requested_target

        if target == "gold":
            source_collection = "prototype"
        else:
            source_collection = "gold"

    else:

        target = "prototype" if source_collection == "gold" else "gold"

    print()
    print("=" * 70)
    print("JEWELMATCH AI - CROSS-COLLECTION SEARCH")
    print("=" * 70)

    print(f"[MATCHER] Gold similarity: " f"{source_info['gold_score']:.4f}")

    print(f"[MATCHER] Prototype similarity: " f"{source_info['prototype_score']:.4f}")

    print(f"[MATCHER] Source collection: " f"{source_collection}")

    print(f"[MATCHER] Target collection: " f"{target}")

    # --------------------------------------------------------
    # STEP 5 - SOURCE CONFIDENCE
    # --------------------------------------------------------

    if source_score < SOURCE_THRESHOLD:

        return {
            "matched": False,
            "reason": "weak_source_evidence",
            "message": (
                "The uploaded image does not have enough "
                "visual evidence to identify it as a "
                "catalogue-compatible jewellery design."
            ),
            "source_collection": source_collection,
            "target_collection": target,
            "source_score": round(
                source_score,
                4,
            ),
            "gold_score": round(
                source_info["gold_score"],
                4,
            ),
            "prototype_score": round(
                source_info["prototype_score"],
                4,
            ),
            "results": [],
        }

    # --------------------------------------------------------
    # STEP 6 - TARGET SEARCH
    # --------------------------------------------------------

    target_entries = index["collections"].get(
        target,
        {},
    )

    catalogue_lookup = _catalogue_lookup()

    candidates = []

    for item_id, record in target_entries.items():

        candidate_path = Path(
            record.get(
                "image_path",
                "",
            )
        )

        if not candidate_path.exists():

            resolved = resolve_image_path(
                record.get(
                    "image",
                    "",
                ),
                target,
            )

            if resolved is None:
                continue

            candidate_path = resolved

        try:

            candidate_embedding = np.asarray(
                record.get("embedding"),
                dtype=np.float32,
            )

            candidate_descriptors = record.get(
                "descriptors",
                {},
            )

            score, details = calculate_match_score(
                query_embedding,
                candidate_embedding,
                query_image_path,
                candidate_path,
                query_descriptors,
                candidate_descriptors,
            )

            candidates.append(
                {
                    "id": item_id,
                    "score": score,
                    "details": details,
                    "image_path": str(candidate_path),
                }
            )

        except Exception as error:

            print(f"[MATCHER] Candidate error " f"{item_id}: {error}")

    if not candidates:

        return {
            "matched": False,
            "reason": "no_candidates",
            "message": (
                "No indexed jewellery is available " "in the target collection."
            ),
            "source_collection": source_collection,
            "target_collection": target,
            "results": [],
        }

    # --------------------------------------------------------
    # STEP 7 - RANK
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    best_candidate = candidates[0]

    best_score = float(best_candidate["score"])

    # --------------------------------------------------------
    # STEP 8 - NO-MATCH GATE
    # --------------------------------------------------------

    if best_score < MATCH_THRESHOLD:

        print(
            f"[MATCHER] NO MATCH - "
            f"best score {best_score:.4f} "
            f"< threshold {MATCH_THRESHOLD:.4f}"
        )

        return {
            "matched": False,
            "reason": "low_similarity",
            "message": ("No reliable jewellery design match " "was found."),
            "source_collection": source_collection,
            "target_collection": target,
            "source_score": round(
                source_score,
                4,
            ),
            "best_similarity": round(
                best_score,
                4,
            ),
            "threshold": MATCH_THRESHOLD,
            "results": [],
        }

    # --------------------------------------------------------
    # STEP 9 - FILTER WEAK SECONDARY RESULTS
    # --------------------------------------------------------

    minimum_result_score = max(
        MATCH_THRESHOLD,
        best_score - RESULT_GAP,
    )

    accepted_candidates = [
        candidate
        for candidate in candidates
        if candidate["score"] >= minimum_result_score
    ]

    accepted_candidates = accepted_candidates[: (top_k or TOP_K)]

    # --------------------------------------------------------
    # STEP 10 - BUILD RESPONSE
    # --------------------------------------------------------

    results = []

    for candidate in accepted_candidates:

        item_id = candidate["id"]

        metadata = catalogue_lookup.get(
            item_id,
            {},
        )

        image_path = Path(candidate["image_path"])

        image_value = metadata.get(
            "image",
            image_path.name,
        )

        similarity = float(candidate["score"])

        result = {
            "id": item_id,
            "jewellery_id": item_id,
            "name": metadata.get(
                "name",
                metadata.get(
                    "title",
                    item_id,
                ),
            ),
            "collection": target,
            "gender": metadata.get(
                "gender",
                "",
            ),
            "type": metadata.get(
                "type",
                "Ring",
            ),
            "subtype": metadata.get(
                "subtype",
                "",
            ),
            "description": metadata.get(
                "description",
                "",
            ),
            "image": str(image_value),
            "image_path": str(image_path),
            "similarity": round(
                similarity,
                4,
            ),
            "score": round(
                similarity,
                4,
            ),
            "match_score": round(
                similarity * 100,
                1,
            ),
            "match_details": candidate["details"],
        }

        results.append(result)

    if not results:

        return {
            "matched": False,
            "reason": "no_reliable_results",
            "message": ("No reliable jewellery match " "was found."),
            "source_collection": source_collection,
            "target_collection": target,
            "results": [],
        }

    print(f"[MATCHER] Returning " f"{len(results)} results")

    print(f"[MATCHER] Best similarity: " f"{best_score:.4f}")

    print(f"[MATCHER] Best match: " f"{results[0]['name']}")

    return {
        "matched": True,
        "message": ("Reliable jewellery matches found."),
        "source_collection": source_collection,
        "target_collection": target,
        "source_score": round(
            source_score,
            4,
        ),
        "best_similarity": round(
            best_score,
            4,
        ),
        "best_match_score": round(
            best_score * 100,
            1,
        ),
        "results": results,
    }


# ============================================================
# COMPATIBILITY ALIAS
# ============================================================


def find_matches(
    query_image_path: str | Path,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Backward-compatible helper for older code.

    Returns only result list.
    """

    response = match_jewellery(
        query_image_path,
        target_collection="auto",
        top_k=top_k,
    )

    return response.get(
        "results",
        [],
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print("Usage:")

        print("python -m backend.services.matcher " "path_to_query_image")

        raise SystemExit(1)

    query = Path(sys.argv[1])

    response = match_jewellery(query)

    print()

    print(
        json.dumps(
            response,
            indent=2,
            default=str,
        )
    )
