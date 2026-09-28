import json
import os
from pathlib import Path

import cv2
import numpy as np

from .embedding import create_embedding


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

CATALOGUE_JSON = BASE_DIR / "database" / "jewellery.json"

GOLD_DIR = BASE_DIR / "catalogue" / "gold"
PROTOTYPE_DIR = BASE_DIR / "catalogue" / "prototype"

INDEX_FILE = BASE_DIR / "database" / "dino_index.npz"


# ============================================================
# MATCHING CONFIGURATION
# ============================================================

# Current working threshold.
DINO_DIRECT_THRESHOLD = 0.35

# Number of DINO candidates to verify.
TOP_CANDIDATES = 8

# Do not perform expensive design verification below this score.
DESIGN_CHECK_MIN_DINO = 0.35

# Final combined score threshold.
FINAL_MATCH_THRESHOLD = 0.62

# Strong design evidence.
STRONG_DESIGN_THRESHOLD = 0.78
STRONG_DESIGN_DINO_MIN = 0.45

# Minimum difference between first and second candidate.
MIN_GAP_FOR_WEAK_MATCH = 0.025


# ============================================================
# COLLECTION HELPERS
# ============================================================


def _normalize_collection(value):
    value = str(value or "").strip().lower()

    if value in {
        "gold",
        "g",
        "finished",
    }:
        return "gold"

    if value in {
        "prototype",
        "p",
        "green",
        "sample",
    }:
        return "prototype"

    return value


def _detect_query_collection(query_path):
    """
    Try to determine whether the uploaded query image belongs
    to the Gold or Prototype collection.

    This is only used for metadata/source reporting.

    IMPORTANT:
    It does NOT restrict 'all' mode.
    """

    try:
        path = Path(query_path).resolve()

        parts = [
            str(part).strip().lower()
            for part in path.parts
        ]

        if "gold" in parts:
            return "gold"

        if "prototype" in parts:
            return "prototype"

        path_string = str(path).lower()

        if "gold_img" in path_string:
            return "gold"

        if "prototype_img" in path_string:
            return "prototype"

        if "gold" in path_string:
            return "gold"

        if "prototype" in path_string:
            return "prototype"

    except Exception:
        pass

    return None


def _normalize_search_mode(value):
    """
    Normalize all supported search mode names.

    Supported canonical modes:

        all
        gold_to_prototype
        prototype_to_gold
    """

    value = str(value or "all").strip().lower()

    aliases = {
        "all": "all",
        "both": "all",
        "all_collections": "all",
        "search_all": "all",

        "gold_to_prototype": "gold_to_prototype",
        "gold-prototype": "gold_to_prototype",
        "gold_to_green": "gold_to_prototype",
        "gold_to_sample": "gold_to_prototype",

        "prototype_to_gold": "prototype_to_gold",
        "prototype-gold": "prototype_to_gold",
        "green_to_gold": "prototype_to_gold",
        "sample_to_gold": "prototype_to_gold",
    }

    return aliases.get(
        value,
        "all",
    )


# ============================================================
# CATALOGUE HELPERS
# ============================================================


def _load_catalogue():

    if not CATALOGUE_JSON.exists():

        print(
            f"[MATCHER] Catalogue not found: "
            f"{CATALOGUE_JSON}"
        )

        return []

    try:

        with open(
            CATALOGUE_JSON,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        if isinstance(data, dict):

            if "items" in data:

                data = data["items"]

            else:

                data = list(data.values())

        if isinstance(data, list):

            return data

        return []

    except Exception as exc:

        print(
            f"[MATCHER] Failed to load catalogue: "
            f"{exc}"
        )

        return []


def _item_id(item):

    return (
        item.get("id")
        or item.get("item_id")
        or item.get("design_id")
        or item.get("sku")
        or ""
    )


def _design_id(item):

    return (
        item.get("design_id")
        or item.get("id")
        or item.get("item_id")
        or item.get("sku")
        or ""
    )


def _item_name(item):
    """
    Get jewellery display name from catalogue JSON.

    Supports multiple possible field names.
    """

    if not item:

        return ""

    return (
        item.get("name")
        or item.get("jewellery_name")
        or item.get("jewelry_name")
        or item.get("title")
        or item.get("product_name")
        or item.get("display_name")
        or ""
    )


def _item_type(item):

    if not item:

        return ""

    return (
        item.get("type")
        or item.get("jewellery_type")
        or item.get("jewelry_type")
        or item.get("category")
        or ""
    )


def _item_subtype(item):

    if not item:

        return ""

    return (
        item.get("subtype")
        or item.get("jewellery_subtype")
        or item.get("jewelry_subtype")
        or ""
    )


def _item_sku(item):

    if not item:

        return ""

    return (
        item.get("sku")
        or item.get("product_code")
        or ""
    )


def _find_catalogue_item(
    item_id="",
    design_id="",
    collection="",
):
    """
    Find the full catalogue record for a matched item.

    Matching order:
    1. item ID + collection
    2. design ID + collection
    3. item ID
    4. design ID
    """

    catalogue = _load_catalogue()

    wanted_id = str(
        item_id or ""
    ).strip()

    wanted_design = str(
        design_id or ""
    ).strip()

    wanted_collection = _normalize_collection(
        collection
    )

    # --------------------------------------------------------
    # First pass: ID + collection
    # --------------------------------------------------------

    if wanted_id:

        for item in catalogue:

            current_id = str(
                _item_id(item)
            ).strip()

            current_collection = _normalize_collection(
                item.get("collection")
                or item.get("type")
                or item.get("category")
            )

            if (
                current_id == wanted_id
                and (
                    not wanted_collection
                    or current_collection == wanted_collection
                )
            ):

                return item

    # --------------------------------------------------------
    # Second pass: design ID + collection
    # --------------------------------------------------------

    if wanted_design:

        for item in catalogue:

            current_design = str(
                _design_id(item)
            ).strip()

            current_collection = _normalize_collection(
                item.get("collection")
                or item.get("type")
                or item.get("category")
            )

            if (
                current_design == wanted_design
                and (
                    not wanted_collection
                    or current_collection == wanted_collection
                )
            ):

                return item

    # --------------------------------------------------------
    # Third pass: ID only
    # --------------------------------------------------------

    if wanted_id:

        for item in catalogue:

            current_id = str(
                _item_id(item)
            ).strip()

            if current_id == wanted_id:

                return item

    # --------------------------------------------------------
    # Fourth pass: design ID only
    # --------------------------------------------------------

    if wanted_design:

        for item in catalogue:

            current_design = str(
                _design_id(item)
            ).strip()

            if current_design == wanted_design:

                return item

    return None


# ============================================================
# IMAGE PATH
# ============================================================


def _resolve_image_path(item):

    collection = _normalize_collection(
        item.get("collection")
        or item.get("type")
        or item.get("category")
    )

    filename = (
        item.get("filename")
        or item.get("image")
        or item.get("image_path")
        or item.get("path")
    )

    if not filename:

        return None

    filename = str(
        filename
    ).replace(
        "\\",
        "/",
    )

    possible = Path(filename)

    # Absolute path
    if possible.is_absolute() and possible.exists():

        return possible

    if collection == "gold":

        base_dir = GOLD_DIR

    elif collection == "prototype":

        base_dir = PROTOTYPE_DIR

    else:

        return None

    # Remove stored collection prefix.
    filename = filename.replace(
        "gold/",
        "",
    )

    filename = filename.replace(
        "prototype/",
        "",
    )

    path = base_dir / filename

    if path.exists():

        return path

    return None


# ============================================================
# IMAGE HELPERS
# ============================================================


def _load_gray(
    image_path,
    size=320,
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    h, w = gray.shape[:2]

    if h == 0 or w == 0:

        return None

    scale = min(
        size / w,
        size / h,
    )

    new_w = max(
        1,
        int(w * scale),
    )

    new_h = max(
        1,
        int(h * scale),
    )

    resized = cv2.resize(
        gray,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (
            size,
            size,
        ),
        dtype=np.uint8,
    )

    x = (
        size - new_w
    ) // 2

    y = (
        size - new_h
    ) // 2

    canvas[
        y : y + new_h,
        x : x + new_w,
    ] = resized

    return canvas


def _normalize_image(gray):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(
            8,
            8,
        ),
    )

    normalized = clahe.apply(
        gray
    )

    normalized = cv2.GaussianBlur(
        normalized,
        (
            3,
            3,
        ),
        0,
    )

    return normalized


# ============================================================
# SHAPE SIMILARITY
# ============================================================


def _shape_similarity(
    query_gray,
    candidate_gray,
):

    try:

        q = _normalize_image(
            query_gray
        )

        c = _normalize_image(
            candidate_gray
        )

        _, q_bin = cv2.threshold(
            q,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU,
        )

        _, c_bin = cv2.threshold(
            c,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU,
        )

        def largest_contour(
            binary,
        ):

            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if not contours:

                return None

            return max(
                contours,
                key=cv2.contourArea,
            )

        q_contour = largest_contour(
            q_bin
        )

        c_contour = largest_contour(
            c_bin
        )

        if (
            q_contour is None
            or c_contour is None
        ):

            return 0.0

        q_area = cv2.contourArea(
            q_contour
        )

        c_area = cv2.contourArea(
            c_contour
        )

        if (
            q_area < 100
            or c_area < 100
        ):

            return 0.0

        q_hu = cv2.HuMoments(
            cv2.moments(
                q_contour
            )
        ).flatten()

        c_hu = cv2.HuMoments(
            cv2.moments(
                c_contour
            )
        ).flatten()

        q_hu = (
            -np.sign(q_hu)
            * np.log10(
                np.abs(q_hu)
                + 1e-12
            )
        )

        c_hu = (
            -np.sign(c_hu)
            * np.log10(
                np.abs(c_hu)
                + 1e-12
            )
        )

        distance = np.linalg.norm(
            q_hu - c_hu
        )

        similarity = (
            1.0
            / (
                1.0
                + distance
            )
        )

        return float(
            np.clip(
                similarity,
                0.0,
                1.0,
            )
        )

    except Exception as exc:

        print(
            f"[DESIGN] Shape comparison error: "
            f"{exc}"
        )

        return 0.0


# ============================================================
# EDGE / STRUCTURE SIMILARITY
# ============================================================


def _edge_similarity(
    query_gray,
    candidate_gray,
):

    try:

        q = _normalize_image(
            query_gray
        )

        c = _normalize_image(
            candidate_gray
        )

        q_edges = cv2.Canny(
            q,
            50,
            150,
        )

        c_edges = cv2.Canny(
            c,
            50,
            150,
        )

        kernel = np.ones(
            (
                2,
                2,
            ),
            np.uint8,
        )

        q_edges = cv2.dilate(
            q_edges,
            kernel,
            iterations=1,
        )

        c_edges = cv2.dilate(
            c_edges,
            kernel,
            iterations=1,
        )

        q_vec = (
            q_edges
            .astype(np.float32)
            .flatten()
        )

        c_vec = (
            c_edges
            .astype(np.float32)
            .flatten()
        )

        q_norm = np.linalg.norm(
            q_vec
        )

        c_norm = np.linalg.norm(
            c_vec
        )

        if (
            q_norm == 0
            or c_norm == 0
        ):

            return 0.0

        cosine = float(
            np.dot(
                q_vec,
                c_vec,
            )
            / (
                q_norm
                * c_norm
            )
        )

        return float(
            np.clip(
                cosine,
                0.0,
                1.0,
            )
        )

    except Exception as exc:

        print(
            f"[DESIGN] Edge comparison error: "
            f"{exc}"
        )

        return 0.0


# ============================================================
# ORB / PATTERN SIMILARITY
# ============================================================


def _orb_similarity(
    query_gray,
    candidate_gray,
):

    try:

        orb = cv2.ORB_create(
            nfeatures=700,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            fastThreshold=10,
        )

        kp1, des1 = orb.detectAndCompute(
            query_gray,
            None,
        )

        kp2, des2 = orb.detectAndCompute(
            candidate_gray,
            None,
        )

        if (
            des1 is None
            or des2 is None
        ):

            return 0.0

        if (
            len(des1) < 3
            or len(des2) < 3
        ):

            return 0.0

        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False,
        )

        matches = matcher.knnMatch(
            des1,
            des2,
            k=2,
        )

        good_matches = []

        for pair in matches:

            if len(pair) < 2:

                continue

            m, n = pair

            if (
                m.distance
                < 0.75 * n.distance
            ):

                good_matches.append(
                    m
                )

        denominator = max(
            1,
            min(
                len(des1),
                len(des2),
            ),
        )

        ratio = (
            len(good_matches)
            / denominator
        )

        return float(
            np.clip(
                ratio * 3.0,
                0.0,
                1.0,
            )
        )

    except Exception as exc:

        print(
            f"[DESIGN] ORB comparison error: "
            f"{exc}"
        )

        return 0.0


# ============================================================
# DESIGN VERIFICATION
# ============================================================


def _calculate_design_features(
    query_path,
    candidate_path,
):

    query_gray = _load_gray(
        query_path
    )

    candidate_gray = _load_gray(
        candidate_path
    )

    if (
        query_gray is None
        or candidate_gray is None
    ):

        return {
            "shape": 0.0,
            "structure": 0.0,
            "pattern": 0.0,
            "design_score": 0.0,
        }

    shape = _shape_similarity(
        query_gray,
        candidate_gray,
    )

    structure = _edge_similarity(
        query_gray,
        candidate_gray,
    )

    pattern = _orb_similarity(
        query_gray,
        candidate_gray,
    )

    design_score = (
        shape * 0.40
        + structure * 0.35
        + pattern * 0.25
    )

    return {
        "shape": float(shape),
        "structure": float(structure),
        "pattern": float(pattern),
        "design_score": float(
            np.clip(
                design_score,
                0.0,
                1.0,
            )
        ),
    }


# ============================================================
# INDEX BUILD
# ============================================================


def build_index(
    force=False,
):

    catalogue = _load_catalogue()

    if not catalogue:

        print(
            "[INDEX] No catalogue items found."
        )

        return {
            "indexed": 0,
            "skipped": 0,
        }

    if (
        INDEX_FILE.exists()
        and not force
    ):

        print(
            f"[INDEX] Existing index found: "
            f"{INDEX_FILE}"
        )

        return load_index()

    print(
        "=" * 70
    )

    print(
        "DINOv2 INDEX BUILD"
    )

    print(
        "=" * 70
    )

    embeddings = []
    ids = []
    collections = []
    image_paths = []
    design_ids = []

    skipped = 0

    for index, item in enumerate(
        catalogue,
        start=1,
    ):

        item_id = _item_id(
            item
        )

        design_id = _design_id(
            item
        )

        collection = _normalize_collection(
            item.get("collection")
            or item.get("type")
            or item.get("category")
        )

        image_path = _resolve_image_path(
            item
        )

        print(
            f"[INDEX] "
            f"{index}/"
            f"{len(catalogue)} "
            f"{item_id} | "
            f"{collection}"
        )

        if (
            not image_path
            or not image_path.exists()
        ):

            print(
                "[INDEX] SKIP - "
                "image not found"
            )

            skipped += 1

            continue

        try:

            embedding = create_embedding(
                str(image_path)
            )

            if embedding is None:

                print(
                    "[INDEX] SKIP - "
                    "embedding failed"
                )

                skipped += 1

                continue

            embedding = np.asarray(
                embedding,
                dtype=np.float32,
            )

            norm = np.linalg.norm(
                embedding
            )

            if norm == 0:

                print(
                    "[INDEX] SKIP - "
                    "zero embedding"
                )

                skipped += 1

                continue

            embedding = (
                embedding / norm
            )

            embeddings.append(
                embedding
            )

            ids.append(
                str(item_id)
            )

            collections.append(
                str(collection)
            )

            image_paths.append(
                str(image_path)
            )

            design_ids.append(
                str(design_id)
            )

        except Exception as exc:

            print(
                f"[INDEX] SKIP - "
                f"{exc}"
            )

            skipped += 1

    if not embeddings:

        raise RuntimeError(
            "No catalogue images "
            "could be embedded."
        )

    matrix = np.vstack(
        embeddings
    ).astype(
        np.float32
    )

    INDEX_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        INDEX_FILE,
        embeddings=matrix,
        ids=np.array(ids),
        collections=np.array(collections),
        image_paths=np.array(image_paths),
        design_ids=np.array(design_ids),
    )

    print(
        "=" * 70
    )

    print(
        f"[INDEX] Indexed: "
        f"{len(embeddings)}"
    )

    print(
        f"[INDEX] Skipped: "
        f"{skipped}"
    )

    print(
        f"[INDEX] Saved: "
        f"{INDEX_FILE}"
    )

    print(
        "=" * 70
    )

    return {
        "indexed": len(embeddings),
        "skipped": skipped,
    }


# ============================================================
# LOAD INDEX
# ============================================================


def load_index():

    if not INDEX_FILE.exists():

        print(
            "[MATCHER] Index does not exist."
        )

        return None

    try:

        data = np.load(
            INDEX_FILE,
            allow_pickle=True,
        )

        index = {
            "embeddings": data[
                "embeddings"
            ].astype(
                np.float32
            ),
            "ids": data[
                "ids"
            ].astype(
                str
            ),
            "collections": data[
                "collections"
            ].astype(
                str
            ),
            "image_paths": data[
                "image_paths"
            ].astype(
                str
            ),
        }

        if "design_ids" in data:

            index["design_ids"] = data[
                "design_ids"
            ].astype(
                str
            )

        else:

            index["design_ids"] = (
                index["ids"]
            )

        print(
            f"[MATCHER] Loaded "
            f"{len(index['ids'])} "
            f"catalogue embeddings"
        )

        return index

    except Exception as exc:

        print(
            f"[MATCHER] Failed to "
            f"load index: {exc}"
        )

        return None


# ============================================================
# COSINE SIMILARITY
# ============================================================


def _cosine_similarity(
    query_embedding,
    matrix,
):

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    query_norm = np.linalg.norm(
        query_embedding
    )

    if query_norm == 0:

        return np.zeros(
            len(matrix),
            dtype=np.float32,
        )

    query_embedding = (
        query_embedding
        / query_norm
    )

    matrix_norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    matrix_norms[
        matrix_norms == 0
    ] = 1.0

    normalized_matrix = (
        matrix
        / matrix_norms
    )

    return np.dot(
        normalized_matrix,
        query_embedding,
    )


# ============================================================
# MAIN MATCHING FUNCTION
# ============================================================


def match_jewellery(
    query_path,
    top_k=8,
    search_mode="all",
):

    print()
    print(
        "=" * 70
    )

    print(
        "JEWELMATCH AI - "
        "DINOv2 CROSS-COLLECTION SEARCH"
    )

    print(
        "=" * 70
    )

    print(
        f"Query image: "
        f"{Path(query_path).name}"
    )

    # --------------------------------------------------------
    # Normalize search mode
    # --------------------------------------------------------

    search_mode = _normalize_search_mode(
        search_mode
    )

    print(
        f"[MATCHER] Search mode: "
        f"{search_mode}"
    )

    # --------------------------------------------------------
    # Detect query source collection
    # --------------------------------------------------------

    source_collection = (
        _detect_query_collection(
            query_path
        )
    )

    if source_collection:

        print(
            f"[MATCHER] Detected query "
            f"collection: "
            f"{source_collection}"
        )

    else:

        print(
            "[MATCHER] Query collection "
            "could not be detected."
        )

    # --------------------------------------------------------
    # Determine target collection(s)
    # --------------------------------------------------------

    if search_mode == "gold_to_prototype":

        target_collections = {
            "prototype"
        }

        target_collection = (
            "prototype"
        )

        print(
            "[MATCHER] MODE: "
            "GOLD → PROTOTYPE"
        )

    elif search_mode == "prototype_to_gold":

        target_collections = {
            "gold"
        }

        target_collection = (
            "gold"
        )

        print(
            "[MATCHER] MODE: "
            "PROTOTYPE → GOLD"
        )

    else:

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # ALL MODE searches BOTH collections.
        #
        # It does NOT automatically select one source
        # collection and search only the opposite collection.
        # ----------------------------------------------------

        target_collections = {
            "gold",
            "prototype",
        }

        target_collection = "both"

        print(
            "[MATCHER] MODE: "
            "ALL COLLECTIONS"
        )

        print(
            "[MATCHER] Searching "
            "Gold + Prototype"
        )

    print(
        f"[MATCHER] Target collection: "
        f"{target_collection}"
    )

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not os.path.exists(
        query_path
    ):

        print(
            "[MATCHER] Query image "
            "not found."
        )

        return {
            "matched": False,
            "results": [],
            "best_similarity": 0.0,
            "source_collection": source_collection,
            "target_collection": target_collection,
            "search_mode": search_mode,
            "message": (
                "Query image not found."
            ),
        }

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    try:

        query_embedding = create_embedding(
            query_path
        )

    except Exception as exc:

        print(
            f"[MATCHER] Query "
            f"embedding failed: "
            f"{exc}"
        )

        return {
            "matched": False,
            "results": [],
            "best_similarity": 0.0,
            "source_collection": source_collection,
            "target_collection": target_collection,
            "search_mode": search_mode,
            "message": (
                "Unable to process "
                "the uploaded image."
            ),
        }

    if query_embedding is None:

        return {
            "matched": False,
            "results": [],
            "best_similarity": 0.0,
            "source_collection": source_collection,
            "target_collection": target_collection,
            "search_mode": search_mode,
            "message": (
                "Unable to create "
                "image embedding."
            ),
        }

    # --------------------------------------------------------
    # Load index
    # --------------------------------------------------------

    index = load_index()

    if index is None:

        return {
            "matched": False,
            "results": [],
            "best_similarity": 0.0,
            "source_collection": source_collection,
            "target_collection": target_collection,
            "search_mode": search_mode,
            "message": (
                "Jewellery search index "
                "is unavailable."
            ),
        }

    embeddings = index[
        "embeddings"
    ]

    collections = index[
        "collections"
    ]

    # --------------------------------------------------------
    # Calculate similarities
    # --------------------------------------------------------

    similarities = _cosine_similarity(
        query_embedding,
        embeddings,
    )

    gold_mask = (
        collections == "gold"
    )

    prototype_mask = (
        collections == "prototype"
    )

    gold_similarity = (
        float(
            np.max(
                similarities[
                    gold_mask
                ]
            )
        )
        if np.any(gold_mask)
        else 0.0
    )

    prototype_similarity = (
        float(
            np.max(
                similarities[
                    prototype_mask
                ]
            )
        )
        if np.any(prototype_mask)
        else 0.0
    )

    print(
        f"[MATCHER] Gold best "
        f"similarity: "
        f"{gold_similarity:.4f}"
    )

    print(
        f"[MATCHER] Prototype best "
        f"similarity: "
        f"{prototype_similarity:.4f}"
    )

    # --------------------------------------------------------
    # Build target mask
    # --------------------------------------------------------

    if (
        "gold" in target_collections
        and "prototype" in target_collections
    ):

        target_mask = (
            gold_mask
            | prototype_mask
        )

    elif "gold" in target_collections:

        target_mask = gold_mask

    elif "prototype" in target_collections:

        target_mask = prototype_mask

    else:

        target_mask = np.zeros(
            len(collections),
            dtype=bool,
        )

    # --------------------------------------------------------
    # Target collection candidates
    # --------------------------------------------------------

    target_indices = np.where(
        target_mask
    )[0]

    if len(target_indices) == 0:

        print(
            "[MATCHER] No target "
            "collection items."
        )

        return {
            "matched": False,
            "results": [],
            "best_similarity": 0.0,
            "source_collection": source_collection,
            "target_collection": target_collection,
            "search_mode": search_mode,
            "message": (
                "No jewellery exists "
                "in the selected target "
                "collection."
            ),
        }

    target_scores = similarities[
        target_indices
    ]

    order = np.argsort(
        target_scores
    )[::-1]

    candidate_count = min(
        TOP_CANDIDATES,
        len(order),
    )

    candidate_indices = [
        int(
            target_indices[i]
        )
        for i in order[
            :candidate_count
        ]
    ]

    # --------------------------------------------------------
    # DINO candidate log
    # --------------------------------------------------------

    print()
    print(
        "[MATCHER] DINO CANDIDATES"
    )

    for rank, idx in enumerate(
        candidate_indices,
        start=1,
    ):

        print(
            f"  {rank}. "
            f"{index['design_ids'][idx]} "
            f"| "
            f"{index['collections'][idx]} "
            f"-> "
            f"{similarities[idx]:.4f}"
        )

    # --------------------------------------------------------
    # Best / second
    # --------------------------------------------------------

    best_idx = candidate_indices[
        0
    ]

    best_similarity = float(
        similarities[
            best_idx
        ]
    )

    second_similarity = (
        float(
            similarities[
                candidate_indices[1]
            ]
        )
        if len(candidate_indices) > 1
        else 0.0
    )

    gap = (
        best_similarity
        - second_similarity
    )

    print()
    print(
        f"[MATCHER] Best similarity: "
        f"{best_similarity:.4f}"
    )

    print(
        f"[MATCHER] Second similarity: "
        f"{second_similarity:.4f}"
    )

    print(
        f"[MATCHER] Result gap: "
        f"{gap:.4f}"
    )

    # ========================================================
    # DESIGN VERIFICATION
    # ========================================================

    verified_results = []

    print()
    print(
        "=" * 70
    )

    print(
        "DESIGN VERIFICATION"
    )

    print(
        "=" * 70
    )

    for rank, idx in enumerate(
        candidate_indices,
        start=1,
    ):

        dino_score = float(
            similarities[idx]
        )

        candidate_path = (
            index["image_paths"][idx]
        )

        candidate_id = (
            index["ids"][idx]
        )

        design_id = (
            index["design_ids"][idx]
        )

        candidate_collection = (
            _normalize_collection(
                index["collections"][idx]
            )
        )

        # ----------------------------------------------------
        # Catalogue metadata
        # ----------------------------------------------------

        catalogue_item = (
            _find_catalogue_item(
                item_id=candidate_id,
                design_id=design_id,
                collection=candidate_collection,
            )
        )

        item_name = _item_name(
            catalogue_item
        )

        item_type = _item_type(
            catalogue_item
        )

        item_subtype = _item_subtype(
            catalogue_item
        )

        item_sku = _item_sku(
            catalogue_item
        )

        # ----------------------------------------------------
        # Design verification
        # ----------------------------------------------------

        if (
            dino_score
            < DESIGN_CHECK_MIN_DINO
        ):

            print(
                f"\n[DESIGN] Candidate "
                f"{rank}: "
                f"{design_id}"
            )

            print(
                f"[DESIGN] Collection: "
                f"{candidate_collection}"
            )

            print(
                f"[DESIGN] DINO: "
                f"{dino_score:.4f}"
            )

            print(
                "[DESIGN] Skipped - "
                "DINO score too low"
            )

            design_features = {
                "shape": 0.0,
                "structure": 0.0,
                "pattern": 0.0,
                "design_score": 0.0,
            }

        else:

            design_features = (
                _calculate_design_features(
                    query_path,
                    candidate_path,
                )
            )

            print(
                f"\n[DESIGN] Candidate "
                f"{rank}: "
                f"{design_id}"
            )

            print(
                f"[DESIGN] Collection: "
                f"{candidate_collection}"
            )

            print(
                f"[DESIGN] Name: "
                f"{item_name or 'Not available'}"
            )

            print(
                f"[DESIGN] DINO: "
                f"{dino_score:.4f}"
            )

            print(
                f"[DESIGN] Shape: "
                f"{design_features['shape']:.4f}"
            )

            print(
                f"[DESIGN] Structure: "
                f"{design_features['structure']:.4f}"
            )

            print(
                f"[DESIGN] Pattern: "
                f"{design_features['pattern']:.4f}"
            )

            print(
                f"[DESIGN] Design score: "
                f"{design_features['design_score']:.4f}"
            )

        design_score = (
            design_features[
                "design_score"
            ]
        )

        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        final_score = (
            dino_score * 0.55
            + design_score * 0.45
        )

        # ----------------------------------------------------
        # Match conditions
        # ----------------------------------------------------

        direct_dino_match = (
            dino_score
            >= DINO_DIRECT_THRESHOLD
        )

        strong_design_match = (
            dino_score
            >= STRONG_DESIGN_DINO_MIN
            and design_score
            >= STRONG_DESIGN_THRESHOLD
            and gap
            >= MIN_GAP_FOR_WEAK_MATCH
        )

        final_match = (
            direct_dino_match
            or (
                final_score
                >= FINAL_MATCH_THRESHOLD
                and gap
                >= MIN_GAP_FOR_WEAK_MATCH
            )
            or strong_design_match
        )

        # ----------------------------------------------------
        # RESULT DATA
        # ----------------------------------------------------

        verified_results.append(
            {
                "rank": rank,

                "id": candidate_id,

                "design_id": design_id,

                "name": item_name,

                "jewellery_name": item_name,

                "jewelry_name": item_name,

                "title": item_name,

                "type": item_type,

                "subtype": item_subtype,

                "sku": item_sku,

                "collection": candidate_collection,

                "image_path": candidate_path,

                "similarity": dino_score,

                "score": final_score,

                "match_score": final_score,

                "similarity_percentage": round(
                    final_score * 100,
                    2,
                ),

                "dino_similarity": dino_score,

                "shape_similarity": (
                    design_features[
                        "shape"
                    ]
                ),

                "structure_similarity": (
                    design_features[
                        "structure"
                    ]
                ),

                "pattern_similarity": (
                    design_features[
                        "pattern"
                    ]
                ),

                "design_score": design_score,

                "search_mode": search_mode,
            }
        )

    # ========================================================
    # SORT RESULTS
    # ========================================================

    verified_results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # Reassign rank after combined sorting.
    for rank, result in enumerate(
        verified_results,
        start=1,
    ):

        result["rank"] = rank

    best_result = (
        verified_results[0]
        if verified_results
        else None
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    matched = False

    if best_result:

        final_score = float(
            best_result["score"]
        )

        best_dino = float(
            best_result[
                "dino_similarity"
            ]
        )

        best_design = float(
            best_result[
                "design_score"
            ]
        )

        print()
        print(
            "=" * 70
        )

        print(
            "FINAL MATCH ANALYSIS"
        )

        print(
            "=" * 70
        )

        print(
            f"[MATCHER] Candidate: "
            f"{best_result['design_id']}"
        )

        print(
            f"[MATCHER] Collection: "
            f"{best_result['collection']}"
        )

        print(
            f"[MATCHER] Name: "
            f"{best_result['name'] or 'Not available'}"
        )

        print(
            f"[MATCHER] DINO similarity: "
            f"{best_dino:.4f}"
        )

        print(
            f"[MATCHER] Shape similarity: "
            f"{best_result['shape_similarity']:.4f}"
        )

        print(
            f"[MATCHER] Structure similarity: "
            f"{best_result['structure_similarity']:.4f}"
        )

        print(
            f"[MATCHER] Pattern similarity: "
            f"{best_result['pattern_similarity']:.4f}"
        )

        print(
            f"[MATCHER] Design score: "
            f"{best_design:.4f}"
        )

        print(
            f"[MATCHER] Final score: "
            f"{final_score:.4f}"
        )

        print(
            f"[MATCHER] DINO threshold: "
            f"{DINO_DIRECT_THRESHOLD:.4f}"
        )

        print(
            f"[MATCHER] Final threshold: "
            f"{FINAL_MATCH_THRESHOLD:.4f}"
        )

        if (
            best_dino
            >= DINO_DIRECT_THRESHOLD
        ):

            matched = True

            print(
                "[MATCHER] STATUS: MATCH"
            )

            print(
                "[MATCHER] REASON: "
                "DINO similarity is "
                "above threshold"
            )

        elif (
            final_score
            >= FINAL_MATCH_THRESHOLD
            and gap
            >= MIN_GAP_FOR_WEAK_MATCH
        ):

            matched = True

            print(
                "[MATCHER] STATUS: MATCH"
            )

            print(
                "[MATCHER] REASON: "
                "Strong combined "
                "design evidence"
            )

        elif (
            best_dino
            >= STRONG_DESIGN_DINO_MIN
            and best_design
            >= STRONG_DESIGN_THRESHOLD
            and gap
            >= MIN_GAP_FOR_WEAK_MATCH
        ):

            matched = True

            print(
                "[MATCHER] STATUS: MATCH"
            )

            print(
                "[MATCHER] REASON: "
                "Strong design evidence"
            )

        else:

            print(
                "[MATCHER] STATUS: "
                "NO MATCH"
            )

            print(
                "[MATCHER] REASON: "
                "Insufficient design evidence"
            )

            print(
                "[MATCHER] Difference "
                "from final threshold: "
                f"{final_score - FINAL_MATCH_THRESHOLD:+.4f}"
            )

    else:

        print(
            "[MATCHER] STATUS: "
            "NO MATCH"
        )

    print(
        "=" * 70
    )

    # ========================================================
    # NO MATCH
    # ========================================================

    if not matched:

        return {
            "matched": False,

            "results": [],

            "best_similarity": (
                best_similarity
            ),

            "source_collection": (
                source_collection
            ),

            "target_collection": (
                target_collection
            ),

            "search_mode": (
                search_mode
            ),

            "second_similarity": (
                second_similarity
            ),

            "result_gap": gap,

            "message": (
                "No reliable "
                "jewellery design "
                "match was found."
            ),
        }

    # ========================================================
    # MATCH
    # ========================================================

    return {
        "matched": True,

        "results": (
            verified_results[:top_k]
        ),

        "best_similarity": (
            best_similarity
        ),

        "source_collection": (
            source_collection
        ),

        "target_collection": (
            target_collection
        ),

        "search_mode": (
            search_mode
        ),

        "second_similarity": (
            second_similarity
        ),

        "result_gap": gap,
    }


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python -m "
            "backend.services.matcher "
            "<image_path>"
        )

        raise SystemExit(1)

    query = sys.argv[1]

    # Optional CLI search mode.
    #
    # Example:
    #
    # python -m backend.services.matcher image.jpg all
    #
    # python -m backend.services.matcher image.jpg gold_to_prototype
    #
    # python -m backend.services.matcher image.jpg prototype_to_gold

    cli_search_mode = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else "all"
    )

    result = match_jewellery(
        query,
        top_k=8,
        search_mode=cli_search_mode,
    )

    print()

    print(
        "FINAL RESULT"
    )

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )