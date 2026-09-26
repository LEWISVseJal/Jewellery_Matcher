"""
JewelMatch AI - Lightweight cross-collection matcher.
"""

from pathlib import Path
import json

import cv2
import numpy as np

from .embedding import create_embedding

BASE_DIR = Path(__file__).resolve().parents[1]

DATABASE_DIR = BASE_DIR / "database"

CATALOGUE_FILE = DATABASE_DIR / "jewellery.json"

INDEX_FILE = DATABASE_DIR / "lightweight_index.npz"

TOP_K = 8

MIN_MATCH_SCORE = 0.30

_INDEX_CACHE = None


def clear_index_cache():
    global _INDEX_CACHE

    _INDEX_CACHE = None


def normalise_collection(value):

    value = str(value or "").strip().lower()

    if value in {
        "gold",
        "g",
    }:
        return "gold"

    if value in {
        "prototype",
        "p",
        "proto",
    }:
        return "prototype"

    return value


def load_index():

    global _INDEX_CACHE

    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    if not INDEX_FILE.exists():

        raise FileNotFoundError(
            "Lightweight index not found. "
            "Run: python -m "
            "backend.scripts.create_lightweight_index"
        )

    data = np.load(
        INDEX_FILE,
        allow_pickle=True,
    )

    _INDEX_CACHE = {
        "features": np.asarray(
            data["features"],
            dtype=np.float32,
        ),
        "ids": np.asarray(
            data["ids"],
            dtype=str,
        ),
        "collections": np.asarray(
            data["collections"],
            dtype=str,
        ),
    }

    return _INDEX_CACHE


def load_catalogue():

    if not CATALOGUE_FILE.exists():
        return []

    with open(
        CATALOGUE_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in (
            "jewellery",
            "items",
            "catalogue",
            "data",
        ):

            if isinstance(
                data.get(key),
                list,
            ):
                return data[key]

    return []


def get_item_id(item):

    return str(
        item.get("id") or item.get("jewellery_id") or item.get("design_id") or ""
    )


def get_item_filename(item):

    value = (
        item.get("filename")
        or item.get("image")
        or item.get("image_filename")
        or item.get("image_path")
        or ""
    )

    if not value:
        return ""

    return Path(str(value)).name


def resolve_item_image(
    item,
    collection,
):

    filename = get_item_filename(item)

    if not filename:
        return None

    path = BASE_DIR / "catalogue" / collection / filename

    if path.exists():
        return path

    return None


def cosine_similarity(a, b):

    a = np.asarray(
        a,
        dtype=np.float32,
    )

    b = np.asarray(
        b,
        dtype=np.float32,
    )

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator < 1e-8:
        return 0.0

    return float(np.dot(a, b) / denominator)


def edge_similarity(
    query_path,
    target_path,
):

    try:

        query = cv2.imread(
            str(query_path),
            cv2.IMREAD_GRAYSCALE,
        )

        target = cv2.imread(
            str(target_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if query is None or target is None:
            return 0.0

        query = cv2.resize(
            query,
            (64, 64),
            interpolation=cv2.INTER_AREA,
        )

        target = cv2.resize(
            target,
            (64, 64),
            interpolation=cv2.INTER_AREA,
        )

        query_edges = (
            cv2.Canny(
                query,
                40,
                120,
            ).astype(np.float32)
            / 255.0
        )

        target_edges = (
            cv2.Canny(
                target,
                40,
                120,
            ).astype(np.float32)
            / 255.0
        )

        return cosine_similarity(
            query_edges.flatten(),
            target_edges.flatten(),
        )

    except Exception:
        return 0.0


def identify_source_collection(query_embedding):

    index = load_index()

    best_scores = {}

    for collection in (
        "gold",
        "prototype",
    ):

        mask = np.char.lower(index["collections"].astype(str)) == collection

        features = index["features"][mask]

        if len(features) == 0:

            best_scores[collection] = 0.0

            continue

        scores = features @ query_embedding

        best_scores[collection] = float(np.max(scores))

    source = max(
        best_scores,
        key=best_scores.get,
    )

    return (
        source,
        best_scores,
    )


def match_jewellery(
    query_image_path,
    top_k=TOP_K,
    target_collection="auto",
):

    index = load_index()

    catalogue = load_catalogue()

    catalogue_by_id = {get_item_id(item): item for item in catalogue}

    query_embedding = create_embedding(query_image_path)

    source_collection, source_scores = identify_source_collection(query_embedding)

    if target_collection == "auto":

        if source_collection == "gold":
            target = "prototype"
        else:
            target = "gold"

    else:

        target = normalise_collection(target_collection)

    collection_values = np.char.lower(index["collections"].astype(str))

    positions = np.where(collection_values == target)[0]

    if len(positions) == 0:

        return {
            "source_collection": source_collection,
            "target_collection": target,
            "source_scores": source_scores,
            "results": [],
            "best_score": 0.0,
            "has_match": False,
            "threshold": MIN_MATCH_SCORE,
        }

    candidates = []

    for position in positions:

        feature = index["features"][position]

        cosine = cosine_similarity(
            query_embedding,
            feature,
        )

        item_id = str(index["ids"][position])

        item = catalogue_by_id.get(
            item_id,
            {},
        )

        image_path = resolve_item_image(
            item,
            target,
        )

        edge_score = 0.0

        if image_path and image_path.exists():

            edge_score = edge_similarity(
                query_image_path,
                image_path,
            )

        final_score = 0.82 * cosine + 0.18 * edge_score

        candidates.append(
            {
                "item": item,
                "id": item_id,
                "similarity": float(
                    np.clip(
                        final_score,
                        0.0,
                        1.0,
                    )
                ),
                "cosine_similarity": float(cosine),
                "edge_similarity": float(edge_score),
            }
        )

    candidates.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    candidates = candidates[: max(1, int(top_k))]

    for result in candidates:

        item = result["item"]

        collection = normalise_collection(item.get("collection") or target)

        filename = get_item_filename(item)

        result["image_url"] = (
            f"/catalogue/" f"{collection}/" f"{filename}" if filename else None
        )

        result["name"] = item.get("name") or item.get("jewellery_name") or result["id"]

        result["collection"] = collection

        result["type"] = item.get("type") or item.get("jewellery_type") or ""

        result["gender"] = item.get("gender") or ""

        result["description"] = item.get("description") or ""

    best_score = candidates[0]["similarity"] if candidates else 0.0

    has_match = best_score >= MIN_MATCH_SCORE

    return {
        "source_collection": source_collection,
        "target_collection": target,
        "source_scores": source_scores,
        "results": candidates if has_match else [],
        "best_score": float(best_score),
        "has_match": has_match,
        "threshold": MIN_MATCH_SCORE,
    }
