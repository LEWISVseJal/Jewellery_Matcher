"""
JewelMatch AI - Lightweight index builder.

Run from the PROJECT ROOT:

    python -m backend.scripts.create_lightweight_index
"""

from pathlib import Path
import json
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from backend.services.embedding import (
    create_embedding,
)

BACKEND_DIR = PROJECT_ROOT / "backend"

DATABASE_DIR = BACKEND_DIR / "database"

CATALOGUE_FILE = DATABASE_DIR / "jewellery.json"

INDEX_FILE = DATABASE_DIR / "lightweight_index.npz"


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


def get_id(item):

    return str(
        item.get("id") or item.get("jewellery_id") or item.get("design_id") or ""
    ).strip()


def get_collection(item):

    value = str(item.get("collection", "")).strip().lower()

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


def get_filename(item):

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


def get_image_path(item):

    collection = get_collection(item)

    filename = get_filename(item)

    if not collection or not filename:
        return None

    image_path = BACKEND_DIR / "catalogue" / collection / filename

    if image_path.exists():
        return image_path

    return None


def main():

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalogue = load_catalogue()

    print()
    print("=" * 70)
    print("JEWELMATCH AI - " "LIGHTWEIGHT INDEX BUILDER")
    print("=" * 70)

    print(f"Catalogue records: " f"{len(catalogue)}")

    print()

    features = []
    ids = []
    collections = []

    failed = []

    for index, item in enumerate(
        catalogue,
        start=1,
    ):

        item_id = get_id(item)

        collection = get_collection(item)

        image_path = get_image_path(item)

        print(
            f"[{index}/{len(catalogue)}] "
            f"{item_id or 'NO-ID'} | "
            f"{collection} | "
            f"{image_path.name if image_path else 'IMAGE NOT FOUND'}"
        )

        if not item_id:

            failed.append(
                (
                    item_id,
                    "Missing ID",
                )
            )

            continue

        if collection not in {
            "gold",
            "prototype",
        }:

            failed.append(
                (
                    item_id,
                    "Invalid collection",
                )
            )

            continue

        if image_path is None:

            failed.append(
                (
                    item_id,
                    "Image not found",
                )
            )

            continue

        try:

            embedding = create_embedding(image_path)

            if embedding.shape != (256,):

                raise ValueError("Invalid feature size: " f"{embedding.shape}")

            features.append(embedding)

            ids.append(item_id)

            collections.append(collection)

        except Exception as exc:

            failed.append(
                (
                    item_id,
                    str(exc),
                )
            )

            print(f"  ERROR: {exc}")

    if features:

        feature_array = np.vstack(features).astype(np.float32)

    else:

        feature_array = np.empty(
            (
                0,
                256,
            ),
            dtype=np.float32,
        )

    np.savez_compressed(
        INDEX_FILE,
        features=feature_array,
        ids=np.asarray(
            ids,
            dtype=str,
        ),
        collections=np.asarray(
            collections,
            dtype=str,
        ),
    )

    print()

    print("=" * 70)

    print(f"Indexed records : " f"{len(ids)}")

    print(f"Failed records  : " f"{len(failed)}")

    print(f"Feature shape   : " f"{feature_array.shape}")

    print(f"Saved to        : " f"{INDEX_FILE}")

    print("=" * 70)

    if failed:

        print()
        print("FAILED RECORDS")

        for item_id, reason in failed:

            print(f"- " f"{item_id or 'UNKNOWN'}: " f"{reason}")


if __name__ == "__main__":
    main()
