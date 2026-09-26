import os
import json
import numpy as np

from backend.services.embedding import create_embedding

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def load_catalogue():

    if not os.path.exists(CATALOGUE_FILE):
        return []

    with open(
        CATALOGUE_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if isinstance(data, dict):

        if "jewellery" in data:
            return data["jewellery"]

        if "items" in data:
            return data["items"]

    return data


def get_collection(item):

    return str(
        item.get(
            "collection",
            item.get("category", ""),
        )
    ).lower()


def get_image_path(item):

    image_path = item.get("image_path")

    if not image_path:
        image_path = item.get("image")

    if not image_path:
        image_path = item.get("path")

    if not image_path:
        return None

    image_path = image_path.replace(
        "\\",
        "/",
    )

    project_dir = os.path.dirname(BASE_DIR)

    candidates = [
        os.path.join(
            project_dir,
            image_path,
        ),
        os.path.join(
            BASE_DIR,
            image_path,
        ),
    ]

    for candidate in candidates:

        if os.path.exists(candidate):
            return candidate

    filename = os.path.basename(image_path)

    for root, _, files in os.walk(project_dir):

        if filename in files:

            return os.path.join(
                root,
                filename,
            )

    return None


def main():

    print("=" * 70)
    print("JEWELMATCH AI - BUILD LIGHTWEIGHT FEATURE INDEX")
    print("=" * 70)

    catalogue = load_catalogue()

    print(f"Catalogue items: {len(catalogue)}")

    features = []
    ids = []
    collections = []

    failed = []

    for index, item in enumerate(
        catalogue,
        start=1,
    ):

        item_id = str(
            item.get(
                "id",
                item.get(
                    "design_id",
                    index,
                ),
            )
        )

        collection = get_collection(item)

        image_path = get_image_path(item)

        print(f"[{index}/{len(catalogue)}] " f"{item_id} | {collection}")

        if not image_path:

            print("  ERROR: Image not found")

            failed.append(item_id)

            continue

        try:

            embedding = create_embedding(image_path)

            features.append(embedding)

            ids.append(item_id)

            collections.append(collection)

            print(f"  OK: {embedding.shape}")

        except Exception as exc:

            print(f"  ERROR: {exc}")

            failed.append(item_id)

    if not features:

        raise RuntimeError("No catalogue features were created.")

    feature_matrix = np.vstack(features).astype(np.float32)

    np.savez_compressed(
        INDEX_FILE,
        features=feature_matrix,
        ids=np.array(
            ids,
            dtype=str,
        ),
        collections=np.array(
            collections,
            dtype=str,
        ),
    )

    print()
    print("=" * 70)
    print("INDEX CREATED")
    print("=" * 70)

    print(f"Items indexed: {len(ids)}")

    print(f"Feature shape: {feature_matrix.shape}")

    print(f"Failed items: {len(failed)}")

    print(f"Saved to: {INDEX_FILE}")

    if failed:

        print()
        print("Failed IDs:")

        for item_id in failed:
            print(f" - {item_id}")


if __name__ == "__main__":
    main()
