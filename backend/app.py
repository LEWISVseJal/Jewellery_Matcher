# ============================================================
# JEWELMATCH AI
# FLASK BACKEND
# ============================================================

import os
import json
import uuid
import shutil

import numpy as np

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
)

# ============================================================
# CONFIGURATION
# ============================================================

from backend.config import (
    UPLOAD_FOLDER,
    JEWELLERY_JSON,
    GOLD_CATALOGUE_DIR,
    PROTOTYPE_CATALOGUE_DIR,
    GOLD_EMBEDDINGS_FILE,
    PROTOTYPE_EMBEDDINGS_FILE,
    GOLD_SEGMENTED_DIR,
    PROTOTYPE_SEGMENTED_DIR,
)

# ============================================================
# FLASK APPLICATION
# IMPORTANT: THIS MUST BE AT MODULE LEVEL
# ============================================================

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


# ============================================================
# CONSTANTS
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp",
}

VALID_COLLECTIONS = {
    "gold",
    "prototype",
}

# DINOv2-base embedding size.
# If later changed to DINOv2-small, change to 384.
EMBEDDING_DIMENSION = 384


# ============================================================
# DIRECTORY SETUP
# ============================================================

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

os.makedirs(GOLD_CATALOGUE_DIR, exist_ok=True)

os.makedirs(PROTOTYPE_CATALOGUE_DIR, exist_ok=True)

os.makedirs(GOLD_SEGMENTED_DIR, exist_ok=True)

os.makedirs(PROTOTYPE_SEGMENTED_DIR, exist_ok=True)

os.makedirs(os.path.dirname(JEWELLERY_JSON), exist_ok=True)


# ============================================================
# BASIC HELPERS
# ============================================================


def allowed_file(filename):

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[-1].lower()

    return extension in ALLOWED_EXTENSIONS


def load_catalogue():

    if not os.path.exists(JEWELLERY_JSON):
        return []

    try:

        with open(JEWELLERY_JSON, "r", encoding="utf-8") as file:

            data = json.load(file)

        if not isinstance(data, list):

            print("[APP] jewellery.json does not contain a list.")

            return []

        return data

    except Exception as exc:

        print("[APP] Failed to load catalogue:")

        print(exc)

        return []


def save_catalogue(data):

    os.makedirs(os.path.dirname(JEWELLERY_JSON), exist_ok=True)

    temporary_file = JEWELLERY_JSON + ".tmp"

    with open(temporary_file, "w", encoding="utf-8") as file:

        json.dump(data, file, indent=4, ensure_ascii=False)

    os.replace(temporary_file, JEWELLERY_JSON)


def get_next_jewellery_id(catalogue):

    numbers = []

    for item in catalogue:

        item_id = str(item.get("id", "")).strip()

        if item_id.upper().startswith("J"):

            try:

                number = int(item_id[1:])

                numbers.append(number)

            except ValueError:

                continue

    if numbers:

        next_number = max(numbers) + 1

    else:

        next_number = 1

    return f"J{next_number:03d}"


def find_catalogue_item(catalogue, jewellery_id):

    jewellery_id = str(jewellery_id).strip()

    for index, item in enumerate(catalogue):

        if str(item.get("id", "")).strip() == jewellery_id:

            return index, item

    return -1, None


def remove_file(path):

    if not path:
        return

    try:

        if os.path.exists(path):

            os.remove(path)

    except Exception as exc:

        print(f"[APP] Could not remove file: {path}")

        print(exc)


# ============================================================
# COLLECTION HELPERS
# ============================================================


def get_collection_directory(collection):

    collection = str(collection).lower().strip()

    if collection == "gold":

        return GOLD_CATALOGUE_DIR

    if collection == "prototype":

        return PROTOTYPE_CATALOGUE_DIR

    return None


def get_segmented_directory(collection):

    collection = str(collection).lower().strip()

    if collection == "gold":

        return GOLD_SEGMENTED_DIR

    if collection == "prototype":

        return PROTOTYPE_SEGMENTED_DIR

    return None


def get_embedding_file(collection):

    collection = str(collection).lower().strip()

    if collection == "gold":

        return GOLD_EMBEDDINGS_FILE

    if collection == "prototype":

        return PROTOTYPE_EMBEDDINGS_FILE

    return None


def get_original_image_path(collection, filename):

    directory = get_collection_directory(collection)

    if not directory:

        return ""

    return os.path.join(directory, filename)


def get_segmented_image_path(collection, filename):

    directory = get_segmented_directory(collection)

    if not directory:

        return ""

    base_name = os.path.splitext(filename)[0]

    return os.path.join(directory, base_name + ".jpg")


# ============================================================
# SEGMENTATION
# ============================================================


def create_segmented_image(image_path, collection, filename):

    # Lazy import.
    # U2-Net is NOT loaded when Flask starts.

    from backend.services.segmentation import segment_jewellery

    segmented_directory = get_segmented_directory(collection)

    if not segmented_directory:

        raise ValueError("Invalid collection.")

    os.makedirs(segmented_directory, exist_ok=True)

    segmented_path = get_segmented_image_path(collection, filename)

    print()
    print("[APP] Creating segmented image:")

    print(image_path)
    print("->", segmented_path)

    segment_jewellery(image_path, segmented_path)

    if not os.path.exists(segmented_path):

        raise RuntimeError("Segmentation did not create the expected image.")

    return segmented_path


# ============================================================
# EMBEDDING
# ============================================================


def create_catalogue_embedding(image_path):

    # Lazy import.
    # PyTorch + DINOv2 are NOT loaded when Flask starts.

    from backend.services.embedding import get_embedding

    print()
    print("[APP] Creating embedding:")

    print(image_path)

    embedding = get_embedding(image_path)

    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.ndim > 1:

        embedding = embedding.reshape(-1)

    return embedding


# ============================================================
# INDEX HELPERS
# ============================================================


def save_collection_embeddings(collection, embeddings):

    embedding_file = get_embedding_file(collection)

    if not embedding_file:

        raise ValueError("Invalid collection.")

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if len(embeddings) == 0:

        embeddings = np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    np.save(embedding_file, embeddings)

    print("[APP] Saved embeddings:", embedding_file)

    print("[APP] Embedding count:", len(embeddings))


def rebuild_collection_index(collection):

    collection = str(collection).lower().strip()

    if collection not in VALID_COLLECTIONS:

        raise ValueError("Invalid collection.")

    print()
    print("=" * 70)
    print(f"[INDEX] Rebuilding {collection.upper()} index")
    print("=" * 70)

    catalogue = load_catalogue()

    collection_items = [
        item
        for item in catalogue
        if str(item.get("collection", "")).lower().strip() == collection
    ]

    embeddings = []

    failed_items = []

    for item in collection_items:

        jewellery_id = str(item.get("id", "")).strip()

        filename = str(item.get("image", "")).strip()

        if not filename:

            print("[INDEX] Missing image:")

            print(jewellery_id)

            failed_items.append(jewellery_id)

            continue

        original_path = get_original_image_path(collection, filename)

        if not os.path.exists(original_path):

            print("[INDEX] Image missing:")

            print(original_path)

            failed_items.append(jewellery_id)

            continue

        try:

            segmented_path = create_segmented_image(original_path, collection, filename)

            embedding = create_catalogue_embedding(segmented_path)

            embeddings.append(embedding)

        except Exception as exc:

            print("[INDEX] Failed:")

            print(jewellery_id, filename)

            print(exc)

            failed_items.append(jewellery_id)

    if embeddings:

        embedding_matrix = np.vstack(embeddings).astype(np.float32)

    else:

        embedding_matrix = np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    save_collection_embeddings(collection, embedding_matrix)

    print()
    print(f"[INDEX] {collection.upper()} rebuild complete.")

    print("[INDEX] Catalogue items:", len(collection_items))

    print("[INDEX] Embeddings:", len(embedding_matrix))

    if failed_items:

        print("[INDEX] Failed items:", failed_items)

    print("=" * 70)

    return {
        "collection": collection,
        "catalogue_count": len(collection_items),
        "embedding_count": len(embedding_matrix),
        "failed_items": failed_items,
    }


def load_embeddings_count(collection):

    embedding_file = get_embedding_file(collection)

    if not embedding_file:

        return []

    if not os.path.exists(embedding_file):

        return []

    try:

        embeddings = np.load(embedding_file)

        if embeddings.ndim == 1:

            return [embeddings]

        return embeddings

    except Exception:

        return []


# ============================================================
# PAGES
# ============================================================


@app.route("/")
def index():

    return render_template("index.html")


@app.route("/catalogue")
def catalogue_page():

    return render_template("catalogue.html")


@app.route("/add-jewellery")
def add_jewellery_page():

    return render_template("add_jewellery.html")


# ============================================================
# HEALTH CHECK
# ============================================================


@app.route("/api/health")
def health():

    return jsonify({"status": "ok"})


# ============================================================
# GET CATALOGUE
# ============================================================


@app.route("/api/jewellery", methods=["GET"])
def get_jewellery():

    try:

        catalogue = load_catalogue()

        items = []

        for item in catalogue:

            result = dict(item)

            collection = str(result.get("collection", "")).lower().strip()

            filename = str(result.get("image", "")).strip()

            if collection in VALID_COLLECTIONS and filename:

                result["image_url"] = f"/catalogue/" f"{collection}/" f"{filename}"

            else:

                result["image_url"] = None

            items.append(result)

        return jsonify({"success": True, "count": len(items), "items": items})

    except Exception as exc:

        print("[APP] Failed to get catalogue:")

        print(exc)

        return (
            jsonify(
                {
                    "success": False,
                    "error": ("Unable to load " "jewellery catalogue."),
                    "details": str(exc),
                }
            ),
            500,
        )


# ============================================================
# SERVE CATALOGUE IMAGES
# ============================================================


@app.route("/catalogue/<collection>/<path:filename>")
def serve_catalogue_image(collection, filename):

    collection = str(collection).lower().strip()

    directory = get_collection_directory(collection)

    if not directory:

        return jsonify({"error": "Invalid collection."}), 400

    return send_from_directory(directory, filename)


# ============================================================
# SERVE UPLOADS
# ============================================================


@app.route("/uploads/<path:filename>")
def serve_uploaded_image(filename):

    return send_from_directory(UPLOAD_FOLDER, filename)


# ============================================================
# MATCH
# ============================================================


@app.route("/api/match", methods=["POST"])
def match():

    if "image" not in request.files:

        return jsonify({"error": "No image was uploaded."}), 400

    image = request.files["image"]

    if not image or not image.filename:

        return jsonify({"error": "Please select an image."}), 400

    if not allowed_file(image.filename):

        return jsonify({"error": "Unsupported image format."}), 400

    extension = image.filename.rsplit(".", 1)[-1].lower()

    query_filename = "query_" + uuid.uuid4().hex + "." + extension

    query_path = os.path.join(UPLOAD_FOLDER, query_filename)

    image.save(query_path)

    print()
    print("=" * 70)
    print("JEWELMATCH AI - CROSS-COLLECTION SEARCH")
    print("=" * 70)

    print("Query image:", query_filename)

    try:

        # Lazy import.
        from backend.services.matcher import match_jewellery

        match_data = match_jewellery(query_path, target_collection="auto")

    except Exception as exc:

        print("[APP] Matching error:")

        print(exc)

        return (
            jsonify(
                {
                    "error": ("Unable to process " "the jewellery image."),
                    "details": str(exc),
                }
            ),
            500,
        )

    finally:

        # Remove temporary query image.
        remove_file(query_path)

    results = []

    for item in match_data.get("results", []):

        result = dict(item)

        collection = str(result.get("collection", "")).lower().strip()

        filename = str(result.get("image", "")).strip()

        if collection in VALID_COLLECTIONS and filename:

            result["image_url"] = f"/catalogue/" f"{collection}/" f"{filename}"

        else:

            result["image_url"] = None

        results.append(result)

    response = {
        "success": True,
        "match_found": match_data.get("match_found", bool(results)),
        "best_similarity": match_data.get("best_similarity", 0.0),
        "best_similarity_percentage": (
            match_data.get("best_similarity_percentage", 0.0)
        ),
        "source_collection": (match_data.get("source_collection")),
        "target_collection": (match_data.get("target_collection")),
        "source_similarity": (match_data.get("source_similarity", 0.0)),
        "gold_similarity": (match_data.get("gold_similarity", 0.0)),
        "prototype_similarity": (match_data.get("prototype_similarity", 0.0)),
        "results": results,
    }

    print()
    print("Source collection:", response["source_collection"])

    print("Target collection:", response["target_collection"])

    print("Results:", len(results))

    print("Best target similarity:", f"{response['best_similarity']:.4f}")

    print("=" * 70)

    return jsonify(response)


# ============================================================
# ADD JEWELLERY
# ============================================================


@app.route("/api/jewellery/add", methods=["POST"])
def add_jewellery():

    if "image" not in request.files:

        return jsonify({"error": ("Jewellery image " "is required.")}), 400

    image = request.files["image"]

    if not image or not image.filename:

        return jsonify({"error": ("Jewellery image " "is required.")}), 400

    if not allowed_file(image.filename):

        return jsonify({"error": "Unsupported image format."}), 400

    name = request.form.get("name", "").strip()

    gender = request.form.get("gender", "").strip()

    jewellery_type = request.form.get("type", "").strip()

    subtype = request.form.get("subtype", "").strip()

    description = request.form.get("description", "").strip()

    collection = request.form.get("collection", "").strip().lower()

    if not name:

        return jsonify({"error": "Jewellery name is required."}), 400

    if collection not in VALID_COLLECTIONS:

        return jsonify({"error": ("Please select " "Gold or Prototype.")}), 400

    if not gender:

        return jsonify({"error": "Gender is required."}), 400

    if not jewellery_type:

        return jsonify({"error": "Jewellery type is required."}), 400

    catalogue = load_catalogue()

    jewellery_id = get_next_jewellery_id(catalogue)

    extension = image.filename.rsplit(".", 1)[-1].lower()

    filename = f"{jewellery_id}." f"{extension}"

    collection_directory = get_collection_directory(collection)

    image_path = os.path.join(collection_directory, filename)

    segmented_path = None

    try:

        # Save original image.
        image.save(image_path)

        # Create segmented image.
        segmented_path = create_segmented_image(image_path, collection, filename)

        # Create embedding.
        create_catalogue_embedding(segmented_path)

        # Create catalogue item.
        item = {
            "id": jewellery_id,
            "name": name,
            "gender": gender,
            "type": jewellery_type,
            "category": jewellery_type,
            "subtype": subtype,
            "description": description,
            "collection": collection,
            "image": filename,
        }

        catalogue.append(item)

        # Save metadata first.
        save_catalogue(catalogue)

        # Rebuild complete index.
        rebuild_collection_index(collection)

        item["image_url"] = f"/catalogue/" f"{collection}/" f"{filename}"

        print()
        print("[APP] Jewellery added:")

        print("ID:", jewellery_id)

        print("Collection:", collection)

        return (
            jsonify(
                {
                    "success": True,
                    "message": ("Jewellery added " "successfully."),
                    "item": item,
                }
            ),
            201,
        )

    except Exception as exc:

        print("[APP] Add jewellery failed:")

        print(exc)

        remove_file(image_path)

        remove_file(segmented_path)

        return (
            jsonify({"error": ("Unable to add " "jewellery."), "details": str(exc)}),
            500,
        )


# ============================================================
# EDIT JEWELLERY
# ============================================================


@app.route("/api/jewellery/<jewellery_id>", methods=["PUT", "POST"])
def edit_jewellery(jewellery_id):

    catalogue = load_catalogue()

    index, item = find_catalogue_item(catalogue, jewellery_id)

    if item is None:

        return jsonify({"error": ("Jewellery item " "not found.")}), 404

    name = request.form.get("name", item.get("name", "")).strip()

    gender = request.form.get("gender", item.get("gender", "")).strip()

    jewellery_type = request.form.get("type", item.get("type", "")).strip()

    subtype = request.form.get("subtype", item.get("subtype", "")).strip()

    description = request.form.get("description", item.get("description", "")).strip()

    new_collection = (
        request.form.get("collection", item.get("collection", "")).strip().lower()
    )

    if not name:

        return jsonify({"error": "Jewellery name is required."}), 400

    if new_collection not in VALID_COLLECTIONS:

        return jsonify({"error": "Invalid collection."}), 400

    if not gender:

        return jsonify({"error": "Gender is required."}), 400

    if not jewellery_type:

        return jsonify({"error": "Jewellery type is required."}), 400

    old_collection = str(item.get("collection", "")).lower().strip()

    old_filename = str(item.get("image", "")).strip()

    new_image = request.files.get("image")

    has_new_image = new_image is not None and bool(new_image.filename)

    if has_new_image:

        if not allowed_file(new_image.filename):

            return jsonify({"error": ("Unsupported image " "format.")}), 400

    collection_changed = old_collection != new_collection

    filename = old_filename

    old_image_path = get_original_image_path(old_collection, old_filename)

    old_segmented_path = get_segmented_image_path(old_collection, old_filename)

    new_image_path = None
    new_segmented_path = None

    try:

        # ----------------------------------------------------
        # NEW IMAGE
        # ----------------------------------------------------

        if has_new_image:

            extension = new_image.filename.rsplit(".", 1)[-1].lower()

            filename = f"{jewellery_id}." f"{extension}"

            new_directory = get_collection_directory(new_collection)

            new_image_path = os.path.join(new_directory, filename)

            new_image.save(new_image_path)

            new_segmented_path = create_segmented_image(
                new_image_path, new_collection, filename
            )

            create_catalogue_embedding(new_segmented_path)

        # ----------------------------------------------------
        # COLLECTION CHANGE
        # ----------------------------------------------------

        elif collection_changed:

            new_directory = get_collection_directory(new_collection)

            new_image_path = os.path.join(new_directory, filename)

            shutil.copy2(old_image_path, new_image_path)

            new_segmented_path = create_segmented_image(
                new_image_path, new_collection, filename
            )

            create_catalogue_embedding(new_segmented_path)

        # ----------------------------------------------------
        # UPDATE METADATA
        # ----------------------------------------------------

        item["name"] = name

        item["gender"] = gender

        item["type"] = jewellery_type

        item["category"] = jewellery_type

        item["subtype"] = subtype

        item["description"] = description

        item["collection"] = new_collection

        item["image"] = filename

        catalogue[index] = item

        # IMPORTANT:
        # Save JSON before rebuilding indexes.

        save_catalogue(catalogue)

        # ----------------------------------------------------
        # REBUILD INDEX
        # ----------------------------------------------------

        if has_new_image or collection_changed:

            rebuild_collection_index(new_collection)

            if old_collection in VALID_COLLECTIONS and old_collection != new_collection:

                rebuild_collection_index(old_collection)

        # ----------------------------------------------------
        # REMOVE OLD FILES
        # ----------------------------------------------------

        if has_new_image or collection_changed:

            if old_image_path and old_image_path != new_image_path:

                remove_file(old_image_path)

            if old_segmented_path and old_segmented_path != new_segmented_path:

                remove_file(old_segmented_path)

        item["image_url"] = f"/catalogue/" f"{new_collection}/" f"{filename}"

        return jsonify(
            {
                "success": True,
                "message": ("Jewellery updated " "successfully."),
                "item": item,
            }
        )

    except Exception as exc:

        print("[APP] Edit failed:")

        print(exc)

        remove_file(new_image_path)

        remove_file(new_segmented_path)

        return (
            jsonify({"error": ("Unable to update " "jewellery."), "details": str(exc)}),
            500,
        )


# ============================================================
# DELETE JEWELLERY
# ============================================================


@app.route("/api/jewellery/<jewellery_id>", methods=["DELETE"])
def delete_jewellery(jewellery_id):

    catalogue = load_catalogue()

    index, item = find_catalogue_item(catalogue, jewellery_id)

    if item is None:

        return jsonify({"error": ("Jewellery item " "not found.")}), 404

    collection = str(item.get("collection", "")).lower().strip()

    filename = str(item.get("image", "")).strip()

    image_path = get_original_image_path(collection, filename)

    segmented_path = get_segmented_image_path(collection, filename)

    # Delete files.

    remove_file(image_path)

    remove_file(segmented_path)

    # Remove JSON item.

    catalogue.pop(index)

    try:

        save_catalogue(catalogue)

    except Exception as exc:

        print("[APP] JSON update failed:")

        print(exc)

        return (
            jsonify(
                {
                    "error": ("Image was removed " "but JSON could not " "be updated."),
                    "details": str(exc),
                }
            ),
            500,
        )

    # Rebuild collection index.

    try:

        rebuild_collection_index(collection)

    except Exception as exc:

        print("[APP] Index rebuild " "after delete failed:")

        print(exc)

    return jsonify(
        {
            "success": True,
            "message": ("Jewellery deleted " "successfully."),
            "deleted_id": jewellery_id,
        }
    )


# ============================================================
# REBUILD INDEX
# ============================================================


@app.route("/api/jewellery/rebuild-index", methods=["POST"])
def rebuild_search_index():

    try:

        gold_result = rebuild_collection_index("gold")

        prototype_result = rebuild_collection_index("prototype")

        return jsonify(
            {
                "success": True,
                "message": ("Search index " "rebuilt successfully."),
                "gold_count": (gold_result["embedding_count"]),
                "prototype_count": (prototype_result["embedding_count"]),
            }
        )

    except Exception as exc:

        print("[APP] Rebuild index failed:")

        print(exc)

        return (
            jsonify(
                {
                    "success": False,
                    "error": ("Unable to rebuild " "search index."),
                    "details": str(exc),
                }
            ),
            500,
        )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================


@app.route("/api/rebuild-index", methods=["POST"])
def rebuild_all_indexes():

    try:

        rebuild_collection_index("gold")

        rebuild_collection_index("prototype")

        return jsonify(
            {
                "success": True,
                "message": ("All search indexes " "rebuilt successfully."),
            }
        )

    except Exception as exc:

        print("[APP] Rebuild all indexes failed:")

        print(exc)

        return (
            jsonify(
                {
                    "success": False,
                    "error": ("Unable to rebuild " "search indexes."),
                    "details": str(exc),
                }
            ),
            500,
        )


# ============================================================
# STATISTICS
# ============================================================


@app.route("/api/jewellery/stats", methods=["GET"])
def jewellery_statistics():

    catalogue = load_catalogue()

    gold_count = 0
    prototype_count = 0

    types = {}

    for item in catalogue:

        collection = str(item.get("collection", "")).lower().strip()

        if collection == "gold":

            gold_count += 1

        elif collection == "prototype":

            prototype_count += 1

        jewellery_type = str(item.get("type", "")).strip()

        if jewellery_type:

            types[jewellery_type] = types.get(jewellery_type, 0) + 1

    return jsonify(
        {
            "success": True,
            "total": len(catalogue),
            "gold": gold_count,
            "prototype": prototype_count,
            "types": types,
        }
    )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("JEWELMATCH AI")
    print("Flask Backend")
    print("=" * 70)

    print()
    print("Catalogue API:")
    print("GET    /api/jewellery")
    print("POST   /api/jewellery/add")
    print("PUT    /api/jewellery/<id>")
    print("DELETE /api/jewellery/<id>")
    print("POST   /api/jewellery/rebuild-index")
    print("POST   /api/match")
    print("=" * 70)

    app.run(host="127.0.0.1", port=5000, debug=False)
