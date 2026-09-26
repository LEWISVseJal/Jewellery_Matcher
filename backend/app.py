"""
JewelMatch AI - Flask Application

Lightweight CPU-only version.

The application does NOT use:

- torch
- torchvision
- transformers
- DINOv2
- rembg
- pymatting
- numba
- onnxruntime

Catalogue management automatically rebuilds the lightweight
OpenCV search index.
"""

from pathlib import Path
from uuid import uuid4

import json
import os
import re
import traceback


from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

BASE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BASE_DIR.parent

DATABASE_DIR = BASE_DIR / "database"

CATALOGUE_FILE = DATABASE_DIR / "jewellery.json"

CATALOGUE_DIR = BASE_DIR / "catalogue"

UPLOADS_DIR = BASE_DIR / "uploads"


ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp",
}


MAX_UPLOAD_MB = 20


app = Flask(
    __name__,
    template_folder=str(PROJECT_DIR / "frontend" / "templates"),
    static_folder=str(PROJECT_DIR / "frontend" / "static"),
)


app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


for directory in (
    DATABASE_DIR,
    CATALOGUE_DIR / "gold",
    CATALOGUE_DIR / "prototype",
    UPLOADS_DIR,
):

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# HELPERS
# ============================================================


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

    return ""


def display_collection(value):

    value = normalise_collection(value)

    if value == "gold":
        return "Gold"

    if value == "prototype":
        return "Prototype"

    return ""


def safe_filename(filename):

    filename = Path(filename or "").name

    filename = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        filename,
    )

    if not filename:

        filename = f"jewellery_" f"{uuid4().hex[:10]}.jpg"

    return filename


def allowed_file(filename):

    extension = Path(filename or "").suffix.lower().lstrip(".")

    return extension in ALLOWED_EXTENSIONS


def load_catalogue():

    if not CATALOGUE_FILE.exists():
        return []

    try:

        with open(
            CATALOGUE_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

    except Exception:

        return []

    if isinstance(
        data,
        list,
    ):

        return data

    if isinstance(
        data,
        dict,
    ):

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


def save_catalogue(items):

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
            indent=2,
            ensure_ascii=False,
        )

    os.replace(
        temporary_file,
        CATALOGUE_FILE,
    )


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


def get_item_image_path(item):

    collection = normalise_collection(item.get("collection"))

    filename = get_item_filename(item)

    if collection not in {
        "gold",
        "prototype",
    }:

        return None

    if not filename:
        return None

    path = CATALOGUE_DIR / collection / filename

    if path.exists():
        return path

    return None


def get_form_value(*names, default=""):

    for name in names:

        value = request.form.get(name)

        if value is not None:

            return value.strip()

    return default


def save_uploaded_image(
    file_storage,
    collection,
    filename,
):

    if file_storage is None or not file_storage.filename:

        raise ValueError("Please select an image.")

    if not allowed_file(file_storage.filename):

        raise ValueError(
            "Unsupported image format. " "Use JPG, JPEG, PNG, WEBP or BMP."
        )

    collection = normalise_collection(collection)

    if collection not in {
        "gold",
        "prototype",
    }:

        raise ValueError("Collection must be Gold or Prototype.")

    filename = safe_filename(filename)

    target = CATALOGUE_DIR / collection / filename

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_storage.save(target)

    return (
        filename,
        target,
    )


# ============================================================
# LIGHTWEIGHT INDEX
# ============================================================


def build_lightweight_index():

    import numpy as np

    from backend.services.embedding import (
        create_embedding,
    )

    items = load_catalogue()

    features = []
    ids = []
    collections = []

    failed = []

    for item in items:

        current_id = get_item_id(item)

        collection = normalise_collection(item.get("collection"))

        image_path = get_item_image_path(item)

        if not current_id:

            failed.append(
                {
                    "id": "",
                    "reason": "Missing jewellery ID",
                }
            )

            continue

        if collection not in {
            "gold",
            "prototype",
        }:

            failed.append(
                {
                    "id": current_id,
                    "reason": "Invalid collection",
                }
            )

            continue

        if image_path is None:

            failed.append(
                {
                    "id": current_id,
                    "reason": "Image not found",
                }
            )

            continue

        try:

            vector = create_embedding(image_path)

            if vector.shape != (256,):

                raise ValueError("Expected 256 features, " f"got {vector.shape}")

            features.append(vector)

            ids.append(current_id)

            collections.append(collection)

        except Exception as exc:

            failed.append(
                {
                    "id": current_id,
                    "reason": str(exc),
                }
            )

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

    index_file = DATABASE_DIR / "lightweight_index.npz"

    np.savez_compressed(
        index_file,
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

    try:

        from backend.services.matcher import (
            clear_index_cache,
        )

        clear_index_cache()

    except Exception:

        pass

    return {
        "indexed": len(ids),
        "total": len(items),
        "failed": failed,
        "index_file": str(index_file),
    }


def delete_old_image(item):

    image_path = get_item_image_path(item)

    if image_path is None:
        return

    try:

        image_path.unlink(missing_ok=True)

    except OSError:

        pass


# ============================================================
# PAGES
# ============================================================


@app.get("/")
def index():

    return render_template("index.html")


@app.get("/catalogue")
def catalogue_page():

    return render_template("catalogue.html")


@app.get("/add-jewellery")
def add_jewellery_page():

    return render_template("add_jewellery.html")


# ============================================================
# IMAGES
# ============================================================


@app.get("/catalogue/<collection>/<path:filename>")
def catalogue_image(
    collection,
    filename,
):

    collection = normalise_collection(collection)

    if collection not in {
        "gold",
        "prototype",
    }:

        return jsonify({"error": "Invalid collection"}), 404

    return send_from_directory(
        CATALOGUE_DIR / collection,
        filename,
    )


@app.get("/uploads/<path:filename>")
def uploaded_image(filename):

    return send_from_directory(
        UPLOADS_DIR,
        filename,
    )


# ============================================================
# HEALTH
# ============================================================


@app.get("/api/health")
def health():

    index_file = DATABASE_DIR / "lightweight_index.npz"

    return jsonify(
        {
            "status": "ok",
            "engine": "opencv-lightweight",
            "catalogue_records": len(load_catalogue()),
            "index_exists": index_file.exists(),
        }
    )


# ============================================================
# GET CATALOGUE
# ============================================================


@app.get("/api/jewellery")
def get_jewellery():

    items = load_catalogue()

    output = []

    for item in items:

        copied = dict(item)

        collection = normalise_collection(copied.get("collection"))

        filename = get_item_filename(copied)

        if collection and filename:

            copied["image_url"] = f"/catalogue/" f"{collection}/" f"{filename}"

        else:

            copied["image_url"] = None

        output.append(copied)

    return jsonify(
        {
            "success": True,
            "items": output,
            "count": len(output),
        }
    )


# ============================================================
# STATISTICS
# ============================================================


@app.get("/api/jewellery/stats")
def jewellery_stats():

    items = load_catalogue()

    gold = sum(
        1 for item in items if normalise_collection(item.get("collection")) == "gold"
    )

    prototype = sum(
        1
        for item in items
        if normalise_collection(item.get("collection")) == "prototype"
    )

    return jsonify(
        {
            "success": True,
            "total": len(items),
            "gold": gold,
            "prototype": prototype,
        }
    )


# ============================================================
# MATCH
# ============================================================


@app.post("/api/match")
def match():

    upload = (
        request.files.get("image")
        or request.files.get("file")
        or request.files.get("query_image")
    )

    if upload is None or not upload.filename:

        return (
            jsonify(
                {
                    "success": False,
                    "error": "Please upload an image.",
                }
            ),
            400,
        )

    if not allowed_file(upload.filename):

        return (
            jsonify(
                {
                    "success": False,
                    "error": "Unsupported image format.",
                }
            ),
            400,
        )

    extension = Path(upload.filename).suffix.lower().lstrip(".")

    filename = f"query_" f"{uuid4().hex}." f"{extension}"

    query_path = UPLOADS_DIR / filename

    try:

        upload.save(query_path)

        from backend.services.matcher import (
            match_jewellery,
        )

        top_k = int(
            request.form.get(
                "top_k",
                8,
            )
        )

        result = match_jewellery(
            query_path,
            top_k=top_k,
            target_collection=request.form.get(
                "target_collection",
                "auto",
            ),
        )

        return jsonify(
            {
                "success": True,
                **result,
            }
        )

    except Exception as exc:

        traceback.print_exc()

        return (
            jsonify(
                {
                    "success": False,
                    "error": str(exc),
                }
            ),
            500,
        )

    finally:

        try:

            query_path.unlink(missing_ok=True)

        except OSError:

            pass


# ============================================================
# ADD JEWELLERY
# ============================================================


@app.post("/api/jewellery/add")
def add_jewellery():

    try:

        upload = (
            request.files.get("image")
            or request.files.get("jewelleryImage")
            or request.files.get("file")
        )

        collection = normalise_collection(
            get_form_value(
                "collection",
                "jewelleryCollection",
            )
        )

        name = get_form_value(
            "name",
            "jewelleryName",
            default="Unnamed Jewellery",
        )

        jewellery_type = get_form_value(
            "type",
            "jewelleryType",
        )

        gender = get_form_value(
            "gender",
            default="Unisex",
        )

        subtype = get_form_value(
            "subtype",
            "jewellerySubtype",
        )

        description = get_form_value("description")

        if collection not in {
            "gold",
            "prototype",
        }:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Please select Gold or Prototype collection.",
                    }
                ),
                400,
            )

        if upload is None or not upload.filename:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Please select a jewellery image.",
                    }
                ),
                400,
            )

        new_id = get_form_value(
            "id",
            "designId",
        )

        if not new_id:

            prefix = "G" if collection == "gold" else "P"

            new_id = f"{prefix}" f"{uuid4().hex[:6].upper()}"

        items = load_catalogue()

        if any(get_item_id(item) == new_id for item in items):

            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Jewellery ID " f"'{new_id}' already exists.",
                    }
                ),
                409,
            )

        extension = Path(upload.filename).suffix.lower()

        filename = f"{new_id}" f"{extension}"

        filename, image_path = save_uploaded_image(
            upload,
            collection,
            filename,
        )

        record = {
            "id": new_id,
            "name": name,
            "collection": display_collection(collection),
            "gender": gender,
            "type": jewellery_type,
            "subtype": subtype,
            "description": description,
            "filename": filename,
            "image": filename,
        }

        items.append(record)

        save_catalogue(items)

        rebuild = build_lightweight_index()

        return jsonify(
            {
                "success": True,
                "message": "Jewellery added successfully.",
                "item": record,
                "image_url": (f"/catalogue/" f"{collection}/" f"{filename}"),
                "index": rebuild,
            }
        )

    except Exception as exc:

        traceback.print_exc()

        return (
            jsonify(
                {
                    "success": False,
                    "error": str(exc),
                }
            ),
            500,
        )


# ============================================================
# EDIT JEWELLERY
# ============================================================


@app.route(
    "/api/jewellery/<jewellery_id>",
    methods=[
        "PUT",
        "POST",
    ],
)
def update_jewellery(jewellery_id):

    items = load_catalogue()

    position = next(
        (i for i, item in enumerate(items) if get_item_id(item) == str(jewellery_id)),
        None,
    )

    if position is None:

        return (
            jsonify(
                {
                    "success": False,
                    "error": "Jewellery item not found.",
                }
            ),
            404,
        )

    current = dict(items[position])

    old_collection = normalise_collection(current.get("collection"))

    old_filename = get_item_filename(current)

    try:

        new_collection = normalise_collection(
            get_form_value(
                "collection",
                "jewelleryCollection",
                default=old_collection,
            )
        )

        if new_collection not in {
            "gold",
            "prototype",
        }:

            raise ValueError("Collection must be Gold or Prototype.")

        current["name"] = get_form_value(
            "name",
            "jewelleryName",
            default=current.get(
                "name",
                "",
            ),
        )

        current["gender"] = get_form_value(
            "gender",
            default=current.get(
                "gender",
                "Unisex",
            ),
        )

        current["type"] = get_form_value(
            "type",
            "jewelleryType",
            default=current.get(
                "type",
                "",
            ),
        )

        current["subtype"] = get_form_value(
            "subtype",
            "jewellerySubtype",
            default=current.get(
                "subtype",
                "",
            ),
        )

        current["description"] = get_form_value(
            "description",
            default=current.get(
                "description",
                "",
            ),
        )

        current["collection"] = display_collection(new_collection)

        upload = (
            request.files.get("image")
            or request.files.get("jewelleryImage")
            or request.files.get("file")
        )

        collection_changed = new_collection != old_collection

        if upload is not None and upload.filename:

            extension = Path(upload.filename).suffix.lower()

            new_filename = f"{jewellery_id}" f"{extension}"

            save_uploaded_image(
                upload,
                new_collection,
                new_filename,
            )

            current["filename"] = new_filename

            current["image"] = new_filename

            if old_filename and (
                old_collection != new_collection or old_filename != new_filename
            ):

                old_path = CATALOGUE_DIR / old_collection / old_filename

                try:

                    old_path.unlink(missing_ok=True)

                except OSError:

                    pass

        elif collection_changed:

            if not old_filename:

                raise ValueError("Current image is missing.")

            old_path = CATALOGUE_DIR / old_collection / old_filename

            new_path = CATALOGUE_DIR / new_collection / old_filename

            new_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if old_path.exists():

                old_path.replace(new_path)

            current["filename"] = old_filename

            current["image"] = old_filename

        items[position] = current

        save_catalogue(items)

        rebuild = build_lightweight_index()

        return jsonify(
            {
                "success": True,
                "message": "Jewellery updated successfully.",
                "item": current,
                "index": rebuild,
            }
        )

    except Exception as exc:

        traceback.print_exc()

        return (
            jsonify(
                {
                    "success": False,
                    "error": str(exc),
                }
            ),
            500,
        )


# ============================================================
# DELETE JEWELLERY
# ============================================================


@app.route(
    "/api/jewellery/<jewellery_id>",
    methods=["DELETE"],
)
def delete_jewellery(jewellery_id):

    items = load_catalogue()

    position = next(
        (i for i, item in enumerate(items) if get_item_id(item) == str(jewellery_id)),
        None,
    )

    if position is None:

        return (
            jsonify(
                {
                    "success": False,
                    "error": "Jewellery item not found.",
                }
            ),
            404,
        )

    removed = items.pop(position)

    try:

        delete_old_image(removed)

        save_catalogue(items)

        rebuild = build_lightweight_index()

        return jsonify(
            {
                "success": True,
                "message": "Jewellery deleted successfully.",
                "deleted_id": str(jewellery_id),
                "index": rebuild,
            }
        )

    except Exception as exc:

        traceback.print_exc()

        return (
            jsonify(
                {
                    "success": False,
                    "error": str(exc),
                }
            ),
            500,
        )


# ============================================================
# REBUILD INDEX
# ============================================================


@app.post("/api/jewellery/rebuild-index")
@app.post("/api/rebuild-index")
def rebuild_index():

    try:

        result = build_lightweight_index()

        return jsonify(
            {
                "success": True,
                "message": "Lightweight search index rebuilt.",
                **result,
            }
        )

    except Exception as exc:

        traceback.print_exc()

        return (
            jsonify(
                {
                    "success": False,
                    "error": str(exc),
                }
            ),
            500,
        )


# ============================================================
# RUN LOCAL
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000,
            )
        ),
        debug=True,
    )
