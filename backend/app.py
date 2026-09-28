"""
JewelMatch AI - Flask Backend

DINOv2-Small / CPU / Render-safe backend.

Main features:
    - Jewellery image matching
    - Search from All
    - Gold -> Prototype
    - Prototype -> Gold
    - Add jewellery
    - Catalogue management
    - Edit jewellery
    - Delete jewellery
    - Rebuild visual index
    - Serve catalogue images
    - Serve uploaded images
"""

from __future__ import annotations

import json
import os
import uuid
import traceback
from pathlib import Path
from typing import Optional

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from flask_cors import CORS
from werkzeug.utils import secure_filename

# ============================================================
# SERVICES
# ============================================================

from .services.matcher import (
    match_jewellery,
    build_index,
    load_index,
)

from .services.segmentation import segment_jewellery

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BASE_DIR.parent

FRONTEND_DIR = PROJECT_DIR / "frontend"

TEMPLATES_DIR = FRONTEND_DIR / "templates"

STATIC_DIR = FRONTEND_DIR / "static"

DATABASE_DIR = BASE_DIR / "database"

CATALOGUE_DIR = BASE_DIR / "catalogue"

UPLOAD_DIR = BASE_DIR / "uploads"

JEWELLERY_JSON = DATABASE_DIR / "jewellery.json"


# ============================================================
# CATALOGUE COLLECTIONS
# ============================================================

GOLD_DIR = CATALOGUE_DIR / "gold"

PROTOTYPE_DIR = CATALOGUE_DIR / "prototype"


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)

CORS(app)


# ============================================================
# FLASK CONFIG
# ============================================================

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

app.config["JSON_SORT_KEYS"] = False


# ============================================================
# ALLOWED FILES
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp",
}


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for directory in [
    DATABASE_DIR,
    CATALOGUE_DIR,
    GOLD_DIR,
    PROTOTYPE_DIR,
    UPLOAD_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# JSON HELPERS
# ============================================================


def load_catalogue() -> list:
    """
    Load jewellery catalogue JSON.
    """

    if not JEWELLERY_JSON.exists():
        return []

    try:
        with open(
            JEWELLERY_JSON,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except Exception as exc:
        print("[CATALOGUE] Failed to load JSON:", exc)
        return []

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in [
            "jewellery",
            "items",
            "catalogue",
            "data",
        ]:
            if isinstance(data.get(key), list):
                return data[key]

    return []


def save_catalogue(catalogue: list) -> None:
    """
    Save catalogue JSON safely.
    """

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = JEWELLERY_JSON.with_suffix(".tmp")

    with open(
        temporary_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            catalogue,
            file,
            indent=4,
            ensure_ascii=False,
        )

    temporary_file.replace(JEWELLERY_JSON)


# ============================================================
# FILE HELPERS
# ============================================================


def allowed_file(filename: str) -> bool:
    """
    Check supported image extension.
    """

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(
        ".",
        1,
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# COLLECTION HELPERS
# ============================================================


def normalize_collection(
    value: Optional[str],
) -> Optional[str]:
    """
    Normalize collection name.

    Supported aliases:

        gold
        gold_img
        finished

        prototype
        prototype_img
        green
    """

    if value is None:
        return None

    value = str(value).strip().lower()

    if value in [
        "gold",
        "gold_img",
        "finished",
    ]:
        return "gold"

    if value in [
        "prototype",
        "prototype_img",
        "green",
    ]:
        return "prototype"

    return None


def get_collection_folder(
    collection: str,
) -> Optional[Path]:
    """
    Return folder for collection.
    """

    collection = normalize_collection(collection)

    if collection == "gold":
        return GOLD_DIR

    if collection == "prototype":
        return PROTOTYPE_DIR

    return None


# ============================================================
# SEARCH MODE
# ============================================================


def normalize_search_mode(
    value: Optional[str],
) -> str:
    """
    Normalize visual search mode.

    Supported modes:

        all
        gold_to_prototype
        prototype_to_gold
    """

    if value is None:
        return "all"

    value = str(value).strip().lower()

    aliases = {
        "all": "all",
        "both": "all",
        "search_all": "all",
        "all_collections": "all",
        "gold_to_prototype": "gold_to_prototype",
        "gold-to-prototype": "gold_to_prototype",
        "gold2prototype": "gold_to_prototype",
        "gold_to_proto": "gold_to_prototype",
        "prototype_to_gold": "prototype_to_gold",
        "prototype-to-gold": "prototype_to_gold",
        "prototype2gold": "prototype_to_gold",
        "proto_to_gold": "prototype_to_gold",
    }

    return aliases.get(
        value,
        "all",
    )


def get_search_mode_label(
    search_mode: str,
) -> str:
    """
    Human-readable search mode.
    """

    search_mode = normalize_search_mode(search_mode)

    if search_mode == "gold_to_prototype":
        return "Gold → Prototype"

    if search_mode == "prototype_to_gold":
        return "Prototype → Gold"

    return "Search from All"


# ============================================================
# FILE NAME
# ============================================================


def generate_unique_filename(
    original_filename: str,
) -> str:
    """
    Generate safe unique image filename.
    """

    original_filename = secure_filename(original_filename)

    extension = ""

    if "." in original_filename:
        extension = (
            "."
            + original_filename.rsplit(
                ".",
                1,
            )[1].lower()
        )

    unique_id = uuid.uuid4().hex[:12]

    return f"{unique_id}{extension}"


# ============================================================
# ID HELPERS
# ============================================================


def generate_design_id(
    collection: str,
    catalogue: list,
) -> str:
    """
    Generate a unique jewellery ID.
    """

    prefix = "GOLD" if collection == "gold" else "PROTOTYPE"

    existing_ids = {str(item.get("id", "")) for item in catalogue}

    number = 1

    while True:

        design_id = f"{prefix}{number:04d}"

        if design_id not in existing_ids:
            return design_id

        number += 1


# ============================================================
# FIND ITEM
# ============================================================


def find_catalogue_item(
    item_id: str,
    catalogue: list,
):
    """
    Find catalogue item by id or design_id.
    """

    item_id = str(item_id)

    for index, item in enumerate(catalogue):

        if str(item.get("id", "")) == item_id:
            return index, item

        if str(item.get("design_id", "")) == item_id:
            return index, item

    return None, None


# ============================================================
# IMAGE PATH FROM ITEM
# ============================================================


def get_item_image_path(
    item: dict,
) -> Optional[Path]:
    """
    Resolve stored image path.
    """

    possible_keys = [
        "image_path",
        "image",
        "path",
        "file_path",
        "filename",
        "file",
    ]

    raw_path = None

    for key in possible_keys:

        if item.get(key):

            raw_path = str(item[key])
            break

    if not raw_path:
        return None

    raw_path = raw_path.replace(
        "\\",
        "/",
    )

    candidate_paths = []

    path = Path(raw_path)

    if path.is_absolute():
        candidate_paths.append(path)

    candidate_paths.extend(
        [
            PROJECT_DIR / raw_path,
            BASE_DIR / raw_path,
            CATALOGUE_DIR / raw_path,
            PROJECT_DIR / "backend" / raw_path,
        ]
    )

    checked = set()

    for candidate in candidate_paths:

        try:
            candidate = candidate.resolve()

        except Exception:
            continue

        candidate_string = str(candidate)

        if candidate_string in checked:
            continue

        checked.add(candidate_string)

        if candidate.exists():
            return candidate

    # Filename fallback.

    filename = Path(raw_path).name

    if filename:

        for folder in [
            GOLD_DIR,
            PROTOTYPE_DIR,
        ]:

            if folder.exists():

                matches = list(folder.rglob(filename))

                if matches:
                    return matches[0]

    return None


# ============================================================
# IMAGE URL
# ============================================================


def image_url_for_item(
    item: dict,
) -> Optional[str]:
    """
    Convert local catalogue image path
    into browser-accessible URL.
    """

    image_path = get_item_image_path(item)

    if image_path is None:
        return None

    try:

        relative = image_path.relative_to(CATALOGUE_DIR)

        parts = relative.parts

        if len(parts) < 2:
            return None

        collection = parts[0]

        filename = "/".join(parts[1:])

        return url_for(
            "catalogue_image",
            collection=collection,
            filename=filename,
        )

    except ValueError:
        return None


# ============================================================
# NORMALIZE ITEM FOR API
# ============================================================


def serialize_catalogue_item(
    item: dict,
) -> dict:
    """
    Return catalogue item safe for frontend.
    """

    result = dict(item)

    image_url = image_url_for_item(item)

    if image_url:

        result["image_url"] = image_url

        # Backward compatibility.
        result["image"] = image_url

    return result


# ============================================================
# HOME
# ============================================================


@app.route(
    "/",
    methods=["GET"],
)
def home():

    return render_template("index.html")


# ============================================================
# ADD JEWELLERY PAGE
# ============================================================


@app.route(
    "/add-jewellery",
    methods=["GET"],
)
def add_jewellery_page():

    return render_template("add_jewellery.html")


# ============================================================
# CATALOGUE PAGE
# ============================================================


@app.route(
    "/catalogue",
    methods=["GET"],
)
def catalogue_page():

    return render_template("catalogue.html")


# ============================================================
# SERVE CATALOGUE IMAGE
# ============================================================


@app.route(
    "/catalogue-image/<collection>/<path:filename>",
    methods=["GET"],
)
def catalogue_image(
    collection,
    filename,
):

    collection = normalize_collection(collection)

    if collection is None:

        return (
            jsonify(
                {
                    "success": False,
                    "message": "Invalid collection.",
                }
            ),
            400,
        )

    folder = get_collection_folder(collection)

    if folder is None:

        return (
            jsonify(
                {
                    "success": False,
                    "message": "Invalid collection.",
                }
            ),
            400,
        )

    return send_from_directory(
        str(folder),
        filename,
    )


# ============================================================
# HEALTH CHECK
# ============================================================


@app.route(
    "/api/health",
    methods=["GET"],
)
def health():

    return jsonify(
        {
            "success": True,
            "status": "healthy",
            "service": "JewelMatch AI",
        }
    )


# ============================================================
# GET CATALOGUE
# ============================================================


def catalogue_response():

    catalogue = load_catalogue()

    collection_filter = request.args.get("collection")

    search = (
        request.args.get(
            "search",
            "",
        )
        .strip()
        .lower()
    )

    normalized_filter = (
        normalize_collection(collection_filter) if collection_filter else None
    )

    results = []

    for item in catalogue:

        item_collection = normalize_collection(
            item.get(
                "collection",
                item.get("category"),
            )
        )

        if normalized_filter and item_collection != normalized_filter:
            continue

        if search:

            searchable = " ".join(
                [
                    str(item.get("id", "")),
                    str(item.get("design_id", "")),
                    str(item.get("name", "")),
                    str(item.get("design_name", "")),
                    str(item.get("description", "")),
                    str(item.get("collection", "")),
                ]
            ).lower()

            if search not in searchable:
                continue

        results.append(serialize_catalogue_item(item))

    gold_count = sum(
        1
        for item in catalogue
        if normalize_collection(
            item.get(
                "collection",
                item.get("category"),
            )
        )
        == "gold"
    )

    prototype_count = sum(
        1
        for item in catalogue
        if normalize_collection(
            item.get(
                "collection",
                item.get("category"),
            )
        )
        == "prototype"
    )

    return jsonify(
        {
            "success": True,
            "items": results,
            "results": results,
            "total": len(results),
            "total_count": len(catalogue),
            "gold_count": gold_count,
            "prototype_count": prototype_count,
        }
    )


@app.route(
    "/api/catalogue",
    methods=["GET"],
)
def get_catalogue():

    return catalogue_response()


# ============================================================
# CATALOGUE API COMPATIBILITY ALIAS
# ============================================================
#
# Your catalogue.js was previously requesting:
#
#     /api/jewellery
#
# while the backend exposed:
#
#     /api/catalogue
#
# Keep both endpoints working.
# ============================================================


@app.route(
    "/api/jewellery",
    methods=["GET"],
)
def get_jewellery_alias():

    return catalogue_response()


# ============================================================
# GET SINGLE CATALOGUE ITEM
# ============================================================


@app.route(
    "/api/catalogue/<item_id>",
    methods=["GET"],
)
def get_catalogue_item(item_id):

    catalogue = load_catalogue()

    index, item = find_catalogue_item(
        item_id,
        catalogue,
    )

    if item is None:

        return (
            jsonify(
                {
                    "success": False,
                    "message": "Jewellery item not found.",
                }
            ),
            404,
        )

    return jsonify(
        {
            "success": True,
            "item": serialize_catalogue_item(item),
        }
    )


# ============================================================
# ADD JEWELLERY
# ============================================================


@app.route(
    "/api/jewellery/add",
    methods=["POST"],
)
def add_jewellery():

    try:

        # ----------------------------------------------------
        # Collection
        # ----------------------------------------------------

        collection = request.form.get("collection") or request.form.get("category")

        collection = normalize_collection(collection)

        if collection is None:

            return (
                jsonify(
                    {
                        "success": False,
                        "message": ("Collection must be " "Gold or Prototype."),
                    }
                ),
                400,
            )

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image_file = request.files.get("image") or request.files.get("file")

        if image_file is None:

            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Please upload an image.",
                    }
                ),
                400,
            )

        if not image_file.filename:

            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Image filename is empty.",
                    }
                ),
                400,
            )

        if not allowed_file(image_file.filename):

            return (
                jsonify(
                    {
                        "success": False,
                        "message": (
                            "Unsupported image format. "
                            "Use JPG, JPEG, PNG, "
                            "WEBP or BMP."
                        ),
                    }
                ),
                400,
            )

        # ----------------------------------------------------
        # Catalogue
        # ----------------------------------------------------

        catalogue = load_catalogue()

        item_id = request.form.get("id") or request.form.get("design_id")

        if not item_id:

            item_id = generate_design_id(
                collection,
                catalogue,
            )

        item_id = str(item_id).strip()

        existing_index, existing_item = find_catalogue_item(
            item_id,
            catalogue,
        )

        if existing_item is not None:

            return (
                jsonify(
                    {
                        "success": False,
                        "message": (f"Jewellery ID " f"'{item_id}' already exists."),
                    }
                ),
                409,
            )

        # ----------------------------------------------------
        # Save original image
        # ----------------------------------------------------

        original_filename = image_file.filename

        safe_filename = generate_unique_filename(original_filename)

        collection_folder = get_collection_folder(collection)

        if collection_folder is None:

            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Invalid collection.",
                    }
                ),
                400,
            )

        collection_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_path = collection_folder / safe_filename

        image_file.save(str(image_path))

        # ----------------------------------------------------
        # Lightweight segmentation
        # ----------------------------------------------------

        segmented_folder = DATABASE_DIR / "segmented_catalogue" / collection

        segmented_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        segmented_filename = Path(safe_filename).stem + "_segmented.jpg"

        segmented_path = segmented_folder / segmented_filename

        try:

            segment_jewellery(
                image_path,
                segmented_path,
            )

        except Exception as exc:

            print(
                "[ADD] Lightweight segmentation failed:",
                exc,
            )

            segmented_path = None

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        relative_image_path = str(image_path.relative_to(PROJECT_DIR)).replace(
            "\\",
            "/",
        )

        item = {
            "id": item_id,
            "design_id": (request.form.get("design_id") or item_id),
            "name": (
                request.form.get("name") or request.form.get("design_name") or item_id
            ),
            "collection": collection,
            "category": collection,
            "description": (request.form.get("description") or ""),
            "image": relative_image_path,
            "image_path": relative_image_path,
            "original_filename": original_filename,
        }

        if segmented_path is not None:

            item["segmented_image"] = str(
                segmented_path.relative_to(PROJECT_DIR)
            ).replace(
                "\\",
                "/",
            )

        # ----------------------------------------------------
        # Custom fields
        # ----------------------------------------------------

        reserved_fields = {
            "collection",
            "category",
            "id",
            "design_id",
            "name",
            "design_name",
            "description",
        }

        for key in request.form:

            if key in reserved_fields:
                continue

            value = request.form.get(key)

            if value is not None:
                item[key] = value

        catalogue.append(item)

        save_catalogue(catalogue)

        # ----------------------------------------------------
        # Rebuild DINO index
        # ----------------------------------------------------

        try:

            build_index(force=True)

        except Exception as exc:

            print(
                "[ADD] DINO index rebuild failed:",
                exc,
            )

            return (
                jsonify(
                    {
                        "success": True,
                        "message": (
                            "Jewellery added, but " "visual index rebuild failed."
                        ),
                        "warning": str(exc),
                        "item": (serialize_catalogue_item(item)),
                    }
                ),
                201,
            )

        return (
            jsonify(
                {
                    "success": True,
                    "message": ("Jewellery added successfully."),
                    "item": (serialize_catalogue_item(item)),
                }
            ),
            201,
        )

    except Exception as exc:

        print(
            "[ADD] ERROR:",
            exc,
        )

        traceback.print_exc()

        return (
            jsonify(
                {
                    "success": False,
                    "message": str(exc),
                }
            ),
            500,
        )


# ============================================================
# EDIT JEWELLERY
# ============================================================


@app.route(
    "/api/jewellery/<item_id>",
    methods=["PUT", "POST"],
)
def update_jewellery(item_id):

    try:

        catalogue = load_catalogue()

        index, item = find_catalogue_item(
            item_id,
            catalogue,
        )

        if item is None:

            return (
                jsonify(
                    {
                        "success": False,
                        "message": ("Jewellery item not found."),
                    }
                ),
                404,
            )

        # ----------------------------------------------------
        # JSON or form data
        # ----------------------------------------------------

        if request.is_json:

            data = request.get_json(silent=True) or {}

        else:

            data = request.form.to_dict()

        # ----------------------------------------------------
        # Collection
        # ----------------------------------------------------

        old_collection = normalize_collection(item.get("collection"))

        new_collection = normalize_collection(
            data.get(
                "collection",
                old_collection,
            )
        )

        if new_collection is None:
            new_collection = old_collection

        # ----------------------------------------------------
        # Update fields
        # ----------------------------------------------------

        editable_fields = [
            "name",
            "design_name",
            "description",
            "design_id",
        ]

        for field in editable_fields:

            if field in data:

                value = data.get(field)

                if value is not None:
                    item[field] = str(value)

        item["collection"] = new_collection

        item["category"] = new_collection

        # ----------------------------------------------------
        # Replace image
        # ----------------------------------------------------

        image_file = request.files.get("image") or request.files.get("file")

        if image_file is not None:

            if not image_file.filename:

                return (
                    jsonify(
                        {
                            "success": False,
                            "message": ("Image filename is empty."),
                        }
                    ),
                    400,
                )

            if not allowed_file(image_file.filename):

                return (
                    jsonify(
                        {
                            "success": False,
                            "message": ("Unsupported image format."),
                        }
                    ),
                    400,
                )

            # Delete old image.

            old_image_path = get_item_image_path(item)

            if old_image_path and old_image_path.exists():

                try:
                    old_image_path.unlink()

                except Exception as exc:

                    print(
                        "[UPDATE] Could not delete old image:",
                        exc,
                    )

            # Save new image.

            collection_folder = get_collection_folder(new_collection)

            if collection_folder is None:

                return (
                    jsonify(
                        {
                            "success": False,
                            "message": ("Invalid collection."),
                        }
                    ),
                    400,
                )

            collection_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            new_filename = generate_unique_filename(image_file.filename)

            new_image_path = collection_folder / new_filename

            image_file.save(str(new_image_path))

            relative_path = str(new_image_path.relative_to(PROJECT_DIR)).replace(
                "\\",
                "/",
            )

            item["image"] = relative_path

            item["image_path"] = relative_path

            item["original_filename"] = image_file.filename

            # Create lightweight processed image.

            segmented_folder = DATABASE_DIR / "segmented_catalogue" / new_collection

            segmented_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            segmented_path = segmented_folder / (
                Path(new_filename).stem + "_segmented.jpg"
            )

            try:

                segment_jewellery(
                    new_image_path,
                    segmented_path,
                )

                item["segmented_image"] = str(
                    segmented_path.relative_to(PROJECT_DIR)
                ).replace(
                    "\\",
                    "/",
                )

            except Exception as exc:

                print(
                    "[UPDATE] Segmentation failed:",
                    exc,
                )

        # ----------------------------------------------------
        # Save catalogue
        # ----------------------------------------------------

        catalogue[index] = item

        save_catalogue(catalogue)

        # ----------------------------------------------------
        # Rebuild DINO index
        # ----------------------------------------------------

        build_index(force=True)

        return jsonify(
            {
                "success": True,
                "message": ("Jewellery updated successfully."),
                "item": (serialize_catalogue_item(item)),
            }
        )

    except Exception as exc:

        print(
            "[UPDATE] ERROR:",
            exc,
        )

        traceback.print_exc()

        return (
            jsonify(
                {
                    "success": False,
                    "message": str(exc),
                }
            ),
            500,
        )


# ============================================================
# DELETE JEWELLERY
# ============================================================


@app.route(
    "/api/jewellery/<item_id>",
    methods=["DELETE"],
)
def delete_jewellery(item_id):

    try:

        catalogue = load_catalogue()

        index, item = find_catalogue_item(
            item_id,
            catalogue,
        )

        if item is None:

            return (
                jsonify(
                    {
                        "success": False,
                        "message": ("Jewellery item not found."),
                    }
                ),
                404,
            )

        # ----------------------------------------------------
        # Delete original image
        # ----------------------------------------------------

        image_path = get_item_image_path(item)

        if image_path and image_path.exists():

            try:
                image_path.unlink()

            except Exception as exc:

                print(
                    "[DELETE] Could not delete image:",
                    exc,
                )

        # ----------------------------------------------------
        # Delete segmented image
        # ----------------------------------------------------

        segmented_raw = item.get("segmented_image")

        if segmented_raw:

            segmented_path = PROJECT_DIR / str(segmented_raw).replace(
                "\\",
                "/",
            )

            if segmented_path.exists():

                try:
                    segmented_path.unlink()

                except Exception as exc:

                    print(
                        "[DELETE] Could not delete segmented image:",
                        exc,
                    )

        # ----------------------------------------------------
        # Remove from JSON
        # ----------------------------------------------------

        deleted_item = catalogue.pop(index)

        save_catalogue(catalogue)

        # ----------------------------------------------------
        # Rebuild DINO index
        # ----------------------------------------------------

        try:

            build_index(force=True)

        except Exception as exc:

            print(
                "[DELETE] DINO index rebuild failed:",
                exc,
            )

            return jsonify(
                {
                    "success": True,
                    "message": ("Jewellery deleted, but " "index rebuild failed."),
                    "warning": str(exc),
                    "item": deleted_item,
                }
            )

        return jsonify(
            {
                "success": True,
                "message": ("Jewellery deleted successfully."),
                "item": deleted_item,
            }
        )

    except Exception as exc:

        print(
            "[DELETE] ERROR:",
            exc,
        )

        traceback.print_exc()

        return (
            jsonify(
                {
                    "success": False,
                    "message": str(exc),
                }
            ),
            500,
        )


# ============================================================
# MATCH API
# ============================================================


@app.route(
    "/api/match",
    methods=["POST"],
)
def match():

    query_path = None

    try:

        # ----------------------------------------------------
        # Receive uploaded file
        # ----------------------------------------------------

        image_file = (
            request.files.get("image")
            or request.files.get("file")
            or request.files.get("query_image")
        )

        if image_file is None:

            return (
                jsonify(
                    {
                        "success": False,
                        "matched": False,
                        "message": ("Please upload a " "jewellery image."),
                        "results": [],
                    }
                ),
                400,
            )

        if not image_file.filename:

            return (
                jsonify(
                    {
                        "success": False,
                        "matched": False,
                        "message": ("Uploaded image has " "no filename."),
                        "results": [],
                    }
                ),
                400,
            )

        if not allowed_file(image_file.filename):

            return (
                jsonify(
                    {
                        "success": False,
                        "matched": False,
                        "message": (
                            "Unsupported image format. "
                            "Use JPG, JPEG, PNG, "
                            "WEBP or BMP."
                        ),
                        "results": [],
                    }
                ),
                400,
            )

        # ----------------------------------------------------
        # Save temporary query image
        # ----------------------------------------------------

        safe_original_name = secure_filename(image_file.filename)

        query_filename = f"query_" f"{uuid.uuid4().hex[:12]}_" f"{safe_original_name}"

        query_path = UPLOAD_DIR / query_filename

        image_file.save(str(query_path))

        # ----------------------------------------------------
        # Top K
        # ----------------------------------------------------

        top_k_raw = request.form.get(
            "top_k",
            "8",
        )

        try:
            top_k = int(top_k_raw)

        except Exception:
            top_k = 8

        top_k = max(
            1,
            min(top_k, 20),
        )

        # ----------------------------------------------------
        # SEARCH MODE
        # ----------------------------------------------------

        search_mode = normalize_search_mode(
            request.form.get(
                "search_mode",
                "all",
            )
        )

        search_mode_label = get_search_mode_label(search_mode)

        print("\n[MATCH API]" f" Search mode: {search_mode}" f" ({search_mode_label})")

        # ----------------------------------------------------
        # MATCH JEWELLERY
        # ----------------------------------------------------
        #
        # IMPORTANT:
        #
        # The frontend sends:
        #
        #   all
        #   gold_to_prototype
        #   prototype_to_gold
        #
        # Pass it directly to matcher.py.
        # ----------------------------------------------------

        result = match_jewellery(
            query_path,
            top_k=top_k,
            search_mode=search_mode,
        )

        # ----------------------------------------------------
        # Normalize matcher response
        # ----------------------------------------------------

        if not isinstance(
            result,
            dict,
        ):

            result = {
                "success": False,
                "matched": False,
                "message": ("Matcher returned " "an invalid response."),
                "results": [],
            }

        if "results" not in result:
            result["results"] = []

        if "success" not in result:
            result["success"] = True

        # ----------------------------------------------------
        # Preserve matcher decision
        # ----------------------------------------------------

        if "matched" not in result:

            result["matched"] = bool(result.get("results"))

        # ----------------------------------------------------
        # Similarity compatibility
        # ----------------------------------------------------

        best_similarity = result.get("best_similarity")

        if best_similarity is None:

            best_similarity = result.get("similarity")

        if best_similarity is None:

            best_similarity = result.get(
                "score",
                0.0,
            )

        try:

            best_similarity = float(best_similarity or 0.0)

        except Exception:

            best_similarity = 0.0

        result["best_similarity"] = best_similarity

        if "similarity" not in result:
            result["similarity"] = best_similarity

        if "score" not in result:
            result["score"] = best_similarity

        # ----------------------------------------------------
        # Search mode information
        # ----------------------------------------------------

        result["search_mode"] = search_mode

        result["search_mode_label"] = search_mode_label

        # ----------------------------------------------------
        # Query information
        # ----------------------------------------------------

        result["query_filename"] = query_filename

        if "source_collection" not in result:
            result["source_collection"] = None

        if "target_collection" not in result:
            result["target_collection"] = None

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        print(
            "\n[MATCH API]"
            f" matched={result.get('matched')}"
            f" best_similarity={best_similarity:.4f}"
            f" mode={search_mode}"
            f" source={result.get('source_collection')}"
            f" target={result.get('target_collection')}"
            f" results={len(result.get('results', []))}"
        )

        return jsonify(result)

    except Exception as exc:

        print("\n" + "=" * 70)

        print("[MATCH API] ERROR")

        print("=" * 70)

        traceback.print_exc()

        return (
            jsonify(
                {
                    "success": False,
                    "matched": False,
                    "message": str(exc),
                    "results": [],
                    "similarity": 0.0,
                    "score": 0.0,
                    "best_similarity": 0.0,
                }
            ),
            500,
        )

    finally:

        # ----------------------------------------------------
        # Delete temporary uploaded image
        # ----------------------------------------------------

        if query_path and query_path.exists():

            try:
                query_path.unlink()

            except Exception as exc:

                print(
                    "[MATCH API] Could not remove " "temporary file:",
                    exc,
                )


# ============================================================
# REBUILD INDEX
# ============================================================


@app.route(
    "/api/rebuild-index",
    methods=["POST", "GET"],
)
def rebuild_index():

    try:

        print("\n[INDEX] Manual DINO index rebuild requested.")

        index = build_index(force=True)

        entries = index.get(
            "entries",
            [],
        )

        gold_count = sum(
            1
            for item in entries
            if normalize_collection(item.get("collection")) == "gold"
        )

        prototype_count = sum(
            1
            for item in entries
            if normalize_collection(item.get("collection")) == "prototype"
        )

        return jsonify(
            {
                "success": True,
                "message": ("Visual index rebuilt successfully."),
                "version": index.get("version"),
                "total": len(entries),
                "gold_count": gold_count,
                "prototype_count": prototype_count,
            }
        )

    except Exception as exc:

        print(
            "[INDEX] Rebuild error:",
            exc,
        )

        traceback.print_exc()

        return (
            jsonify(
                {
                    "success": False,
                    "message": str(exc),
                }
            ),
            500,
        )


# ============================================================
# INDEX STATUS
# ============================================================


@app.route(
    "/api/index/status",
    methods=["GET"],
)
def index_status():

    try:

        index = load_index()

        entries = index.get(
            "entries",
            [],
        )

        gold_count = sum(
            1
            for item in entries
            if normalize_collection(item.get("collection")) == "gold"
        )

        prototype_count = sum(
            1
            for item in entries
            if normalize_collection(item.get("collection")) == "prototype"
        )

        return jsonify(
            {
                "success": True,
                "version": index.get("version"),
                "total": len(entries),
                "gold_count": gold_count,
                "prototype_count": prototype_count,
            }
        )

    except Exception as exc:

        print(
            "[INDEX] Status error:",
            exc,
        )

        return (
            jsonify(
                {
                    "success": False,
                    "message": str(exc),
                }
            ),
            500,
        )


# ============================================================
# GENERIC UPLOAD
# ============================================================


@app.route(
    "/api/upload",
    methods=["POST"],
)
def upload():

    try:

        image_file = request.files.get("image") or request.files.get("file")

        if image_file is None:

            return (
                jsonify(
                    {
                        "success": False,
                        "message": ("No image uploaded."),
                    }
                ),
                400,
            )

        if not image_file.filename:

            return (
                jsonify(
                    {
                        "success": False,
                        "message": ("Image filename is empty."),
                    }
                ),
                400,
            )

        if not allowed_file(image_file.filename):

            return (
                jsonify(
                    {
                        "success": False,
                        "message": ("Unsupported image format."),
                    }
                ),
                400,
            )

        filename = generate_unique_filename(image_file.filename)

        path = UPLOAD_DIR / filename

        image_file.save(str(path))

        return jsonify(
            {
                "success": True,
                "filename": filename,
                "path": str(path),
            }
        )

    except Exception as exc:

        print(
            "[UPLOAD] ERROR:",
            exc,
        )

        return (
            jsonify(
                {
                    "success": False,
                    "message": str(exc),
                }
            ),
            500,
        )


# ============================================================
# ERROR HANDLERS
# ============================================================


@app.errorhandler(413)
def file_too_large(error):

    return (
        jsonify(
            {
                "success": False,
                "message": ("Image is too large. " "Maximum upload size is 20 MB."),
            }
        ),
        413,
    )


@app.errorhandler(404)
def not_found(error):

    if request.path.startswith("/api/"):

        return (
            jsonify(
                {
                    "success": False,
                    "message": ("API endpoint not found."),
                }
            ),
            404,
        )

    return (
        "Page not found.",
        404,
    )


# ============================================================
# STARTUP
# ============================================================


def startup():

    print("\n" + "=" * 70)

    print("JEWELMATCH AI BACKEND")

    print("=" * 70)

    print(
        "Project:",
        PROJECT_DIR,
    )

    print(
        "Catalogue:",
        JEWELLERY_JSON,
    )

    print(
        "Gold:",
        GOLD_DIR,
    )

    print(
        "Prototype:",
        PROTOTYPE_DIR,
    )

    print(
        "Uploads:",
        UPLOAD_DIR,
    )

    print(
        "Matching:",
        "DINOv2-Small / Visual Search",
    )

    print(
        "Search modes:",
        "All / Gold → Prototype / Prototype → Gold",
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":

    startup()

    port = int(
        os.getenv(
            "PORT",
            "5000",
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
    )

else:

    startup()
