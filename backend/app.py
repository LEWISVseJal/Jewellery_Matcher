# ============================================================
# JEWELMATCH AI
# FLASK BACKEND
# ============================================================

import os
import json
import uuid

import numpy as np

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
)

from backend.config import (
    UPLOAD_FOLDER,
    JEWELLERY_JSON,
    GOLD_CATALOGUE_DIR,
    PROTOTYPE_CATALOGUE_DIR,
    GOLD_EMBEDDINGS_FILE,
    PROTOTYPE_EMBEDDINGS_FILE,
)

from backend.services.matcher import (
    match_jewellery,
)

from backend.services.embedding import (
    get_embedding,
)

from backend.services.segmentation import (
    segment_jewellery,
)


# ============================================================
# APP
# ============================================================

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
)

app.config["MAX_CONTENT_LENGTH"] = (
    20 * 1024 * 1024
)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    GOLD_CATALOGUE_DIR,
    exist_ok=True
)

os.makedirs(
    PROTOTYPE_CATALOGUE_DIR,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp",
}


def allowed_file(filename):

    if not filename:
        return False

    extension = (
        filename.rsplit(
            ".",
            1
        )[-1]
        .lower()
    )

    return extension in ALLOWED_EXTENSIONS


def load_catalogue():

    if not os.path.exists(
        JEWELLERY_JSON
    ):

        return []


    try:

        with open(
            JEWELLERY_JSON,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)


        if not isinstance(data, list):

            return []


        return data

    except Exception as exc:

        print(
            f"[APP] Failed to load catalogue: {exc}"
        )

        return []


def save_catalogue(data):

    with open(
        JEWELLERY_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


def get_next_jewellery_id(catalogue):

    numbers = []


    for item in catalogue:

        item_id = str(
            item.get("id", "")
        ).strip()


        if item_id.upper().startswith("J"):

            try:

                numbers.append(
                    int(item_id[1:])
                )

            except ValueError:

                pass


    next_number = (
        max(numbers) + 1
        if numbers
        else 1
    )


    return (
        f"J{next_number:03d}"
    )


def get_collection_directory(
    collection
):

    if collection == "gold":

        return GOLD_CATALOGUE_DIR

    if collection == "prototype":

        return PROTOTYPE_CATALOGUE_DIR

    return None


def get_embedding_file(
    collection
):

    if collection == "gold":

        return GOLD_EMBEDDINGS_FILE

    if collection == "prototype":

        return PROTOTYPE_EMBEDDINGS_FILE

    return None


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# ADD JEWELLERY PAGE
# ============================================================

@app.route("/add-jewellery")
def add_jewellery_page():

    return render_template(
        "add_jewellery.html"
    )


# ============================================================
# HEALTH
# ============================================================

@app.route("/api/health")
def health():

    return jsonify({
        "status": "ok"
    })


# ============================================================
# GET CATALOGUE
# ============================================================

@app.route("/api/jewellery")
def get_jewellery():

    return jsonify(
        load_catalogue()
    )


# ============================================================
# SERVE CATALOGUE IMAGE
# ============================================================

@app.route(
    "/catalogue/<collection>/<path:filename>"
)
def serve_catalogue_image(
    collection,
    filename
):

    collection = collection.lower()


    if collection == "gold":

        directory = GOLD_CATALOGUE_DIR

    elif collection == "prototype":

        directory = PROTOTYPE_CATALOGUE_DIR

    else:

        return (
            jsonify({
                "error":
                "Invalid collection."
            }),
            400
        )


    return send_from_directory(
        directory,
        filename
    )


# ============================================================
# SERVE UPLOADED IMAGE
# ============================================================

@app.route(
    "/uploads/<path:filename>"
)
def serve_uploaded_image(
    filename
):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ============================================================
# MATCH JEWELLERY
# ============================================================

@app.route(
    "/api/match",
    methods=["POST"]
)
def match():

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "error":
            "No image was uploaded."
        }), 400


    image = request.files["image"]


    if not image or not image.filename:

        return jsonify({
            "error":
            "Please select an image."
        }), 400


    if not allowed_file(
        image.filename
    ):

        return jsonify({
            "error":
            "Unsupported image format."
        }), 400


    # --------------------------------------------------------
    # Save query image
    # --------------------------------------------------------

    extension = (
        image.filename
        .rsplit(
            ".",
            1
        )[-1]
        .lower()
    )


    query_filename = (
        f"query_"
        f"{uuid.uuid4().hex}."
        f"{extension}"
    )


    query_path = os.path.join(
        UPLOAD_FOLDER,
        query_filename
    )


    image.save(
        query_path
    )


    print(
        "\n========================================"
    )

    print(
        "JEWELLERY CROSS-COLLECTION SEARCH"
    )

    print(
        "========================================"
    )

    print(
        f"Query image: {query_filename}"
    )


    # --------------------------------------------------------
    # Always use AUTO mode.
    #
    # Gold -> Prototype
    # Prototype -> Gold
    # --------------------------------------------------------

    try:

        match_data = match_jewellery(
            query_path,
            target_collection="auto"
        )

    except Exception as exc:

        print(
            "[APP] Matching error:"
        )

        print(exc)


        return jsonify({
            "error":
            "Unable to process the jewellery image.",
            "details":
            str(exc),
        }), 500


    # --------------------------------------------------------
    # Add frontend image URLs
    # --------------------------------------------------------

    results = []


    for item in match_data.get(
        "results",
        []
    ):

        result = dict(item)


        collection = (
            str(
                result.get(
                    "collection",
                    ""
                )
            ).lower()
        )


        filename = result.get(
            "image",
            ""
        )


        if (
            collection in {
                "gold",
                "prototype"
            }
            and filename
        ):

            result["image_url"] = (
                f"/catalogue/"
                f"{collection}/"
                f"{filename}"
            )

        else:

            result["image_url"] = None


        results.append(
            result
        )


    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    response = {

        "match_found":
            match_data.get(
                "match_found",
                False
            ),

        "best_similarity":
            match_data.get(
                "best_similarity",
                0.0
            ),

        "best_similarity_percentage":
            match_data.get(
                "best_similarity_percentage",
                0.0
            ),

        "source_collection":
            match_data.get(
                "source_collection"
            ),

        "target_collection":
            match_data.get(
                "target_collection"
            ),

        "source_similarity":
            match_data.get(
                "source_similarity",
                0.0
            ),

        "results":
            results,
    }


    print(
        f"Source collection: "
        f"{response['source_collection']}"
    )

    print(
        f"Target collection: "
        f"{response['target_collection']}"
    )

    print(
        f"Best target similarity: "
        f"{response['best_similarity']:.4f}"
    )


    return jsonify(
        response
    )


# ============================================================
# ADD JEWELLERY
# ============================================================

@app.route(
    "/api/jewellery/add",
    methods=["POST"]
)
def add_jewellery():

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "error":
            "Jewellery image is required."
        }), 400


    image = request.files["image"]


    if not image or not image.filename:

        return jsonify({
            "error":
            "Jewellery image is required."
        }), 400


    if not allowed_file(
        image.filename
    ):

        return jsonify({
            "error":
            "Unsupported image format."
        }), 400


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    name = (
        request.form.get(
            "name",
            ""
        ).strip()
    )

    gender = (
        request.form.get(
            "gender",
            ""
        ).strip()
    )

    jewellery_type = (
        request.form.get(
            "type",
            ""
        ).strip()
    )

    subtype = (
        request.form.get(
            "subtype",
            ""
        ).strip()
    )

    description = (
        request.form.get(
            "description",
            ""
        ).strip()
    )

    collection = (
        request.form.get(
            "collection",
            ""
        ).strip()
        .lower()
    )


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not name:

        return jsonify({
            "error":
            "Jewellery name is required."
        }), 400


    if collection not in {
        "gold",
        "prototype"
    }:

        return jsonify({
            "error":
            "Please select Gold or Prototype."
        }), 400


    if not gender:

        return jsonify({
            "error":
            "Gender is required."
        }), 400


    if not jewellery_type:

        return jsonify({
            "error":
            "Jewellery type is required."
        }), 400


    # --------------------------------------------------------
    # Catalogue
    # --------------------------------------------------------

    catalogue = load_catalogue()


    jewellery_id = (
        get_next_jewellery_id(
            catalogue
        )
    )


    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    extension = (
        image.filename
        .rsplit(
            ".",
            1
        )[-1]
        .lower()
    )


    filename = (
        f"{jewellery_id}."
        f"{extension}"
    )


    collection_directory = (
        get_collection_directory(
            collection
        )
    )


    image_path = os.path.join(
        collection_directory,
        filename
    )


    image.save(
        image_path
    )


    # --------------------------------------------------------
    # Segment
    # --------------------------------------------------------

    segmented_directory = (
        GOLD_CATALOGUE_DIR
        if collection == "gold"
        else PROTOTYPE_CATALOGUE_DIR
    )


    # Actual segmented directory is handled
    # by segmentation/matcher configuration.
    #
    # We only create the embedding here.

    try:

        from backend.config import (
            GOLD_SEGMENTED_DIR,
            PROTOTYPE_SEGMENTED_DIR,
        )


        segmented_directory = (
            GOLD_SEGMENTED_DIR
            if collection == "gold"
            else PROTOTYPE_SEGMENTED_DIR
        )


        os.makedirs(
            segmented_directory,
            exist_ok=True
        )


        segmented_path = os.path.join(
            segmented_directory,
            f"{jewellery_id}_segmented.jpg"
        )


        segment_jewellery(
            image_path,
            segmented_path
        )


    except Exception as exc:

        print(
            "[APP] Segmentation failed:"
        )

        print(exc)

        segmented_path = image_path


    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    try:

        embedding = get_embedding(
            segmented_path
        )

    except Exception as exc:

        if os.path.exists(
            image_path
        ):

            os.remove(
                image_path
            )


        return jsonify({
            "error":
            "Unable to create jewellery embedding.",
            "details":
            str(exc),
        }), 500


    # --------------------------------------------------------
    # Update embedding file
    # --------------------------------------------------------

    embedding_file = (
        get_embedding_file(
            collection
        )
    )


    try:

        if os.path.exists(
            embedding_file
        ):

            existing_embeddings = (
                np.load(
                    embedding_file
                )
            )


            if existing_embeddings.ndim == 1:

                existing_embeddings = (
                    existing_embeddings.reshape(
                        1,
                        -1
                    )
                )


            updated_embeddings = (
                np.vstack([
                    existing_embeddings,
                    embedding
                ])
            )

        else:

            updated_embeddings = (
                np.array(
                    [embedding],
                    dtype=np.float32
                )
            )


        np.save(
            embedding_file,
            updated_embeddings
        )


    except Exception as exc:

        if os.path.exists(
            image_path
        ):

            os.remove(
                image_path
            )


        return jsonify({
            "error":
            "Unable to update jewellery index.",
            "details":
            str(exc),
        }), 500


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    item = {

        "id":
            jewellery_id,

        "name":
            name,

        "gender":
            gender,

        "type":
            jewellery_type,

        "category":
            jewellery_type,

        "subtype":
            subtype,

        "description":
            description,

        "collection":
            collection,

        "image":
            filename,
    }


    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    try:

        catalogue.append(
            item
        )

        save_catalogue(
            catalogue
        )

    except Exception as exc:

        return jsonify({
            "error":
            "Jewellery image was processed, "
            "but metadata could not be saved.",
            "details":
            str(exc),
        }), 500


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    item["image_url"] = (
        f"/catalogue/"
        f"{collection}/"
        f"{filename}"
    )


    return jsonify({

        "success": True,

        "message":
            "Jewellery added successfully.",

        "item":
            item,
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )