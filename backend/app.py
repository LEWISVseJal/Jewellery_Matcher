"""
============================================================
JEWELLERY AI MATCHING
Flask Backend + Frontend
============================================================

Routes:

GET  /
GET  /api/health
POST /api/match
GET  /catalogue/<filename>

The Flask server also serves the frontend.
============================================================
"""

import os
import uuid

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    render_template
)

from flask_cors import CORS

from config import (
    UPLOAD_FOLDER,
    CATALOGUE_DIR
)

from services.matcher import (
    find_matches,
    is_jewellery_image
)


# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BACKEND_DIR
)

FRONTEND_TEMPLATE_DIR = os.path.join(
    PROJECT_DIR,
    "frontend",
    "templates"
)

FRONTEND_STATIC_DIR = os.path.join(
    PROJECT_DIR,
    "frontend",
    "static"
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=FRONTEND_TEMPLATE_DIR,
    static_folder=FRONTEND_STATIC_DIR,
    static_url_path="/static"
)

CORS(app)


# ============================================================
# CREATE UPLOAD FOLDER
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# FRONTEND
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "success",

        "message":
            "Jewellery Matching API is running"

    })


# ============================================================
# JEWELLERY MATCHING
# ============================================================

@app.route(
    "/api/match",
    methods=["POST"]
)
def match_jewellery():

    # --------------------------------------------------------
    # Check uploaded image
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({

            "status":
                "error",

            "message":
                "No image uploaded"

        }), 400


    uploaded_file = request.files[
        "image"
    ]


    if uploaded_file.filename == "":

        return jsonify({

            "status":
                "error",

            "message":
                "No image selected"

        }), 400


    # --------------------------------------------------------
    # Get file extension
    # --------------------------------------------------------

    extension = os.path.splitext(
        uploaded_file.filename
    )[1].lower()


    # --------------------------------------------------------
    # Generate unique filename
    # --------------------------------------------------------

    filename = (
        str(uuid.uuid4())
        + extension
    )


    upload_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    uploaded_file.save(
        upload_path
    )


    try:

        # ====================================================
        # STEP 1
        # CHECK WHETHER IMAGE IS JEWELLERY
        # ====================================================

        jewellery_detected, validation_score = (
            is_jewellery_image(
                upload_path
            )
        )


        print()
        print(
            "Jewellery validation score:",
            round(validation_score, 4)
        )


        # ====================================================
        # NON-JEWELLERY
        # ====================================================

        if not jewellery_detected:

            print(
                "Non-jewellery image detected."
            )


            return jsonify({

                "status":
                    "success",

                "match_found":
                    False,

                "jewellery_detected":
                    False,

                "message":
                    "No jewellery detected. No match present.",

                "results":
                    []

            })


        # ====================================================
        # STEP 2
        # FIND JEWELLERY MATCHES
        # ====================================================

        print(
            "Jewellery image detected."
        )


        results = find_matches(
            upload_path
        )


        # ====================================================
        # CHECK FINAL MATCH
        # ====================================================

        match_found = (

            len(results) > 0

            and

            results[0]["match"]

        )


        if match_found:

            message = (
                "Matching jewellery found."
            )

        else:

            message = (
                "No matching jewellery found."
            )


        return jsonify({

            "status":
                "success",

            "match_found":
                match_found,

            "jewellery_detected":
                True,

            "message":
                message,

            "results":
                results

        })


    except Exception as error:

        print()
        print(
            "Matching error:",
            error
        )


        return jsonify({

            "status":
                "error",

            "message":
                str(error)

        }), 500


# ============================================================
# CATALOGUE IMAGE
# ============================================================

@app.route(
    "/catalogue/<path:filename>"
)
def catalogue_image(filename):

    return send_from_directory(
        CATALOGUE_DIR,
        filename
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(
        "JEWELLERY AI MATCHING APPLICATION"
    )
    print("=" * 60)

    print()
    print(
        "Application:"
    )

    print(
        "http://127.0.0.1:5000/"
    )

    print()
    print(
        "Health:"
    )

    print(
        "http://127.0.0.1:5000/api/health"
    )

    print("=" * 60)


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )