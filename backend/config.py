import os


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)


# ============================================================
# DATABASE
# ============================================================

DATABASE_DIR = os.path.join(
    BASE_DIR,
    "database"
)

JEWELLERY_JSON = os.path.join(
    DATABASE_DIR,
    "jewellery.json"
)


# ============================================================
# CATALOGUE DIRECTORIES
# ============================================================

CATALOGUE_DIR = os.path.join(
    BASE_DIR,
    "catalogue"
)

GOLD_CATALOGUE_DIR = os.path.join(
    CATALOGUE_DIR,
    "gold"
)

PROTOTYPE_CATALOGUE_DIR = os.path.join(
    CATALOGUE_DIR,
    "prototype"
)


# ============================================================
# EMBEDDING FILES
# ============================================================

GOLD_EMBEDDINGS_FILE = os.path.join(
    DATABASE_DIR,
    "gold_embeddings.npy"
)

PROTOTYPE_EMBEDDINGS_FILE = os.path.join(
    DATABASE_DIR,
    "prototype_embeddings.npy"
)


# ============================================================
# SEGMENTED CATALOGUE
# ============================================================

SEGMENTED_CATALOGUE_DIR = os.path.join(
    DATABASE_DIR,
    "segmented_catalogue"
)

GOLD_SEGMENTED_DIR = os.path.join(
    SEGMENTED_CATALOGUE_DIR,
    "gold"
)

PROTOTYPE_SEGMENTED_DIR = os.path.join(
    SEGMENTED_CATALOGUE_DIR,
    "prototype"
)


# ============================================================
# UPLOADS
# ============================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)


# ============================================================
# MATCHING SETTINGS
# ============================================================

TOP_K = 5

# Temporary starting threshold.
# We will tune this using real Gold/Prototype pairs.
MATCH_THRESHOLD = 0.15


# ============================================================
# ALLOWED IMAGE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp"
}


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

directories = [
    DATABASE_DIR,
    CATALOGUE_DIR,
    GOLD_CATALOGUE_DIR,
    PROTOTYPE_CATALOGUE_DIR,
    SEGMENTED_CATALOGUE_DIR,
    GOLD_SEGMENTED_DIR,
    PROTOTYPE_SEGMENTED_DIR,
    UPLOAD_FOLDER,
]

for directory in directories:
    os.makedirs(
        directory,
        exist_ok=True
    )