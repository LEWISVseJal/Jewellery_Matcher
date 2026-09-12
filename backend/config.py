"""
Configuration file for the Jewellery Matching AI project.

Keeping paths in one place makes the project easier to maintain.
"""

import os


# ---------------------------------------------------------
# BACKEND DIRECTORY
# ---------------------------------------------------------

# Folder containing this config.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------
# CATALOGUE
# ---------------------------------------------------------

# Folder containing all jewellery catalogue images
CATALOGUE_DIR = os.path.join(
    BASE_DIR,
    "catalogue"
)


# ---------------------------------------------------------
# DATABASE
# ---------------------------------------------------------

# JSON file containing jewellery information
JEWELLERY_JSON = os.path.join(
    BASE_DIR,
    "database",
    "jewellery.json"
)


# Numpy file containing catalogue embeddings
EMBEDDINGS_FILE = os.path.join(
    BASE_DIR,
    "database",
    "embeddings.npy"
)


# ---------------------------------------------------------
# UPLOADS
# ---------------------------------------------------------

# Folder where uploaded query images will temporarily be stored
UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)


# ---------------------------------------------------------
# MATCHING SETTINGS
# ---------------------------------------------------------

# Number of results returned to the frontend
TOP_K = 5


# Minimum similarity required to consider something a match.
#
# We will tune this later after testing with real jewellery.
MATCH_THRESHOLD = 0.60