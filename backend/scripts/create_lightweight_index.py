"""
Build JewelMatch AI lightweight visual index.

Run from project root:

    python backend/scripts/create_lightweight_index.py

or:

    python -m backend.scripts.create_lightweight_index
"""

from pathlib import Path
import sys

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# IMPORT
# ============================================================

from backend.services.matcher import build_index

# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 70)
    print("JEWELMATCH AI - LIGHTWEIGHT INDEX BUILDER")
    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")

    index = build_index(verbose=True)

    gold_count = len(index["collections"]["gold"])

    prototype_count = len(index["collections"]["prototype"])

    errors = index.get(
        "errors",
        [],
    )

    print()
    print("=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)

    print(f"Gold indexed      : {gold_count}")

    print(f"Prototype indexed : {prototype_count}")

    print(f"Errors             : {len(errors)}")

    if errors:

        print()
        print("INDEX ERRORS:")

        for error in errors:

            print(f" - {error}")

    print()
    print("The visual search index is ready.")

    print()


if __name__ == "__main__":
    main()
