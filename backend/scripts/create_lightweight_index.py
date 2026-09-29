"""
Compatibility wrapper for the JewelMatch AI catalogue index.

The active project index is:

    backend/database/dino_index.npz

This file is retained so older commands do not break.

Run from project root:

    python -m backend.scripts.create_lightweight_index
"""

from backend.services.matcher import build_index


def main():

    print()
    print("=" * 70)
    print("JEWELMATCH AI - CATALOGUE INDEX")
    print("=" * 70)

    result = build_index(
        force=True,
        verbose=True,
    )

    print()
    print("=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)

    print(f"Gold indexed      : " f"{len(result['collections']['gold'])}")

    print(f"Prototype indexed : " f"{len(result['collections']['prototype'])}")

    print(f"Errors            : " f"{len(result.get('errors', []))}")

    print()


if __name__ == "__main__":
    main()
