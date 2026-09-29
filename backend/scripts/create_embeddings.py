"""
Legacy compatibility script.

JewelMatch AI now uses one catalogue index:

    backend/database/dino_index.npz

The old lightweight_index.npz workflow is no longer used
by Matcher V4.

This script now redirects to the active DINO index builder.

Run from project root:

    python -m backend.scripts.create_embeddings
"""

from backend.services.matcher import build_index


def main():

    print()
    print("=" * 70)
    print("JEWELMATCH AI - ACTIVE DINO INDEX BUILDER")
    print("=" * 70)

    print("The old lightweight embedding index is no longer " "used by Matcher V4.")

    print()
    print("Building the active DINO catalogue index...")

    result = build_index(
        force=True,
        verbose=True,
    )

    print()
    print("=" * 70)
    print("DINO INDEX BUILD COMPLETE")
    print("=" * 70)

    print(f"Gold indexed      : " f"{len(result['collections']['gold'])}")

    print(f"Prototype indexed : " f"{len(result['collections']['prototype'])}")

    print(f"Errors            : " f"{len(result.get('errors', []))}")

    print()


if __name__ == "__main__":
    main()
