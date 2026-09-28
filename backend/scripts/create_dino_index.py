"""
Rebuild the complete DINOv2 catalogue index.

Run from project root:

python -m backend.scripts.create_dino_index
"""

from backend.services.matcher import build_index

if __name__ == "__main__":

    print("=" * 70)
    print("JEWELMATCH AI - DINO CATALOGUE INDEX")
    print("=" * 70)

    build_index(force=True)

    print("=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)
