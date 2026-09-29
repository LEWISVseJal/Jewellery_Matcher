from pathlib import Path

from backend.services.matcher import build_index


def main():
    print()
    print("=" * 70)
    print("JEWELMATCH AI - DINO CATALOGUE INDEX")
    print("=" * 70)

    result = build_index(force=True, verbose=True)

    if not result:
        print()
        print("INDEX BUILD FAILED")
        return

    collections = result["collections"]

    gold_count = sum(
        1 for collection in collections if str(collection).lower() == "gold"
    )

    prototype_count = sum(
        1 for collection in collections if str(collection).lower() == "prototype"
    )

    total_count = len(collections)

    print()
    print("=" * 70)
    print("DINOv2-BASE INDEX CREATED")
    print("=" * 70)
    print(f"Total indexed : {total_count}")
    print(f"Gold indexed  : {gold_count}")
    print(f"Prototype     : {prototype_count}")
    print(f"Embedding dim : {result['embeddings'].shape[1]}")
    print(f"Errors        : {len(result.get('errors', []))}")
    print(f"Index file    : {result['index_file']}")
    print("=" * 70)

    print()
    print("=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
