"""
Dataset Pipeline — Build FAISS index from a new image dataset using FashionCLIP.

Expected dataset structure:
    dataset_dir/
        Shirts/
            image1.jpg
            image2.jpg
        Pants/
            image1.jpg
        Shoes/
            ...

Outputs (written to output_dir, default = project root):
    embeddings.pkl      — numpy array of shape (N, 512)
    img_paths.pkl       — list of N image paths
    flatIndex.index     — FAISS flat L2 index

Usage:
    python pipeline/build_index.py --dataset path/to/dataset
    python pipeline/build_index.py --dataset path/to/dataset --remove-bg --output .
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def load_fclip():
    from fashion_clip.fashion_clip import FashionCLIP
    print("Loading FashionCLIP model...")
    return FashionCLIP('fashion-clip')


def remove_background(image_array):
    from rembg import remove
    pil_img = Image.fromarray(image_array)
    no_bg = remove(pil_img)
    white_bg = Image.new("RGB", no_bg.size, (255, 255, 255))
    white_bg.paste(no_bg, mask=no_bg.split()[3])
    return np.array(white_bg)


def collect_images(dataset_dir):
    """Walk dataset_dir subdirectories (each subdir = one category)."""
    items = []
    for category_dir in sorted(Path(dataset_dir).iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for img_file in sorted(category_dir.iterdir()):
            if img_file.suffix.lower() in SUPPORTED_EXTENSIONS:
                items.append((str(img_file), category))
    return items


def build_index(dataset_dir, output_dir, remove_bg=False, batch_size=32):
    items = collect_images(dataset_dir)
    if not items:
        print(f"No images found in {dataset_dir}")
        sys.exit(1)

    categories = sorted({c for _, c in items})
    print(f"Found {len(items)} images across {len(categories)} categories: {categories}")

    fclip = load_fclip()

    all_embeddings = []
    all_paths = []
    failed = 0

    # Process in batches per category for efficiency
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        pil_images = []
        valid_items = []

        for img_path, category in batch:
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                if remove_bg:
                    img = remove_background(img)
                pil_images.append(Image.fromarray(img))
                valid_items.append(img_path)
            except Exception as e:
                print(f"\nSkipping {img_path}: {e}")
                failed += 1

        if not pil_images:
            continue

        embeddings = fclip.encode_images(pil_images, batch_size=batch_size)
        all_embeddings.append(embeddings)
        all_paths.extend(valid_items)

        print(f"\r  Processed {min(i + batch_size, len(items))}/{len(items)} images", end="")

    print(f"\nFailed: {failed}")

    embeddings_array = np.vstack(all_embeddings).astype('float32')
    print(f"Embeddings shape: {embeddings_array.shape}")

    # Build FAISS flat index
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_array)

    # Save outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings_array, f)
    with open(out / "img_paths.pkl", "wb") as f:
        pickle.dump(all_paths, f)
    faiss.write_index(index, str(out / "flatIndex.index"))

    print(f"\nSaved to {output_dir}:")
    print(f"  embeddings.pkl  ({embeddings_array.shape[0]} vectors, dim={dim})")
    print(f"  img_paths.pkl")
    print(f"  flatIndex.index")
    print("\nDone. Restart the app to load the new index.")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index using FashionCLIP")
    parser.add_argument("--dataset", required=True,
                        help="Dataset directory (subdirs = category names)")
    parser.add_argument("--output", default=".",
                        help="Output directory (default: project root)")
    parser.add_argument("--remove-bg", action="store_true",
                        help="Remove background from each image before embedding")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Images per FashionCLIP batch (default: 32)")
    args = parser.parse_args()

    build_index(args.dataset, args.output, args.remove_bg, args.batch_size)


if __name__ == "__main__":
    main()
