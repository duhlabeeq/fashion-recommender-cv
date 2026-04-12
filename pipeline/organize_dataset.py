"""
Organise the Myntra/Kaggle flat dataset into category subfolders
so pipeline/build_index.py can process it.

Usage:
    python pipeline/organize_dataset.py \
        --csv    archive/styles.csv \
        --images archive/images \
        --output fashion-dataset
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

# Map Kaggle articleType → our category folder name (must match COMPLEMENTS keys)
CATEGORY_MAP = {
    'Tshirts':               'Shirts & Tops',
    'Shirts':                'Shirts & Tops',
    'Tops':                  'Shirts & Tops',
    'Kurtas':                'Shirts & Tops',
    'Blouses':               'Shirts & Tops',
    'Tunics':                'Shirts & Tops',
    'Jeans':                 'Pants',
    'Trousers':              'Pants',
    'Track Pants':           'Pants',
    'Shorts':                'Shorts',
    'Skirts':                'Skirts',
    'Dresses':               'Dresses',
    'Jumpsuit':              'Jumpsuits',
    'Casual Shoes':          'Shoes',
    'Sports Shoes':          'Shoes',
    'Formal Shoes':          'Shoes',
    'Heels':                 'Shoes',
    'Flip Flops':            'Shoes',
    'Sandals':               'Shoes',
    'Handbags':              'Handbags',
    'Clutches':              'Handbags',
    'Backpacks':             'Handbags',
    'Wallets':               'Handbags',
    'Belts':                 'Belts',
    'Watches':               'Watches',
    'Sunglasses':            'Sunglasses',
    'Scarves':               'Scarves & Shawls',
    'Stoles':                'Scarves & Shawls',
    'Caps':                  'Hats',
    'Hats':                  'Hats',
    'Jewellery Set':         'Jewelry',
    'Earrings':              'Jewelry',
    'Necklace and Chains':   'Jewelry',
    'Bracelet':              'Jewelry',
    'Ring':                  'Jewelry',
}


def organize(csv_path, images_dir, output_dir):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    counts = {}
    skipped = 0

    with open(csv_path, encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Processing {len(rows)} entries...")

    for row in rows:
        article_type = row.get('articleType', '').strip()
        img_id = row.get('id', '').strip()

        category = CATEGORY_MAP.get(article_type)
        if not category:
            skipped += 1
            continue

        src = images_dir / f"{img_id}.jpg"
        if not src.exists():
            skipped += 1
            continue

        dest_dir = output_dir / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / f"{img_id}.jpg")
        counts[category] = counts.get(category, 0) + 1

    print(f"\nOrganised images:")
    for cat, n in sorted(counts.items()):
        print(f"  {n:5d}  {cat}")
    print(f"\n  Skipped: {skipped}")
    print(f"  Output:  {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',    default='archive/styles.csv')
    parser.add_argument('--images', default='archive/images')
    parser.add_argument('--output', default='fashion-dataset')
    args = parser.parse_args()
    organize(args.csv, args.images, args.output)


if __name__ == '__main__':
    main()
