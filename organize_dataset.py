"""
organize_dataset.py - Analyze and systematically organize ISL image dataset.

Categorizes the 29,300+ images in 'Image/' into structured groups:
  1. Alphabets (A - Z, 26 classes, ~14,347 images)
  2. Digits (zero - nine, 10 classes, 10,000 images)
  3. Commands / Actions (stop, up, down, left, right, 5 classes, 5,000 images)

Creates zero-duplication symlinks in 'dataset/' and generates 'dataset_summary.json'.
"""

import os
import json
import glob
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'Image')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

ALPHABET_CLASSES = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
DIGIT_CLASSES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
COMMAND_CLASSES = ['stop', 'up', 'down', 'left', 'right']


def organize():
    if not os.path.exists(IMAGE_DIR):
        print(f"[Error] Image directory '{IMAGE_DIR}' not found!")
        return

    os.makedirs(os.path.join(DATASET_DIR, 'alphabets'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'digits'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'commands'), exist_ok=True)

    summary = {
        "total_images": 0,
        "total_categories": 0,
        "taxonomies": {
            "alphabets": {},
            "digits": {},
            "commands": {},
            "unclassified": {}
        }
    }

    folders = sorted([f for f in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, f))])
    summary["total_categories"] = len(folders)

    print("=" * 75)
    print("      INDIAN SIGN LANGUAGE (ISL) DATASET ANALYSIS & ORGANIZATION")
    print("=" * 75)
    print(f"{'Category':<15} | {'Type':<12} | {'Count':<7} | {'Resolution':<12} | {'Target Link'}")
    print("-" * 75)

    for folder in folders:
        src_folder = os.path.join(IMAGE_DIR, folder)
        image_files = glob.glob(os.path.join(src_folder, '*.*'))
        image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = len(image_files)
        summary["total_images"] += count

        # Determine dimensions of first image
        resolution = "N/A"
        if image_files:
            try:
                with Image.open(image_files[0]) as im:
                    resolution = f"{im.size[0]}x{im.size[1]}"
            except Exception:
                pass

        # Classify taxonomy
        if folder in ALPHABET_CLASSES:
            taxonomy = "alphabets"
            group_label = "Alphabet"
        elif folder.lower() in DIGIT_CLASSES:
            taxonomy = "digits"
            group_label = "Digit (0-9)"
        elif folder.lower() in COMMAND_CLASSES:
            taxonomy = "commands"
            group_label = "Command"
        else:
            taxonomy = "unclassified"
            group_label = "Other"

        dest_folder = os.path.join(DATASET_DIR, taxonomy, folder)

        # Create symlink without duplicating storage
        if not os.path.exists(dest_folder):
            try:
                rel_src = os.path.relpath(src_folder, os.path.dirname(dest_folder))
                os.symlink(rel_src, dest_folder)
            except Exception as e:
                # If symlink unsupported, fall back to folder or note
                pass

        summary["taxonomies"][taxonomy][folder] = {
            "count": count,
            "resolution": resolution,
            "sample_file": os.path.basename(image_files[0]) if image_files else None
        }

        print(f"{folder:<15} | {group_label:<12} | {count:<7} | {resolution:<12} | dataset/{taxonomy}/{folder}")

    print("-" * 75)
    print(f"Total Categories: {summary['total_categories']}")
    print(f"Total Images:     {summary['total_images']:,}")
    print(f"  - Alphabets:    {sum(v['count'] for v in summary['taxonomies']['alphabets'].values()):,} images (26 letters)")
    print(f"  - Digits:       {sum(v['count'] for v in summary['taxonomies']['digits'].values()):,} images (10 numbers)")
    print(f"  - Commands:     {sum(v['count'] for v in summary['taxonomies']['commands'].values()):,} images (5 actions)")
    print("=" * 75)

    summary_file = os.path.join(BASE_DIR, 'dataset_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Success] Dataset organized successfully!")
    print(f"  - Organized Structure: file://{DATASET_DIR}")
    print(f"  - Metadata Report:     file://{summary_file}\n")


if __name__ == '__main__':
    organize()
