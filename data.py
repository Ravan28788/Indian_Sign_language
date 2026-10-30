import os
import glob
from function import (
    DATA_PATH, IMAGE_PATH, DATASET_PATH,
    actions, ACTIONS_ALPHABETS, ACTIONS_COMMANDS_AND_DIGITS, ACTIONS_ALL,
    no_sequences, sequence_length, images_per_action
)


def verify_dataset(action_set=None):
    """Verify collected dataset in IMAGE_PATH, DATASET_PATH, and DATA_PATH."""
    if action_set is None:
        action_set = actions

    print("=" * 65)
    print("           ISL DATASET STATUS & VALIDATION REPORT")
    print("=" * 65)
    print(f"Target Action Set: {len(action_set)} classes")
    print(f"Classes: {action_set}")
    print("-" * 65)

    # 1. Verify Image Dataset
    print(f"1. IMAGE DIRECTORY STATUS ('{IMAGE_PATH}'):")
    total_found_images = 0
    missing_classes = []

    if not os.path.exists(IMAGE_PATH):
        print(f"   [Error] Directory '{IMAGE_PATH}' does not exist.")
    else:
        for action in action_set:
            action_dir = os.path.join(IMAGE_PATH, action)
            if not os.path.exists(action_dir):
                missing_classes.append(action)
                continue
            valid_imgs = [
                f for f in os.listdir(action_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            total_found_images += len(valid_imgs)

        print(f"   Classes Present: {len(action_set) - len(missing_classes)}/{len(action_set)}")
        print(f"   Total Sample Images for Active Set: {total_found_images:,}")
        if missing_classes:
            print(f"   Missing Classes: {missing_classes}")

    print("-" * 65)

    # 2. Verify Structured Dataset Directory
    print(f"2. ORGANIZED DATASET DIRECTORY ('{DATASET_PATH}'):")
    if os.path.exists(DATASET_PATH):
        for sub in ['alphabets', 'digits', 'commands']:
            sub_path = os.path.join(DATASET_PATH, sub)
            if os.path.exists(sub_path):
                sub_cats = [d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d))]
                print(f"   - {sub.capitalize():<10}: {len(sub_cats)} categories linked")
    else:
        print("   Notice: 'dataset/' organized view not found. Run 'python organize_dataset.py'.")

    print("-" * 65)

    # 3. Verify Keypoint Dataset
    print(f"3. KEYPOINT DATASET ('{DATA_PATH}'):")
    total_kp_files = 0
    if os.path.exists(DATA_PATH):
        for action in action_set:
            action_path = os.path.join(DATA_PATH, action)
            if os.path.exists(action_path):
                npy_files = glob.glob(os.path.join(action_path, '**', '*.npy'), recursive=True)
                total_kp_files += len(npy_files)
        print(f"   Total Extracted Keypoint Files: {total_kp_files:,}")
    else:
        print(f"   Directory '{DATA_PATH}' does not exist yet.")

    print("=" * 65)
    return total_found_images > 0 or total_kp_files > 0


if __name__ == '__main__':
    verify_dataset()
