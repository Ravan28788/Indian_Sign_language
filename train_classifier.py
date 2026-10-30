"""
train_classifier.py - High-precision Keypoint Extraction and Gesture Classifier Trainer.

Supports:
  - Static Image datasets (.jpg, .jpeg, .png)
  - Video datasets (.avi, .mp4) - automatically extracts frames
  - Dynamic discovery of custom user-recorded gestures
  - 60+ FPS real-time wrist-normalized keypoint Neural Network
"""

import os
import json
import glob
import time
import numpy as np

# Cache environment config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.cache', 'matplotlib'))
os.environ.setdefault('KERAS_HOME', os.path.join(BASE_DIR, '.cache', 'keras'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)

import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, BatchNormalization
    from keras.utils import to_categorical
    from keras.callbacks import EarlyStopping

from function import extract_normalized_keypoints, IMAGE_PATH


def discover_available_classes():
    """Discover all folders in Image/ containing either images or videos."""
    if not os.path.exists(IMAGE_PATH):
        return []
    
    classes = []
    # Ignore alphabets if they have low detectability unless requested
    for d in sorted(os.listdir(IMAGE_PATH)):
        full_p = os.path.join(IMAGE_PATH, d)
        if not os.path.isdir(full_p):
            continue
        
        has_imgs = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(full_p))
        has_videos = any(f.lower().endswith(('.avi', '.mp4')) for f in os.listdir(full_p))
        if has_imgs or has_videos:
            classes.append(d)
            
    return classes


def extract_keypoints_from_folder(folder_path, hands, max_samples=150):
    """Extract keypoints from a folder containing images OR avi/mp4 videos."""
    X_samples = []
    
    # 1. Check for video files (.avi, .mp4)
    video_files = glob.glob(os.path.join(folder_path, '*.avi')) + glob.glob(os.path.join(folder_path, '*.mp4'))
    
    if video_files:
        # Sample frames across video files
        frames_per_video = max(3, max_samples // max(1, len(video_files)))
        for vpath in video_files:
            if len(X_samples) >= max_samples:
                break
            cap = cv2.VideoCapture(vpath)
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_f <= 0:
                cap.release()
                continue
            
            # Sample evenly spaced frames
            step = max(1, total_f // (frames_per_video * 2))
            frame_idx = 0
            while cap.isOpened() and len(X_samples) < max_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = hands.process(rgb)
                    if res.multi_hand_landmarks:
                        kp = extract_normalized_keypoints(res)
                        if np.any(kp):
                            X_samples.append(kp)
                frame_idx += 1
            cap.release()
            
        return X_samples

    # 2. Check for image files (.jpg, .jpeg, .png)
    image_files = glob.glob(os.path.join(folder_path, '*.*'))
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for ipath in image_files:
        if len(X_samples) >= max_samples:
            break
        img = cv2.imread(ipath)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            kp = extract_normalized_keypoints(res)
            if np.any(kp):
                X_samples.append(kp)
                
    return X_samples


def build_and_train(classes=None, samples_per_class=120, epochs=45):
    """Trains neural network on keypoints from images and videos."""
    if classes is None:
        # Standard default: digits + commands + any video/custom classes
        all_discovered = discover_available_classes()
        # Filter out individual raw letters by default unless only letters exist
        candidates = [c for c in all_discovered if len(c) > 1 or c in ['0', '1', '2']]
        classes = candidates if candidates else all_discovered

    print("=" * 65)
    print(f"TRAINING GESTURE CLASSIFIER ({len(classes)} Classes)")
    print(f"Target Classes: {classes}")
    print("=" * 65)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.25
    )

    X_list, y_list = [], []
    valid_classes = []
    start_time = time.time()

    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(IMAGE_PATH, class_name)
        if not os.path.exists(class_dir):
            continue

        samples = extract_keypoints_from_folder(class_dir, hands, max_samples=samples_per_class)
        if len(samples) >= 15:
            class_idx = len(valid_classes)
            valid_classes.append(class_name)
            for s in samples:
                X_list.append(s)
                y_list.append(class_idx)
            print(f"  [{class_name:<16}] Extracted {len(samples)} valid keypoint samples")
        else:
            print(f"  [Skipped {class_name:<10}] Only {len(samples)} valid hand detections")

    hands.close()
    elapsed = time.time() - start_time
    print(f"\nKeypoint extraction completed in {elapsed:.1f}s.")
    print(f"Successfully loaded {len(valid_classes)} classes with {len(X_list)} total samples.")

    if len(valid_classes) < 2 or len(X_list) < 30:
        print("[Error] Not enough classes or samples to train a model.")
        return False

    X = np.array(X_list, dtype=np.float32)
    y_indices = np.array(y_list, dtype=np.int32)
    y = to_categorical(y_indices, num_classes=len(valid_classes))

    # Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y_indices
    )

    print("\nBuilding and training Neural Network...")
    input_dim = 63  # 21 landmarks x 3
    num_classes = len(valid_classes)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
    print("=" * 65)
    print(f"TRAINING SUCCESS! Validation Accuracy: {val_acc * 100:.2f}%")
    print("=" * 65)

    # Save model architecture & weights
    model_json = model.to_json()
    with open(os.path.join(BASE_DIR, "model.json"), "w") as f:
        f.write(model_json)
    model.save(os.path.join(BASE_DIR, "model.h5"))

    # Save active classes
    clean_labels = [c.replace("_Cropped", "") for c in valid_classes]
    with open(os.path.join(BASE_DIR, "actions.json"), "w") as af:
        json.dump(clean_labels, af, indent=2)

    print(f"[Success] Saved 'model.json', 'model.h5', and 'actions.json'")
    return True


if __name__ == '__main__':
    build_and_train()
