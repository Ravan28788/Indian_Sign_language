import os
import cv2
import numpy as np

# Ensure cache directories are inside workspace to avoid permission issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.cache', 'matplotlib'))
os.environ.setdefault('KERAS_HOME', os.path.join(BASE_DIR, '.cache', 'keras'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

DATA_PATH = os.path.join(BASE_DIR, 'MP_Data')
IMAGE_PATH = os.path.join(BASE_DIR, 'Image')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')

# Standard Action Sets
ACTIONS_COMMANDS_AND_DIGITS = [
    'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine',
    'stop', 'up', 'down', 'left', 'right'
]

ACTIONS_ALPHABETS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]

ACTIONS_ALL = ACTIONS_ALPHABETS + ACTIONS_COMMANDS_AND_DIGITS

# Default active actions for detection/training
# Can be changed to ACTIONS_ALL or ACTIONS_ALPHABETS
actions = ACTIONS_COMMANDS_AND_DIGITS

no_sequences = 30
sequence_length = 30
images_per_action = 100


def mediapipe_detection(image, model):
    """Processes image through MediaPipe Hands model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    """Draws styled landmarks on the frame."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )


draw_styles_landmarks = draw_styled_landmarks


def extract_keypoints(results):
    """Extract raw 63 keypoints (21 landmarks x 3 coords: x, y, z)."""
    if results.multi_hand_landmarks:
        rh = np.array([
            [res.x, res.y, res.z]
            for res in results.multi_hand_landmarks[0].landmark
        ]).flatten()
        return rh
    return np.zeros(21 * 3)


def extract_normalized_keypoints(results):
    """
    Extract wrist-relative, scale-normalized 63 keypoints.
    Makes detection translation-invariant and distance-invariant.
    """
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        wrist = landmarks[0]
        wx, wy, wz = wrist.x, wrist.y, wrist.z

        # Center on wrist
        coords = np.array([
            [lm.x - wx, lm.y - wy, lm.z - wz]
            for lm in landmarks
        ])

        # Normalize scale by maximum distance from wrist
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist > 1e-6:
            coords = coords / max_dist

        return coords.flatten()

    return np.zeros(21 * 3)


def get_hand_bbox(results, frame_shape):
    """Calculate tight bounding box around detected hand for visual indicator."""
    if not results.multi_hand_landmarks:
        return None
    h, w = frame_shape[:2]
    landmarks = results.multi_hand_landmarks[0].landmark
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x_min, x_max = max(0, int(min(xs)) - 20), min(w, int(max(xs)) + 20)
    y_min, y_max = max(0, int(min(ys)) - 20), min(h, int(max(ys)) + 20)
    return x_min, y_min, x_max, y_max


if __name__ == '__main__':
    print("function.py loaded and verified successfully!")
    print(f"Data Path: {DATA_PATH}")
    print(f"Image Path: {IMAGE_PATH}")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Active Actions ({len(actions)}): {actions}")
    print(f"Available Action Sets:")
    print(f"  - Commands & Digits: {len(ACTIONS_COMMANDS_AND_DIGITS)} classes")
    print(f"  - Alphabets (A-Z):   {len(ACTIONS_ALPHABETS)} classes")
    print(f"  - All Combined:      {len(ACTIONS_ALL)} classes")