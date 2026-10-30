"""
app.py - Real-Time Indian Sign Language (ISL) Gesture Detection with Sound System.

Key Capabilities:
  - Real-time live camera gesture classification
  - Wrist-relative normalized 63 keypoints for robust distance & position invariance
  - Non-blocking background Text-to-Speech (TTS) audio engine
  - Keyboard Controls:
      * [S] : Toggle Audio Voice ON / OFF
      * [C] : Clear Recognized Sentence History
      * [Q] / [ESC] : Exit Application
  - Temporal consistency smoothing to eliminate flicker
  - Modern, responsive HUD with bounding box and landmark mesh
"""

import os
import sys
import time
import json
from collections import deque
import cv2
import numpy as np

# Ensure workspace-local cache to prevent macOS sandbox / permission blocks
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.cache', 'matplotlib'))
os.environ.setdefault('KERAS_HOME', os.path.join(BASE_DIR, '.cache', 'keras'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)

from function import (
    mediapipe_detection,
    draw_styled_landmarks,
    extract_normalized_keypoints,
    get_hand_bbox,
    mp_hands,
    actions as fallback_actions
)
from sound_engine import sound_player

try:
    from tensorflow.keras.models import model_from_json
except ImportError:
    from keras.models import model_from_json


def load_model_and_labels():
    """Load model architecture, weights, and class labels."""
    model_json_path = os.path.join(BASE_DIR, "model.json")
    model_weights_path = os.path.join(BASE_DIR, "model.h5")
    actions_json_path = os.path.join(BASE_DIR, "actions.json")

    if not os.path.exists(model_json_path) or not os.path.exists(model_weights_path):
        print("=" * 65)
        print("ERROR: Trained model files ('model.json' or 'model.h5') not found!")
        print("Please run 'python train_classifier.py' to train the model first.")
        print("=" * 65)
        return None, None

    print("[App] Loading model architecture and weights...")
    with open(model_json_path, "r") as f:
        model_json = f.read()

    model = model_from_json(model_json)
    model.load_weights(model_weights_path)
    print("[App] Model loaded successfully!")

    # Load trained class labels
    if os.path.exists(actions_json_path):
        with open(actions_json_path, "r") as af:
            actions = json.load(af)
        print(f"[App] Loaded {len(actions)} gesture classes: {actions}")
    else:
        actions = fallback_actions
        print(f"[App] Using fallback classes ({len(actions)}): {actions}")

    return model, actions


def run_app():
    model, actions = load_model_and_labels()
    if model is None:
        return

    # Camera Setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open webcam. Please verify camera permissions.")
        return

    # Configure optimal resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 65)
    print("REAL-TIME GESTURE DETECTION STARTED")
    print("=" * 65)
    print("Controls:")
    print("  [S] : Toggle voice sound feedback ON / OFF")
    print("  [C] : Clear recognized sentence history")
    print("  [Q] : Exit application")
    print("=" * 65)

    recent_predictions = deque(maxlen=7)
    sentence = []
    current_stable_gesture = ""
    current_confidence = 0.0

    threshold = 0.70
    min_stability_count = 5

    prev_frame_time = time.time()
    fps = 0.0

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[App] Failed to read frame from webcam.")
                break

            # Mirror view for natural interaction
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Calculate FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_frame_time, 1e-5))
            prev_frame_time = now

            # Process MediaPipe
            image, results = mediapipe_detection(frame, hands)

            hand_detected = (results.multi_hand_landmarks is not None)

            if hand_detected:
                # Draw skeleton landmarks
                draw_styled_landmarks(frame, results)

                # Get bounding box
                bbox = get_hand_bbox(results, frame.shape)
                if bbox:
                    bx1, by1, bx2, by2 = bbox
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 128), 2)

                # Extract normalized 63 keypoints
                keypoints = extract_normalized_keypoints(results)

                # Predict gesture
                probs = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])
                pred_action = actions[pred_idx]

                recent_predictions.append(pred_action)

                # Temporal smoothing filter
                if len(recent_predictions) >= min_stability_count:
                    # Check most frequent prediction in window
                    counts = {act: recent_predictions.count(act) for act in set(recent_predictions)}
                    most_common, freq = max(counts.items(), key=lambda item: item[1])

                    if freq >= min_stability_count and conf >= threshold:
                        current_stable_gesture = most_common
                        current_confidence = conf

                        # Trigger Voice Sound Output
                        sound_player.speak_gesture(
                            current_stable_gesture,
                            confidence=current_confidence,
                            min_confidence=threshold
                        )

                        # Update sentence history
                        if not sentence or sentence[-1] != current_stable_gesture:
                            sentence.append(current_stable_gesture)
                            if len(sentence) > 8:
                                sentence = sentence[-8:]

                # Draw label directly above hand bounding box
                if bbox and current_stable_gesture:
                    bx1, by1, _, _ = bbox
                    tag = f"{current_stable_gesture.upper()} ({current_confidence * 100:.1f}%)"
                    cv2.putText(
                        frame, tag,
                        (bx1, max(30, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA
                    )

            else:
                recent_predictions.clear()
                current_stable_gesture = ""
                current_confidence = 0.0

            # ----------------- Modern HUD UI Overlay -----------------
            # Top Banner (semi-transparent)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)

            # Bottom Banner
            cv2.rectangle(overlay, (0, h - 55), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Top Banner Contents: Detected Gesture & Sound Status
            if current_stable_gesture:
                gesture_display = f"Gesture: {current_stable_gesture.upper()} [{current_confidence * 100:.1f}%]"
                cv2.putText(frame, gesture_display, (15, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                status_msg = "Hand: ACTIVE (Hold gesture steady)" if hand_detected else "Hand: None (Show hand to camera)"
                cv2.putText(frame, status_msg, (15, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)

            # Sound Toggle Badge
            sound_on = sound_player.enabled
            sound_text = "[SOUND: ON (S)]" if sound_on else "[SOUND: OFF (S)]"
            sound_color = (0, 255, 0) if sound_on else (0, 0, 255)
            cv2.putText(frame, sound_text, (w - 380, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, sound_color, 2, cv2.LINE_AA)

            # FPS Indicator
            cv2.putText(frame, f"{int(fps)} FPS", (w - 95, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

            # Bottom Banner Contents: Sentence History & Shortcuts
            sentence_str = " ".join(sentence) if sentence else "(none)"
            cv2.putText(frame, f"Sentence: {sentence_str}", (15, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            shortcuts_hint = "[S] Voice Toggle  |  [C] Clear  |  [Q] Exit"
            cv2.putText(frame, shortcuts_hint, (w - 470, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            # Render Window
            cv2.imshow("Indian Sign Language (ISL) Real-Time Detector", frame)

            # Key Handling
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s') or key == ord('S'):  # Toggle Sound
                sound_player.toggle()
            elif key == ord('c') or key == ord('C'):  # Clear Sentence
                sentence.clear()
                recent_predictions.clear()

        cap.release()
        cv2.destroyAllWindows()
        sound_player.shutdown()
        print("[App] Application closed gracefully.")


if __name__ == '__main__':
    run_app()
