"""
collectdata.py - Interactive Camera Gesture Collector with Auto-Folder Creation & Immediate Training.

Workflow:
  1. Asks for gesture name in terminal (e.g., 'water', 'hello', 'doctor')
  2. Automatically creates the target directory in 'Image/<gesture>/'
  3. Opens camera with centered ergonomic capture box
     - Press [SPACE] to capture individual photos
     - Press [R] to auto-record a continuous burst of 40 photos (hands-free!)
     - Press [Q] when finished
  4. Asks in terminal: 'Do you want to train the model now? (y/n): '
     - If 'y': automatically trains and updates model.h5 / actions.json!
"""

import os
import sys
import time
import cv2
import numpy as np

# Cache environment config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.cache', 'matplotlib'))
os.environ.setdefault('KERAS_HOME', os.path.join(BASE_DIR, '.cache', 'keras'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)

from function import IMAGE_PATH, DATASET_PATH, mediapipe_detection, draw_styled_landmarks, mp_hands
from sound_engine import sound_player
from train_classifier import build_and_train


def get_image_count(folder_path):
    """Return number of valid image files in folder."""
    if not os.path.exists(folder_path):
        return 0
    return len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])


def run_collector():
    print("=" * 65)
    print("      ISL REAL-TIME GESTURE COLLECTOR & AUTO-TRAINER")
    print("=" * 65)

    # 1. Ask for gesture name
    try:
        gesture_input = input("Enter the name of the gesture to record (e.g., water, hello, doctor): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return

    if not gesture_input:
        gesture_input = "custom_gesture"

    # Sanitize folder name
    gesture_name = "".join(c for c in gesture_input if c.isalnum() or c in ('_', '-')).lower()
    target_dir = os.path.join(IMAGE_PATH, gesture_name)
    os.makedirs(target_dir, exist_ok=True)

    # Link into dataset/custom
    custom_dataset_dir = os.path.join(DATASET_PATH, 'custom')
    os.makedirs(custom_dataset_dir, exist_ok=True)
    link_path = os.path.join(custom_dataset_dir, gesture_name)
    if not os.path.exists(link_path):
        try:
            rel = os.path.relpath(target_dir, custom_dataset_dir)
            os.symlink(rel, link_path)
        except Exception:
            pass

    print(f"\n[Directory Ready] Saving samples into: {target_dir}")
    print("Launching camera...")
    print("Controls in camera window:")
    print("  [SPACE]     : Snap a single image")
    print("  [R]         : Auto-record a burst of 40 images (with 3-sec countdown)")
    print("  [Q] / [ESC] : Save and proceed to training option")
    print("=" * 65)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    feedback_text = "Press SPACE to snap or 'R' for auto-burst"
    feedback_color = (255, 255, 255)
    feedback_frames = 0
    burst_mode = False
    burst_target = 40
    burst_collected = 0
    countdown_start = 0

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Ergonomic center ROI box (360 x 360)
            box_size = 360
            cx, cy = w // 2, h // 2
            x1, y1 = cx - box_size // 2, cy - box_size // 2
            x2, y2 = cx + box_size // 2, cy + box_size // 2

            cropframe = frame[y1:y2, x1:x2]
            clean_crop = cropframe.copy()

            _, results = mediapipe_detection(cropframe, hands)
            draw_styled_landmarks(cropframe, results)

            hand_detected = (results.multi_hand_landmarks is not None)
            box_color = (0, 255, 0) if hand_detected else (0, 140, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            current_count = get_image_count(target_dir)

            # Auto-Burst Logic
            now = time.time()
            if countdown_start > 0:
                elapsed = now - countdown_start
                remaining = int(3.5 - elapsed)
                if remaining > 0:
                    cv2.putText(frame, f"GET READY IN: {remaining}", (cx - 160, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    countdown_start = 0
                    burst_mode = True
                    burst_collected = 0

            elif burst_mode and hand_detected:
                if burst_collected < burst_target:
                    save_path = os.path.join(target_dir, f"{current_count}.jpg")
                    cv2.imwrite(save_path, clean_crop)
                    burst_collected += 1
                    feedback_text = f"BURST RECORDING: {burst_collected}/{burst_target}"
                    feedback_color = (0, 255, 0)
                    feedback_frames = 10
                    time.sleep(0.08)  # Sample every 80ms
                else:
                    burst_mode = False
                    sound_player.speak(f"{gesture_name} recorded", force=True)
                    feedback_text = f"Burst finished! Captured {burst_target} images."
                    feedback_color = (0, 255, 255)
                    feedback_frames = 40

            # Top HUD
            cv2.rectangle(frame, (0, 0), (w, 50), (25, 25, 25), -1)
            cv2.putText(frame, f"Target: [{gesture_name.upper()}]", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, f"Saved: {current_count} images", (360, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            hand_status = "Hand: READY" if hand_detected else "Hand: MOVE INTO BOX"
            h_color = (0, 255, 0) if hand_detected else (0, 100, 255)
            cv2.putText(frame, hand_status, (w - 330, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, h_color, 2, cv2.LINE_AA)

            # Bottom Feedback / Controls Hint
            cv2.rectangle(frame, (0, h - 50), (w, h), (25, 25, 25), -1)
            if feedback_frames > 0:
                cv2.putText(frame, feedback_text, (20, h - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, feedback_color, 2, cv2.LINE_AA)
                feedback_frames -= 1
            else:
                hint = "[SPACE] Snap 1  |  [R] Auto Burst (40 imgs)  |  [Q] Save & Finish"
                cv2.putText(frame, hint, (20, h - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Gesture Data Collector", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' ') or key == 32:  # Snap
                save_path = os.path.join(target_dir, f"{current_count}.jpg")
                cv2.imwrite(save_path, clean_crop)
                feedback_text = f"Saved: {current_count}.jpg"
                feedback_color = (0, 255, 0)
                feedback_frames = 20
            elif key == ord('r') or key == ord('R'):  # Trigger Burst
                countdown_start = time.time()

        cap.release()
        cv2.destroyAllWindows()

    total_saved = get_image_count(target_dir)
    print(f"\n[Finished Recording] Total {total_saved} images stored for '{gesture_name}' in: {target_dir}")

    # 4. Terminal Training Prompt
    print("=" * 65)
    train_answer = input("Do you want to train the model now with the updated dataset? (y/n): ").strip().lower()

    if train_answer in ('y', 'yes'):
        print("\nStarting automated training...")
        success = build_and_train()
        if success:
            print("\n" + "=" * 65)
            print(f"[READY] Your new gesture '{gesture_name.upper()}' is now trained and active!")
            print("Run 'python app.py' to test it in real time!")
            print("=" * 65)
    else:
        print(f"\nData saved for '{gesture_name}'. You can train anytime later with 'python train_classifier.py'.")


if __name__ == '__main__':
    run_collector()
