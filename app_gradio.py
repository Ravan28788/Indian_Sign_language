"""
app.py - High-Quality Real-Time Indian Sign Language (ISL) Workspace for Hugging Face Gradio (100% Free Tier)

Features:
  - Participant Onboarding Gate (Name, Age, Country, Phone)
  - Persistent Participant Profile HUD
  - Real-Time Webcam Streaming with MediaPipe Landmarks and Neural Network Classification
  - Full In-Browser Speech Synthesis (Web Speech API) with Mute toggle & Rate controls
  - Sentence Builder, History Log, and Confidence Threshold tuning
"""

import os
import json
import time
from collections import deque
import cv2
import numpy as np
import gradio as gr

# Ensure workspace-local cache
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

try:
    from tensorflow.keras.models import model_from_json
except ImportError:
    from keras.models import model_from_json


def load_model_and_labels():
    model_json_path = os.path.join(BASE_DIR, "model.json")
    model_weights_path = os.path.join(BASE_DIR, "model.h5")
    actions_json_path = os.path.join(BASE_DIR, "actions.json")

    if not os.path.exists(model_json_path) or not os.path.exists(model_weights_path):
        return None, None

    with open(model_json_path, "r") as f:
        model_json = f.read()

    model = model_from_json(model_json)
    model.load_weights(model_weights_path)

    if os.path.exists(actions_json_path):
        with open(actions_json_path, "r") as af:
            actions = json.load(af)
    else:
        actions = fallback_actions

    return model, actions


model, actions = load_model_and_labels()

# Global hands detection instance
hands_detector = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Global streaming state
state_tracker = {
    "recent": deque(maxlen=6),
    "sentence": [],
    "last_spoken": "",
    "last_spoken_time": 0.0
}


def create_profile_html(name, age, country, phone):
    initials = "".join([p[0] for p in name.split()[:2]]).upper() if name else "IS"
    return f"""
    <div style="background: linear-gradient(135deg, #111827 0%, #1E293B 100%);
                border: 1px solid rgba(0, 255, 163, 0.4); border-radius: 14px;
                padding: 16px 20px; margin-bottom: 15px; color: #fff;
                display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 15px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: linear-gradient(135deg, #00FFA3, #00D2FF); color: #000;
                        font-weight: 800; font-size: 20px; width: 46px; height: 46px;
                        border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                {initials}
            </div>
            <div>
                <h3 style="margin: 0; font-size: 18px; color: #fff;">{name}</h3>
                <span style="font-size: 12px; color: #94A3B8;">Verified Participant • Active Session</span>
            </div>
        </div>
        <div style="display: flex; gap: 10px; flex-wrap: wrap; font-size: 13px;">
            <span style="background: rgba(255,255,255,0.06); padding: 5px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">🎂 Age: <strong>{age}</strong></span>
            <span style="background: rgba(255,255,255,0.06); padding: 5px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">🌍 Country: <strong>{country}</strong></span>
            <span style="background: rgba(255,255,255,0.06); padding: 5px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">📱 Phone: <strong>{phone}</strong></span>
            <span style="background: rgba(0,255,163,0.15); color: #00FFA3; font-weight: 600; padding: 5px 12px; border-radius: 8px; border: 1px solid rgba(0,255,163,0.4);">🟢 System Live</span>
        </div>
    </div>
    """


def register_user(name, age, country, phone):
    if not name or len(name.strip()) < 2:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "<p style='color: #EF4444;'>Please enter a valid Full Name (at least 2 letters).</p>",
            ""
        )
    if not phone or len(phone.strip()) < 5:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "<p style='color: #EF4444;'>Please enter a valid Phone Number.</p>",
            ""
        )

    profile_card = create_profile_html(name.strip(), age, country.strip(), phone.strip())
    # Return: hide onboarding, show workspace, clear error, set profile
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        "",
        profile_card
    )


def process_frame(frame, threshold, mute_sound, speech_rate):
    """Real-time webcam frame inference handler."""
    if frame is None or model is None:
        return None, "--", "--", " ".join(state_tracker["sentence"]) or "(none)", ""

    # Mirror for natural interaction
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Convert RGB to BGR for OpenCV / MediaPipe
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    image, results = mediapipe_detection(bgr, hands_detector)

    gesture_text = "--"
    conf_text = "--"
    speech_trigger = ""

    if results.multi_hand_landmarks:
        draw_styled_landmarks(bgr, results)
        bbox = get_hand_bbox(results, bgr.shape)
        if bbox:
            bx1, by1, bx2, by2 = bbox
            cv2.rectangle(bgr, (bx1, by1), (bx2, by2), (0, 255, 163), 2)

        keypoints = extract_normalized_keypoints(results)
        probs = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        pred_action = actions[pred_idx]

        state_tracker["recent"].append(pred_action)

        if len(state_tracker["recent"]) >= 4:
            counts = {a: state_tracker["recent"].count(a) for a in set(state_tracker["recent"])}
            most_common, freq = max(counts.items(), key=lambda it: it[1])

            if freq >= 4 and conf >= threshold:
                gesture_text = most_common.upper()
                conf_text = f"{conf * 100:.1f}%"

                # Check speech trigger cooldown
                now = time.time()
                if (not mute_sound) and (most_common != state_tracker["last_spoken"] or (now - state_tracker["last_spoken_time"]) > 2.5):
                    state_tracker["last_spoken"] = most_common
                    state_tracker["last_spoken_time"] = now
                    # Generate browser speech script
                    speech_trigger = f"""
                    <script>
                    (function() {{
                        if ('speechSynthesis' in window) {{
                            window.speechSynthesis.cancel();
                            const u = new SpeechSynthesisUtterance({json.dumps(most_common)});
                            u.rate = {float(speech_rate)};
                            window.speechSynthesis.speak(u);
                        }}
                    }})();
                    </script>
                    """

                # Update sentence
                if not state_tracker["sentence"] or state_tracker["sentence"][-1] != most_common:
                    state_tracker["sentence"].append(most_common)
                    if len(state_tracker["sentence"]) > 10:
                        state_tracker["sentence"] = state_tracker["sentence"][-10:]

    # HUD Banner inside video
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, bgr, 0.25, 0, bgr)

    hud_msg = f"GESTURE: {gesture_text} [{conf_text}]" if gesture_text != "--" else "Hold hand steady in camera view"
    cv2.putText(bgr, hud_msg, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 163), 2, cv2.LINE_AA)

    rgb_out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    current_sentence = " ".join(state_tracker["sentence"]) or "(none)"

    return rgb_out, gesture_text, conf_text, current_sentence, speech_trigger


def clear_sentence():
    state_tracker["sentence"].clear()
    state_tracker["recent"].clear()
    state_tracker["last_spoken"] = ""
    return "(none)"


def read_sentence_aloud(speech_rate, mute_sound):
    if mute_sound or not state_tracker["sentence"]:
        return ""
    full_text = " ".join(state_tracker["sentence"])
    return f"""
    <script>
    if ('speechSynthesis' in window) {{
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance({json.dumps(full_text)});
        u.rate = {float(speech_rate)};
        window.speechSynthesis.speak(u);
    }}
    </script>
    """


# ----------------- Modern Gradio UI Layout -----------------
theme = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate"
)

with gr.Blocks(theme=theme, title="ISL Vision Pro") as demo:
    gr.Markdown("# 🤟 ISL Vision Pro - Real-Time Indian Sign Language AI")

    # Error alert
    err_box = gr.HTML("")

    # ==================== STEP 1: ONBOARDING GATE ====================
    with gr.Column(visible=True) as onboarding_col:
        with gr.Group():
            gr.Markdown("### 📋 Participant Registration Gate\nPlease fill in your participant details before entering the live workspace.")
            with gr.Row():
                in_name = gr.Textbox(label="Full Name *", placeholder="e.g. Prabhat Singh")
                in_age = gr.Number(label="Age *", value=22, precision=0)
            with gr.Row():
                in_country = gr.Dropdown(
                    label="Country *",
                    choices=["India", "United States", "United Kingdom", "Canada", "Australia", "Germany", "Other"],
                    value="India"
                )
                in_phone = gr.Textbox(label="Phone Number *", placeholder="+91 98765 43210")
            
            btn_launch = gr.Button("🚀 Launch ISL Workspace", variant="primary", size="lg")

    # ==================== STEP 2: ISL LIVE WORKSPACE ====================
    with gr.Column(visible=False) as workspace_col:
        profile_display = gr.HTML("")
        audio_carrier = gr.HTML("")

        with gr.Row():
            # Left: Camera Viewport
            with gr.Column(scale=5):
                gr.Markdown("### 🎥 Live Camera Stream")
                webcam_input = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="Camera Feed"
                )

            # Right: Controls & Transcription
            with gr.Column(scale=4):
                gr.Markdown("### 📝 Live Transcription & Controls")
                with gr.Row():
                    out_gesture = gr.Textbox(label="Current Gesture", value="--", interactive=False)
                    out_confidence = gr.Textbox(label="Confidence", value="--", interactive=False)

                out_sentence = gr.Textbox(label="Recognized Sentence Stream", value="(none)", interactive=False)

                with gr.Row():
                    btn_read = gr.Button("🔊 Read Sentence", variant="secondary")
                    btn_clear = gr.Button("🗑️ Clear History", variant="stop")

                gr.Markdown("#### 🎛️ Sound & Sensitivity Controls")
                mute_toggle = gr.Checkbox(label="🔇 Mute Voice Output", value=False)
                rate_slider = gr.Slider(minimum=0.5, maximum=1.8, value=1.0, step=0.1, label="Speaking Speed (Rate)")
                thresh_slider = gr.Slider(minimum=0.5, maximum=0.95, value=0.70, step=0.05, label="Confidence Threshold")

        # Streaming frame event
        webcam_input.stream(
            fn=process_frame,
            inputs=[webcam_input, thresh_slider, mute_toggle, rate_slider],
            outputs=[webcam_input, out_gesture, out_confidence, out_sentence, audio_carrier]
        )

        btn_clear.click(fn=clear_sentence, inputs=[], outputs=[out_sentence])
        btn_read.click(fn=read_sentence_aloud, inputs=[rate_slider, mute_toggle], outputs=[audio_carrier])

    # Gate submission event
    btn_launch.click(
        fn=register_user,
        inputs=[in_name, in_age, in_country, in_phone],
        outputs=[onboarding_col, workspace_col, err_box, profile_display]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)
