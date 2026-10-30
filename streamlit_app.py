"""
streamlit_app.py - High-Quality Real-Time Indian Sign Language (ISL) Vision & Vocalizer Workspace

Features:
  - Participant Onboarding Gate (Name, Age, Country, Phone Number)
  - Persistent Participant Identity HUD & Sidebar Profile Card
  - Live Real-Time Camera Stream with Hand Landmark Mesh & HUD Overlays
  - Comprehensive Camera Controls: Camera ON/OFF toggle, Mirror Feed, Mode selector
  - Advanced Sound & Vocalizer System:
      * Master Sound Toggle (Mute / Unmute)
      * Volume Slider (0% - 100%)
      * Speech Rate / Speed Control (0.5x - 2.0x)
      * Speech Pitch Control (0.5 - 1.5)
      * Browser Native Web Speech API (Works 100% on Cloud / Hugging Face / Streamlit Cloud)
      * Test Audio voice button
  - Real-Time Sentence Builder with Read Aloud & Clear History
  - Detection Tuning: Confidence Threshold & Smoothing Window
  - Clean, responsive glassmorphic cyber-aesthetic UI
"""

import os
import sys
import time
import json
from collections import deque
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# ----------------- Workspace Caches -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.cache', 'matplotlib'))
os.environ.setdefault('KERAS_HOME', os.path.join(BASE_DIR, '.cache', 'keras'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)

# ----------------- Conditional WebRTC Import -----------------
try:
    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode,
        RTCConfiguration,
        VideoProcessorBase
    )
    import av
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

# ----------------- Helper Imports from Local Project -----------------
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


# ----------------- Page Config -----------------
st.set_page_config(
    page_title="ISL Vision Pro | Real-Time Sign Translation",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ----------------- Custom Modern Styling -----------------
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Participant Banner */
    .user-hud-banner {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(30, 41, 59, 0.92) 100%);
        border: 1px solid rgba(0, 255, 163, 0.35);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 15px rgba(0, 255, 163, 0.15);
        border-radius: 16px;
        padding: 18px 24px;
        margin-bottom: 24px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
    }
    .hud-title-col {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .hud-avatar {
        background: linear-gradient(135deg, #00FFA3, #00B8D9);
        color: #0d1117;
        font-weight: 800;
        font-size: 22px;
        width: 52px;
        height: 52px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 12px rgba(0, 255, 163, 0.4);
    }
    .hud-user-name {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
    }
    .hud-user-subtitle {
        font-size: 13px;
        color: #94A3B8;
        margin: 0;
    }
    .hud-badges-col {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 12px;
    }
    .hud-badge {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 13px;
        color: #E2E8F0;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .hud-badge-active {
        background: rgba(0, 255, 163, 0.12);
        border: 1px solid rgba(0, 255, 163, 0.4);
        color: #00FFA3;
        font-weight: 600;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #00FFA3;
        border-radius: 50%;
        box-shadow: 0 0 8px #00FFA3;
        display: inline-block;
        animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.7; }
        50% { transform: scale(1.3); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.7; }
    }

    /* Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #00FFA3;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 12px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Sentence Box */
    .sentence-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.4);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        border-radius: 14px;
        padding: 20px;
        margin-top: 14px;
    }
    .sentence-text {
        font-size: 22px;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: 0.03em;
        min-height: 40px;
        word-break: break-word;
    }
    
    /* Registration Portal */
    .portal-hero {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(0, 255, 163, 0.2);
        border-radius: 20px;
        padding: 40px;
        margin: 20px auto;
        max-width: 720px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6);
    }
    .portal-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #00FFA3, #00D2FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 8px;
    }
    .portal-desc {
        text-align: center;
        color: #94A3B8;
        font-size: 15px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- Web Speech Audio Bridge -----------------
def play_browser_speech(text: str, volume: float = 1.0, rate: float = 1.0, pitch: float = 1.0, muted: bool = False):
    """
    Renders a zero-height client-side JavaScript snippet that synthesizes
    speech directly through the user's web browser using the HTML5 Web Speech API.
    Works anywhere (locally or on cloud/Hugging Face).
    """
    if muted or not text:
        return

    clean_text = json.dumps(str(text).strip())
    js = f"""
    <script>
    (function() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const textToSpeak = {clean_text};
            const msg = new SpeechSynthesisUtterance(textToSpeak);
            msg.volume = {max(0.0, min(1.0, float(volume)))};
            msg.rate = {max(0.5, min(2.0, float(rate)))};
            msg.pitch = {max(0.5, min(1.5, float(pitch)))};
            msg.lang = 'en-US';
            
            // Try to select high quality natural voice if available
            const voices = window.speechSynthesis.getVoices();
            const preferred = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha')));
            if (preferred) msg.voice = preferred;

            window.speechSynthesis.speak(msg);
        }}
    }})();
    </script>
    """
    components.html(js, height=0, width=0)


# ----------------- Model & Resource Caching -----------------
@st.cache_resource(show_spinner="Loading ISL gesture neural network...")
def load_model_and_labels():
    """Loads the trained model architecture, weights, and action labels once."""
    model_json_path = os.path.join(BASE_DIR, "model.json")
    model_weights_path = os.path.join(BASE_DIR, "model.h5")
    actions_json_path = os.path.join(BASE_DIR, "actions.json")

    if not os.path.exists(model_json_path) or not os.path.exists(model_weights_path):
        st.error("Trained model files ('model.json' or 'model.h5') not found in workspace.")
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


# ----------------- Session State Initialization -----------------
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None

if "sentence" not in st.session_state:
    st.session_state.sentence = []

if "history_log" not in st.session_state:
    st.session_state.history_log = []

if "current_gesture" not in st.session_state:
    st.session_state.current_gesture = ""

if "current_confidence" not in st.session_state:
    st.session_state.current_confidence = 0.0

if "camera_enabled" not in st.session_state:
    st.session_state.camera_enabled = True

if "sound_muted" not in st.session_state:
    st.session_state.sound_muted = False

if "sound_volume" not in st.session_state:
    st.session_state.sound_volume = 1.0

if "sound_rate" not in st.session_state:
    st.session_state.sound_rate = 1.0

if "sound_pitch" not in st.session_state:
    st.session_state.sound_pitch = 1.0

if "pending_speech" not in st.session_state:
    st.session_state.pending_speech = None


# =========================================================================
# PHASE 1: PARTICIPANT ONBOARDING PORTAL (If User Info Not Set)
# =========================================================================
if st.session_state.user_profile is None:
    st.markdown("""
    <div class="portal-hero">
        <div class="portal-title">🤟 ISL VISION PRO</div>
        <div class="portal-desc">
            Real-Time Indian Sign Language AI Recognition & Speech Synthesis System.<br>
            Please provide your details below before launching the operational workspace.
        </div>
    """, unsafe_allow_html=True)

    with st.form("onboarding_form", clear_on_submit=False):
        col_name, col_age = st.columns([3, 1])
        with col_name:
            user_name = st.text_input("Full Name *", placeholder="e.g. Prabhat Singh")
        with col_age:
            user_age = st.number_input("Age *", min_value=5, max_value=120, value=22, step=1)

        col_country, col_phone = st.columns([2, 2])
        with col_country:
            countries_list = [
                "India", "United States", "United Kingdom", "Canada",
                "Australia", "Germany", "France", "United Arab Emirates",
                "Singapore", "Other"
            ]
            user_country = st.selectbox("Country / Region *", countries_list, index=0)
            if user_country == "Other":
                user_country = st.text_input("Specify Country", placeholder="Enter country name")

        with col_phone:
            user_phone = st.text_input("Phone Number *", placeholder="+91 98765 43210")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Launch ISL Workspace", use_container_width=True)

        if submitted:
            if not user_name.strip():
                st.error("Please enter your name to proceed.")
            elif not user_phone.strip():
                st.error("Please enter a valid phone number.")
            else:
                st.session_state.user_profile = {
                    "name": user_name.strip(),
                    "age": int(user_age),
                    "country": user_country.strip(),
                    "phone": user_phone.strip(),
                    "session_start": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                # Initial welcome speech
                st.session_state.pending_speech = f"Welcome {user_name}! ISL Workspace is ready."
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# =========================================================================
# PHASE 2: FULLY OPERATIONAL WORKSPACE (When User Profile is Set)
# =========================================================================

# Load ML Model
model, actions = load_model_and_labels()
if model is None:
    st.error("Model initialization failed. Please check trained model files.")
    st.stop()

# ----------------- Top Header Participant HUD -----------------
profile = st.session_state.user_profile
initials = "".join([part[0] for part in profile["name"].split()[:2]]).upper() or "IS"

col_hud, col_edit = st.columns([6, 1])
with col_hud:
    st.markdown(f"""
    <div class="user-hud-banner">
        <div class="hud-title-col">
            <div class="hud-avatar">{initials}</div>
            <div>
                <p class="hud-user-name">{profile['name']}</p>
                <p class="hud-user-subtitle">Session Active Since: {profile['session_start']}</p>
            </div>
        </div>
        <div class="hud-badges-col">
            <div class="hud-badge">🎂 <strong>Age:</strong> {profile['age']} yrs</div>
            <div class="hud-badge">🌍 <strong>Country:</strong> {profile['country']}</div>
            <div class="hud-badge">📱 <strong>Phone:</strong> {profile['phone']}</div>
            <div class="hud-badge hud-badge-active"><span class="pulse-dot"></span> System Live</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_edit:
    st.write("")
    if st.button("✏️ Edit Profile", use_container_width=True):
        st.session_state.user_profile = None
        st.rerun()


# ----------------- Sidebar Control Panel -----------------
with st.sidebar:
    st.markdown("### 🎛️ Control Center")

    # 1. Camera Controls
    st.markdown("#### 📷 Camera Controls")
    camera_on = st.toggle("Camera Feed Active", value=st.session_state.camera_enabled, key="toggle_cam")
    st.session_state.camera_enabled = camera_on

    mirror_feed = st.checkbox("Mirror Camera View", value=True, help="Flips the horizontal axis for intuitive mirror-like gesturing")

    st.markdown("---")

    # 2. Sound & Vocalizer Controls
    st.markdown("#### 🔊 Sound & Voice Controller")
    sound_mute = st.toggle("Mute Voice Output", value=st.session_state.sound_muted, key="toggle_mute")
    st.session_state.sound_muted = sound_mute

    st.session_state.sound_volume = st.slider(
        "Volume", min_value=0.0, max_value=1.0, value=st.session_state.sound_volume, step=0.05
    )

    st.session_state.sound_rate = st.slider(
        "Speaking Speed (Rate)", min_value=0.5, max_value=1.8, value=st.session_state.sound_rate, step=0.1,
        help="1.0 is standard human conversation speed (~145 WPM)"
    )

    st.session_state.sound_pitch = st.slider(
        "Pitch", min_value=0.7, max_value=1.4, value=st.session_state.sound_pitch, step=0.1
    )

    if st.button("🔊 Test Voice Output", use_container_width=True):
        st.session_state.pending_speech = f"Audio check complete. Hello {profile['name']}!"

    st.markdown("---")

    # 3. Detection AI Parameters
    st.markdown("#### 🧠 Detection Tuning")
    threshold = st.slider("Confidence Threshold", min_value=0.50, max_value=0.95, value=0.70, step=0.05)
    min_stability_frames = st.slider("Smoothing Window (Frames)", min_value=3, max_value=10, value=5, step=1)

    st.markdown("---")
    with st.expander(f"📚 Supported Gestures ({len(actions)})"):
        st.write(", ".join([f"`{a}`" for a in actions]))


# ----------------- Execute Pending Audio Synthesis -----------------
if st.session_state.pending_speech:
    play_browser_speech(
        st.session_state.pending_speech,
        volume=st.session_state.sound_volume,
        rate=st.session_state.sound_rate,
        pitch=st.session_state.sound_pitch,
        muted=st.session_state.sound_muted
    )
    st.session_state.pending_speech = None


# ----------------- Main Interface Workspace -----------------
col_cam, col_data = st.columns([5, 4])

# Shared detection state tracker
class DetectionTracker:
    def __init__(self):
        self.recent = deque(maxlen=7)
        self.last_spoken_gesture = None
        self.last_spoken_time = 0.0

tracker = DetectionTracker()


def process_video_frame(frame_bgr: np.ndarray, hands_solution) -> tuple:
    """
    Process raw frame with MediaPipe, normalize keypoints, run neural network inference,
    and draw stylish HUD bounding box and labels.
    """
    if mirror_feed:
        frame_bgr = cv2.flip(frame_bgr, 1)

    h, w = frame_bgr.shape[:2]
    image, results = mediapipe_detection(frame_bgr, hands_solution)

    pred_action = ""
    conf = 0.0
    is_stable = False

    if results.multi_hand_landmarks:
        # Draw skeleton mesh
        draw_styled_landmarks(frame_bgr, results)

        # Get hand bounding box
        bbox = get_hand_bbox(results, frame_bgr.shape)
        if bbox:
            bx1, by1, bx2, by2 = bbox
            cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), (0, 255, 163), 2)

        # Extract normalized 63 keypoints
        keypoints = extract_normalized_keypoints(results)

        # Infer gesture
        probs = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        pred_action = actions[pred_idx]

        tracker.recent.append(pred_action)

        # Temporal stability smoothing
        if len(tracker.recent) >= min_stability_frames:
            counts = {a: tracker.recent.count(a) for a in set(tracker.recent)}
            most_common, freq = max(counts.items(), key=lambda it: it[1])
            if freq >= min_stability_frames and conf >= threshold:
                is_stable = True
                pred_action = most_common

                # Overlay tag above bbox
                if bbox:
                    tag = f"{pred_action.upper()} ({conf * 100:.1f}%)"
                    cv2.putText(
                        frame_bgr, tag, (bx1, max(30, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 163), 2, cv2.LINE_AA
                    )

    # Top banner overlay inside video frame
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, frame_bgr, 0.25, 0, frame_bgr)

    status_text = f"GESTURE: {pred_action.upper()} [{conf*100:.1f}%]" if is_stable else "Hold hand steady in camera view"
    text_color = (0, 255, 163) if is_stable else (180, 180, 180)
    cv2.putText(frame_bgr, status_text, (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

    return frame_bgr, pred_action, conf, is_stable


# ----------------- Left Column: Camera Viewport -----------------
with col_cam:
    st.markdown("### 🎥 Live Camera Viewport")

    if not st.session_state.camera_enabled:
        st.info("📷 **Camera is Paused.** Toggle **'Camera Feed Active'** in the sidebar to resume live detection.")
    else:
        # Preferred WebRTC streaming if available
        if HAS_WEBRTC:
            class ISLVideoProcessor(VideoProcessorBase):
                def __init__(self):
                    self.hands = mp_hands.Hands(
                        model_complexity=0,
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )

                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img = frame.to_ndarray(format="bgr24")
                    processed_img, action, conf, stable = process_video_frame(img, self.hands)
                    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

            webrtc_ctx = webrtc_streamer(
                key="isl-live-stream",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ),
                video_processor_factory=ISLVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        else:
            # Fallback High-Quality Snapshot / Live Upload mode if WebRTC is not installed
            st.info("💡 Real-Time WebRTC engine loading. You can also use direct frame capture below:")
            cam_image = st.camera_input("Capture ISL Gesture Frame")
            if cam_image is not None:
                bytes_data = cam_image.getvalue()
                cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

                with mp_hands.Hands(model_complexity=0, max_num_hands=1) as hands_engine:
                    processed_img, action, conf, stable = process_video_frame(cv_img, hands_engine)
                    st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                    if stable and action:
                        st.session_state.current_gesture = action
                        st.session_state.current_confidence = conf

                        # Update sentence history
                        if not st.session_state.sentence or st.session_state.sentence[-1] != action:
                            st.session_state.sentence.append(action)
                            st.session_state.history_log.insert(0, {
                                "time": time.strftime("%H:%M:%S"),
                                "gesture": action,
                                "confidence": f"{conf*100:.1f}%"
                            })
                            st.session_state.pending_speech = action
                            st.rerun()


# ----------------- Right Column: Live Transcription & Stats -----------------
with col_data:
    st.markdown("### 📝 Live Transcription & History")

    # Metrics row
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.current_gesture.upper() or '--'}</div>
            <div class="metric-label">Current Gesture</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        conf_display = f"{st.session_state.current_confidence*100:.1f}%" if st.session_state.current_confidence > 0 else "--"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{conf_display}</div>
            <div class="metric-label">Model Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    # Sentence Box
    sentence_joined = " ".join(st.session_state.sentence) if st.session_state.sentence else "(Waiting for gestures...)"
    st.markdown(f"""
    <div class="sentence-container">
        <div style="font-size: 12px; color: #94A3B8; text-transform: uppercase; margin-bottom: 6px;">
            Recognized Sentence Stream:
        </div>
        <div class="sentence-text">{sentence_joined}</div>
    </div>
    """, unsafe_allow_html=True)

    # Sentence Controls
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("🔊 Read Aloud", use_container_width=True):
            if st.session_state.sentence:
                st.session_state.pending_speech = " ".join(st.session_state.sentence)
                st.rerun()
            else:
                st.toast("Sentence is currently empty!")
    with btn_col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.sentence.clear()
            st.session_state.current_gesture = ""
            st.session_state.current_confidence = 0.0
            st.rerun()
    with btn_col3:
        if st.button("📋 Download", use_container_width=True):
            st.download_button(
                label="Save Text",
                data=" ".join(st.session_state.sentence),
                file_name="isl_transcription.txt",
                mime="text/plain"
            )

    # History Table
    st.markdown("#### 🕒 Recent Detection Log")
    if st.session_state.history_log:
        st.dataframe(
            st.session_state.history_log[:8],
            column_config={
                "time": "Time",
                "gesture": "Recognized Gesture",
                "confidence": "Confidence"
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.caption("No gestures detected yet in this session.")
