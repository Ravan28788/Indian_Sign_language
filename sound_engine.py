"""
sound_engine.py - Threaded, Non-blocking Audio Speech System for ISL Gesture Recognition.

Tuned for:
  - Calm, natural human speaking speed (~145 WPM - peaceful cadence)
  - Clear, concise pronunciation (short, not drawn-out)
  - High-quality natural voices (Samantha / Rishi on macOS)
  - Asynchronous background worker (zero webcam frame-rate drops)
  - Keyboard toggleable state ([S] key)
  - Intelligent debounce / repetition cooldown
"""

import os
import platform
import subprocess
import threading
import time
import queue

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False


class SoundEngine:
    """Threaded audio speech synthesizer with natural human cadence and cooldown."""

    def __init__(self, enabled=True, cooldown=2.2, speech_rate=145, voice="Samantha"):
        self.enabled = enabled
        self.cooldown = cooldown
        self.speech_rate = speech_rate  # 140-150 WPM: peaceful, natural human pace
        self.is_mac = (platform.system() == "Darwin")
        self.voice = voice

        # Check preferred macOS voice
        if self.is_mac:
            self._select_best_mac_voice()

        self.last_spoken_gesture = None
        self.last_spoken_time = 0.0

        self.speech_queue = queue.Queue(maxsize=3)
        self.lock = threading.Lock()

        # Dedicated pyttsx3 engine for non-mac fallback
        self._pyttsx_engine = None
        if not self.is_mac and HAS_PYTTSX3:
            try:
                self._pyttsx_engine = pyttsx3.init()
                self._pyttsx_engine.setProperty('rate', self.speech_rate)
            except Exception:
                self._pyttsx_engine = None

        self._running = True
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()

    def _select_best_mac_voice(self):
        """Pick a smooth, peaceful voice available on macOS."""
        try:
            res = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=False)
            output = res.stdout
            # Prefer Samantha or Rishi if installed
            if "Samantha" in output:
                self.voice = "Samantha"
            elif "Rishi" in output:
                self.voice = "Rishi"
            elif "Daniel" in output:
                self.voice = "Daniel"
            else:
                self.voice = None
        except Exception:
            self.voice = None

    def toggle(self):
        """Toggle audio feedback ON or OFF."""
        with self.lock:
            self.enabled = not self.enabled
            status = "ON" if self.enabled else "OFF"
            print(f"[SoundEngine] Audio feedback turned {status}")
            return self.enabled

    def set_enabled(self, state: bool):
        """Explicitly set audio state."""
        with self.lock:
            self.enabled = state

    def speak(self, text: str, force: bool = False):
        """Enqueue speech utterance asynchronously."""
        if not self.enabled and not force:
            return

        text = str(text).strip()
        if not text:
            return

        # Keep utterance short and concise
        text = text.replace("_Cropped", "").replace("_", " ").title()

        try:
            self.speech_queue.put_nowait(text)
        except queue.Full:
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.put_nowait(text)
            except Exception:
                pass

    def speak_gesture(self, gesture: str, confidence: float = 1.0, min_confidence: float = 0.70):
        """Speak detected gesture with confidence filter and peaceful cooldown."""
        if not self.enabled or confidence < min_confidence:
            return False

        gesture_clean = gesture.strip().replace("_Cropped", "").replace("_", " ").title()
        if not gesture_clean:
            return False

        now = time.time()
        with self.lock:
            is_same = (gesture_clean.lower() == str(self.last_spoken_gesture).lower())
            time_since_last = now - self.last_spoken_time

            # Only repeat same sign after cooldown interval (peaceful pace)
            if is_same and time_since_last < self.cooldown:
                return False

            self.last_spoken_gesture = gesture_clean
            self.last_spoken_time = now

        self.speak(gesture_clean)
        return True

    def _speech_worker(self):
        """Worker running in background daemon thread."""
        while self._running:
            try:
                text = self.speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if self.is_mac:
                    # Native macOS peaceful voice with controlled cadence
                    cmd = ["say", "-r", str(self.speech_rate)]
                    if self.voice:
                        cmd.extend(["-v", self.voice])
                    cmd.append(text)

                    subprocess.run(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False
                    )
                elif self._pyttsx_engine is not None:
                    self._pyttsx_engine.say(text)
                    self._pyttsx_engine.runAndWait()
                else:
                    print(f"\a[Sound]: {text}")
            except Exception as e:
                print(f"[SoundEngine Error]: {e}")
            finally:
                self.speech_queue.task_done()

    def shutdown(self):
        """Stop worker cleanly."""
        self._running = False


# Module-level shared singleton
sound_player = SoundEngine(enabled=True, speech_rate=145)


if __name__ == "__main__":
    print(f"Testing SoundEngine at peaceful pace (rate={sound_player.speech_rate}, voice={sound_player.voice})...")
    sound_player.speak_gesture("Help", confidence=0.95)
    time.sleep(1.8)
    sound_player.speak_gesture("Doctor", confidence=0.95)
    time.sleep(2.0)
    print("Sound test complete.")
