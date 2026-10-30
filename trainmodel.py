"""
trainmodel.py - Unified Training Dispatcher for ISL Models.

Automatically detects whether to train:
  1. Image Keypoint Classifier (from 29,300+ images in 'Image/' via train_classifier.py)
  2. Temporal Sequence Model (from LSTM sequences in 'MP_Data/')
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.cache', 'matplotlib'))
os.environ.setdefault('KERAS_HOME', os.path.join(BASE_DIR, '.cache', 'keras'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)

from function import actions, DATA_PATH, IMAGE_PATH, no_sequences, sequence_length


def check_sequence_data():
    """Check if MP_Data contains valid sequence files."""
    if not os.path.exists(DATA_PATH):
        return False
    for action in actions:
        action_dir = os.path.join(DATA_PATH, str(action))
        if os.path.exists(action_dir):
            for seq in os.listdir(action_dir):
                seq_dir = os.path.join(action_dir, seq)
                if os.path.isdir(seq_dir) and any(f.endswith('.npy') for f in os.listdir(seq_dir)):
                    return True
    return False


def train_sequence_lstm():
    """Train LSTM model on MP_Data sequences."""
    try:
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.callbacks import TensorBoard
    except ImportError:
        from keras.utils import to_categorical
        from keras.models import Sequential
        from keras.layers import LSTM, Dense
        from keras.callbacks import TensorBoard

    from sklearn.model_selection import train_test_split

    label_map = {label: num for num, label in enumerate(actions)}

    print("[trainmodel] Loading dataset from sequence keypoint files in MP_Data...")
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                npy_path = os.path.join(DATA_PATH, str(action), str(sequence), f"{frame_num}.npy")
                if os.path.exists(npy_path):
                    res = np.load(npy_path, allow_pickle=True)
                    window.append(res)
                else:
                    window.append(np.zeros(21 * 3))
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    if len(X) < 2:
        print("[Error] Not enough sequence samples to train.")
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    print("Building LSTM model...")
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 21 * 3)),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(actions), activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    log_dir = os.path.join(BASE_DIR, 'Logs')
    os.makedirs(log_dir, exist_ok=True)
    tb_callback = TensorBoard(log_dir=log_dir)

    print("Training LSTM model...")
    model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback], validation_data=(X_test, y_test))

    model_json = model.to_json()
    with open(os.path.join(BASE_DIR, "model.json"), "w") as json_file:
        json_file.write(model_json)
    model.save(os.path.join(BASE_DIR, "model.h5"))
    print("[Success] LSTM model trained and saved to 'model.json' & 'model.h5'.")


def main():
    has_sequences = check_sequence_data()

    if has_sequences:
        print("[trainmodel] Found sequence keypoint data in 'MP_Data/'. Training LSTM...")
        train_sequence_lstm()
    else:
        print("[trainmodel] No temporal sequences in 'MP_Data/'. Detected image dataset in 'Image/'.")
        print("[trainmodel] Dispatching to high-accuracy image keypoint classifier (train_classifier.py)...")
        from train_classifier import build_and_train
        build_and_train()


if __name__ == '__main__':
    main()
