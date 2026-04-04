import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D,
    Dropout, Dense, LSTM, Bidirectional   # ← added Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------------------------
# Paths
# ---------------------------
dataset_paths = {
    "Double Blinks": "/Users/bikramsingh/Documents/muse_data/Double_Blinks",
    "Long Blinks":   "/Users/bikramsingh/Documents/muse_data/Long_Blinks",
    "Noise":         "/Users/bikramsingh/Documents/muse_data/Noise",
    "Triple Blinks": "/Users/bikramsingh/Documents/muse_data/Triple_Blinks",
}

label_map = {name: idx for idx, name in enumerate(dataset_paths.keys())}
num_classes = len(label_map)

# ---------------------------
# Windowing
# ---------------------------
FS = 256
WINDOW_SEC = 1.5
WINDOW = int(FS * WINDOW_SEC)
STRIDE = int(FS * 0.25)

CH1_NAME = "AF7"
CH2_NAME = "AF8"

# ---------------------------
# Load one clip
# ---------------------------
def load_two_channels_file(path):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")

    if CH1_NAME not in df.columns or CH2_NAME not in df.columns:
        raise ValueError(f"Missing {CH1_NAME}/{CH2_NAME} in {path}")

    x1 = pd.to_numeric(df[CH1_NAME], errors="coerce").to_numpy(dtype=np.float32)
    x2 = pd.to_numeric(df[CH2_NAME], errors="coerce").to_numpy(dtype=np.float32)

    mask = np.isfinite(x1) & np.isfinite(x2)
    x1, x2 = x1[mask], x2[mask]

    return np.stack([x1, x2], axis=-1)

# ---------------------------
# Windowize
# ---------------------------
def windowize(X, label, window, stride):
    xs, ys = [], []
    n = X.shape[0]
    if n < window:
        return xs, ys

    for start in range(0, n - window + 1, stride):
        w = X[start:start + window].copy()
        mu = w.mean(axis=0, keepdims=True)
        sd = w.std(axis=0, keepdims=True) + 1e-6
        w = (w - mu) / sd
        xs.append(w)
        ys.append(label)

    return xs, ys

# ---------------------------
# Build file list
# ---------------------------
file_items = []
for cls_name, folder in dataset_paths.items():
    for name in os.listdir(folder):
        if name.startswith("."):
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.getsize(path) > 50:
            file_items.append((path, label_map[cls_name]))

file_items.sort(key=lambda t: t[0])

print("Total files:", len(file_items))
if len(file_items) == 0:
    raise SystemExit("0 files found")

# ---------------------------
# Split by file
# ---------------------------
train_files, val_files = train_test_split(
    file_items,
    test_size=0.2,
    random_state=42,
    stratify=[y for _, y in file_items]
)

# ---------------------------
# Build window datasets
# ---------------------------
X_train, y_train = [], []
for path, y in train_files:
    try:
        X = load_two_channels_file(path)
        xs, ys = windowize(X, y, WINDOW, STRIDE)
        X_train.extend(xs)
        y_train.extend(ys)
    except Exception:
        pass

X_val, y_val = [], []
for path, y in val_files:
    try:
        X = load_two_channels_file(path)
        xs, ys = windowize(X, y, WINDOW, STRIDE)
        X_val.extend(xs)
        y_val.extend(ys)
    except Exception:
        pass

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)
X_val = np.array(X_val, dtype=np.float32)
y_val = np.array(y_val, dtype=np.int64)

print("Train windows:", X_train.shape)
print("Val windows:", X_val.shape)
if X_train.shape[0] == 0 or X_val.shape[0] == 0:
    raise SystemExit("0 windows created")

# ---------------------------
# Model: 1D CNN + BiLSTM
# ---------------------------
inp = Input(shape=(WINDOW, 2))

x = Conv1D(64, 11, padding="same")(inp)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling1D(2)(x)

x = Conv1D(128, 9, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.10)(x)

x = Conv1D(256, 7, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.10)(x)

x = Conv1D(512, 5, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)

# ← only change here
x = Bidirectional(LSTM(160, return_sequences=False))(x)

# classifier
x = Dense(256, activation="relu")(x)
x = Dropout(0.30)(x)
out = Dense(num_classes, activation="softmax")(x)

model = Model(inp, out)

model.compile(
    optimizer=Adam(learning_rate=7e-4, clipnorm=1.0),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True,
    mode="max"
)

lr_plateau = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-5,
    verbose=1
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=140,
    batch_size=32,
    callbacks=[early_stop, lr_plateau]
)

# ---------------------------
# Save + evaluate
# ---------------------------
final_path = "/Users/bikramsingh/Documents/muse_models/MUSE_final2.keras"
os.makedirs(os.path.dirname(final_path), exist_ok=True)
model.save(final_path)

final_win_loss, final_win_acc = model.evaluate(X_val, y_val, verbose=1)

correct = 0
total = 0
for path, y in val_files:
    try:
        X = load_two_channels_file(path)
        xs, _ = windowize(X, y, WINDOW, STRIDE)
        if len(xs) == 0:
            continue
        Xw = np.array(xs, dtype=np.float32)
        probs = model.predict(Xw, batch_size=256, verbose=0)
        pred = int(np.argmax(probs.mean(axis=0)))
        total += 1
        if pred == y:
            correct += 1
    except Exception:
        continue

final_file_acc = correct / max(total, 1)

print("FINAL window accuracy:", final_win_acc)
print("FINAL file accuracy:", final_file_acc)
print("Saved final:", final_path)