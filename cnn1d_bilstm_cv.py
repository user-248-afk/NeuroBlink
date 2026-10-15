import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D,
    Dropout, GlobalAveragePooling1D, Dense, LSTM, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

BASE = "/Applications/Neuro/muse_data_4s"
MODEL_DIR = "/Applications/Neuro/muse_models"

dataset_paths = {
    "Double Blinks": f"{BASE}/Double_Blinks",
    "Long Blinks":   f"{BASE}/Long_Blinks",
    "Noise":         f"{BASE}/Noise",
    "Triple Blinks": f"{BASE}/Triple_Blinks",
}

label_map = {name: idx for idx, name in enumerate(dataset_paths.keys())}
num_classes = len(label_map)

FS = 256
WINDOW_SEC = 1.5
WINDOW = int(FS * WINDOW_SEC)
STRIDE = int(FS * 0.25)

CHANNELS = ["TP9", "TP10"]

NUM_FOLDS = 5
SEED = 42
EPOCHS = 140
BATCH_SIZE = 32
LEARNING_RATE = 7e-4

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_file(path):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    cols = [pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float32) for c in CHANNELS]
    mask = np.all([np.isfinite(c) for c in cols], axis=0)
    return np.stack([c[mask] for c in cols], axis=-1)


def windowize(X, window, stride):
    """Slice into overlapping windows, z-scoring each window per channel."""
    xs = []
    for start in range(0, X.shape[0] - window + 1, stride):
        w = X[start:start + window].copy()
        w = (w - w.mean(axis=0, keepdims=True)) / (w.std(axis=0, keepdims=True) + 1e-6)
        xs.append(w)
    return xs


def build_dataset(files):
    X, y = [], []
    for path, label in files:
        try:
            xs = windowize(load_file(path), WINDOW, STRIDE)
        except Exception as e:
            print("Skipped file:", path, e)
            continue
        X.extend(xs)
        y.extend([label] * len(xs))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def build_model():
    inp = Input(shape=(WINDOW, len(CHANNELS)))

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

    x = Bidirectional(LSTM(160, return_sequences=False))(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.30)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def file_level_accuracy(model, files):
    """Average window probabilities over each file, then take the argmax."""
    correct, total = 0, 0
    for path, label in files:
        try:
            xs = windowize(load_file(path), WINDOW, STRIDE)
        except Exception:
            continue
        if not xs:
            continue
        probs = model.predict(np.array(xs, dtype=np.float32), batch_size=256, verbose=0)
        total += 1
        correct += int(np.argmax(probs.mean(axis=0)) == label)
    return correct / max(total, 1)


all_files, all_labels = [], []
for cls_name, folder in dataset_paths.items():
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if name.startswith(".") or not os.path.isfile(path) or os.path.getsize(path) <= 50:
            continue
        all_files.append((path, label_map[cls_name]))
        all_labels.append(label_map[cls_name])

print("Total files:", len(all_files))

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
window_accs, file_accs = [], []

for fold, (tr_idx, va_idx) in enumerate(skf.split(all_files, all_labels), start=1):
    train_files = [all_files[i] for i in tr_idx]
    val_files = [all_files[i] for i in va_idx]

    X_train, y_train = build_dataset(train_files)
    X_val, y_val = build_dataset(val_files)
    print(f"\n[fold {fold}] train windows {X_train.shape}  val windows {X_val.shape}")

    model = build_model()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, mode="max"),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
        ],
        verbose=2,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/MUSE_CNN_BiLSTM_fold{fold}.keras")

    _, win_acc = model.evaluate(X_val, y_val, verbose=0)
    file_acc = file_level_accuracy(model, val_files)
    window_accs.append(win_acc)
    file_accs.append(file_acc)
    print(f"[fold {fold}] window acc {win_acc:.4f}  file acc {file_acc:.4f}")

print("\n=== 1D-CNN + BiLSTM {}-FOLD CV ({}) ===".format(NUM_FOLDS, "+".join(CHANNELS)))
print(f"Window accuracy: {np.mean(window_accs):.4f} +/- {np.std(window_accs):.4f}")
print(f"File accuracy:   {np.mean(file_accs):.4f} +/- {np.std(file_accs):.4f}")
