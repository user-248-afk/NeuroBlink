"""
Fresh TCN training on the single-blink data (TP9 + TP10, matching the
combo classifier's channels and the project's established electrode
finding), evaluated with proper file-level stratified 5-fold CV rather
than a single fixed train/test folder split. This gives an honest mean +/-
std before moving on to participant-level validation.

Data: the 4 pooled class folders (Double_Blinks, Triple_Blinks,
Long_Blinks, Noise), NOT the pre-split _ML/_Testing folders (those have
lost participant identity and are the old fixed-split recipe).
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, ReLU, Dropout,
    GlobalAveragePooling1D, Dense, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold

FS = 256
WINDOW_SEC = 1.5
WINDOW = int(FS * WINDOW_SEC)
STRIDE = int(FS * 0.25)
CHANNELS = ["TP9", "TP10"]

BASE = "/Users/bikramsingh/Documents/muse_data_4s"
CLASS_FOLDERS = {
    "double_blink": f"{BASE}/Double_Blinks",
    "long_blink":   f"{BASE}/Long_Blinks",
    "noise":        f"{BASE}/Noise",
    "triple_blink": f"{BASE}/Triple_Blinks",
}
CLASS_NAMES = list(CLASS_FOLDERS.keys())
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
L2REG = 7e-4
SEED = 0


def collect_files():
    items = []
    for cls_name, folder in CLASS_FOLDERS.items():
        for name in sorted(os.listdir(folder)):
            if name.startswith(".") or not name.endswith(".txt"):
                continue
            p = os.path.join(folder, name)
            if os.path.getsize(p) > 50:
                items.append((p, LABEL_MAP[cls_name], cls_name))
    return items


def load_channels_file(path):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    cols = []
    for ch in CHANNELS:
        cols.append(pd.to_numeric(df[ch], errors="coerce").to_numpy(dtype=np.float32))
    X = np.stack(cols, axis=-1)
    mask = np.all(np.isfinite(X), axis=1)
    return X[mask]


def windowize(X, label, window=WINDOW, stride=STRIDE):
    xs, ys = [], []
    n = X.shape[0]
    if n < window:
        return xs, ys
    for start in range(0, n - window + 1, stride):
        w = X[start:start + window].copy()
        mu = w.mean(axis=0, keepdims=True)
        sd = w.std(axis=0, keepdims=True) + 1e-6
        xs.append((w - mu) / sd)
        ys.append(label)
    return xs, ys


def build_windows(file_items):
    X_all, y_all = [], []
    for path, y, _ in file_items:
        try:
            X = load_channels_file(path)
            xs, ys = windowize(X, y)
            X_all.extend(xs)
            y_all.extend(ys)
        except Exception as e:
            print(f"  skip {path}: {e}", flush=True)
    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.int64)


def residual_block(x, dilation_rate, filters, kernel_size=3, dropout=0.2, l2reg=L2REG):
    from tensorflow.keras.regularizers import l2
    prev = x
    h = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding="same", kernel_regularizer=l2(l2reg))(x)
    h = BatchNormalization()(h); h = ReLU()(h); h = Dropout(dropout)(h)
    h = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding="same", kernel_regularizer=l2(l2reg))(h)
    h = BatchNormalization()(h); h = ReLU()(h); h = Dropout(dropout)(h)
    if prev.shape[-1] != filters:
        prev = Conv1D(filters, 1, padding="same")(prev)
    return Add()([prev, h])


def build_tcn(input_len, n_channels, num_classes, dilations=(1, 2, 4, 8, 16),
              filters_per_block=(32, 32, 64, 64, 128)):
    inp = Input(shape=(input_len, n_channels))
    x = inp
    for d, f in zip(dilations, filters_per_block):
        x = residual_block(x, dilation_rate=d, filters=f)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def file_level_eval(model, file_items):
    correct = 0
    total = 0
    for path, y, _ in file_items:
        try:
            X = load_channels_file(path)
            xs, _ = windowize(X, y)
            if not xs:
                continue
            probs = model.predict(np.array(xs, dtype=np.float32), batch_size=256, verbose=0)
            pred = int(np.argmax(probs.mean(axis=0)))
            total += 1
            correct += (pred == y)
        except Exception:
            continue
    return correct / max(total, 1)


def main(n_splits=5, epochs=100):
    file_items = collect_files()
    labels = [f[1] for f in file_items]
    print(f"Total files: {len(file_items)}  per class: "
          f"{ {c: labels.count(LABEL_MAP[c]) for c in CLASS_NAMES} }", flush=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    idx = np.arange(len(file_items))
    fold_accs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(idx, labels)):
        train_files = [file_items[i] for i in train_idx]
        test_files = [file_items[i] for i in test_idx]

        X_train, y_train = build_windows(train_files)
        X_test, y_test = build_windows(test_files)
        print(f"[fold {fold_idx}] train_files={len(train_files)} test_files={len(test_files)} "
              f"train_windows={len(X_train)} test_windows={len(X_test)}", flush=True)

        model = build_tcn(WINDOW, len(CHANNELS), len(CLASS_NAMES))
        es = EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True, mode="max")
        rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=0)
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=epochs, batch_size=64, callbacks=[es, rlr], verbose=0)

        file_acc = file_level_eval(model, test_files)
        print(f"[fold {fold_idx}] file_acc={file_acc:.1%}", flush=True)
        fold_accs.append(file_acc)
        model.save(f"/Users/bikramsingh/eeg_project/scratch/tcn_optimized_fold{fold_idx}.keras")

    fold_accs = np.array(fold_accs)
    print(f"\n=== TCN (TP9+TP10) 5-FOLD CV ===")
    print(f"Per-fold: {[f'{a:.1%}' for a in fold_accs]}")
    print(f"MEAN: {fold_accs.mean():.1%}  STD: {fold_accs.std():.1%}  MIN: {fold_accs.min():.1%}  MAX: {fold_accs.max():.1%}")


if __name__ == "__main__":
    main()
