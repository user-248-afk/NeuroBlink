"""
Honest final validation: 5-fold CV of the v7+v2+RF blend, using a FIXED
weight combo (w_v7=0.40, w_v2=0.45, w_rf=0.15) chosen ahead of time from a
separate 3-seed exploration (ensemble_cnn_peak.py). No weight tuning
happens against these folds' test sets -- v7 and v2 are trained fresh per
fold, RF features/fit fresh per fold, then blended with the fixed weights
and evaluated once. This is the number that actually means something.
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from scipy.signal import find_peaks, peak_widths
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_combo_cnn7 import build_model as build_v7_model, augment as augment_v7
from train_combo_cnn2 import build_model as build_v2_model, augment as augment_v2, normalize as normalize_v2

FS = 256
FIXED_LEN = 1536
CH1, CH2 = "TP9", "TP10"
PARTICIPANTS = ["mom1", "mom2", "mom3", "mom4", "mom5", "mom6", "mom7",
                "mom8", "mom9", "mom10", "mom11"]
COMBO_BASE = Path("/Users/bikramsingh/Documents/muse_data_4s")
BLINK_TYPES = ["double_blink", "long_blink", "triple_blink"]
TYPE_TO_IDX = {t: i for i, t in enumerate(BLINK_TYPES)}

DETREND_WIN = 128
ENV_SMOOTH_WIN = 64
PEAK_SMOOTH_WIN = 32
MIN_PEAK_DISTANCE_SEC = 0.22

W1 = (0.48, 0.32, 0.20)  # (w_v7, w_v2, w_rf) for slot1 -- chosen ahead of time from a separate 3-seed search
W2 = (0.45, 0.55, 0.00)  # (w_v7, w_v2, w_rf) for slot2
N_AUGMENTS = 25
L2REG = 7e-4


def combo_command_files(participant):
    root = COMBO_BASE / f"{participant}_combinatoric"
    pattern = re.compile(rf"^{re.escape(participant)}_combinatoric_(.+)_\d+$")
    out = []
    for f in sorted(root.glob("*.txt")):
        m = pattern.match(f.stem)
        if m:
            out.append((f, m.group(1)))
    return out


def load_two_channels(path):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    x1 = pd.to_numeric(df[CH1], errors="coerce").to_numpy(dtype=np.float32)
    x2 = pd.to_numeric(df[CH2], errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(x1) & np.isfinite(x2)
    return np.stack([x1[mask], x2[mask]], axis=-1)


def fit_to_length(X, length):
    n = X.shape[0]
    if n == length:
        return X
    if n > length:
        start = (n - length) // 2
        return X[start:start + length]
    pad = length - n
    left = pad // 2
    right = pad - left
    return np.pad(X, ((left, right), (0, 0)), mode="edge")


def moving_average(x, win):
    kernel = np.ones(win, dtype=np.float32) / win
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = np.convolve(x[:, c], kernel, mode="same")
    return out


def add_envelope(X):
    detrended = X - moving_average(X, DETREND_WIN)
    rectified = np.abs(detrended)
    envelope = moving_average(rectified, ENV_SMOOTH_WIN)
    return np.concatenate([X, envelope], axis=-1)


def normalize(w):
    mu = w.mean(axis=0, keepdims=True)
    sd = w.std(axis=0, keepdims=True) + 1e-6
    return (w - mu) / sd


def moving_average_1d(x, win):
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, kernel, mode="same")


def envelope_1ch(x):
    detrended = x - moving_average_1d(x, DETREND_WIN)
    rectified = np.abs(detrended)
    return moving_average_1d(rectified, PEAK_SMOOTH_WIN)


def detect_events(x1, x2):
    env = np.maximum(envelope_1ch(x1), envelope_1ch(x2))
    thresh = env.mean() + 1.2 * env.std()
    peaks, _ = find_peaks(env, height=thresh, distance=int(MIN_PEAK_DISTANCE_SEC * FS))
    combined_detrended = (x1 - moving_average_1d(x1, DETREND_WIN)) + (x2 - moving_average_1d(x2, DETREND_WIN))
    return peaks, env, combined_detrended


def split_two_clusters(peaks):
    if len(peaks) < 2:
        return peaks, np.array([], dtype=int)
    gaps = np.diff(peaks)
    split_at = np.argmax(gaps)
    return peaks[:split_at + 1], peaks[split_at + 1:]


def cluster_features(cluster_peaks, env, detrended):
    n = len(cluster_peaks)
    margin = int(0.3 * FS)
    if n == 0:
        return [0.0] * 7
    heights = env[cluster_peaks]
    mean_height = float(heights.mean())
    start = max(0, cluster_peaks[0] - margin)
    end = min(len(env), cluster_peaks[-1] + margin)
    window = env[start:end]
    mean_energy = float(window.mean())
    total_energy = float(window.sum()) / FS
    sig = detrended[start:end]
    zero_crossings = float(np.sum(np.diff(np.sign(sig)) != 0))
    if n == 1:
        widths, _, _, _ = peak_widths(env, cluster_peaks, rel_height=0.5)
        span_sec = float(widths[0]) / FS
        mean_gap_sec = 0.0
    else:
        span_sec = (cluster_peaks[-1] - cluster_peaks[0]) / FS
        mean_gap_sec = float(np.mean(np.diff(cluster_peaks))) / FS
    return [float(n), span_sec, mean_height, mean_gap_sec, mean_energy, total_energy, zero_crossings]


def extract_peak_features(x1, x2):
    peaks, env, detrended = detect_events(x1, x2)
    c1, c2 = split_two_clusters(peaks)
    return cluster_features(c1, env, detrended), cluster_features(c2, env, detrended)


def train_cnn(build_fn, augment_fn, add_env, norm_fn, train_recs, test_recs, seed, n_augments, l2reg, epochs=300):
    rng = np.random.default_rng(seed)
    X_train, y1_train, y2_train = [], [], []
    for X, t0, t1, _, _, _ in train_recs:
        for Xa in augment_fn(X, rng, n_augments=n_augments):
            Xf = add_env(Xa) if add_env else Xa
            X_train.append(norm_fn(Xf))
            y1_train.append(TYPE_TO_IDX[t0])
            y2_train.append(TYPE_TO_IDX[t1])
    X_train = np.array(X_train, dtype=np.float32)
    y1_train = np.array(y1_train, dtype=np.int64)
    y2_train = np.array(y2_train, dtype=np.int64)

    X_val = np.array([norm_fn(add_env(X) if add_env else X) for X, _, _, _, _, _ in test_recs], dtype=np.float32)
    y1_val = np.array([TYPE_TO_IDX[r[1]] for r in test_recs], dtype=np.int64)
    y2_val = np.array([TYPE_TO_IDX[r[2]] for r in test_recs], dtype=np.int64)

    model = build_fn(FIXED_LEN, l2reg=l2reg)
    early_stop = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, mode="min")
    lr_plateau = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=0)
    model.fit(
        X_train, {"event1": y1_train, "event2": y2_train},
        validation_data=(X_val, {"event1": y1_val, "event2": y2_val}),
        epochs=epochs, batch_size=32,
        callbacks=[early_stop, lr_plateau],
        verbose=0,
    )
    return model


def main(n_splits=5, seed=0):
    records = []
    for p in PARTICIPANTS:
        for path, label in combo_command_files(p):
            X = load_two_channels(path)
            Xfixed = fit_to_length(X, FIXED_LEN)
            t0, t1 = label.split("_then_")
            records.append((Xfixed, t0, t1, label, path.name, p))

    # Participant-level folds: 2 held-out participants per fold, 5 folds,
    # covering 10 of 11 participants. The 11th (whichever one) never lands
    # in a test set, only ever in training. Fixed-seed shuffle, printed for
    # reproducibility.
    rng = np.random.default_rng(seed)
    shuffled = list(PARTICIPANTS)
    rng.shuffle(shuffled)
    held_out_forever = shuffled[10]
    fold_participants = [shuffled[i * 2:i * 2 + 2] for i in range(n_splits)]
    print(f"Participant order (shuffled, seed={seed}): {shuffled}", flush=True)
    print(f"Fold test-participant pairs: {fold_participants}", flush=True)
    print(f"Never held out (always in training): {held_out_forever}", flush=True)

    full_accs = []
    oof_names, oof_true_labels, oof_pred1, oof_pred2 = [], [], [], []
    for fold_idx, test_participants in enumerate(fold_participants):
        train_recs = [r for r in records if r[5] not in test_participants]
        test_recs = [r for r in records if r[5] in test_participants]

        y1_test = np.array([TYPE_TO_IDX[r[1]] for r in test_recs], dtype=np.int64)
        y2_test = np.array([TYPE_TO_IDX[r[2]] for r in test_recs], dtype=np.int64)

        # v7 (envelope input, 3-block)
        model7 = train_cnn(build_v7_model, augment_v7, add_envelope, normalize, train_recs, test_recs, seed, N_AUGMENTS, L2REG)
        X_test7 = np.array([normalize(add_envelope(X)) for X, _, _, _, _, _ in test_recs], dtype=np.float32)
        p1_v7, p2_v7 = model7.predict(X_test7, verbose=0)

        # v2 (raw input, 2-block, hard split)
        model2 = train_cnn(build_v2_model, augment_v2, None, normalize_v2, train_recs, test_recs, seed, N_AUGMENTS, L2REG)
        X_test2 = np.array([normalize_v2(X) for X, _, _, _, _, _ in test_recs], dtype=np.float32)
        p1_v2, p2_v2 = model2.predict(X_test2, verbose=0)

        # RF peak features
        train_f1, train_f2, train_y1, train_y2 = [], [], [], []
        for X, t0, t1, _, _, _ in train_recs:
            f1, f2 = extract_peak_features(X[:, 0], X[:, 1])
            train_f1.append(f1); train_f2.append(f2)
            train_y1.append(TYPE_TO_IDX[t0]); train_y2.append(TYPE_TO_IDX[t1])
        test_f1, test_f2 = [], []
        for X, t0, t1, _, _, _ in test_recs:
            f1, f2 = extract_peak_features(X[:, 0], X[:, 1])
            test_f1.append(f1); test_f2.append(f2)

        clf1 = RandomForestClassifier(n_estimators=200, random_state=seed)
        clf2 = RandomForestClassifier(n_estimators=200, random_state=seed)
        clf1.fit(train_f1, train_y1)
        clf2.fit(train_f2, train_y2)
        p1_rf_raw = clf1.predict_proba(test_f1)
        p2_rf_raw = clf2.predict_proba(test_f2)
        p1_rf = np.zeros_like(p1_v7); p2_rf = np.zeros_like(p2_v7)
        for j, cls in enumerate(clf1.classes_):
            p1_rf[:, cls] = p1_rf_raw[:, j]
        for j, cls in enumerate(clf2.classes_):
            p2_rf[:, cls] = p2_rf_raw[:, j]

        blend1 = W1[0] * p1_v7 + W1[1] * p1_v2 + W1[2] * p1_rf
        blend2 = W2[0] * p2_v7 + W2[1] * p2_v2 + W2[2] * p2_rf
        pred1 = np.argmax(blend1, axis=1); pred2 = np.argmax(blend2, axis=1)
        full_acc = ((pred1 == y1_test) & (pred2 == y2_test)).mean()

        full_v7 = ((np.argmax(p1_v7, axis=1) == y1_test) & (np.argmax(p2_v7, axis=1) == y2_test)).mean()
        full_v2 = ((np.argmax(p1_v2, axis=1) == y1_test) & (np.argmax(p2_v2, axis=1) == y2_test)).mean()
        full_rf = ((np.argmax(p1_rf, axis=1) == y1_test) & (np.argmax(p2_rf, axis=1) == y2_test)).mean()

        print(f"[fold {fold_idx}] test={test_participants} n_test={len(test_recs)}  v7={full_v7:.1%} v2={full_v2:.1%} rf={full_rf:.1%}  "
              f"BLEND={full_acc:.1%}", flush=True)
        full_accs.append(full_acc)

        oof_names.extend([r[4] for r in test_recs])
        oof_true_labels.extend([r[3] for r in test_recs])
        oof_pred1.extend(pred1.tolist())
        oof_pred2.extend(pred2.tolist())

    full_accs = np.array(full_accs)
    print(f"\n=== FINAL {n_splits}-FOLD CV, BLEND WEIGHTS w1={W1} w2={W2} ===")
    print(f"Per-fold: {[f'{a:.1%}' for a in full_accs]}")
    print(f"MEAN: {full_accs.mean():.1%}  STD: {full_accs.std():.1%}  MIN: {full_accs.min():.1%}  MAX: {full_accs.max():.1%}")

    idx_to_type = {v: k for k, v in TYPE_TO_IDX.items()}
    pred_labels = [f"{idx_to_type[p1]}_then_{idx_to_type[p2]}" for p1, p2 in zip(oof_pred1, oof_pred2)]
    df_oof = pd.DataFrame({"file": oof_names, "true_label": oof_true_labels, "pred_label": pred_labels})
    out_path = "/Users/bikramsingh/eeg_project/scratch/oof_predictions_participant_tp9tp10.csv"
    df_oof.to_csv(out_path, index=False)
    print(f"Saved out-of-fold predictions (for confusion matrix) to {out_path}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed=seed)
