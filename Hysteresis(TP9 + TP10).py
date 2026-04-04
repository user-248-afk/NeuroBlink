import os
import numpy as np
from scipy.signal import savgol_filter, medfilt

FS = 256

TP9_COL = 1
TP10_COL = 4

SAVGOL_WINDOW = 51
SAVGOL_POLY = 3
BASELINE_MEDFILT_KERNEL = 257

GROUP_GAP_MS = 520

DOUBLE_MIN_MS = 180.0
DOUBLE_MAX_MS = 650.0

LONG_MIN_DEPTH = 0.05
LONG_DEPTH_SIGMA_MULT = 9.0

NOISE_MIN_PTP = 0.08

MAX_SHORT_TROUGH_WIDTH_MS = 360.0

DOUBLE_CLOSED_RUN_FRAC = 0.20
DOUBLE_MAX_CLOSED_RUN_MS = 120.0

HYST_HIGH = -0.35
HYST_LOW = -0.12
REFRACTORY_MS = 120


def load_csv_numeric(file_path):
    try:
        return np.genfromtxt(file_path, delimiter=",", skip_header=1)
    except Exception:
        return None


def pick_eeg_channel(data):
    return (data[:, TP9_COL] + data[:, TP10_COL]) / 2.0


def robust_sigma(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return 1.4826 * mad


def min_window_sigma(x, win_sec=0.25, step_sec=0.10):
    win = int(FS * win_sec)
    step = max(1, int(FS * step_sec))

    if len(x) <= win:
        return robust_sigma(x)

    sigs = []
    for s in range(0, len(x) - win + 1, step):
        sigs.append(robust_sigma(x[s:s + win]))

    return np.percentile(sigs, 20)


def preprocess_and_normalize(eeg):
    x = eeg.astype(float)

    k = BASELINE_MEDFILT_KERNEL
    if k % 2 == 0:
        k += 1

    if len(x) >= k:
        base = medfilt(x, kernel_size=k)
        x = x - base
    else:
        x = x - np.median(x)

    if len(x) >= SAVGOL_WINDOW:
        x = savgol_filter(x, SAVGOL_WINDOW, SAVGOL_POLY)

    scale = max(np.max(x), -np.min(x))
    if scale <= 1e-6:
        return x, False

    return x / scale, True


def detect_troughs_with_props(x):
    events = []

    in_blink = False
    start = None

    refractory = int(FS * REFRACTORY_MS / 1000)
    ref_count = 0

    for i, v in enumerate(x):
        if ref_count > 0:
            ref_count -= 1
            continue

        if not in_blink and v < HYST_HIGH:
            in_blink = True
            start = i

        elif in_blink and v > HYST_LOW:
            end = i

            trough = start + np.argmin(x[start:end + 1])
            width = end - start
            prom_val = -np.min(x[start:end + 1])

            events.append((trough, prom_val, width))

            in_blink = False
            ref_count = refractory

    if len(events) == 0:
        return np.array([]), {"prominences": [], "widths": []}

    troughs = np.array([e[0] for e in events])
    proms = np.array([e[1] for e in events])
    widths = np.array([e[2] for e in events])

    props = {"prominences": proms, "widths": widths}
    return troughs.astype(int), props


def group_indices(idxs):
    if len(idxs) == 0:
        return []

    groups = []
    cur = [idxs[0]]

    for t in idxs[1:]:
        dt = (t - cur[-1]) / FS * 1000

        if dt <= GROUP_GAP_MS:
            cur.append(t)
        else:
            groups.append(cur)
            cur = [t]

    groups.append(cur)
    return groups


def trough_maps(troughs, props):
    proms = props.get("prominences", np.zeros(len(troughs)))
    widths = props.get("widths", np.zeros(len(troughs)))

    idx_to_prom = {int(troughs[i]): float(proms[i]) for i in range(len(troughs))}
    idx_to_w = {int(troughs[i]): float(widths[i]) for i in range(len(troughs))}

    return idx_to_prom, idx_to_w


def width_ok_samples(w):
    return w <= FS * (MAX_SHORT_TROUGH_WIDTH_MS / 1000)


def double_reopen_ok(x, t0, t1):
    seg = x[t0:t1 + 1]
    depth = max(-x[t0], -x[t1], 1e-9)
    thr = -DOUBLE_CLOSED_RUN_FRAC * depth

    best = 0
    cur = 0

    for v in seg < thr:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0

    dur_ms = best / FS * 1000
    return dur_ms <= DOUBLE_MAX_CLOSED_RUN_MS


def classify_group(group, idx_to_prom, idx_to_w, x):
    if len(group) < 2:
        return "noise"

    times = sorted(group)
    widths = [idx_to_w.get(t, 0.0) for t in times]

    if len(times) >= 3:
        for i in range(len(times) - 2):
            t0 = times[i]
            t1 = times[i + 1]
            t2 = times[i + 2]

            g1 = (t1 - t0) / FS * 1000
            g2 = (t2 - t1) / FS * 1000
            span = (t2 - t0) / FS * 1000

            if (
                90 <= g1 <= 700 and
                90 <= g2 <= 700 and
                240 <= span <= 1250 and
                width_ok_samples(widths[i]) and
                width_ok_samples(widths[i + 1]) and
                width_ok_samples(widths[i + 2])
            ):
                return "triple_blink"

    for i in range(len(times) - 1):
        t0 = times[i]
        t1 = times[i + 1]

        d = (t1 - t0) / FS * 1000

        if not (DOUBLE_MIN_MS <= d <= DOUBLE_MAX_MS):
            continue

        if not (
            width_ok_samples(widths[i]) and
            width_ok_samples(widths[i + 1])
        ):
            continue

        if not double_reopen_ok(x, t0, t1):
            continue

        return "double_blink"

    return "noise"


def is_long_blink(x):
    sig = min_window_sigma(x)
    depth = -np.min(x)

    min_depth_strict = max(LONG_MIN_DEPTH, LONG_DEPTH_SIGMA_MULT * sig)

    if depth >= min_depth_strict:
        return True

    thr = -0.24 * depth
    closed = x < thr

    run = 0
    best = 0

    for v in closed:
        if v:
            run += 1
            best = max(best, run)
        else:
            run = 0

    dur_ms = best / FS * 1000
    if dur_ms >= 220:
        return True

    valley = x < (-0.12 * depth)
    width = np.sum(valley) / FS * 1000
    if width >= 320:
        return True

    return False


def process_file(file_path):
    # normal classification for every file, including noise files
    data = load_csv_numeric(file_path)
    if data is None:
        return "noise"

    if data.ndim < 2 or data.shape[0] < 10:
        return "noise"

    eeg = pick_eeg_channel(data)
    x, ok = preprocess_and_normalize(eeg)

    if not ok or np.ptp(x) < NOISE_MIN_PTP:
        return "noise"

    troughs, props = detect_troughs_with_props(x)

    if len(troughs) > 0:
        groups = group_indices(troughs)
        idx_to_prom, idx_to_w = trough_maps(troughs, props)

        best = max(groups, key=lambda g: sum(idx_to_prom.get(t, 0) for t in g))
        label = classify_group(best, idx_to_prom, idx_to_w, x)

        if label != "noise":
            return label

    if is_long_blink(x):
        return "long_blink"

    return "noise"


def true_label_from_path(path):
    folder = os.path.basename(os.path.dirname(path)).lower()

    if "double" in folder:
        return "double_blink"
    if "triple" in folder:
        return "triple_blink"
    if "long" in folder:
        return "long_blink"
    if "noise" in folder:
        return "noise"

    return None


def process_all_files(directory, results):
    if not os.path.isdir(directory):
        return

    for f in sorted(os.listdir(directory)):
        if f.startswith("."):
            continue

        path = os.path.join(directory, f)

        if os.path.isfile(path):
            pred = process_file(path)
            results.append((path, pred))


if __name__ == "__main__":
    base_dir = "/Users/bikramsingh/Documents/muse_data"

    folders = [
        "Double_Blinks",
        "Triple_Blinks",
        "Long_Blinks",
        "Noise",
    ]

    results = []

    for rel in folders:
        process_all_files(os.path.join(base_dir, rel), results)

    if len(results) == 0:
        print("No files found.")
    else:
        for path, pred in results:
            print(path, pred)

    labels = ["long_blink", "double_blink", "triple_blink", "noise"]

    correct = 0
    total = 0
    per = {lab: [0, 0] for lab in labels}

    for path, pred in results:
        gt = true_label_from_path(path)

        if gt not in labels:
            continue

        total += 1
        per[gt][1] += 1

        if pred == gt:
            correct += 1
            per[gt][0] += 1

    if total > 0:
        print("OVERALL ACC:", correct / total)

    for lab in labels:
        c, t = per[lab]
        acc = (c / t) if t else 0.0
        print(lab, "ACC:", acc, "(", c, "/", t, ")")