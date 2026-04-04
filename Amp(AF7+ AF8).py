import os
import numpy as np
from scipy.signal import savgol_filter, find_peaks, medfilt

FS = 256

AF7_COL = 3
AF8_COL = 4

SAVGOL_WINDOW = 51
SAVGOL_POLY = 3
BASELINE_MEDFILT_KERNEL = 257

MIN_PEAK_DISTANCE_MS = 130
GROUP_GAP_MS = 520

DOUBLE_MIN_MS = 180.0
DOUBLE_MAX_MS = 650.0

MAX_SHORT_TROUGH_WIDTH_MS = 360.0

LONG_MIN_DEPTH = 0.05
LONG_DEPTH_SIGMA_MULT = 9.0

NOISE_MIN_PTP = 0.08

MIN_PROM_ABS = 0.04
PROM_SIGMA_MULT = 5.0

KEEP_PROM_FRAC = 0.55

DOUBLE_CLOSED_RUN_FRAC = 0.20
DOUBLE_MAX_CLOSED_RUN_MS = 120.0


def load_csv_numeric(file_path):
    try:
        return np.genfromtxt(file_path, delimiter=",", skip_header=1)
    except Exception:
        return None


def pick_eeg_channel(data):
    return (data[:, AF7_COL] + data[:, AF8_COL]) / 2.0


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
    # baseline remove + smooth + min-max normalize, then center
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

    x_min = np.min(x)
    x_max = np.max(x)
    denom = x_max - x_min
    if denom <= 1e-9:
        return x, False

    x01 = (x - x_min) / denom
    x_centered = x01 - 0.5
    return x_centered, True


def prominence_from_signal(x):
    sig = min_window_sigma(x)
    return max(MIN_PROM_ABS, PROM_SIGMA_MULT * sig)


def detect_troughs_with_props(x, prom):
    y = -x
    min_dist = int(FS * (MIN_PEAK_DISTANCE_MS / 1000))
    troughs, props = find_peaks(y, prominence=prom, distance=min_dist, width=1)
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


def prune_tremor_troughs(group, group_proms):
    if len(group) <= 2:
        return group

    dom = max(group_proms) if group_proms else 0.0
    if dom <= 0:
        return group

    kept = []
    for t, p in zip(group, group_proms):
        if p >= KEEP_PROM_FRAC * dom:
            kept.append(int(t))

    if len(group) >= 3 and len(kept) < 3:
        order = np.argsort(group_proms)[::-1]
        top3 = [int(group[i]) for i in order[:3]]
        top3.sort()
        return top3

    if len(kept) < 2:
        order = np.argsort(group_proms)[::-1]
        top2 = [int(group[i]) for i in order[:2]]
        top2.sort()
        return top2

    kept.sort()
    return kept


def double_ok(t0, t1):
    d = (t1 - t0) / FS * 1000
    return DOUBLE_MIN_MS <= d <= DOUBLE_MAX_MS


def width_ok_samples(w):
    return w <= FS * (MAX_SHORT_TROUGH_WIDTH_MS / 1000)


def max_run_below_ms(seg, thr):
    best = 0
    cur = 0
    for v in seg < thr:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best / FS * 1000


def double_reopen_ok(x, t0, t1):
    seg = x[t0:t1 + 1]
    depth = max(-x[t0], -x[t1], 1e-9)
    thr = -DOUBLE_CLOSED_RUN_FRAC * depth
    return max_run_below_ms(seg, thr) <= DOUBLE_MAX_CLOSED_RUN_MS


def classify_group(group, idx_to_prom, idx_to_w, x):
    # returns: "triple_blink" | "double_blink" | "noise"
    if len(group) < 2:
        return "noise"

    times = sorted(group)
    proms = [idx_to_prom.get(t, 0.0) for t in times]
    widths = [idx_to_w.get(t, 0.0) for t in times]

    # triple
    if len(times) >= 3:
        for i in range(len(times) - 2):
            t0, t1, t2 = times[i], times[i + 1], times[i + 2]

            g1 = (t1 - t0) / FS * 1000
            g2 = (t2 - t1) / FS * 1000
            span = (t2 - t0) / FS * 1000
            if not (70 <= g1 <= 700 and 70 <= g2 <= 700 and 260 <= span <= 1250):
                continue

            if not (
                width_ok_samples(widths[i]) and
                width_ok_samples(widths[i + 1]) and
                width_ok_samples(widths[i + 2])
            ):
                continue

            pmax = max(proms[i], proms[i + 1], proms[i + 2])
            if pmax <= 0:
                continue

            if not (
                proms[i] > 0.30 * pmax and
                proms[i + 1] > 0.30 * pmax and
                proms[i + 2] > 0.30 * pmax
            ):
                continue

            return "triple_blink"

    # double
    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        if not double_ok(t0, t1):
            continue
        if not (width_ok_samples(widths[i]) and width_ok_samples(widths[i + 1])):
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
            if run > best:
                best = run
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


def true_label_from_path(file_path):
    parent = os.path.basename(os.path.dirname(file_path)).lower()
    if "noise" in parent:
        return "noise"
    if "long" in parent:
        return "long_blink"
    if "double" in parent:
        return "double_blink"
    if "triple" in parent:
        return "triple_blink"
    return "unknown"


def process_file(file_path):
    # normal classification for every file, including noise files
    data = load_csv_numeric(file_path)
    if data is None or data.ndim < 2 or data.shape[0] < 10:
        return "noise"

    eeg = pick_eeg_channel(data)
    x, ok = preprocess_and_normalize(eeg)

    if not ok or np.ptp(x) < NOISE_MIN_PTP:
        return "noise"

    prom = prominence_from_signal(x)
    troughs, props = detect_troughs_with_props(x, prom)

    if len(troughs) > 0:
        groups = group_indices(troughs)
        idx_to_prom, idx_to_w = trough_maps(troughs, props)

        best = None
        best_score = -1.0
        for g in groups:
            sc = sum(idx_to_prom.get(t, 0.0) for t in g)
            if sc > best_score:
                best_score = sc
                best = g

        if best is not None:
            gp = [idx_to_prom.get(t, 0.0) for t in best]
            g2 = prune_tremor_troughs(best, gp)

            label = classify_group(g2, idx_to_prom, idx_to_w, x)
            if label != "noise":
                return label

    if is_long_blink(x):
        return "long_blink"

    return "noise"


def process_all_files(directory):
    out = []
    if not os.path.isdir(directory):
        return out

    for f in sorted(os.listdir(directory)):
        if f.startswith("."):
            continue
        path = os.path.join(directory, f)
        if os.path.isfile(path):
            pred = process_file(path)
            out.append((path, pred))
    return out


if __name__ == "__main__":
    base_dir = "/Users/bikramsingh/Documents/muse_data"

    folders = [
        "Long_Blinks",
        "Triple_Blinks",
        "Double_Blinks",
        "Noise",
    ]

    results = []
    for rel in folders:
        results.extend(process_all_files(os.path.join(base_dir, rel)))

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