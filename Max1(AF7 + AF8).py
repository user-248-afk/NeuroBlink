import os
import numpy as np
from scipy.signal import savgol_filter, find_peaks, medfilt

FS = 256
AF7_COL = 2
AF8_COL = 3

SAVGOL_WINDOW = 51
SAVGOL_POLY = 3
BASELINE_MEDFILT_KERNEL = 257

MIN_PEAK_DISTANCE_MS = 130
GROUP_GAP_MS = 520

DOUBLE_MIN_MS = 180.0
DOUBLE_MAX_MS = 650.0

TRIPLE_MIN_MS = 380.0
TRIPLE_MAX_MS = 980.0
TRIPLE_GAP_MIN_MS = 120.0
TRIPLE_GAP_MAX_MS = 520.0

LONG_MIN_MS = 180.0
LONG_MAX_MS = 5000.0

LONG_MIN_DEPTH = 0.05
LONG_DEPTH_SIGMA_MULT = 9.0

NOISE_MIN_PTP = 0.08

MIN_PROM_ABS = 0.04
PROM_SIGMA_MULT = 5.0

KEEP_PROM_FRAC = 0.55
MAX_SHORT_TROUGH_WIDTH_MS = 360.0

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


def triple_energy_ok(p0, p1, p2):
    pmax = max(p0, p1, p2)
    if pmax <= 0:
        return False
    return (
        p0 > 0.30 * pmax and
        p1 > 0.30 * pmax and
        p2 > 0.30 * pmax
    )


def triple_timing_ok(t0, t1, t2):
    d = (t2 - t0) / FS * 1000
    g1 = (t1 - t0) / FS * 1000
    g2 = (t2 - t1) / FS * 1000
    if g1 <= 0 or g2 <= 0:
        return False
    ratio = min(g1, g2) / max(g1, g2)
    return (
        260 <= d <= 1250 and
        70 <= g1 <= 700 and
        70 <= g2 <= 700 and
        ratio >= 0.45
    )


def classify_group(group, idx_to_prom, idx_to_w, x):
    if len(group) < 2:
        return "noise"

    times = sorted(group)
    proms = [idx_to_prom.get(t, 0.0) for t in times]
    widths = [idx_to_w.get(t, 0.0) for t in times]

    if len(times) >= 3:
        for i in range(len(times) - 2):
            for j in range(i + 1, len(times) - 1):
                for k in range(j + 1, len(times)):
                    t0, t1, t2 = times[i], times[j], times[k]

                    if not triple_timing_ok(t0, t1, t2):
                        continue
                    if not (
                        width_ok_samples(widths[i]) and
                        width_ok_samples(widths[j]) and
                        width_ok_samples(widths[k])
                    ):
                        continue
                    if not triple_energy_ok(proms[i], proms[j], proms[k]):
                        continue

                    return "triple_blink"

    for i in range(len(times) - 1):
        for j in range(i + 1, len(times)):
            t0, t1 = times[i], times[j]
            if not double_ok(t0, t1):
                continue
            if not (
                width_ok_samples(widths[i]) and
                width_ok_samples(widths[j])
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


def process_file(file_path):
    data = load_csv_numeric(file_path)
    if data is None:
        print(file_path, "noise")
        return "noise"

    eeg = pick_eeg_channel(data)
    x, ok = preprocess_and_normalize(eeg)

    if not ok or np.ptp(x) < NOISE_MIN_PTP:
        print(file_path, "noise")
        return "noise"

    prom = prominence_from_signal(x)
    troughs, props = detect_troughs_with_props(x, prom)

    if len(troughs) > 0:
        groups = group_indices(troughs)
        idx_to_prom, idx_to_w = trough_maps(troughs, props)

        best = None
        best_score = -1
        for g in groups:
            sc = sum(idx_to_prom.get(t, 0) for t in g)
            if sc > best_score:
                best_score = sc
                best = g

        gp = [idx_to_prom.get(t, 0) for t in best]
        g2 = prune_tremor_troughs(best, gp)
        label = classify_group(g2, idx_to_prom, idx_to_w, x)

        if label != "noise":
            print(file_path, label)
            return label

    if is_long_blink(x):
        print(file_path, "long_blink")
        return "long_blink"

    print(file_path, "noise")
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