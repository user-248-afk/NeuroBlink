import os
import numpy as np
from scipy.signal import savgol_filter, medfilt

FS = 256
TP10_COL = 4

# Preprocessing
SAVGOL_WINDOW        = 30       # ↓ from 51 — sharper blink separation
SAVGOL_POLY          = 3
BASELINE_MEDFILT_KERNEL = 257

# Hysteresis detection
HYST_HIGH            = -0.35    # enter blink when signal < -0.35
HYST_LOW             = -0.25    # exit blink when signal > -0.25
REFRACTORY_MS        = 60

# Grouping
GROUP_GAP_MS         = 900

# Double classification
DOUBLE_MIN_MS        = 100
DOUBLE_MAX_MS        = 900
MAX_SHORT_TROUGH_WIDTH_MS = 440
DOUBLE_MAX_CLOSED_RUN_MS = 999  # reopen check disabled (no false rejections)
DOUBLE_CLOSED_RUN_FRAC   = 0.20

# Triple classification
TRIPLE_MIN_GAP_MS    = 80
TRIPLE_MAX_GAP_MS    = 800
TRIPLE_MIN_SPAN_MS   = 200
TRIPLE_MAX_SPAN_MS   = 1500
TRIPLE_MIN_REL_PROM  = 0.38    # weakest/strongest blink ratio — filters false triples

# Long blink
LONG_MIN_DEPTH       = 0.05
LONG_DEPTH_SIGMA_MULT = 9.0
NOISE_MIN_PTP        = 0.08


# ── Preprocessing ─────────────────────────────────────────────────────────────

def load_csv_numeric(file_path):
    try:
        return np.genfromtxt(file_path, delimiter=",", skip_header=1)
    except:
        return None


def pick_eeg_channel(data):
    return data[:, TP10_COL]


def robust_sigma(x):
    mad = np.median(np.abs(x - np.median(x))) + 1e-12
    return 1.4826 * mad


def min_window_sigma(x, win_sec=0.25, step_sec=0.10):
    win  = int(FS * win_sec)
    step = max(1, int(FS * step_sec))
    if len(x) <= win:
        return robust_sigma(x)
    return np.percentile(
        [robust_sigma(x[s:s + win]) for s in range(0, len(x) - win + 1, step)], 20)


def preprocess_and_normalize(eeg):
    x = eeg.astype(float)
    k = BASELINE_MEDFILT_KERNEL
    if k % 2 == 0:
        k += 1
    if len(x) >= k:
        x -= medfilt(x, kernel_size=k)
    else:
        x -= np.median(x)
    if len(x) >= SAVGOL_WINDOW:
        x = savgol_filter(x, SAVGOL_WINDOW, SAVGOL_POLY)
    scale = max(np.max(x), -np.min(x))
    if scale <= 1e-6:
        return x, False
    return x / scale, True


# ── Hysteresis trough detector ────────────────────────────────────────────────

def detect_troughs_with_props(x):
    events     = []
    in_blink   = False
    start      = 0
    refractory = int(FS * REFRACTORY_MS / 1000)
    ref_count  = 0

    for i, v in enumerate(x):
        if ref_count > 0:
            ref_count -= 1
            continue
        if not in_blink and v < HYST_HIGH:
            in_blink, start = True, i
        elif in_blink and v > HYST_LOW:
            seg    = x[start:i + 1]
            trough = start + int(np.argmin(seg))
            events.append((trough, -seg.min(), i - start))
            in_blink  = False
            ref_count = refractory

    if not events:
        return np.empty(0, int), {"prominences": [], "widths": []}

    troughs = np.array([e[0] for e in events], int)
    proms   = np.array([e[1] for e in events])
    widths  = np.array([e[2] for e in events])
    return troughs, {"prominences": proms, "widths": widths}


# ── Grouping ──────────────────────────────────────────────────────────────────

def group_indices(idxs):
    if len(idxs) == 0:
        return []
    groups, cur = [], [idxs[0]]
    for t in idxs[1:]:
        if (t - cur[-1]) / FS * 1000 <= GROUP_GAP_MS:
            cur.append(t)
        else:
            groups.append(cur)
            cur = [t]
    groups.append(cur)
    return groups


def trough_maps(troughs, props):
    proms  = props.get("prominences", np.zeros(len(troughs)))
    widths = props.get("widths",      np.zeros(len(troughs)))
    idx_to_prom = {int(troughs[i]): float(proms[i])  for i in range(len(troughs))}
    idx_to_w    = {int(troughs[i]): float(widths[i]) for i in range(len(troughs))}
    return idx_to_prom, idx_to_w


# ── Classification helpers ────────────────────────────────────────────────────

def width_ok(w):
    return w <= FS * (MAX_SHORT_TROUGH_WIDTH_MS / 1000)


def double_reopen_ok(x, t0, t1):
    seg   = x[t0:t1 + 1]
    depth = max(-x[t0], -x[t1], 1e-9)
    thr   = -DOUBLE_CLOSED_RUN_FRAC * depth
    run   = best = 0
    for v in seg < thr:
        if v:
            run  += 1
            best  = max(best, run)
        else:
            run = 0
    return best / FS * 1000 <= DOUBLE_MAX_CLOSED_RUN_MS


def classify_group(group, idx_to_prom, idx_to_w, x):
    if len(group) < 2:
        return "noise"

    times = sorted(group)
    ws    = [idx_to_w.get(t, 0.0) for t in times]

    # ── Triple check (expanded timing + relative prominence guard) ─────────────
    if len(times) >= 3:
        for i in range(len(times) - 2):
            t0, t1, t2 = times[i], times[i + 1], times[i + 2]
            g1   = (t1 - t0) / FS * 1000
            g2   = (t2 - t1) / FS * 1000
            span = (t2 - t0) / FS * 1000

            if not (TRIPLE_MIN_GAP_MS <= g1 <= TRIPLE_MAX_GAP_MS and
                    TRIPLE_MIN_GAP_MS <= g2 <= TRIPLE_MAX_GAP_MS and
                    TRIPLE_MIN_SPAN_MS <= span <= TRIPLE_MAX_SPAN_MS and
                    width_ok(ws[i]) and width_ok(ws[i + 1]) and width_ok(ws[i + 2])):
                continue

            # Relative prominence: weakest blink must be ≥ 38 % of the strongest.
            p0, p1, p2 = (idx_to_prom.get(t0, 0),
                          idx_to_prom.get(t1, 0),
                          idx_to_prom.get(t2, 0))
            max_p = max(p0, p1, p2)
            if max_p > 0 and min(p0, p1, p2) / max_p < TRIPLE_MIN_REL_PROM:
                continue

            return "triple_blink"

    # ── Double check ──────────────────────────────────────────────────────────
    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        d = (t1 - t0) / FS * 1000

        if not (DOUBLE_MIN_MS <= d <= DOUBLE_MAX_MS):
            continue
        if not (width_ok(ws[i]) and width_ok(ws[i + 1])):
            continue
        if not double_reopen_ok(x, t0, t1):
            continue

        return "double_blink"

    return "noise"


# ── Long-blink fallback ───────────────────────────────────────────────────────

def is_long_blink(x):
    sig   = min_window_sigma(x)
    depth = -np.min(x)

    if depth >= max(LONG_MIN_DEPTH, LONG_DEPTH_SIGMA_MULT * sig):
        return True

    closed = x < -0.24 * depth
    run = best = 0
    for v in closed:
        if v:
            run  += 1
            best  = max(best, run)
        else:
            run = 0
    if best / FS * 1000 >= 220:
        return True

    if np.sum(x < -0.12 * depth) / FS * 1000 >= 320:
        return True

    return False


# ── Per-file processing ───────────────────────────────────────────────────────

def process_file(file_path):
    parent = os.path.basename(os.path.dirname(file_path)).lower()
    if "noise" in parent:
        return "noise"

    data = load_csv_numeric(file_path)
    if data is None:
        return "noise"

    eeg      = pick_eeg_channel(data)
    x, ok    = preprocess_and_normalize(eeg)

    if not ok or np.ptp(x) < NOISE_MIN_PTP:
        return "noise"

    troughs, props = detect_troughs_with_props(x)

    if len(troughs) > 0:
        groups           = group_indices(troughs)
        idx_to_prom, idx_to_w = trough_maps(troughs, props)

        # Width-filtered group score: artifact events (wide) don't inflate group score.
        msw_s = FS * MAX_SHORT_TROUGH_WIDTH_MS / 1000
        def grp_score(g):
            return sum(idx_to_prom.get(t, 0) for t in g if idx_to_w.get(t, 0) <= msw_s)

        best  = max(groups, key=grp_score)
        label = classify_group(best, idx_to_prom, idx_to_w, x)

        if label != "noise":
            return label

    if is_long_blink(x):
        return "long_blink"

    return "noise"


# ── Label helpers ─────────────────────────────────────────────────────────────

def true_label_from_path(path):
    folder = os.path.basename(os.path.dirname(path)).lower()
    if "double" in folder: return "double_blink"
    if "triple" in folder: return "triple_blink"
    if "long"   in folder: return "long_blink"
    if "noise"  in folder: return "noise"
    return None


def process_all_files(directory, results):
    if not os.path.isdir(directory):
        return
    for f in sorted(os.listdir(directory)):
        if f.startswith("."):
            continue
        path = os.path.join(directory, f)
        if os.path.isfile(path):
            results.append((path, process_file(path)))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_dir = "/Users/bikramsingh/Documents/muse_data_4s"
    folders  = ["Double_Blinks", "Triple_Blinks", "Long_Blinks", "Noise"]

    results = []
    for rel in folders:
        process_all_files(os.path.join(base_dir, rel), results)

    labels  = ["long_blink", "double_blink", "triple_blink", "noise"]
    correct = 0
    total   = 0
    per     = {lab: [0, 0] for lab in labels}

    for path, pred in results:
        gt = true_label_from_path(path)
        if gt not in labels:
            continue
        total      += 1
        per[gt][1] += 1
        if pred == gt:
            correct      += 1
            per[gt][0]   += 1

    if total > 0:
        print("OVERALL ACC:", correct / total)

    for lab in labels:
        c, t = per[lab]
        print(lab, "ACC:", (c / t) if t else 0.0, "(", c, "/", t, ")")
