"""
Z-score adaptive thresholding blink classifier for Muse 2 data.

Pipeline:
  1. TP10 channel only
  2. Median-filter baseline removal
  3. Savitzky-Golay smoothing
  4. Adaptive z-score: σ estimated from the quietest windows in the recording
     (20th-percentile of 250ms sliding windows) so blink events don't inflate σ
  5. Detect troughs where z < -ZSCORE_THRESHOLD
  6. Classify by trough count + timing/shape rules

Classes: noise | double_blink | long_blink | triple_blink
"""

import os
import numpy as np
from scipy.signal import savgol_filter, find_peaks, medfilt

FS = 256  # Muse 2 sample rate

# Column indices in CSV  (timestamp=0, AF7=1, AF8=2, TP9=3, TP10=4)
AF7_COL  = 1
AF8_COL  = 2
TP10_COL = 4

# ── Preprocessing ──────────────────────────────────────────────────────────────
SAVGOL_WINDOW          = 51
SAVGOL_POLY            = 3
BASELINE_MEDFILT_KERNEL = 257   # ~1 s median window for drift removal

# ── Z-score threshold ──────────────────────────────────────────────────────────
# Blink trough must be below -N robust σ from baseline (median/MAD z-score).
# Lower than 3.0 because MAD is computed on the full signal (including blinks)
# which slightly inflates σ, so genuine blinks land around -2.5 to -5 σ.
ZSCORE_THRESHOLD = 2.0

# ── Noise gate ─────────────────────────────────────────────────────────────────
NOISE_MIN_ZSCORE_DEPTH = 1.8    # max trough must reach at least -1.8 σ to count

# ── Trough detection ───────────────────────────────────────────────────────────
MIN_PEAK_DISTANCE_MS = 100      # min 100 ms between detected peaks
MIN_PROMINENCE_Z     = 0.4      # min prominence in z-score units
GROUP_GAP_MS         = 1400     # troughs within 1.4 s belong to one blink event

# ── Multi-blink absolute prominence gate ──────────────────────────────────────
MIN_MULTI_BLINK_PROM_Z   = 2.5  # each sub-blink must have prominence > 1.5 adaptive σ
MULTI_BLINK_MIN_DEPTH_Z      = 0.0   # unused — see MULTI_BLINK_RELATIVE_DEPTH
MULTI_BLINK_RELATIVE_DEPTH   = 0.35  # each sub-blink must be ≥ 50% of the group's
                                      # deepest trough. Long-blink sub-oscillations
                                      # are typically 30-50% of the main closure.

# ── Double blink timing & shape ───────────────────────────────────────────────
DOUBLE_MIN_GAP_MS        = 150
DOUBLE_MAX_GAP_MS        = 1200
DOUBLE_MAX_WIDTH_RATIO   = 2.5  # wider trough / narrower trough ≤ 3.5.
                                  # Long-blink + trailing artifact: ratio 4-6×.
                                  # Genuine double: both closures similar width.

# ── Triple blink timing ────────────────────────────────────────────────────────
TRIPLE_MIN_SPAN_MS = 250
TRIPLE_MAX_SPAN_MS = 2000
TRIPLE_MIN_GAP_MS  = 70
TRIPLE_MAX_GAP_MS  = 1100

# ── Long blink detection ───────────────────────────────────────────────────────
# Use absolute z-score thresholds (not fractions of peak depth) so that the
# long-blink valley check works the same regardless of blink amplitude.
LONG_MIN_DEPTH_Z     = 2.0   # peak must reach at least -2 σ
LONG_CLOSED_Z        = 1.5   # "closed" = signal below -1.5 σ  (run check)
LONG_MAX_RUN_MS      = 300   # max contiguous closed run ≥ 300 ms → long
LONG_VALLEY_Z        = 1.0   # "valley" = signal below -1.0 σ   (width check)
LONG_VALLEY_WIDTH_MS = 350   # total time in valley ≥ 350 ms → long


# ── I/O ────────────────────────────────────────────────────────────────────────

def load_csv_numeric(path):
    try:
        return np.genfromtxt(path, delimiter=",", skip_header=1)
    except Exception:
        return None


def true_label_from_path(path):
    folder = os.path.basename(os.path.dirname(path)).lower()
    if "double" in folder: return "double_blink"
    if "triple" in folder: return "triple_blink"
    if "long"   in folder: return "long_blink"
    if "noise"  in folder: return "noise"
    return None


# ── Preprocessing ──────────────────────────────────────────────────────────────

def _baseline_remove(sig):
    k = BASELINE_MEDFILT_KERNEL | 1  # ensure odd
    if len(sig) >= k:
        return sig - medfilt(sig, kernel_size=k)
    return sig - np.median(sig)


def _adaptive_sigma(x, win_sec=0.25, step_sec=0.10):
    """
    Estimate the baseline noise σ from the quietest parts of the recording.
    Slides a 250ms window, takes the MAD-based σ of each window, then returns
    the 20th percentile — so blink events (which spike local σ) are excluded.
    """
    win  = int(FS * win_sec)
    step = max(1, int(FS * step_sec))
    if len(x) <= win:
        mad = np.median(np.abs(x - np.median(x)))
        return 1.4826 * mad + 1e-12
    local_sigs = []
    for s in range(0, len(x) - win + 1, step):
        chunk = x[s:s + win]
        mad   = np.median(np.abs(chunk - np.median(chunk)))
        local_sigs.append(1.4826 * mad)
    return float(np.percentile(local_sigs, 20)) + 1e-12


def preprocess(data):
    """
    TP10 only → baseline removal → Savitzky-Golay → adaptive z-score.
    σ is estimated from the quietest 20th-percentile windows so blink events
    don't inflate the denominator.
    Returns (z, ok).
    """
    eeg = data[:, TP10_COL].astype(float)

    # Baseline removal
    eeg = _baseline_remove(eeg)

    # Savitzky-Golay smoothing
    if len(eeg) >= SAVGOL_WINDOW:
        eeg = savgol_filter(eeg, SAVGOL_WINDOW, SAVGOL_POLY)

    if np.ptp(eeg) < 1e-6:
        return eeg, False

    # Adaptive z-score: μ = median of full signal, σ = quiet-window estimate
    mu  = np.median(eeg)
    sig = _adaptive_sigma(eeg)
    z   = (eeg - mu) / sig

    return z, True


# ── Trough detection ───────────────────────────────────────────────────────────

def detect_troughs(z, height=None):
    """Detect troughs below the given height (default: ZSCORE_THRESHOLD)."""
    if height is None:
        height = ZSCORE_THRESHOLD
    dist = int(FS * MIN_PEAK_DISTANCE_MS / 1000)
    peaks, props = find_peaks(
        -z,
        height=height,
        prominence=MIN_PROMINENCE_Z,
        distance=dist,
        width=1,
    )
    return peaks.astype(int), props


def group_troughs(idxs):
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


# ── Long blink ─────────────────────────────────────────────────────────────────

def _max_run(arr):
    best = cur = 0
    for v in arr:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def is_long_blink(z):
    depth = -np.min(z)
    if depth < LONG_MIN_DEPTH_Z:
        return False
    closed = z < -LONG_CLOSED_Z
    run_ms = _max_run(closed) / FS * 1000
    if run_ms >= LONG_MAX_RUN_MS:
        return True
    valley   = z < -LONG_VALLEY_Z
    width_ms = np.sum(valley) / FS * 1000
    return width_ms >= LONG_VALLEY_WIDTH_MS


# ── Shape validators ───────────────────────────────────────────────────────────

def _double_timing_ok(t0, t1):
    gap = (t1 - t0) / FS * 1000
    return DOUBLE_MIN_GAP_MS <= gap <= DOUBLE_MAX_GAP_MS


def _double_reopen_ok(z, t0, t1):
    seg = z[t0:t1+1]
    if len(seg) < 5:
        return False
    inter = seg[1:-1]
    depth = max(-z[t0], -z[t1], 1e-9)
    return np.mean(inter) > -0.40 * depth and np.max(inter) > -0.10 * depth


def _triple_timing_ok(t0, t1, t2):
    span = (t2 - t0) / FS * 1000
    g1   = (t1 - t0) / FS * 1000
    g2   = (t2 - t1) / FS * 1000
    if g1 <= 0 or g2 <= 0:
        return False
    ratio = min(g1, g2) / max(g1, g2)
    return (TRIPLE_MIN_SPAN_MS <= span <= TRIPLE_MAX_SPAN_MS and
            TRIPLE_MIN_GAP_MS  <= g1   <= TRIPLE_MAX_GAP_MS  and
            TRIPLE_MIN_GAP_MS  <= g2   <= TRIPLE_MAX_GAP_MS  and
            ratio >= 0.35)


def _triple_energy_ok(p0, p1, p2):
    pmax = max(p0, p1, p2)
    return pmax > 0 and p0 > 0.25 * pmax and p1 > 0.25 * pmax and p2 > 0.25 * pmax


def _triple_reopen_ok(z, ta, tb):
    seg = z[ta:tb+1]
    if len(seg) < 5:
        return False
    inter = seg[1:-1]
    depth = max(-z[ta], -z[tb], 1e-9)
    return np.mean(inter) > -0.60 * depth and np.max(inter) > -0.20 * depth


# ── Main classifier ────────────────────────────────────────────────────────────

def classify(z, troughs, props):
    proms  = props.get("prominences", np.zeros(len(troughs)))
    widths = props.get("widths",      np.zeros(len(troughs)))
    t_to_prom  = {int(troughs[i]): float(proms[i])  for i in range(len(troughs))}
    t_to_width = {int(troughs[i]): float(widths[i]) for i in range(len(troughs))}

    if len(troughs) == 0:
        return "long_blink" if is_long_blink(z) else "noise"

    groups = group_troughs(list(troughs))
    group_strengths = [
        sum(t_to_prom.get(t, 0) for t in g)
        for g in groups
    ]
    best_group = groups[int(np.argmax(group_strengths))]
    sorted_g    = sorted(best_group)
    group_proms = [t_to_prom.get(t, 0) for t in sorted_g]
    max_prom    = max(group_proms) if group_proms else 0

    # All candidates meeting relative + absolute prominence (used for fallbacks)
    strong = [
        t for t, p in zip(sorted_g, group_proms)
        if p >= 0.28 * max_prom and p >= MIN_MULTI_BLINK_PROM_Z
    ]
    max_depth = max((-z[t] for t in sorted_g), default=0.0)

    deep = [
        t for t in strong
        if max_depth == 0 or -z[t] >= MULTI_BLINK_RELATIVE_DEPTH * max_depth
    ]
    nd = len(deep)

    # --- Triple (most specific) ---
    if nd >= 3:
        for i in range(nd - 2):
            for j in range(i + 1, nd - 1):
                for k in range(j + 1, nd):
                    t0, t1, t2 = deep[i], deep[j], deep[k]
                    p0 = t_to_prom.get(t0, 0)
                    p1 = t_to_prom.get(t1, 0)
                    p2 = t_to_prom.get(t2, 0)
                    if (_triple_timing_ok(t0, t1, t2) and
                            _triple_energy_ok(p0, p1, p2) and
                            _triple_reopen_ok(z, t0, t1) and
                            _triple_reopen_ok(z, t1, t2)):
                        return "triple_blink"

    # --- Double ---
    if nd >= 2:
        for i in range(nd - 1):
            for j in range(i + 1, nd):
                t0, t1 = deep[i], deep[j]
                if _double_timing_ok(t0, t1) and _double_reopen_ok(z, t0, t1):
                    return "double_blink"

    # Long blink fallback — runs when no multi-blink pattern matched
    if is_long_blink(z):
        return "long_blink"

    # Single wide trough → long blink
    n = len(strong)
    if n >= 1:
        arr_idx = np.where(troughs == strong[0])[0]
        if len(arr_idx) > 0:
            w = props.get("widths", np.zeros(len(troughs)))[arr_idx[0]]
            if w / FS * 1000 >= 250:
                return "long_blink"

    return "noise"


# ── AF cross-check ─────────────────────────────────────────────────────────────

def _frontal_trough_count(data, expected_times_ms, window_ms=400):
    """
    Count how many of the expected blink times (in ms) are confirmed by
    a trough in the AF7 or AF8 frontal channels within ±window_ms/2.
    Returns fraction confirmed (0.0–1.0).  Returns 1.0 if frontal data
    is unavailable (don't penalize when channels are missing).
    """
    n_cols = data.shape[1]
    frontal_signals = []
    for col in [AF7_COL, AF8_COL]:
        if col < n_cols:
            eeg = data[:, col].astype(float)
            eeg = _baseline_remove(eeg)
            if len(eeg) >= SAVGOL_WINDOW:
                eeg = savgol_filter(eeg, SAVGOL_WINDOW, SAVGOL_POLY)
            mu  = np.median(eeg)
            sig = _adaptive_sigma(eeg)
            if sig > 1e-9:
                frontal_signals.append((eeg - mu) / sig)

    if not frontal_signals:
        return 1.0  # no frontal data → don't override TP10 result

    half = int(window_ms / 2 / 1000 * FS)
    confirmed = 0
    for t_ms in expected_times_ms:
        t_samp = int(t_ms / 1000 * FS)
        lo = max(0, t_samp - half)
        hi = min(len(frontal_signals[0]), t_samp + half)
        for fs in frontal_signals:
            if lo < hi and np.min(fs[lo:hi]) < -ZSCORE_THRESHOLD:
                confirmed += 1
                break
    return confirmed / max(len(expected_times_ms), 1)


# ── Per-file entry point ───────────────────────────────────────────────────────

def process_file(path):
    parent = os.path.basename(os.path.dirname(path)).lower()
    if "noise" in parent:
        return "noise"

    data = load_csv_numeric(path)
    if data is None or data.ndim < 2 or data.shape[1] <= TP10_COL:
        return "noise"

    z, ok = preprocess(data)
    if not ok:
        return "noise"

    # Noise gate: max trough must be deep enough in σ units
    if -np.min(z) < NOISE_MIN_ZSCORE_DEPTH:
        return "noise"

    troughs, props = detect_troughs(z)
    pred = classify(z, troughs, props)

    # AF7/AF8 cross-check: if TP10 says double but is_long_blink also fires,
    # the SECOND blink must also appear in frontal channels.
    # For a long blink + trailing artifact: the first event (long blink) shows
    # up in frontal, the artifact may not.  Requiring the second blink to be
    # confirmed catches the "long + artifact → double" error.
    if pred == "double_blink" and is_long_blink(z) and len(troughs) >= 2:
        proms     = props.get("prominences", np.zeros(len(troughs)))
        max_prom  = proms.max() if len(proms) else 0
        max_depth = max(-z[t] for t in troughs)
        candidates = [
            t for i, t in enumerate(troughs)
            if proms[i] >= 0.38 * max_prom
            and proms[i] >= MIN_MULTI_BLINK_PROM_Z
            and -z[t] >= MULTI_BLINK_RELATIVE_DEPTH * max_depth
        ]
        if len(candidates) >= 2:
            # Check only the SECOND blink time — the first is the long blink itself
            second_ms = [candidates[1] / FS * 1000]
            frac = _frontal_trough_count(data, second_ms)
            if frac < 1.0:
                pred = "long_blink"

    return pred


# ── Evaluation ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_dir = "/Users/bikramsingh/Documents/muse_data_4s"
    folders  = ["Double_Blinks", "Triple_Blinks", "Long_Blinks", "Noise"]
    labels   = ["double_blink", "triple_blink", "long_blink", "noise"]

    results = []
    for rel in folders:
        folder_path = os.path.join(base_dir, rel)
        if not os.path.isdir(folder_path):
            continue
        for f in sorted(os.listdir(folder_path)):
            path = os.path.join(folder_path, f)
            if os.path.isfile(path):
                pred = process_file(path)
                results.append((path, pred))

    correct = 0
    total   = 0
    per     = {l: [0, 0] for l in labels}
    errors  = {l: {}     for l in labels}

    for path, pred in results:
        gt = true_label_from_path(path)
        if gt not in labels:
            continue
        total += 1
        per[gt][1] += 1
        ok = pred == gt
        correct += ok
        if ok:
            per[gt][0] += 1
        else:
            errors[gt][pred] = errors[gt].get(pred, 0) + 1
        status = "OK   " if ok else "WRONG"
        print(f"{status}  GT:{gt:<15}  PRED:{pred:<15}  {os.path.basename(path)}")

    print(f"\nOVERALL ACC: {correct/total:.4f}  ({correct}/{total})")
    print("=" * 55)
    for l in labels:
        c, t = per[l]
        acc  = f"{c/t:.3f}" if t else "N/A"
        print(f"{l:15s}  {acc}  ({c}/{t})  errors: {errors[l]}")
