"""
Classify beats (N, V, PAC) on a custom ECG CSV using the same pipeline as mitdb_windows.py (PAC = A+a in MIT-BIH training).

mitdb_windows.py uses MIT-BIH *reference annotations* for R-peaks. Your recording has no labels,
so this script:
  1) Loads your CSV signal
  2) Optionally resamples to 360 Hz (MIT-BIH rate — matches the trained model)
  3) Detects R-peaks with a bandpass + peak finder
  4) Extracts 180-sample windows centered on each peak (90 before / 90 after)
  5) Loads or trains a LogisticRegression on MIT-BIH (180 morphology + 2 RR-ratio features),
     saved as improved_lr_mitdb.joblib
  6) Runs predictions on each beat (optional --confidence for conservative abnormal calls)

Usage (from Capstone folder):
    python classify_custom_ecg.py --csv Subject_Z2_ECG.csv
    python classify_custom_ecg.py --csv Subject_Z2_ECG.csv --input-fs 250
    python classify_custom_ecg.py --csv Subject_Z2_ECG.csv --retrain

Important:
    - Set --input-fs to your recorder's real sample rate (default 360). Wrong Fs breaks windows
      and peak detection.
    - The classifier is trained on MIT-BIH (single-lead clinical DB). Consumer/hardware ECG may
      look different; predictions can be biased — treat as experimental unless you validate.
    - Lead polarity often differs from MIT-BIH MLII. By default this script picks +/- polarity
      by whichever gives higher mean P(N) from the model; use --no-auto-polarity or --invert
      to override.
"""

import argparse
import csv
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

from ecg_preprocessing import (
    get_mitdb_data_path,
    load_record_names,
    compute_class_weights,
    correct_baseline_wander,
    normalize_beat,
)

try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False

try:
    from scipy.signal import butter, filtfilt, find_peaks, resample
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

MODEL_FS = 180  # window length in samples; training uses 360 Hz -> 0.5 s per beat
HALF = 90
TARGET_SAMPLE_RATE_HZ = 360  # must match MIT-BIH preprocessing the model was trained on
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SCRIPT_DIR, "improved_lr_mitdb.joblib")

LABEL_NAMES = {0: "N", 1: "V", 2: "a"}

BEAT_SYMBOLS = set("NLRejAaJSFVE/fQ")
# PAC (class 2): A = atrial premature, a = aberrant atrial premature (MIT-BIH)
KEEP_LABELS = {"N": 0, "V": 1, "A": 2, "a": 2}


def load_subject_csv(csv_path):
    """Load ECG column from CSV (uses 'signal' column if present, else first column)."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return np.array([], dtype=float)
        header_lower = [h.strip().lower() for h in header]
        try:
            sig_col = header_lower.index("signal")
        except ValueError:
            sig_col = 0
        values = []
        for row in reader:
            if not row:
                continue
            try:
                values.append(float(row[sig_col]))
            except (IndexError, ValueError):
                continue
    sig = np.asarray(values, dtype=float)
    sig = sig[np.isfinite(sig)]
    return sig


def resample_signal(sig, fs_in, fs_out):
    """Resample whole trace so R-peak spacing matches MIT-BIH training rate."""
    if fs_in == fs_out:
        return sig.astype(float)
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for resampling when --input-fs != 360")
    n_out = max(2, int(len(sig) * fs_out / fs_in))
    return resample(sig.astype(float), n_out)


def detect_r_peaks_ecg(sig, fs):
    """
    Simple R-peak detector: bandpass roughly QRS band, then find peaks on |filt|^2.
    Using energy avoids missing R-waves when the lead is inverted (positive QRS -> negative
    in hardware); peak times match the underlying filtered signal.
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for R-peak detection (install scipy)")

    sig = sig.astype(float)
    sig = sig - np.nanmean(sig)
    nyq = 0.5 * fs
    low, high = 5.0 / nyq, min(40.0 / nyq, 0.99)
    if high <= low:
        high = 0.99
    b, a = butter(2, [low, high], btype="band")
    filt = filtfilt(b, a, sig)

    env = filt.astype(float) ** 2
    duration_s = len(sig) / float(fs)
    # Short consumer clips often pick T-waves / noise with the default threshold; stricter
    # rules improve RR stability and beat alignment for ResNet+RR and logistic baselines.
    if duration_s < 30.0:
        prom_mult = 0.58
        distance = max(int(0.32 * fs), 1)
    else:
        prom_mult = 0.35
        distance = max(int(0.25 * fs), 1)
    prominence = np.std(env) * prom_mult
    peaks, _ = find_peaks(env, distance=distance, prominence=max(prominence, 1e-18))
    return peaks


def windows_from_peaks(sig, peaks):
    """Extract (n_peaks, 180) windows centered on each peak index."""
    X = []
    used_peaks = []
    for p in peaks:
        start, end = p - HALF, p + HALF
        if start < 0 or end > len(sig):
            continue
        w = sig[start:end]
        if len(w) != MODEL_FS:
            continue
        X.append(w)
        used_peaks.append(p)
    return np.array(X, dtype=float), np.array(used_peaks, dtype=int)


def preprocess_windows_only(X_raw):
    """Same per-beat pipeline as preprocess_beats (detrend + normalize)."""
    X_out = []
    for i in range(len(X_raw)):
        w = correct_baseline_wander(X_raw[i])
        w = normalize_beat(w)
        X_out.append(w)
    return np.array(X_out, dtype=float)


def compute_rr_features(peak_indices):
    """
    RR-interval ratio features for each beat: [RR_pre/median, RR_post/median].
    PVCs are premature (ratio << 1) with compensatory pauses (ratio >> 1);
    regular sinus beats have both ratios ~1.0.
    """
    n = len(peak_indices)
    if n < 2:
        return np.ones((n, 2), dtype=float)
    rr = np.diff(peak_indices).astype(float)
    med = np.median(rr)
    if med == 0:
        med = 1.0
    feats = np.ones((n, 2), dtype=float)
    for i in range(n):
        feats[i, 0] = (rr[i - 1] / med) if i > 0 else (rr[0] / med)
        feats[i, 1] = (rr[i] / med) if i < n - 1 else (rr[-1] / med)
    return feats


def build_features(X_raw, peak_indices, polarity=1):
    """Preprocess windows, compute RR features, and concatenate into feature matrix."""
    X_morph = preprocess_windows_only(X_raw * polarity)
    rr = compute_rr_features(peak_indices)
    return np.hstack([X_morph, rr])


def choose_polarity_sign(model, X_raw, peak_indices):
    """
    Try raw windows as-is and negated; return +1 or -1 that maximizes mean P(N).
    Uses full feature vector (morphology + RR) so evaluation matches prediction.
    """
    if not hasattr(model, "predict_proba"):
        return 1
    best_sign, best_score = 1, -1.0
    for sign in (1, -1):
        X = build_features(X_raw, peak_indices, polarity=sign)
        pr = model.predict_proba(X)
        score = float(np.mean(pr[:, 0]))
        if score > best_score:
            best_score = score
            best_sign = sign
    return best_sign


def _load_mitdb_with_rr():
    """
    Load all MIT-BIH records and return windows, labels, RR features, and record IDs.
    RR intervals are computed from ALL beat annotations (not just N/V/a) so that
    intervals around non-kept beats are still physiologically correct.
    """
    if not HAS_WFDB:
        raise RuntimeError("wfdb is required for MIT-BIH training (pip install wfdb)")
    data_path = get_mitdb_data_path(_SCRIPT_DIR)
    record_names = load_record_names(data_path)

    X_list, y_list, rr_list, rid_list = [], [], [], []
    orig_dir = os.getcwd()

    for rname in record_names:
        try:
            os.chdir(data_path)
            rec = wfdb.rdrecord(rname)
            ann = wfdb.rdann(rname, "atr")
            os.chdir(orig_dir)

            ecg = rec.p_signal[:, 0]
            symbols = np.array(ann.symbol)
            samples = ann.sample

            beat_mask = np.array([s in BEAT_SYMBOLS for s in symbols])
            beat_samples = samples[beat_mask]
            beat_symbols = symbols[beat_mask]

            rr_all = compute_rr_features(beat_samples)

            for i, (pos, sym) in enumerate(zip(beat_samples, beat_symbols)):
                if sym not in KEEP_LABELS:
                    continue
                s, e = pos - HALF, pos + HALF
                if s < 0 or e > len(ecg):
                    continue
                w = ecg[s:e]
                if len(w) != MODEL_FS:
                    continue
                X_list.append(w)
                y_list.append(KEEP_LABELS[sym])
                rr_list.append(rr_all[i])
                rid_list.append(rname)
        except Exception as exc:
            os.chdir(orig_dir)
            print(f"  Skip {rname}: {exc}")

    os.chdir(orig_dir)
    return (np.array(X_list), np.array(y_list),
            np.array(rr_list), np.array(rid_list))


def train_mitdb_improved():
    """
    Train on MIT-BIH with morphology + RR-interval features.
    Logistic regression trains quickly and uses RR ratios to separate premature / ectopic timing
    from regular sinus rhythm.
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score as sk_f1

    print("Loading MIT-BIH records with RR features...")
    X_raw, y, rr, record_ids = _load_mitdb_with_rr()
    X_morph = preprocess_windows_only(X_raw)
    X = np.hstack([X_morph, rr])
    print(f"  {len(X)} beats, feature dim {X.shape[1]} (180 morph + 2 RR)")

    unique_recs = np.unique(record_ids)
    np.random.seed(42)
    np.random.shuffle(unique_recs)
    n = len(unique_recs)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_recs = set(unique_recs[:n_train])
    val_recs = set(unique_recs[n_train:n_train + n_val])
    test_recs = set(unique_recs[n_train + n_val:])

    tr = np.array([r in train_recs for r in record_ids])
    va = np.array([r in val_recs for r in record_ids])
    te = np.array([r in test_recs for r in record_ids])
    X_train, y_train = X[tr], y[tr]
    X_val, y_val = X[va], y[va]
    X_test, y_test = X[te], y[te]

    cw = compute_class_weights(y_train)
    print(f"  Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

    model = LogisticRegression(
        class_weight=cw, solver="lbfgs", C=1.0, max_iter=2000, random_state=42
    )
    print("Training LogisticRegression (morph + RR)...")
    model.fit(X_train, y_train)

    for name, Xp, yp in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
        pred = model.predict(Xp)
        ba = balanced_accuracy_score(yp, pred, adjusted=False)
        mf1 = sk_f1(yp, pred, average="macro", zero_division=0, labels=[0, 1, 2])
        print(f"  {name:4s}  balanced-acc={ba:.4f}  macro-F1={mf1:.4f}")

    return model


def load_or_train_model(retrain):
    if not retrain and os.path.isfile(MODEL_PATH):
        print(f"Loading saved model: {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    print("Training improved model on MIT-BIH (may take a minute)...")
    model = train_mitdb_improved()
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Classify custom ECG CSV (N / V / a)")
    parser.add_argument("--csv", default="Subject_Z2_ECG.csv", help="Path to CSV with ECG column")
    parser.add_argument(
        "--input-fs",
        type=float,
        default=360,
        help="Sampling rate of your CSV in Hz. Default 360. If your device is e.g. 250, pass 250.",
    )
    parser.add_argument("--retrain", action="store_true", help="Retrain and overwrite saved baseline model")
    parser.add_argument("--out", default=None, help="Optional path to save beat predictions CSV")
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Force negative polarity on beat windows (after peak alignment). Overrides auto.",
    )
    parser.add_argument(
        "--no-auto-polarity",
        action="store_true",
        help="Do not search +/- polarity; use raw window sign (+1).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.0,
        help="Min probability to accept an abnormal (V/a) prediction; below this default to N. "
             "Try 0.7 for conservative screening. 0 = disabled (use model argmax).",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.isfile(csv_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(script_dir, os.path.basename(csv_path))
        if os.path.isfile(alt):
            csv_path = alt
        else:
            raise FileNotFoundError(f"CSV not found: {args.csv} (also tried {alt})")

    sig = load_subject_csv(csv_path)
    print(f"Loaded {csv_path}: {len(sig)} samples")

    fs_in = args.input_fs
    if fs_in != TARGET_SAMPLE_RATE_HZ:
        print(f"Resampling {fs_in} Hz -> {TARGET_SAMPLE_RATE_HZ} Hz for model compatibility")
        sig = resample_signal(sig, fs_in, TARGET_SAMPLE_RATE_HZ)
        fs = TARGET_SAMPLE_RATE_HZ
    else:
        fs = fs_in

    peaks = detect_r_peaks_ecg(sig, fs)
    print(f"Detected {len(peaks)} candidate R-peaks")

    X_raw, peak_idx = windows_from_peaks(sig, peaks)
    print(f"Extracted {len(X_raw)} full 180-sample windows")

    if len(X_raw) == 0:
        print("No valid beats — check --input-fs or signal quality.")
        return

    model = load_or_train_model(args.retrain)
    if args.invert:
        polarity = -1
        print("Beat-window polarity: -1 (--invert)")
    elif args.no_auto_polarity:
        polarity = 1
        print("Beat-window polarity: +1 (--no-auto-polarity)")
    else:
        polarity = choose_polarity_sign(model, X_raw, peak_idx)
        print(f"Beat-window polarity: {polarity:+d} (auto: max mean P(N))")

    X_feat = build_features(X_raw, peak_idx, polarity)
    y_pred = model.predict(X_feat)
    proba = model.predict_proba(X_feat) if hasattr(model, "predict_proba") else None

    conf = args.confidence
    if conf > 0 and proba is not None:
        for i in range(len(y_pred)):
            if y_pred[i] != 0 and proba[i, int(y_pred[i])] < conf:
                y_pred[i] = 0
        print(f"Confidence threshold: {conf} (abnormal beats below this -> N)")

    counts = {LABEL_NAMES[k]: int(np.sum(y_pred == k)) for k in LABEL_NAMES}
    print("\nPredicted beat counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    print("\nPer-beat (sample_index@360Hz, label):")
    for i, (idx, lab) in enumerate(zip(peak_idx, y_pred)):
        line = f"  beat {i+1:4d}  peak_sample={idx:6d}  -> {LABEL_NAMES[int(lab)]}"
        if proba is not None:
            line += f"  P(N,V,a)={proba[i].round(3)}"
        print(line)

    if args.out:
        rows = []
        for i, (idx, lab) in enumerate(zip(peak_idx, y_pred)):
            row = {"beat_index": i + 1, "peak_sample_360hz": idx, "label": LABEL_NAMES[int(lab)]}
            if proba is not None:
                row["prob_N"] = proba[i, 0]
                row["prob_V"] = proba[i, 1]
                row["prob_a"] = proba[i, 2]
            rows.append(row)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
