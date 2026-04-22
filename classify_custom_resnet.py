"""
Run a trained ResNet-1D+RR (from train_cnn_resnet.py) on a custom ECG CSV.

Waveform branch matches training preprocess; second input is RR ratios (2-D) scaled with
resnet1d_mitdb_rr_scaler.joblib (fit on train only during training).

Train / refresh artifacts:
    python train_cnn_resnet.py

Then classify:
    python classify_custom_resnet.py --csv Subject_Z2_ECG.csv

Short or noisy strips: use --neutral-rr to feed the train-mean RR vector (zeros after scaling)
so morphology dominates when RR timing is unreliable.

Legacy single-input .keras (waveform only) still loads if you omit the scaler requirement.
"""

import argparse
import csv
import os

import joblib
import numpy as np

try:
    from tensorflow import keras
except ImportError:
    raise SystemExit("TensorFlow is required: pip install tensorflow")

from classify_custom_ecg import (
    MODEL_FS,
    TARGET_SAMPLE_RATE_HZ,
    detect_r_peaks_ecg,
    load_subject_csv,
    preprocess_windows_only,
    resample_signal,
    windows_from_peaks,
)
from ecg_preprocessing import compute_rr_features
from train_cnn_resnet import (
    proba_to_labels,
    resnet_model_path,
    resnet_pac_threshold_path,
    resnet_rr_scaler_path,
)

LABEL_NAMES = {0: "N", 1: "V", 2: "a"}


def _model_has_rr_input(model):
    return hasattr(model, "inputs") and len(model.inputs) == 2


def _predict_batched(model, X_wave, rr_scaled, batch_size=256):
    """X_wave: (n, 180, 1); rr_scaled: (n, 2) or None for single-input models."""
    if rr_scaled is None:
        return model.predict(X_wave, verbose=0, batch_size=batch_size)
    return model.predict(
        {"wave": X_wave, "rr": rr_scaled},
        verbose=0,
        batch_size=batch_size,
    )


def choose_polarity_resnet(
    model, X_raw, peak_idx, rr_scaled, n_boost, rr_low, rr_high, use_rr
):
    """
    Pick +1 or -1 on raw windows to maximize mean P(N); ResNet is not sign-invariant.
    RR does not depend on waveform sign. Optional sinus prior when n_boost is non-zero
    (see classify_custom_ecg-style boosts if wired; kept for API compat).
    """
    del n_boost, rr_low, rr_high  # optional post-hoc; defaults off in CLI
    rr_for_pred = rr_scaled if use_rr else None
    best_sign, best_score = 1, -1.0
    for sign in (1, -1):
        X_pre = preprocess_windows_only(X_raw * sign)
        Xr = X_pre.reshape(-1, MODEL_FS, 1)
        pr = _predict_batched(model, Xr, rr_for_pred)
        score = float(np.mean(pr[:, 0]))
        if score > best_score:
            best_score = score
            best_sign = sign
    return best_sign


def main():
    parser = argparse.ArgumentParser(
        description="Classify beats with trained ResNet-1D+RR (MIT-BIH weights)"
    )
    parser.add_argument("--csv", default="Subject_Z2_ECG.csv", help="ECG CSV (column 'signal')")
    parser.add_argument(
        "--input-fs",
        type=float,
        default=360,
        help="Sampling rate of the CSV in Hz (default 360). Resamples to 360 for the model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Path to saved Keras model (default: {resnet_model_path()})",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Force negative polarity on beat windows (overrides auto).",
    )
    parser.add_argument(
        "--no-auto-polarity",
        action="store_true",
        help="Use +1 polarity only (no search).",
    )
    parser.add_argument("--out", default=None, help="Optional CSV path for per-beat predictions")
    parser.add_argument(
        "--neutral-rr",
        action="store_true",
        help="Ignore computed RR: use scaler mean (zeros scaled) so predictions rely on waveform.",
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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model or resnet_model_path(script_dir)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"No saved ResNet at {model_path}. Run from the Capstone folder:\n"
            "  python train_cnn_resnet.py\n"
            "That trains on MIT-BIH and writes resnet1d_mitdb.keras next to this script."
        )

    print(f"Loading model: {model_path}")
    # compile=False: custom label-smoothing loss from training is not deserializable.
    model = keras.models.load_model(model_path, compile=False)
    use_rr = _model_has_rr_input(model)
    scaler_path = resnet_rr_scaler_path(script_dir)
    rr_scaler = None
    if use_rr:
        aux_w = int(model.inputs[1].shape[1])
        if aux_w != 2:
            raise ValueError(
                f"Expected RR auxiliary width 2, got {aux_w}. "
                "Retrain with train_cnn_resnet.py for a matching checkpoint."
            )
        if not os.path.isfile(scaler_path):
            raise FileNotFoundError(
                f"Two-input model requires RR scaler: {scaler_path}\n"
                "Run python train_cnn_resnet.py to generate it."
            )
        rr_scaler = joblib.load(scaler_path)
        if int(rr_scaler.n_features_in_) != 2:
            raise ValueError(
                f"Scaler expects {rr_scaler.n_features_in_} features; need 2. Regenerate with train_cnn_resnet.py."
            )
        print(f"Loaded RR scaler: {scaler_path}")
    else:
        print("Single-input checkpoint (waveform only); RR scaler not used.")

    pac_threshold = None
    thr_path = resnet_pac_threshold_path(script_dir)
    if os.path.isfile(thr_path):
        pac_threshold = float(joblib.load(thr_path)["pac_threshold"])
        print(
            f"Loaded PAC threshold tau={pac_threshold:.4f} ({os.path.basename(thr_path)})"
        )

    sig = load_subject_csv(csv_path)
    print(f"Loaded {csv_path}: {len(sig)} samples")

    fs_in = args.input_fs
    if fs_in != TARGET_SAMPLE_RATE_HZ:
        print(f"Resampling {fs_in} Hz -> {TARGET_SAMPLE_RATE_HZ} Hz")
        sig = resample_signal(sig, fs_in, TARGET_SAMPLE_RATE_HZ)
        fs = TARGET_SAMPLE_RATE_HZ
    else:
        fs = fs_in

    peaks = detect_r_peaks_ecg(sig, fs)
    print(f"Detected {len(peaks)} candidate R-peaks")

    X_raw, peak_idx = windows_from_peaks(sig, peaks)
    print(f"Extracted {len(X_raw)} full {MODEL_FS}-sample windows")

    if len(X_raw) == 0:
        print("No valid beats — check --input-fs or signal quality.")
        return

    rr_raw = compute_rr_features(np.asarray(peak_idx, dtype=int))
    if rr_scaler is not None:
        if args.neutral_rr:
            rr_scaled = np.zeros((len(peak_idx), 2), dtype=np.float32)
            print(
                "RR branch: using train mean (neutral-rr) — waveform drives classification."
            )
        else:
            rr_scaled = rr_scaler.transform(rr_raw).astype(np.float32)
    else:
        rr_scaled = None

    if args.invert:
        polarity = -1
        print("Beat-window polarity: -1 (--invert)")
    elif args.no_auto_polarity:
        polarity = 1
        print("Beat-window polarity: +1 (--no-auto-polarity)")
    else:
        polarity = choose_polarity_resnet(
            model, X_raw, peak_idx, rr_scaled, 0, 0.0, 0.0, use_rr
        )
        print(f"Beat-window polarity: {polarity:+d} (auto: max mean P(N))")

    X_pre = preprocess_windows_only(X_raw * polarity)
    Xr = X_pre.reshape(-1, MODEL_FS, 1)
    proba = _predict_batched(model, Xr, rr_scaled if use_rr else None)
    y_pred = proba_to_labels(proba, pac_threshold)

    counts = {LABEL_NAMES[k]: int(np.sum(y_pred == k)) for k in LABEL_NAMES}
    arch = "ResNet-1D+RR" if use_rr else "ResNet-1D"
    print(f"\nPredicted beat counts ({arch}):")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    print("\nPer-beat (sample_index @ 360 Hz, label):")
    for i, (idx, lab) in enumerate(zip(peak_idx, y_pred)):
        print(
            f"  beat {i+1:4d}  peak_sample={idx:6d}  -> {LABEL_NAMES[int(lab)]}  "
            f"P(N,V,a)={np.round(proba[i], 3)}"
        )

    if args.out:
        rows = []
        for i, (idx, lab) in enumerate(zip(peak_idx, y_pred)):
            rows.append(
                {
                    "beat_index": i + 1,
                    "peak_sample_360hz": int(idx),
                    "label": LABEL_NAMES[int(lab)],
                    "prob_N": float(proba[i, 0]),
                    "prob_V": float(proba[i, 1]),
                    "prob_a": float(proba[i, 2]),
                }
            )
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
