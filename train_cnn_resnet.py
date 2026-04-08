"""
ResNet-1D training for ECG beat classification (N, V, a) with RR auxiliary input.

MIT-BIH — same protocol as train_cnn.py, plus per-beat RR-ratio features (2-D) fused
after the conv trunk. StandardScaler is fit on TRAIN RR only and saved for inference.

After training:
  * resnet1d_mitdb.keras
  * resnet1d_mitdb_rr_scaler.joblib
  * resnet1d_mitdb_pac_threshold.joblib  (tau on proba['a'] for PAC vs N/V)

Custom CSV inference:
  python classify_custom_resnet.py --csv Subject_Z2_ECG.csv
"""

import os

import joblib
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from ecg_preprocessing import (
    compute_class_weights,
    get_mitdb_data_path,
    load_record_names,
    preprocess_beats,
    rr_features_for_labeled_beats,
    segment_record,
    split_by_record,
)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
except ImportError:
    print("ERROR: TensorFlow not found. Please install: pip install tensorflow")
    raise SystemExit(1)


def sparse_categorical_crossentropy_with_label_smoothing(smoothing):
    """
    Label smoothing for integer labels; works on TF versions that do not support
    SparseCategoricalCrossentropy(label_smoothing=...).
    """

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        one_hot = tf.one_hot(y_true, num_classes)
        one_hot = tf.cast(one_hot, tf.float32)
        sm = tf.cast(smoothing, tf.float32)
        nc = tf.cast(num_classes, tf.float32)
        smooth_labels = one_hot * (1.0 - sm) + sm / nc
        return tf.reduce_mean(
            keras.losses.categorical_crossentropy(smooth_labels, y_pred)
        )

    return loss_fn

from train_cnn import print_class_weights, print_split_statistics

# Training tweaks for minority PAC (class 2) — see plan: test balanced accuracy target.
LABEL_SMOOTHING = 0.02
PAC_OVERSAMPLE_EXTRA_COPIES = 2
# class_weight multipliers on top of inverse-frequency weights from compute_class_weights
CLASS_WEIGHT_N_MULT = 0.5
CLASS_WEIGHT_V_MULT = 2.0
CLASS_WEIGHT_A_MULT = 4.0
TRAIN_BATCH_SIZE = 256

RESNET_MODEL_FILENAME = "resnet1d_mitdb.keras"
RESNET_RR_SCALER_FILENAME = "resnet1d_mitdb_rr_scaler.joblib"
RESNET_PAC_THRESHOLD_FILENAME = "resnet1d_mitdb_pac_threshold.joblib"


def resnet_model_path(script_dir=None):
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, RESNET_MODEL_FILENAME)


def resnet_rr_scaler_path(script_dir=None):
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, RESNET_RR_SCALER_FILENAME)


def resnet_pac_threshold_path(script_dir=None):
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, RESNET_PAC_THRESHOLD_FILENAME)


def proba_to_labels(proba, pac_threshold=None):
    """
    Map softmax outputs to class labels. If pac_threshold is set, predict PAC when
    proba[2] >= pac_threshold; otherwise choose between N and V only via argmax on [:2].
    """
    if pac_threshold is None:
        return np.argmax(proba, axis=1)
    pred_nv = np.argmax(proba[:, :2], axis=1)
    return np.where(proba[:, 2] >= pac_threshold, 2, pred_nv).astype(np.int64)


def tune_pac_decision_threshold(
    model, X_val, y_val, RR_val, min_recall_normal=0.82
):
    """
    Pick tau (proba['a'] >= tau -> PAC) on validation. Constrain val recall(N) so
    thresholding does not trade almost all normals for PAC (rare on val, easy to overfit).
    """
    X_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    proba = model.predict(
        {"wave": X_reshaped, "rr": RR_val}, verbose=0, batch_size=512
    )
    best_tau, best_bacc = None, -1.0
    for tau in np.linspace(0.05, 0.55, 51):
        y_pred = proba_to_labels(proba, tau)
        recalls = recall_score(y_val, y_pred, average=None, zero_division=0)
        if recalls[0] < min_recall_normal:
            continue
        bacc = balanced_accuracy_score(y_val, y_pred)
        if bacc > best_bacc:
            best_bacc = bacc
            best_tau = float(tau)
    if best_tau is None:
        print(
            f"\nPAC threshold: no tau met val recall(N)>={min_recall_normal:.2f}; "
            "using plain argmax."
        )
        return None
    print(
        f"\nPAC decision threshold (val): tau={best_tau:.4f} "
        f"(val balanced accuracy: {best_bacc:.4f}, min N-recall constraint)"
    )
    return best_tau


def resnet_conv_block(x, out_filters, kernel_size, stride, use_projection):
    """Residual block: two Conv1D, BN, ReLU, skip."""
    if use_projection:
        shortcut = layers.Conv1D(
            out_filters, 1, strides=stride, padding="same", use_bias=False
        )(x)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = x

    y = layers.Conv1D(
        out_filters, kernel_size, strides=stride, padding="same", use_bias=False
    )(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv1D(out_filters, kernel_size, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.add([shortcut, y])
    y = layers.Activation("relu")(y)
    return y


def build_resnet1d_rr_model(n_classes=3, base_filters=24):
    """ResNet-1D on (180, 1) plus scaled RR (2,) fused before the softmax."""
    wave_in = layers.Input(shape=(180, 1), name="wave")
    rr_in = layers.Input(shape=(2,), name="rr")

    x = layers.Conv1D(base_filters, 5, padding="same", use_bias=False)(wave_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)

    x = resnet_conv_block(x, base_filters, 3, 1, use_projection=False)
    x = resnet_conv_block(x, base_filters * 2, 3, 2, use_projection=True)
    x = resnet_conv_block(x, base_filters * 2, 3, 1, use_projection=False)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Small RR branch so morphology stays primary; RR mainly disambiguates PVC/PAC timing.
    rr = layers.Dense(4, activation="relu")(rr_in)
    rr = layers.Dropout(0.15)(rr)
    fused = layers.Concatenate()([x, rr])
    fused = layers.Dense(32, activation="relu")(fused)
    fused = layers.Dropout(0.2)(fused)
    outputs = layers.Dense(n_classes, activation="softmax")(fused)
    return models.Model(inputs=[wave_in, rr_in], outputs=outputs)


def oversample_train_pac(X_train, y_train, RR_train, n_extra_copies=PAC_OVERSAMPLE_EXTRA_COPIES):
    """Append extra copies of PAC (label 2) beats and shuffle (train-only augmentation)."""
    idx = np.where(y_train == 2)[0]
    if len(idx) == 0:
        return X_train, y_train, RR_train
    xs, ys, rrs = [X_train], [y_train], [RR_train]
    for _ in range(n_extra_copies):
        xs.append(X_train[idx].copy())
        ys.append(y_train[idx].copy())
        rrs.append(RR_train[idx].copy())
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    rr = np.concatenate(rrs, axis=0)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], rr[perm]


class BalancedAccuracyValMonitor(keras.callbacks.Callback):
    """Track validation balanced accuracy; early-stop and restore best weights by it."""

    def __init__(self, X_val_reshaped, y_val, RR_val, patience=5):
        super().__init__()
        self.X_val = X_val_reshaped
        self.y_val = np.asarray(y_val)
        self.RR_val = RR_val
        self.patience = patience
        self.best = 0.0
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        proba = self.model.predict(
            {"wave": self.X_val, "rr": self.RR_val}, verbose=0, batch_size=512
        )
        y_pred = np.argmax(proba, axis=1)
        bacc = balanced_accuracy_score(self.y_val, y_pred)
        logs["val_balanced_accuracy"] = float(bacc)
        print(f" val_balanced_accuracy: {bacc:.4f}")

        if bacc > self.best + 1e-7:
            self.best = bacc
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(
                    f"Early stopping on val_balanced_accuracy "
                    f"(best={self.best:.4f}, patience={self.patience})"
                )

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(f"Restored weights from best val_balanced_accuracy: {self.best:.4f}")


def train_resnet1d_rr_model(
    X_train,
    y_train,
    RR_train,
    X_val,
    y_val,
    RR_val,
    class_weights,
    epochs=25,
    batch_size=128,
):
    """Train two-input ResNet with dict inputs (wave + RR)."""
    print("\n" + "=" * 50)
    print("STEP 4: TRAINING RESNET-1D+RR MODEL (lightweight, fast)")
    print("=" * 50)

    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    print("Building ResNet-1D+RR architecture...")
    model = build_resnet1d_rr_model(n_classes=3)

    class_weight_dict = {
        0: class_weights[0] * CLASS_WEIGHT_N_MULT,
        1: class_weights[1] * CLASS_WEIGHT_V_MULT,
        2: class_weights[2] * CLASS_WEIGHT_A_MULT,
    }

    print("Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss=sparse_categorical_crossentropy_with_label_smoothing(LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    bacc_monitor = BalancedAccuracyValMonitor(
        X_val_reshaped, y_val, RR_val, patience=6
    )
    callbacks = [
        bacc_monitor,
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=2,
            factor=0.5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nModel architecture:")
    model.summary()

    print(f"\nTraining parameters:")
    print(f"  Epochs (max): {epochs}")
    print(f"  Batch size: {batch_size}")
    print("  Class weights:")
    for k in sorted(class_weight_dict.keys()):
        print(f"    {k}: {float(class_weight_dict[k])}")
    print(f"  Label smoothing (in loss): {LABEL_SMOOTHING}")
    print(f"  Optimizer: Adam (lr=0.0003)")
    print(f"  Loss: sparse categorical cross-entropy (with label smoothing)")
    print(
        f"  Callbacks: early stop on val_balanced_accuracy (patience=6), "
        f"ReduceLROnPlateau(val_loss, patience=2)"
    )

    print("\nStarting training...")
    history = model.fit(
        {"wave": X_train_reshaped, "rr": RR_train},
        y_train,
        validation_data=(
            {"wave": X_val_reshaped, "rr": RR_val},
            y_val,
        ),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    print("Training complete!")
    return model, history


def evaluate_resnet_rr(
    model,
    X,
    y,
    RR,
    split_name,
    class_names,
    architecture_name="ResNet-1D+RR",
    pac_threshold=None,
):
    X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
    y_pred_proba = model.predict(
        {"wave": X_reshaped, "rr": RR}, verbose=0, batch_size=512
    )
    y_pred = proba_to_labels(y_pred_proba, pac_threshold)

    cm = confusion_matrix(y, y_pred)
    precision = precision_score(y, y_pred, average=None, zero_division=0)
    recall = recall_score(y, y_pred, average=None, zero_division=0)
    f1 = f1_score(y, y_pred, average=None, zero_division=0)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(y, y_pred)

    print("\n" + "=" * 50)
    print(f"STEP 4: {architecture_name.upper()} MODEL - {split_name.upper()} SET")
    if pac_threshold is not None:
        print(f"(PAC threshold tau={pac_threshold:.4f} on proba['a'])")
    print("=" * 50)

    print("\nConfusion Matrix:")
    print("              Predicted")
    print("              ", end="")
    class_labels = ["N", "V", "a"]
    for label in class_labels:
        print(f"{label:>6s}", end="")
    print()
    print("Actual ", end="")
    for label in class_labels:
        print(f"{label:>6s}", end="")
    print()
    for i, _ in enumerate([0, 1, 2]):
        print(f"      {class_labels[i]:>3s}", end="")
        for j in range(3):
            print(f"{cm[i, j]:>6d}", end="")
        print()

    print("\nPer-class metrics:")
    for class_id in [0, 1, 2]:
        name = class_names[class_id]
        print(
            f"  {name:20s}: Precision={precision[class_id]:.4f}, "
            f"Recall={recall[class_id]:.4f}, F1={f1[class_id]:.4f}"
        )

    print("\nOverall metrics:")
    print(f"  Macro-F1:          {macro_f1:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")


if __name__ == "__main__":
    print("=" * 50)
    print("RESNET-1D+RR ECG BEAT CLASSIFIER TRAINING")
    print("=" * 50)

    print("\n" + "=" * 50)
    print("STEP 1: LOADING AND SEGMENTING MIT-BIH RECORDS")
    print("=" * 50)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = get_mitdb_data_path(script_dir)

    record_names = load_record_names(data_path)
    print(f"\nFound {len(record_names)} MIT-BIH records to process")
    print(f"Records: {', '.join(record_names[:10])}{'...' if len(record_names) > 10 else ''}")

    X_raw_all = []
    y_raw_all = []
    record_ids_all = []
    peak_raw_all = []

    for record_name in record_names:
        X_raw_record, y_raw_record, record_ids_record, peaks_record, n_beats = segment_record(
            record_name, data_path
        )
        X_raw_all.extend(X_raw_record)
        y_raw_all.extend(y_raw_record)
        record_ids_all.extend(record_ids_record)
        peak_raw_all.extend(peaks_record)
        print(f"  {record_name}: {n_beats} beats extracted")

    X_raw = np.array(X_raw_all)
    y_raw = np.array(y_raw_all)
    record_ids = np.array(record_ids_all)
    peak_raw = np.array(peak_raw_all, dtype=int)

    print(f"\nTotal beats extracted: {len(X_raw)}")

    print("\n" + "=" * 50)
    print("STEP 2: PREPROCESSING ECG BEATS")
    print("=" * 50)

    X, y, record_ids, stats, peak_samples = preprocess_beats(
        X_raw, y_raw, record_ids, peak_raw
    )

    print(f"\nPreprocessing complete:")
    print(f"  Input beats:  {stats['n_input']}")
    print(f"  Output beats: {stats['n_output']}")
    print(f"  Baseline corrected: {stats['baseline_corrected']}")
    print(f"  Normalized: {stats['normalized']}")
    print(f"\nClass distribution after preprocessing:")
    for class_id in [0, 1, 2]:
        count = stats["class_counts"][class_id]
        percentage = (count / stats["n_output"] * 100) if stats["n_output"] > 0 else 0
        class_names_map = {0: "N (Normal)", 1: "V (PVC)", 2: "a (PAC)"}
        print(f"  {class_names_map[class_id]:15s}: {count:6d} ({percentage:5.1f}%)")

    print("\n" + "=" * 50)
    print("STEP 3: DATASET STATISTICS, WEIGHTS, AND SPLIT")
    print("=" * 50)

    (
        X_train,
        y_train,
        record_ids_train,
        X_val,
        y_val,
        record_ids_val,
        X_test,
        y_test,
        record_ids_test,
        split_info,
    ) = split_by_record(
        X,
        y,
        record_ids,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )

    print_split_statistics(split_info, y_train, y_val, y_test)
    print(
        "\nMitdb_windows-style protocol: fit on TRAIN; val metrics during training = VAL; "
        "TEST is held out until final evaluation below."
    )

    class_weights = compute_class_weights(y_train)
    print_class_weights(class_weights)

    train_recs = set(split_info["train_records"])
    val_recs = set(split_info["val_records"])
    test_recs = set(split_info["test_records"])
    train_mask = np.array([rid in train_recs for rid in record_ids])
    val_mask = np.array([rid in val_recs for rid in record_ids])
    test_mask = np.array([rid in test_recs for rid in record_ids])

    peak_train = peak_samples[train_mask]
    peak_val = peak_samples[val_mask]
    peak_test = peak_samples[test_mask]

    RR_train_raw = rr_features_for_labeled_beats(peak_train, record_ids_train)
    RR_val_raw = rr_features_for_labeled_beats(peak_val, record_ids_val)
    RR_test_raw = rr_features_for_labeled_beats(peak_test, record_ids_test)

    rr_scaler = StandardScaler()
    RR_train = rr_scaler.fit_transform(RR_train_raw).astype(np.float32)
    RR_val = rr_scaler.transform(RR_val_raw).astype(np.float32)
    RR_test = rr_scaler.transform(RR_test_raw).astype(np.float32)

    scaler_path = resnet_rr_scaler_path(script_dir)
    joblib.dump(rr_scaler, scaler_path)
    print(f"\nSaved RR StandardScaler (fit on train only): {scaler_path}")

    print("\n" + "=" * 50)
    print("FINAL OUTPUT READY FOR RESNET-1D+RR TRAINING")
    print("=" * 50)
    print(f"Training set:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  record_ids_train: {len(np.unique(record_ids_train))} unique records")
    print(f"\nValidation set:")
    print(f"  X_val shape: {X_val.shape}")
    print(f"  y_val shape: {y_val.shape}")
    print(f"  record_ids_val: {len(np.unique(record_ids_val))} unique records")
    print(f"\nTest set:")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  record_ids_test: {len(np.unique(record_ids_test))} unique records")
    print("\nClass weights:")
    for k in sorted(class_weights.keys()):
        print(f"  {k}: {float(class_weights[k])}")

    n_pac_train = int(np.sum(y_train == 2))
    X_fit, y_fit, RR_fit = oversample_train_pac(X_train, y_train, RR_train)
    print(
        f"\nPAC oversampling: {n_pac_train} PAC beats → "
        f"{int(np.sum(y_fit == 2))} in fit set "
        f"(1 + {PAC_OVERSAMPLE_EXTRA_COPIES} extra copies per PAC beat)"
    )

    model, history = train_resnet1d_rr_model(
        X_fit,
        y_fit,
        RR_fit,
        X_val,
        y_val,
        RR_val,
        class_weights,
        epochs=35,
        batch_size=TRAIN_BATCH_SIZE,
    )

    pac_tau = tune_pac_decision_threshold(model, X_val, y_val, RR_val)
    threshold_path = resnet_pac_threshold_path(script_dir)
    if pac_tau is not None:
        joblib.dump({"pac_threshold": pac_tau}, threshold_path)
        print(f"Saved PAC decision threshold: {threshold_path}")
    elif os.path.isfile(threshold_path):
        os.remove(threshold_path)
        print("Removed stale PAC threshold file (using argmax).")

    class_names = {
        0: "Normal (N)",
        1: "PVC (V)",
        2: "PAC (a)",
    }

    evaluate_resnet_rr(
        model,
        X_train,
        y_train,
        RR_train,
        "Train",
        class_names,
        pac_threshold=pac_tau,
    )
    evaluate_resnet_rr(
        model, X_val, y_val, RR_val, "Val", class_names, pac_threshold=pac_tau
    )
    evaluate_resnet_rr(
        model, X_test, y_test, RR_test, "Test", class_names, pac_threshold=pac_tau
    )

    out_path = resnet_model_path(script_dir)
    model.save(out_path)
    print(f"\nSaved trained ResNet-1D+RR for custom inference: {out_path}")

    print("\n" + "=" * 50)
    print("RESNET-1D+RR TRAINING COMPLETE")
    print("=" * 50)
