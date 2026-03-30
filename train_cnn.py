"""
1D CNN Training Script for ECG Beat Classification

This script trains a 1D Convolutional Neural Network for classifying ECG beats
into three classes: Normal (N), PVC (V), and PAC (a).

Reuses the same preprocessing pipeline and patient-safe train/val/test split
as the baseline logistic regression model.
"""

import os
import numpy as np

# Import preprocessing utilities
from ecg_preprocessing import (
    get_mitdb_data_path,
    load_record_names,
    segment_record,
    preprocess_beats,
    split_by_record,
    compute_class_weights,
    compute_class_distribution,
)

# Try to import TensorFlow/Keras for CNN
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("ERROR: TensorFlow not found. Please install: pip install tensorflow")
    exit(1)

# Import sklearn metrics for evaluation
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)


def print_split_statistics(split_info, y_train=None, y_val=None, y_test=None):
    """
    Print detailed statistics about the record-level split.
    
    Args:
        split_info: Dictionary with split information
        y_train: Training labels (optional, for class verification)
        y_val: Validation labels (optional, for class verification)
        y_test: Test labels (optional, for class verification)
    """
    print("\n" + "="*50)
    print("STEP 3: RECORD-LEVEL SPLIT STATISTICS")
    print("="*50)
    
    print(f"\nRecord-level split:")
    print(f"  Total records: {split_info['n_total_records']}")
    print(f"  Train records: {split_info['n_train_records']} ({split_info['n_train_records']/split_info['n_total_records']*100:.1f}%)")
    print(f"  Val records:   {split_info['n_val_records']} ({split_info['n_val_records']/split_info['n_total_records']*100:.1f}%)")
    print(f"  Test records:  {split_info['n_test_records']} ({split_info['n_test_records']/split_info['n_total_records']*100:.1f}%)")
    
    print(f"\nSample-level split:")
    total_samples = (split_info['n_train_samples'] + 
                     split_info['n_val_samples'] + 
                     split_info['n_test_samples'])
    print(f"  Total samples: {total_samples}")
    print(f"  Train samples: {split_info['n_train_samples']} ({split_info['n_train_samples']/total_samples*100:.1f}%)")
    print(f"  Val samples:   {split_info['n_val_samples']} ({split_info['n_val_samples']/total_samples*100:.1f}%)")
    print(f"  Test samples:  {split_info['n_test_samples']} ({split_info['n_test_samples']/total_samples*100:.1f}%)")
    
    print(f"\nRecord IDs in each split:")
    train_records_str = [str(r) for r in split_info['train_records']]
    val_records_str = [str(r) for r in split_info['val_records']]
    test_records_str = [str(r) for r in split_info['test_records']]
    print(f"  Train: {train_records_str}")
    print(f"  Val:   {val_records_str}")
    print(f"  Test:  {test_records_str}")
    
    # Verify no overlap
    train_set = set(split_info['train_records'])
    val_set = set(split_info['val_records'])
    test_set = set(split_info['test_records'])
    
    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set
    
    print(f"\nOverlap verification:")
    print(f"  Train ∩ Val:  {len(overlap_train_val)} records (should be 0)")
    print(f"  Train ∩ Test: {len(overlap_train_test)} records (should be 0)")
    print(f"  Val ∩ Test:   {len(overlap_val_test)} records (should be 0)")
    
    if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
        print(f"  ✓ Zero overlap confirmed - patient-safe split!")
    else:
        print(f"  ✗ WARNING: Overlap detected!")
    
    # Verify class distribution in each split
    if y_train is not None and y_val is not None and y_test is not None:
        print(f"\nClass distribution in each split:")
        for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            unique_classes = sorted(np.unique(y_split))
            class_names_map = {0: "N", 1: "V", 2: "a"}
            class_labels = [class_names_map[c] for c in unique_classes]
            counts = {c: np.sum(y_split == c) for c in [0, 1, 2]}
            print(f"  {split_name:8s}: Classes present: {class_labels}, Counts - N:{counts[0]:6d}, V:{counts[1]:6d}, a:{counts[2]:6d}")
        
        # Verify test set has all classes
        test_classes = set(np.unique(y_test))
        required_classes = {0, 1, 2}
        if required_classes.issubset(test_classes):
            print(f"\n  ✓ Test set contains all three classes (N, V, a)")
        else:
            missing = required_classes - test_classes
            class_names_map = {0: "N", 1: "V", 2: "a"}
            missing_names = [class_names_map[c] for c in missing]
            print(f"\n  ✗ WARNING: Test set missing classes: {missing_names}")


def print_class_weights(class_weights):
    """
    Print class weights computed from training set.
    
    Args:
        class_weights: Dictionary {class_id: weight}
    """
    print("\n" + "="*50)
    print("STEP 3: CLASS WEIGHTS (from training set)")
    print("="*50)
    
    class_names = {
        0: ("Normal (N)", "N → 0"),
        1: ("PVC (V)", "V → 1"),
        2: ("PAC (a)", "a → 2")
    }
    
    print(f"\nClass weights (inverse-frequency):")
    for class_id in [0, 1, 2]:
        weight = class_weights[class_id]
        name, mapping = class_names[class_id]
        print(f"  {name:20s} ({mapping:6s}): {weight:.6f}")
    
    print(f"\nNote: Higher weights indicate minority classes that need more emphasis during training.")


def build_1d_cnn_model(input_shape=(180, 1), n_classes=3):
    """
    Build a 1D CNN model for ECG beat classification.
    
    Architecture:
    - Input: 180-sample ECG window with 1 feature channel
    - Conv1D layers with BatchNorm, padding='same', MaxPool, Dropout
    - GlobalAveragePooling1D instead of Flatten
    - Simplified dense layers (64, then 32)
    
    Args:
        input_shape: Shape of input (sequence_length, features) - should be (180, 1)
        n_classes: Number of output classes
    Returns:
        model: Compiled Keras model
    """
    model = models.Sequential([
        # Input layer with proper shape for Conv1D (batch, timesteps, features)
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv1D(32, kernel_size=7, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Second convolutional block
        layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Third convolutional block
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Global average pooling and dense layers
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(n_classes, activation='softmax')
    ])
    
    return model


def train_cnn_model(X_train, y_train, X_val, y_val, class_weights, epochs=50, batch_size=64):
    """
    Train a 1D CNN model with class weights.
    
    Args:
        X_train: Training features, shape (n_samples, 180)
        y_train: Training labels, shape (n_samples,)
        X_val: Validation features, shape (n_samples, 180)
        y_val: Validation labels, shape (n_samples,)
        class_weights: Dictionary {class_id: weight}
        epochs: Number of training epochs
        batch_size: Batch size for training
    Returns:
        model: Trained Keras model
        history: Training history
    """
    print("\n" + "="*50)
    print("STEP 4: TRAINING 1D CNN MODEL")
    print("="*50)
    
    # Reshape data for Conv1D: (n_samples, 180) -> (n_samples, 180, 1)
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    
    # Build model
    print("Building 1D CNN architecture...")
    model = build_1d_cnn_model(input_shape=(180, 1), n_classes=3)
    
    # Stronger class weights: PAC 4x, PVC 2x, Normal 0.5x
    class_weight_dict = {
        0: class_weights[0] * 0.5,
        1: class_weights[1] * 2.0,
        2: class_weights[2] * 4.0
    }
    
    # Compile model
    print("Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=5,
            factor=0.5,
            verbose=1
        )
    ]
    
    # Print model summary
    print("\nModel architecture:")
    model.summary()
    
    print(f"\nTraining parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Class weights: {class_weight_dict}")
    print(f"  Optimizer: Adam (lr=0.0001)")
    print(f"  Loss: sparse_categorical_crossentropy")
    print(f"  Callbacks: EarlyStopping(patience=15), ReduceLROnPlateau(patience=5, factor=0.5)")
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        X_train_reshaped, y_train,
        validation_data=(X_val_reshaped, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    print("Training complete!")
    
    return model, history


def evaluate_model(model, X, y, split_name, class_names):
    """
    Evaluate model and print comprehensive metrics.
    
    Args:
        model: Trained Keras model
        X: Features, shape (n_samples, 180)
        y: True labels, shape (n_samples,)
        split_name: String name of split ('Train', 'Val', 'Test')
        class_names: Dictionary mapping class_id to name
    """
    # Reshape for Conv1D: (n_samples, 180) -> (n_samples, 180, 1)
    X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Get predictions
    y_pred_proba = model.predict(X_reshaped, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Compute per-class metrics
    precision = precision_score(y, y_pred, average=None, zero_division=0)
    recall = recall_score(y, y_pred, average=None, zero_division=0)
    f1 = f1_score(y, y_pred, average=None, zero_division=0)
    
    # Compute overall metrics
    macro_f1 = f1_score(y, y_pred, average='macro', zero_division=0)
    balanced_acc = balanced_accuracy_score(y, y_pred)
    
    # Print results
    print("\n" + "="*50)
    print(f"STEP 4: 1D CNN MODEL - {split_name.upper()} SET")
    print("="*50)
    
    # Confusion matrix
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
    for i, class_id in enumerate([0, 1, 2]):
        print(f"      {class_labels[i]:>3s}", end="")
        for j in range(3):
            print(f"{cm[i, j]:>6d}", end="")
        print()
    
    # Per-class metrics
    print("\nPer-class metrics:")
    for class_id in [0, 1, 2]:
        name = class_names[class_id]
        prec = precision[class_id]
        rec = recall[class_id]
        f1_val = f1[class_id]
        print(f"  {name:20s}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1_val:.4f}")
    
    # Overall metrics
    print("\nOverall metrics:")
    print(f"  Macro-F1:         {macro_f1:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

if __name__ == "__main__":
    print("="*50)
    print("1D CNN ECG BEAT CLASSIFIER TRAINING")
    print("="*50)
    
    # ========================================================================
    # STEP 1: LOAD AND SEGMENT MIT-BIH RECORDS
    # ========================================================================
    print("\n" + "="*50)
    print("STEP 1: LOADING AND SEGMENTING MIT-BIH RECORDS")
    print("="*50)
    
    # Get data path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = get_mitdb_data_path(script_dir)
    
    # Load list of available records
    record_names = load_record_names(data_path)
    print(f"\nFound {len(record_names)} MIT-BIH records to process")
    print(f"Records: {', '.join(record_names[:10])}{'...' if len(record_names) > 10 else ''}")
    
    # Segment all records
    X_raw_all = []
    y_raw_all = []
    record_ids_all = []
    
    for record_name in record_names:
        X_raw_record, y_raw_record, record_ids_record, n_beats = segment_record(record_name, data_path)
        X_raw_all.extend(X_raw_record)
        y_raw_all.extend(y_raw_record)
        record_ids_all.extend(record_ids_record)
        print(f"  {record_name}: {n_beats} beats extracted")
    
    # Convert to numpy arrays
    X_raw = np.array(X_raw_all)
    y_raw = np.array(y_raw_all)
    record_ids = np.array(record_ids_all)
    
    print(f"\nTotal beats extracted: {len(X_raw)}")
    
    # ========================================================================
    # STEP 2: PREPROCESS BEATS
    # ========================================================================
    print("\n" + "="*50)
    print("STEP 2: PREPROCESSING ECG BEATS")
    print("="*50)
    
    X, y, record_ids_filtered, stats = preprocess_beats(X_raw, y_raw, record_ids)
    
    print(f"\nPreprocessing complete:")
    print(f"  Input beats:  {stats['n_input']}")
    print(f"  Output beats: {stats['n_output']}")
    print(f"  Baseline corrected: {stats['baseline_corrected']}")
    print(f"  Normalized: {stats['normalized']}")
    print(f"\nClass distribution after preprocessing:")
    for class_id in [0, 1, 2]:
        count = stats['class_counts'][class_id]
        percentage = (count / stats['n_output'] * 100) if stats['n_output'] > 0 else 0
        class_names_map = {0: "N (Normal)", 1: "V (PVC)", 2: "a (PAC)"}
        print(f"  {class_names_map[class_id]:15s}: {count:6d} ({percentage:5.1f}%)")
    
    # ========================================================================
    # STEP 3: DATASET STATS + WEIGHTS + SPLIT
    # ========================================================================
    print("\n" + "="*50)
    print("STEP 3: DATASET STATISTICS, WEIGHTS, AND SPLIT")
    print("="*50)
    
    # Step 3.1: Compute and print class distribution
    class_counts, class_percentages = compute_class_distribution(y)
    
    # Step 3.2: Split dataset by record ID (patient-safe)
    X_train, y_train, record_ids_train, \
    X_val, y_val, record_ids_val, \
    X_test, y_test, record_ids_test, \
    split_info = split_by_record(X, y, record_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Print split statistics (with class verification)
    print_split_statistics(split_info, y_train, y_val, y_test)
    
    # Step 3.3: Compute class weights from training set only
    class_weights = compute_class_weights(y_train)
    print_class_weights(class_weights)
    
    # Final output summary
    print("\n" + "="*50)
    print("FINAL OUTPUT READY FOR CNN TRAINING")
    print("="*50)
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
    class_weights_display = {k: float(v) for k, v in class_weights.items()}
    print(f"\nClass weights: {class_weights_display}")
    
    # ========================================================================
    # STEP 4: CNN MODEL TRAINING AND EVALUATION
    # ========================================================================
    
    # Train model
    model, history = train_cnn_model(
        X_train, y_train,
        X_val, y_val,
        class_weights,
        epochs=100,
        batch_size=64
    )
    
    # Class name mapping
    class_names = {
        0: "Normal (N)",
        1: "PVC (V)",
        2: "PAC (a)"
    }
    
    # Evaluate on all splits
    evaluate_model(model, X_train, y_train, "Train", class_names)
    evaluate_model(model, X_val, y_val, "Val", class_names)
    evaluate_model(model, X_test, y_test, "Test", class_names)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
