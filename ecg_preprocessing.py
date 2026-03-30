"""
ECG Preprocessing Utilities for MIT-BIH Arrhythmia Database

This module provides reusable functions for:
- Loading and segmenting MIT-BIH ECG records
- Preprocessing ECG beat windows (baseline correction, normalization)
- Patient-safe dataset splitting
- Class weight computation

Can be imported and used by both baseline and CNN training scripts.
"""

import os
import numpy as np
import wfdb

# Try to import scipy for detrending, otherwise use manual implementation
try:
    from scipy.signal import detrend
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def get_mitdb_data_path(script_dir=None):
    """
    Get the path to the MIT-BIH database directory.
    
    Args:
        script_dir: Optional directory where the script is located.
                   If None, uses the directory of this file.
    Returns:
        data_path: Path to the MIT-BIH database directory
    """
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    data_path = os.path.join(
        script_dir,
        "mit-bih-arrhythmia-database-1.0.0",
        "mit-bih-arrhythmia-database-1.0.0",
    )
    return data_path


def load_record_names(data_path):
    """
    Load the list of available MIT-BIH record names from RECORDS file.
    
    Args:
        data_path: Path to the MIT-BIH database directory
    Returns:
        record_names: List of record name strings (e.g., ['100', '101', ...])
    """
    records_file = os.path.join(data_path, "RECORDS")
    with open(records_file, 'r') as f:
        record_names = [line.strip() for line in f if line.strip()]
    return record_names


def segment_record(record_name, data_path):
    """
    Load a single MIT-BIH record and segment it into beat windows.
    
    Args:
        record_name: String name of the record (e.g., "100")
        data_path: Path to the MIT-BIH database directory
    Returns:
        X_raw: List of raw ECG windows (180 samples each)
        y_raw: List of labels (0, 1, or 2)
        record_ids: List of record IDs (same length as X_raw)
        n_beats: Number of beats extracted from this record
    """
    X_raw_record = []
    y_raw_record = []
    record_ids_record = []
    
    # Window size: 180 samples (90 before + 90 after the R-peak)
    window_size = 180
    half_window = 90
    
    # Change to the data directory temporarily (some wfdb versions don't support base_dir)
    original_dir = os.getcwd()
    os.chdir(data_path)
    
    try:
        # Load ECG signal
        record = wfdb.rdrecord(record_name)
        ecg_signal = record.p_signal[:, 0]
        
        # Load annotations
        annotations = wfdb.rdann(record_name, 'atr')
    finally:
        # Change back to the original directory
        os.chdir(original_dir)
    
    # Get R-peak indices and beat types
    r_peak_indices = annotations.sample
    beat_types = annotations.symbol
    
    # Loop through each R-peak annotation
    for i in range(len(r_peak_indices)):
        r_peak_pos = r_peak_indices[i]
        beat_type = beat_types[i]
        
        # Only keep beat types N, V, or a
        if beat_type not in ['N', 'V', 'a']:
            continue
        
        # Extract window centered on R-peak
        window_start = r_peak_pos - half_window
        window_end = r_peak_pos + half_window
        
        # Skip if window goes out of bounds
        if window_start < 0 or window_end > len(ecg_signal):
            continue
        
        # Extract the 180-sample window
        window = ecg_signal[window_start:window_end]
        
        # Safety check
        if len(window) != window_size:
            continue
        
        # Convert beat type to number: N → 0, V → 1, a → 2
        if beat_type == 'N':
            label = 0
        elif beat_type == 'V':
            label = 1
        elif beat_type == 'a':
            label = 2
        else:
            continue
        
        # Store window, label, and record ID
        X_raw_record.append(window)
        y_raw_record.append(label)
        record_ids_record.append(record_name)
    
    n_beats = len(X_raw_record)
    return X_raw_record, y_raw_record, record_ids_record, n_beats


def correct_baseline_wander(window):
    """
    Remove baseline wander using linear detrending.
    Real-time compatible: works on single 180-sample window.
    
    Args:
        window: 1D array of shape (180,)
    Returns:
        detrended_window: 1D array of shape (180,)
    """
    if HAS_SCIPY:
        return detrend(window)
    else:
        # Manual linear detrending: subtract linear trend
        # Create linear trend from first to last sample
        n = len(window)
        trend = np.linspace(window[0], window[-1], n)
        return window - trend


def normalize_beat(window):
    """
    Normalize beat window using mean-std normalization.
    Real-time compatible: works on single beat.
    
    Args:
        window: 1D array of shape (180,)
    Returns:
        normalized_window: 1D array, mean=0, std=1
    """
    window_mean = np.mean(window)
    window_std = np.std(window)
    
    # Handle edge case: constant signal (std=0)
    if window_std == 0:
        # Return zero-centered signal
        return window - window_mean
    
    return (window - window_mean) / window_std


def preprocess_beats(X_raw, y_raw, record_ids):
    """
    Apply preprocessing pipeline to beat windows.
    
    Steps:
    1. Baseline wander correction (detrending)
    2. Per-beat mean-std normalization
    3. Filter to ensure exactly 3 classes (N=0, V=1, a=2)
    
    Args:
        X_raw: numpy array of shape (n_beats, 180)
        y_raw: numpy array of shape (n_beats,) with labels
        record_ids: numpy array of shape (n_beats,) with record IDs
    Returns:
        X: Preprocessed beats, shape (n_beats, 180)
        y: Filtered labels, shape (n_beats,)
        record_ids_filtered: Filtered record IDs, shape (n_beats,)
        stats: Dictionary with preprocessing statistics
    """
    n_beats = len(X_raw)
    X_processed = []
    y_processed = []
    record_ids_processed = []
    
    # Track statistics
    stats = {
        'n_input': n_beats,
        'n_output': 0,
        'baseline_corrected': True,
        'normalized': True,
        'class_counts': {0: 0, 1: 0, 2: 0}
    }
    
    # Process each beat
    for i in range(n_beats):
        window = X_raw[i]
        label = y_raw[i]
        record_id = record_ids[i]
        
        # Step 1: Baseline wander correction
        window = correct_baseline_wander(window)
        
        # Step 2: Per-beat normalization
        window = normalize_beat(window)
        
        # Step 3: Ensure valid class (should already be filtered, but double-check)
        if label not in [0, 1, 2]:
            continue
        
        X_processed.append(window)
        y_processed.append(label)
        record_ids_processed.append(record_id)
        stats['class_counts'][label] += 1
    
    # Convert to numpy arrays
    X = np.array(X_processed)
    y = np.array(y_processed)
    record_ids_filtered = np.array(record_ids_processed)
    stats['n_output'] = len(X)
    
    return X, y, record_ids_filtered, stats


def split_by_record(X, y, record_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split dataset by record ID (patient-safe splitting).
    All beats from the same record go to the same split.
    Ensures test set contains all three classes (N, V, a).
    
    Args:
        X: numpy array of shape (n_samples, 180)
        y: numpy array of shape (n_samples,)
        record_ids: numpy array of shape (n_samples,) with record IDs
        train_ratio: Proportion of records for training
        val_ratio: Proportion of records for validation
        test_ratio: Proportion of records for testing
        random_seed: Random seed for reproducibility
    Returns:
        X_train, y_train, record_ids_train
        X_val, y_val, record_ids_val
        X_test, y_test, record_ids_test
        split_info: Dictionary with split statistics
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Get unique record IDs
    unique_records = np.unique(record_ids)
    n_records = len(unique_records)
    
    # Find which records contain which classes
    record_classes = {}
    for record_id in unique_records:
        record_mask = record_ids == record_id
        record_y = y[record_mask]
        record_classes[record_id] = set(np.unique(record_y))
    
    # Find records that contain each class
    records_with_class = {0: [], 1: [], 2: []}  # N, V, a
    for record_id, classes in record_classes.items():
        for class_id in [0, 1, 2]:
            if class_id in classes:
                records_with_class[class_id].append(record_id)
    
    # Ensure test set has all three classes
    # Select at least one record with each class for test set
    np.random.seed(random_seed)
    test_records = set()
    
    # Pick one record with each class for test set
    for class_id in [0, 1, 2]:
        if len(records_with_class[class_id]) > 0:
            # Randomly select one record with this class
            available = [r for r in records_with_class[class_id] if r not in test_records]
            if len(available) > 0:
                selected = np.random.choice(available)
                test_records.add(selected)
    
    # Calculate target number of records for each split
    n_train_records = int(n_records * train_ratio)
    n_val_records = int(n_records * val_ratio)
    n_test_records_target = n_records - n_train_records - n_val_records
    
    # Fill remaining test records randomly (up to target)
    remaining_records = [r for r in unique_records if r not in test_records]
    np.random.shuffle(remaining_records)
    
    # Add more records to test set if needed (up to target)
    while len(test_records) < n_test_records_target and len(remaining_records) > 0:
        test_records.add(remaining_records.pop(0))
    
    # Split remaining records between train and val
    np.random.shuffle(remaining_records)
    train_records = set(remaining_records[:n_train_records])
    val_records = set(remaining_records[n_train_records:n_train_records + n_val_records])
    
    # Handle any remaining records (due to rounding)
    for record in remaining_records[n_train_records + n_val_records:]:
        if len(train_records) < n_train_records:
            train_records.add(record)
        elif len(val_records) < n_val_records:
            val_records.add(record)
        else:
            test_records.add(record)
    
    # Verify no overlap
    assert len(train_records & val_records) == 0, "Train and Val have overlapping records!"
    assert len(train_records & test_records) == 0, "Train and Test have overlapping records!"
    assert len(val_records & test_records) == 0, "Val and Test have overlapping records!"
    
    # Create masks for each split
    train_mask = np.array([rid in train_records for rid in record_ids])
    val_mask = np.array([rid in val_records for rid in record_ids])
    test_mask = np.array([rid in test_records for rid in record_ids])
    
    # Split data
    X_train = X[train_mask]
    y_train = y[train_mask]
    record_ids_train = record_ids[train_mask]
    
    X_val = X[val_mask]
    y_val = y[val_mask]
    record_ids_val = record_ids[val_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    record_ids_test = record_ids[test_mask]
    
    # Verify test set has all three classes
    test_classes = set(np.unique(y_test))
    required_classes = {0, 1, 2}  # N, V, a
    if not required_classes.issubset(test_classes):
        missing_classes = required_classes - test_classes
        class_names_map = {0: "N (Normal)", 1: "V (PVC)", 2: "a (PAC)"}
        missing_names = [class_names_map[c] for c in missing_classes]
        print(f"\nWARNING: Test set is missing classes: {missing_names}")
        print(f"  Test set classes present: {sorted(test_classes)}")
        print(f"  Attempting to fix by adding records with missing classes...")
        
        # Try to add records with missing classes to test set
        for missing_class in missing_classes:
            # Find a record with this class that's not already in test
            available_records = [r for r in records_with_class[missing_class] 
                               if r not in test_records and r not in train_records and r not in val_records]
            if len(available_records) == 0:
                # Try taking from train or val
                for r in records_with_class[missing_class]:
                    if r in train_records and len(train_records) > n_train_records:
                        train_records.remove(r)
                        test_records.add(r)
                        break
                    elif r in val_records and len(val_records) > n_val_records:
                        val_records.remove(r)
                        test_records.add(r)
                        break
        
        # Re-split with updated record assignments
        train_mask = np.array([rid in train_records for rid in record_ids])
        val_mask = np.array([rid in val_records for rid in record_ids])
        test_mask = np.array([rid in test_records for rid in record_ids])
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        record_ids_train = record_ids[train_mask]
        
        X_val = X[val_mask]
        y_val = y[val_mask]
        record_ids_val = record_ids[val_mask]
        
        X_test = X[test_mask]
        y_test = y[test_mask]
        record_ids_test = record_ids[test_mask]
        
        # Verify again
        test_classes = set(np.unique(y_test))
        if required_classes.issubset(test_classes):
            print(f"  ✓ Fixed! Test set now has all classes: {sorted(test_classes)}")
        else:
            print(f"  ✗ Could not fix. Test set still missing: {sorted(required_classes - test_classes)}")
    
    # Collect split information
    split_info = {
        'n_total_records': n_records,
        'n_train_records': len(train_records),
        'n_val_records': len(val_records),
        'n_test_records': len(test_records),
        'train_records': sorted(train_records),
        'val_records': sorted(val_records),
        'test_records': sorted(test_records),
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val),
        'n_test_samples': len(X_test)
    }
    
    return (X_train, y_train, record_ids_train,
            X_val, y_val, record_ids_val,
            X_test, y_test, record_ids_test,
            split_info)


def compute_class_weights(y_train):
    """
    Generate class weights from training set only using inverse-frequency method.
    
    Formula: weight[class] = n_samples / (n_classes * count[class])
    This gives higher weight to minority classes.
    
    Args:
        y_train: numpy array of shape (n_train,) with training labels
    Returns:
        class_weights: Dictionary {class_id: weight}
    """
    n_samples = len(y_train)
    n_classes = 3
    
    class_weights = {}
    for class_id in [0, 1, 2]:
        count = np.sum(y_train == class_id)
        if count > 0:
            weight = n_samples / (n_classes * count)
        else:
            # Handle edge case: class not present in training set
            weight = 0.0
        class_weights[class_id] = weight
    
    return class_weights


def compute_class_distribution(y):
    """
    Compute class distribution for 3-class labels.
    
    Args:
        y: numpy array of shape (n_samples,) with labels
    Returns:
        class_counts: Dictionary with class counts
        class_percentages: Dictionary with class percentages
    """
    n_total = len(y)
    class_counts = {0: 0, 1: 0, 2: 0}
    
    for class_id in [0, 1, 2]:
        count = np.sum(y == class_id)
        class_counts[class_id] = count
    
    class_percentages = {}
    for class_id in [0, 1, 2]:
        percentage = (class_counts[class_id] / n_total * 100) if n_total > 0 else 0
        class_percentages[class_id] = percentage
    
    return class_counts, class_percentages
