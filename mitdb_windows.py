import os
import numpy as np
import wfdb

# Try to import scipy for detrending, otherwise use manual implementation
try:
    from scipy.signal import detrend
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Scikit-learn imports for baseline ML model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# Data path: MIT-BIH folder is in the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(
    script_dir,
    "mit-bih-arrhythmia-database-1.0.0",
    "mit-bih-arrhythmia-database-1.0.0",
)

# Load list of available MIT-BIH records
records_file = os.path.join(data_path, "RECORDS")
with open(records_file, 'r') as f:
    record_names = [line.strip() for line in f if line.strip()]

print(f"Found {len(record_names)} MIT-BIH records to process")
print(f"Records: {', '.join(record_names[:10])}{'...' if len(record_names) > 10 else ''}")


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

# ============================================================================
# STEP 1: SEGMENTATION (Multiple Records)
# ============================================================================
# Process all MIT-BIH records and aggregate beat windows
# Extract 180-sample windows centered on MIT-BIH annotated R-peaks
# Filter to 3 classes (N, V, a)
# Output: Raw windows X_raw and labels y_raw (before preprocessing)

# Initialize lists to store windows, labels, and record IDs (aggregated across all records)
X_raw = []  # List to store raw ECG windows (each window is 180 samples)
y_raw = []  # List to store labels (0, 1, or 2)
record_ids = []  # List to store record ID for each beat (for patient-safe splitting)

# Process each record
print("\nProcessing records...")
for idx, record_name in enumerate(record_names, 1):
    try:
        X_raw_record, y_raw_record, record_ids_record, n_beats = segment_record(record_name, data_path)
        
        # Aggregate results
        X_raw.extend(X_raw_record)
        y_raw.extend(y_raw_record)
        record_ids.extend(record_ids_record)
        
        print(f"  [{idx}/{len(record_names)}] Record {record_name}: {n_beats} beats extracted")
    except Exception as e:
        print(f"  [{idx}/{len(record_names)}] Record {record_name}: ERROR - {str(e)}")
        continue

# Convert lists to numpy arrays for easier handling
X_raw = np.array(X_raw)
y_raw = np.array(y_raw)
record_ids = np.array(record_ids)  # Array of record IDs (strings)

# Print aggregated Step 1 results
print("\n" + "="*50)
print("STEP 1: SEGMENTATION RESULTS (All Records)")
print("="*50)
print(f"Total records processed: {len(record_names)}")
print(f"Total windows created: {len(X_raw)}")
print(f"Window shape: {X_raw.shape}")
print(f"\nClass distribution (counts per label):")
print(f"  Class 0 (Normal, N):     {np.sum(y_raw == 0):6d}")
print(f"  Class 1 (PVC, V):        {np.sum(y_raw == 1):6d}")
print(f"  Class 2 (PAC, a):        {np.sum(y_raw == 2):6d}")
print(f"\nTotal beats: {len(y_raw)}")

# ============================================================================
# STEP 2: PREPROCESSING
# ============================================================================

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


def print_preprocessing_stats(X, y, stats):
    """
    Print dataset statistics after preprocessing.
    
    Args:
        X: Preprocessed beats, shape (n_beats, 180)
        y: Labels, shape (n_beats,)
        stats: Dictionary with preprocessing statistics
    """
    n_total = len(X)
    
    print("\n" + "="*50)
    print("STEP 2: PREPROCESSING RESULTS")
    print("="*50)
    print(f"Total samples: {n_total}")
    print(f"Window shape: {X.shape}")
    
    # Class distribution
    print(f"\nClass distribution:")
    class_names = {
        0: ("Normal (N)", "N → 0"),
        1: ("PVC (V)", "V → 1"),
        2: ("PAC (a)", "a → 2")
    }
    
    for class_id in [0, 1, 2]:
        count = stats['class_counts'][class_id]
        percentage = (count / n_total * 100) if n_total > 0 else 0
        name, mapping = class_names[class_id]
        print(f"  {name:20s} ({mapping:6s}): {count:6d} ({percentage:5.1f}%)")
    
    # Preprocessing details
    print(f"\nPreprocessing applied:")
    print(f"  ✓ Baseline wander correction (detrending)")
    print(f"  ✓ Per-beat mean-std normalization")
    print(f"  ✓ Class filtering (3 classes)")
    
    # Verify normalization
    sample_means = np.mean(X, axis=1)
    sample_stds = np.std(X, axis=1)
    print(f"\nNormalization verification:")
    print(f"  Mean of beat means: {np.mean(sample_means):.6f} (target: ~0)")
    print(f"  Mean of beat stds:  {np.mean(sample_stds):.6f} (target: ~1)")


# Apply preprocessing pipeline
X, y, record_ids, preprocessing_stats = preprocess_beats(X_raw, y_raw, record_ids)

# Print preprocessing statistics
print_preprocessing_stats(X, y, preprocessing_stats)

# ============================================================================
# STEP 3: DATASET STATS + WEIGHTS + SPLIT
# ============================================================================

def compute_class_distribution(y):
    """
    Compute and print class distribution for 3-class labels.
    
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
    
    # Print distribution
    print("\n" + "="*50)
    print("STEP 3: CLASS DISTRIBUTION")
    print("="*50)
    print(f"Total samples: {n_total}")
    
    class_names = {
        0: ("Normal (N)", "N → 0"),
        1: ("PVC (V)", "V → 1"),
        2: ("PAC (a)", "a → 2")
    }
    
    print(f"\nClass distribution:")
    for class_id in [0, 1, 2]:
        count = class_counts[class_id]
        percentage = class_percentages[class_id]
        name, mapping = class_names[class_id]
        print(f"  {name:20s} ({mapping:6s}): {count:6d} ({percentage:5.1f}%)")
    
    return class_counts, class_percentages


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
    # Convert numpy strings to regular strings for cleaner display
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
print("FINAL OUTPUT READY FOR ML TRAINING")
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
# Format class weights for cleaner display (remove numpy type prefixes)
class_weights_display = {k: float(v) for k, v in class_weights.items()}
print(f"\nClass weights: {class_weights_display}")

# ============================================================================
# STEP 4: BASELINE MODEL TRAINING AND EVALUATION
# ============================================================================

def train_baseline_model(X_train, y_train, class_weights):
    """
    Train a multiclass logistic regression model with class weights.
    
    Args:
        X_train: Training features, shape (n_samples, 180)
        y_train: Training labels, shape (n_samples,)
        class_weights: Dictionary {class_id: weight}
    Returns:
        model: Trained LogisticRegression model
    """
    print("\n" + "="*50)
    print("STEP 4: TRAINING BASELINE MODEL")
    print("="*50)
    
    # Create model with improved settings for imbalanced classes
    # Use manual class weights with better hyperparameters
    # Lower C (0.1) for stronger regularization to help minority classes
    # Use 'lbfgs' solver which handles multiclass well
    # Higher max_iter to ensure convergence
    
    print("Training logistic regression model...")
    print(f"  Using computed class weights: {class_weights}")
    print(f"  Solver: lbfgs")
    print(f"  Regularization (C): 0.1 (stronger regularization for imbalanced classes)")
    print(f"  Max iterations: 2000")
    
    model = LogisticRegression(
        class_weight=class_weights,  # Use computed weights
        solver='lbfgs',
        C=0.1,  # Lower C = stronger regularization (helps minority classes)
        max_iter=2000,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Check prediction distribution on training set
    y_train_pred = model.predict(X_train)
    train_class_counts = {i: np.sum(y_train_pred == i) for i in [0, 1, 2]}
    print(f"\nTraining set prediction distribution:")
    print(f"  Class 0 (N): {train_class_counts[0]}")
    print(f"  Class 1 (V): {train_class_counts[1]}")
    print(f"  Class 2 (a): {train_class_counts[2]}")
    print("Training complete!")
    
    return model


def evaluate_model(model, X, y, split_name, class_names):
    """
    Evaluate model and print comprehensive metrics.
    
    Args:
        model: Trained model
        X: Features, shape (n_samples, 180)
        y: True labels, shape (n_samples,)
        split_name: String name of split ('Train', 'Val', 'Test')
        class_names: Dictionary mapping class_id to name
    """
    # Get predictions
    y_pred = model.predict(X)
    
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
    print(f"STEP 4: BASELINE MODEL - {split_name.upper()} SET")
    print("="*50)
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("              Predicted")
    print("              ", end="")
    # Print class labels: N, V, a
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


# Train model
model = train_baseline_model(X_train, y_train, class_weights)

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
