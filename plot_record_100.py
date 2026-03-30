"""
Visualize MIT-BIH Record 100 with All Annotations

Displays the ECG signal from record 100 with all R-peak annotations overlaid.
Shows proper axes labels (time in seconds, amplitude in mV).

Usage:
    python plot_record_100.py              # Show first 15 seconds, interactive
    python plot_record_100.py --save       # Save to file instead of displaying
"""

import argparse
import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ecg_preprocessing import get_mitdb_data_path


def load_and_plot_record(record_name="100", duration_sec=15, start_sec=0, save_path=None):
    """
    Load a MIT-BIH record and plot ECG with all annotations.
    
    Args:
        record_name: Record ID (e.g., "100")
        duration_sec: Duration of segment to plot in seconds
        start_sec: Start time of segment in seconds
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = get_mitdb_data_path(script_dir)
    
    # Change to data directory (for wfdb compatibility)
    original_dir = os.getcwd()
    os.chdir(data_path)
    
    try:
        # Load record
        record = wfdb.rdrecord(record_name)
        annotations = wfdb.rdann(record_name, "atr")
    finally:
        os.chdir(original_dir)
    
    # Get signal and metadata
    fs = record.fs  # Sampling frequency (typically 360 Hz)
    ecg_signal = record.p_signal[:, 0]  # First channel (usually MLII)
    signal_units = "mV"
    if hasattr(record, "units") and record.units is not None:
        try:
            signal_units = record.units[0]
        except (IndexError, TypeError):
            pass
    
    # Annotation data
    ann_sample = annotations.sample  # Sample indices
    ann_symbol = annotations.symbol  # Beat type symbols (N, V, a, etc.)
    
    # Compute segment
    start_sample = int(start_sec * fs)
    end_sample = int((start_sec + duration_sec) * fs)
    end_sample = min(end_sample, len(ecg_signal))
    
    signal_segment = ecg_signal[start_sample:end_sample]
    time_axis = np.arange(len(signal_segment)) / fs + start_sec
    
    # Filter annotations to those within the segment
    idx = np.where((ann_sample >= start_sample) & (ann_sample < end_sample))[0]
    ann_sample_seg = ann_sample[idx]
    ann_symbol_seg = [ann_symbol[i] for i in idx]
    
    # Convert annotation sample indices to time for plotting
    ann_time = (ann_sample_seg - start_sample) / fs + start_sec
    ann_amplitude = ecg_signal[ann_sample_seg]
    
    # Create figure with clear axes
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Plot ECG signal
    ax.plot(time_axis, signal_segment, "b-", linewidth=1.0, label="ECG signal")
    
    # Plot annotations as markers with labels
    colors = {"N": "green", "V": "red", "a": "orange", "L": "purple", "R": "brown",
              "A": "darkorange", "/": "gray", "f": "pink", "j": "cyan", "E": "magenta",
              "J": "teal", "Q": "coral", "F": "navy", "?": "gray"}
    
    for t, amp, sym in zip(ann_time, ann_amplitude, ann_symbol_seg):
        sym_str = str(sym).strip() if hasattr(sym, "strip") else str(sym)
        color = colors.get(sym_str, "black")
        ax.scatter(t, amp, c=color, s=50, zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(
            sym_str,
            (t, amp),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=9,
            ha="center",
            fontweight="bold",
        )
    
    # Axis labels and title
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel(f"Amplitude ({signal_units})", fontsize=12)
    ax.set_title(f"MIT-BIH Record {record_name} — ECG with All Annotations (t = {start_sec}s to {start_sec + duration_sec}s)", fontsize=14)
    
    # Grid
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Legend for annotation symbols in this segment
    unique_syms = sorted(set(str(s).strip() if hasattr(s, "strip") else str(s) for s in ann_symbol_seg))
    legend_elements = [plt.Line2D([0], [0], color="b", linewidth=2, label="ECG signal")]
    for s in unique_syms:
        legend_elements.append(
            Patch(facecolor=colors.get(s, "black"), edgecolor="black", label=f"'{s}'")
        )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    # Print summary to console
    print(f"\nRecord {record_name}:")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Total length: {len(ecg_signal) / fs:.1f} seconds")
    print(f"  Annotations in segment: {len(ann_sample_seg)}")
    print(f"  Annotation types in segment: {unique_syms}")
    print(f"  All annotation types in full record: {sorted(set(str(s).strip() for s in ann_symbol))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MIT-BIH record 100 with annotations")
    parser.add_argument("--save", action="store_true", help="Save plot to file instead of displaying")
    parser.add_argument("--record", default="100", help="Record ID (default: 100)")
    parser.add_argument("--duration", type=float, default=15, help="Duration in seconds (default: 15)")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds (default: 0)")
    args = parser.parse_args()

    save_path = "record_100_annotated.png" if args.save else None
    load_and_plot_record(
        record_name=args.record,
        duration_sec=args.duration,
        start_sec=args.start,
        save_path=save_path,
    )
