#!/usr/bin/env python3
"""
Blimp Trajectory Visualization Script

Reads the trajectory log produced by blimp.cpp (fixed_fusion_trajectory.csv)
and generates diagnostic plots.

CSV columns (from blimp.cpp fprintf):
    time, x, y, z,
    yaw_deg, cmd_yaw_deg, forward_speed,
    bearing_angle, bearing_valid, bearing_weight,
    grad_angle, grad_mag, grad_valid, grad_weight,
    map_size, total_light, dist_to_target

Plots produced:
    1. blimp_trajectory_3d.png         -- 3D path coloured by phase
    2. blimp_trajectory_2d.png         -- Top-down view with direction arrows
    3. blimp_position_vs_time.png      -- X, Y, Z and yaw over time
    4. blimp_speed_analysis.png        -- Forward speed and heading error
    5. blimp_navigation_analysis.png   -- Weights, gradient, map size, light

Usage:
    python trajectory_analyzer.py
    python trajectory_analyzer.py path/to/fixed_fusion_trajectory.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR   = os.path.join(SCRIPT_DIR, "..", "logs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")

CANDIDATE_PATHS = [
    os.path.join(LOGS_DIR,   "fixed_fusion_trajectory.csv"),
    os.path.join(SCRIPT_DIR, "fixed_fusion_trajectory.csv"),
    os.path.join(LOGS_DIR,   "trajectory_debug.csv"),
    os.path.join(SCRIPT_DIR, "trajectory_debug.csv"),
]

COL_BEARING = "#F44336"
COL_FUSED   = "#4CAF50"
COL_SOURCE  = "#FFD700"

TARGET_X = 5.1988
TARGET_Y = 5.329

# =============================================================================
# DATA LOADING
# =============================================================================

def load_trajectory_data(path=None):
    if path and os.path.exists(path):
        log_path = path
    else:
        log_path = None
        for c in CANDIDATE_PATHS:
            if os.path.exists(os.path.abspath(c)):
                log_path = os.path.abspath(c)
                break

    if log_path is None:
        print("[ERROR] Could not find trajectory CSV.")
        print("Searched:")
        for c in CANDIDATE_PATHS:
            print(f"  {os.path.abspath(c)}")
        print("Usage:  python trajectory_analyzer.py <file.csv>")
        return None

    print(f"Looking for log file at: {log_path}")
    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

    print(f"Loaded {len(df)} data points from trajectory log")
    print(f"Data columns: {list(df.columns)}")

    required = ["time", "x", "y", "z", "yaw_deg", "cmd_yaw_deg", "forward_speed"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return None

    # Derive a phase column (BEARING_ONLY vs FUSED) from grad_weight.
    # Once fusion has activated it stays active -- use cummax on a 0/1 flag
    # so that any step where grad_weight briefly drops back to 0 (e.g. a
    # rejected plane fit) is still labelled FUSED rather than snapping back
    # to BEARING_ONLY and drawing a red line through the green section.
    if "grad_weight" in df.columns:
        ever_fused = (df["grad_weight"] > 0).cummax()
        df["phase"] = ever_fused.map({True: "FUSED", False: "BEARING_ONLY"})
    else:
        df["phase"] = "BEARING_ONLY"

    # Heading error wrapped to [-180, 180]
    df["heading_error"] = ((df["cmd_yaw_deg"] - df["yaw_deg"] + 180) % 360) - 180

    return df


def create_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved -> {path}")

# =============================================================================
# PLOT 1 -- 3D trajectory coloured by phase
# =============================================================================

def phase_xy(df, phase):
    """
    Return x, y arrays with NaN where the phase is NOT active, so matplotlib
    draws separate segments instead of connecting non-contiguous rows with a
    straight line in the wrong colour.
    """
    x = df["x"].copy().astype(float)
    y = df["y"].copy().astype(float)
    mask = df["phase"] != phase
    x[mask] = np.nan
    y[mask] = np.nan
    return x.values, y.values


def phase_xyz(df, phase):
    """Same as phase_xy but also returns z."""
    x = df["x"].copy().astype(float)
    y = df["y"].copy().astype(float)
    z = df["z"].copy().astype(float)
    mask = df["phase"] != phase
    x[mask] = np.nan
    y[mask] = np.nan
    z[mask] = np.nan
    return x.values, y.values, z.values


def plot_3d_trajectory(df):
    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection="3d")

    for phase, color in [("BEARING_ONLY", COL_BEARING), ("FUSED", COL_FUSED)]:
        if (df["phase"] == phase).any():
            x, y, z = phase_xyz(df, phase)
            ax.plot(x, y, z, color=color, linewidth=2, alpha=0.85,
                    label=f"Phase: {phase}")

    ax.scatter(df["x"].iloc[0],  df["y"].iloc[0],  df["z"].iloc[0],
               color="green", s=150, marker="o", edgecolors="black",
               zorder=5, label="Start")
    ax.scatter(df["x"].iloc[-1], df["y"].iloc[-1], df["z"].iloc[-1],
               color="red",   s=150, marker="s", edgecolors="black",
               zorder=5, label="End")

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title("Blimp 3D Trajectory\n(red = bearing-only, green = fused)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ranges = np.array([df["x"].max()-df["x"].min(),
                       df["y"].max()-df["y"].min(),
                       df["z"].max()-df["z"].min()])
    half  = max(ranges.max() / 2.0, 0.1)
    mid_x = (df["x"].max() + df["x"].min()) * 0.5
    mid_y = (df["y"].max() + df["y"].min()) * 0.5
    mid_z = (df["z"].max() + df["z"].min()) * 0.5
    ax.set_xlim(mid_x - half, mid_x + half)
    ax.set_ylim(mid_y - half, mid_y + half)
    ax.set_zlim(mid_z - half, mid_z + half)

    plt.tight_layout()
    return fig

# =============================================================================
# PLOT 2 -- 2D top-down trajectory
# =============================================================================

def plot_2d_trajectory(df):
    fig, ax = plt.subplots(figsize=(12, 10))

    for phase, color in [("BEARING_ONLY", COL_BEARING), ("FUSED", COL_FUSED)]:
        if (df["phase"] == phase).any():
            x, y = phase_xy(df, phase)
            ax.plot(x, y, color=color, linewidth=2.5, alpha=0.85,
                    label=f"Phase: {phase}")

    n_arrows = min(15, max(1, len(df) // 20))
    for idx in np.linspace(0, len(df) - 2, n_arrows, dtype=int):
        dx = df["x"].iloc[idx+1] - df["x"].iloc[idx]
        dy = df["y"].iloc[idx+1] - df["y"].iloc[idx]
        if np.sqrt(dx**2 + dy**2) > 0.01:
            ax.arrow(df["x"].iloc[idx], df["y"].iloc[idx], dx*2, dy*2,
                     head_width=0.06, head_length=0.06,
                     fc="darkblue", ec="darkblue", alpha=0.6)

    ax.scatter(df["x"].iloc[0],  df["y"].iloc[0],
               color="green", s=150, marker="o", zorder=5,
               edgecolors="black", label="Start")
    ax.scatter(df["x"].iloc[-1], df["y"].iloc[-1],
               color="red",   s=150, marker="s", zorder=5,
               edgecolors="black", label="End")
    ax.scatter(TARGET_X, TARGET_Y,
               color=COL_SOURCE, s=250, marker="*", zorder=6,
               edgecolors="black", linewidths=0.8, label="Light source")

    fused = df[df["phase"] == "FUSED"]
    if not fused.empty:
        first = fused.iloc[0]
        ax.scatter(first["x"], first["y"], s=200, color="white",
                   edgecolors="black", lw=2, zorder=7,
                   label=f"Fusion starts (t={first['time']:.1f}s)")

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Blimp 2D Trajectory (Top View)", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig

# =============================================================================
# PLOT 3 -- Position and yaw vs time
# =============================================================================

def plot_position_vs_time(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    t = df["time"]

    axes[0,0].plot(t, df["x"], "b-", linewidth=2, label="Actual X")
    axes[0,0].set_xlabel("Time (s)"); axes[0,0].set_ylabel("X Position (m)")
    axes[0,0].set_title("X Position vs Time", fontweight="bold")
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(t, df["y"], "g-", linewidth=2, label="Actual Y")
    axes[0,1].set_xlabel("Time (s)"); axes[0,1].set_ylabel("Y Position (m)")
    axes[0,1].set_title("Y Position vs Time", fontweight="bold")
    axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    axes[1,0].plot(t, df["z"], "m-", linewidth=2, label="Actual Z (altitude)")
    axes[1,0].set_xlabel("Time (s)"); axes[1,0].set_ylabel("Z Position (m)")
    axes[1,0].set_title("Altitude vs Time", fontweight="bold")
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    # yaw_deg is already in degrees -- no np.degrees() needed
    axes[1,1].plot(t, df["yaw_deg"],     "c-",  linewidth=2, label="Actual Yaw")
    axes[1,1].plot(t, df["cmd_yaw_deg"], "r--", linewidth=2, alpha=0.7,
                   label="Commanded Yaw")
    axes[1,1].set_xlabel("Time (s)"); axes[1,1].set_ylabel("Yaw (degrees)")
    axes[1,1].set_title("Yaw Angle vs Time", fontweight="bold")
    axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# =============================================================================
# PLOT 4 -- Speed and heading error
# =============================================================================

def plot_speed_analysis(df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    t = df["time"]

    axes[0].plot(t, df["forward_speed"], "b-", linewidth=2, label="Forward Speed")
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Forward Speed vs Time", fontsize=14, fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, df["heading_error"], "r-", linewidth=2, label="Heading Error")
    axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1].fill_between(t, df["heading_error"], alpha=0.25, color="red")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Heading Error (degrees)")
    axes[1].set_title("Heading Tracking Error  (cmd_yaw - actual_yaw)",
                      fontsize=14, fontweight="bold")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    rms = np.sqrt(np.mean(df["heading_error"]**2))
    axes[1].text(0.02, 0.98, f"RMS Error: {rms:.2f}deg",
                 transform=axes[1].transAxes, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    return fig

# =============================================================================
# PLOT 5 -- Navigation analysis
# =============================================================================

def plot_navigation_analysis(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    t = df["time"]

    fused = df[df["phase"] == "FUSED"]
    fused_t = fused["time"].iloc[0] if not fused.empty else t.iloc[-1]

    def shade(ax):
        ax.axvspan(t.iloc[0], fused_t,    alpha=0.06, color=COL_BEARING)
        ax.axvspan(fused_t,   t.iloc[-1], alpha=0.06, color=COL_FUSED)
        ax.axvline(fused_t, color="black", lw=1.2, ls="--", alpha=0.5)

    # Panel 1: weights
    ax = axes[0,0]
    if "bearing_weight" in df.columns:
        ax.plot(t, df["bearing_weight"], color=COL_BEARING, lw=2,
                label="Bearing weight")
    if "grad_weight" in df.columns:
        ax.plot(t, df["grad_weight"], color=COL_FUSED, lw=2,
                label="Gradient weight")
    shade(ax)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Weight")
    ax.set_title("Navigation Weights (Bearing vs Gradient)", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 2: gradient magnitude + bearing angle
    ax = axes[0,1]
    if "grad_mag" in df.columns:
        ax.plot(t, df["grad_mag"], color=COL_FUSED, lw=1.8,
                label="Gradient magnitude")
        ax.set_ylabel("Gradient magnitude")
    if "bearing_angle" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(t, df["bearing_angle"], color=COL_BEARING, lw=1.4,
                 alpha=0.7, label="Bearing angle (deg)")
        ax2.set_ylabel("Bearing angle (deg)", color=COL_BEARING)
        ax2.tick_params(axis="y", labelcolor=COL_BEARING)
    shade(ax)
    ax.set_xlabel("Time (s)")
    ax.set_title("Gradient Magnitude & Bearing Angle vs Time", fontweight="bold")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.25)

    # Panel 3: map size
    ax = axes[1,0]
    if "map_size" in df.columns:
        ax.plot(t, df["map_size"], color="#9C27B0", lw=2,
                label="Map cells filled")
        ax.fill_between(t, df["map_size"], alpha=0.15, color="#9C27B0")
    shade(ax)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Map size (cells)")
    ax.set_title("Spatial Map Accumulation", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 4: light + distance
    ax = axes[1,1]
    if "total_light" in df.columns:
        ax.plot(t, df["total_light"], color="#FF9800", lw=1.8,
                label="Total light intensity")
        ax.set_ylabel("Total light intensity")
    if "dist_to_target" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(t, df["dist_to_target"], color="#3F51B5", lw=1.8,
                 label="Distance to target (m)")
        ax2.set_ylabel("Distance to target (m)", color="#3F51B5")
        ax2.tick_params(axis="y", labelcolor="#3F51B5")
    shade(ax)
    ax.set_xlabel("Time (s)")
    ax.set_title("Light Intensity & Distance to Target", fontweight="bold")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.25)

    plt.tight_layout()
    return fig

# =============================================================================
# STATISTICS
# =============================================================================

def print_statistics(df):
    print("\n" + "="*60)
    print("BLIMP TRAJECTORY ANALYSIS REPORT")
    print("="*60)

    print(f"Total simulation time: {df['time'].iloc[-1]:.2f} seconds")
    print(f"Total data points: {len(df)}")
    print(f"Average sampling rate: {len(df)/df['time'].iloc[-1]:.1f} Hz")

    print(f"\nPosition Range:")
    for col in ["x", "y", "z"]:
        lo, hi = df[col].min(), df[col].max()
        print(f"  {col.upper()}: {lo:.3f} to {hi:.3f} m "
              f"(range: {hi-lo:.3f} m)")

    dx = np.diff(df["x"]); dy = np.diff(df["y"]); dz = np.diff(df["z"])
    total_dist = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
    print(f"\nTotal distance traveled: {total_dist:.3f} m")

    print(f"\nSpeed Statistics:")
    print(f"  Max speed: {df['forward_speed'].max():.3f} m/s")
    print(f"  Average speed: {df['forward_speed'].mean():.3f} m/s")
    print(f"  Speed standard deviation: {df['forward_speed'].std():.3f} m/s")

    print(f"\nHeading Tracking:")
    rms_h = np.sqrt(np.mean(df["heading_error"]**2))
    print(f"  RMS heading error: {rms_h:.2f} deg")
    print(f"  Max heading error: {df['heading_error'].abs().max():.2f} deg")

    if "dist_to_target" in df.columns:
        print(f"\nDistance to target:")
        print(f"  Start: {df['dist_to_target'].iloc[0]:.2f} m")
        print(f"  End:   {df['dist_to_target'].iloc[-1]:.2f} m")

    if "map_size" in df.columns:
        print(f"\nMap size at end: {int(df['map_size'].iloc[-1])} cells")

    print(f"\nController Modes Used:")
    phase_counts = df["phase"].value_counts()
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count/len(df)*100:.1f}% ({count} data points)")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Blimp Trajectory Visualization and Analysis")
    print("=" * 50)

    create_output_dir()

    path = sys.argv[1] if len(sys.argv) > 1 else None
    df   = load_trajectory_data(path)
    if df is None:
        return

    print_statistics(df)

    print(f"\nGenerating plots in: {os.path.abspath(OUTPUT_DIR)}")

    print("Generating 3D trajectory plot...")
    save(plot_3d_trajectory(df), "blimp_trajectory_3d.png")
    plt.close("all")
    print("  3D trajectory plot saved")

    print("Generating 2D trajectory plot...")
    save(plot_2d_trajectory(df), "blimp_trajectory_2d.png")
    plt.close("all")
    print("  2D trajectory plot saved")

    print("Generating position vs time plots...")
    save(plot_position_vs_time(df), "blimp_position_vs_time.png")
    plt.close("all")
    print("  Position vs time plot saved")

    print("Generating speed analysis plots...")
    save(plot_speed_analysis(df), "blimp_speed_analysis.png")
    plt.close("all")
    print("  Speed analysis plot saved")

    print("Generating navigation analysis plots...")
    save(plot_navigation_analysis(df), "blimp_navigation_analysis.png")
    plt.close("all")
    print("  Navigation analysis plot saved")

    print(f"\nAll plots generated successfully in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()