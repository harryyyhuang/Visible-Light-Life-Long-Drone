#!/usr/bin/env python3
"""
Blimp Trajectory Visualization Script

This script reads the trajectory log CSV file and creates various plots:
1. 3D trajectory plot
2. 2D trajectory (top-down view)
3. Position vs time plots
4. Speed vs time plot
5. Yaw angle vs time plot

Usage:
    python trajectory_analyzer.py

Make sure you have the required libraries installed:
    pip install matplotlib pandas numpy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

# Configuration - Use relative paths to logs directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "..", "logs")
LOG_FILE_PATH = os.path.join(LOGS_DIR, "trajectory_debug.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")

def create_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def load_trajectory_data():
    """Load trajectory data from CSV file"""
    try:
        abs_log_path = os.path.abspath(LOG_FILE_PATH)
        print(f"Looking for log file at: {abs_log_path}")
        
        if not os.path.exists(abs_log_path):
            print(f"Error: Log file not found at {abs_log_path}")
            print("Make sure to run the simulation with DEBUG=1 to generate log data")
            return None
            
        df = pd.read_csv(abs_log_path)
        print(f"Loaded {len(df)} data points from trajectory log")
        print(f"Data columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_3d_trajectory(df):
    """Create a 3D plot of the blimp trajectory"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by mode
    modes = df['mode'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(modes)))
    
    for i, mode in enumerate(modes):
        mode_data = df[df['mode'] == mode]
        ax.plot(mode_data['x'], mode_data['y'], mode_data['z'], 
                color=colors[i], label=f'Mode: {mode}', linewidth=2, alpha=0.8)
    
    # Plot setpoints as reference
    ax.plot(df['setpoint_x'], df['setpoint_y'], df['setpoint_z'], 
            'k--', alpha=0.5, linewidth=1, label='Setpoint Path')
    
    # Mark start and end points
    ax.scatter(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0], 
               color='green', s=150, label='Start', marker='o', edgecolors='black')
    ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1], 
               color='red', s=150, label='End', marker='s', edgecolors='black')
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Blimp 3D Trajectory', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
    ax.grid(True, alpha=0.3)
    
    # Make axes equal
    max_range = np.array([df['x'].max()-df['x'].min(), 
                         df['y'].max()-df['y'].min(), 
                         df['z'].max()-df['z'].min()]).max() / 2.0
    mid_x = (df['x'].max()+df['x'].min()) * 0.5
    mid_y = (df['y'].max()+df['y'].min()) * 0.5
    mid_z = (df['z'].max()+df['z'].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig

def plot_2d_trajectory(df):
    """Create a 2D top-down view of the trajectory"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by mode
    modes = df['mode'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(modes)))
    
    for i, mode in enumerate(modes):
        mode_data = df[df['mode'] == mode]
        ax.plot(mode_data['x'], mode_data['y'], 
                color=colors[i], label=f'Mode: {mode}', linewidth=3, alpha=0.8)
    
    # Plot setpoints as reference
    ax.plot(df['setpoint_x'], df['setpoint_y'], 
            'k--', alpha=0.5, linewidth=2, label='Setpoint Path')
    
    # Mark start and end points
    ax.scatter(df['x'].iloc[0], df['y'].iloc[0], 
               color='green', s=150, label='Start', marker='o', zorder=5, edgecolors='black')
    ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], 
               color='red', s=150, label='End', marker='s', zorder=5, edgecolors='black')
    
    # Add arrows to show direction
    n_arrows = min(15, len(df)//20)  # Show max 15 arrows
    if n_arrows > 0:
        arrow_indices = np.linspace(0, len(df)-2, n_arrows, dtype=int)
        
        for idx in arrow_indices:
            dx = df['x'].iloc[idx+1] - df['x'].iloc[idx]
            dy = df['y'].iloc[idx+1] - df['y'].iloc[idx]
            if np.sqrt(dx**2 + dy**2) > 0.01:  # Only draw if movement is significant
                ax.arrow(df['x'].iloc[idx], df['y'].iloc[idx], dx*2, dy*2,
                        head_width=0.05, head_length=0.05, fc='darkblue', ec='darkblue', alpha=0.7)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Blimp 2D Trajectory (Top View)', fontsize=16, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def plot_position_vs_time(df):
    """Plot position components vs time"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # X position
    axes[0,0].plot(df['time'], df['x'], 'b-', label='Actual X', linewidth=2)
    axes[0,0].plot(df['time'], df['setpoint_x'], 'r--', label='Setpoint X', alpha=0.7, linewidth=2)
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('X Position (m)')
    axes[0,0].set_title('X Position vs Time', fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Y position
    axes[0,1].plot(df['time'], df['y'], 'g-', label='Actual Y', linewidth=2)
    axes[0,1].plot(df['time'], df['setpoint_y'], 'r--', label='Setpoint Y', alpha=0.7, linewidth=2)
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Y Position (m)')
    axes[0,1].set_title('Y Position vs Time', fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Z position (altitude)
    axes[1,0].plot(df['time'], df['z'], 'm-', label='Actual Z', linewidth=2)
    axes[1,0].plot(df['time'], df['setpoint_z'], 'r--', label='Setpoint Z', alpha=0.7, linewidth=2)
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Z Position (m)')
    axes[1,0].set_title('Altitude vs Time', fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Yaw angle (convert from radians to degrees)
    axes[1,1].plot(df['time'], np.degrees(df['yaw']), 'c-', label='Actual Yaw', linewidth=2)
    axes[1,1].plot(df['time'], np.degrees(df['setpoint_yaw']), 'r--', label='Setpoint Yaw', alpha=0.7, linewidth=2)
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Yaw Angle (degrees)')
    axes[1,1].set_title('Yaw Angle vs Time', fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_speed_analysis(df):
    """Plot speed analysis"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Speed vs time
    axes[0].plot(df['time'], df['current_speed'], 'b-', label='Actual Speed', linewidth=2)
    axes[0].plot(df['time'], df['setpoint_speed'], 'r--', label='Setpoint Speed', alpha=0.7, linewidth=2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Speed (m/s)')
    axes[0].set_title('Forward Speed vs Time', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Speed error
    speed_error = df['setpoint_speed'] - df['current_speed']
    axes[1].plot(df['time'], speed_error, 'r-', linewidth=2, label='Speed Error')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].fill_between(df['time'], speed_error, alpha=0.3, color='red')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Speed Error (m/s)')
    axes[1].set_title('Speed Tracking Error', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add RMS error text
    rms_error = np.sqrt(np.mean(speed_error**2))
    axes[1].text(0.02, 0.98, f'RMS Error: {rms_error:.4f} m/s', 
                transform=axes[1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_tracking_errors(df):
    """Plot position and yaw tracking errors"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Position errors
    x_error = df['setpoint_x'] - df['x']
    y_error = df['setpoint_y'] - df['y']
    z_error = df['setpoint_z'] - df['z']
    yaw_error = np.degrees(df['setpoint_yaw'] - df['yaw'])
    
    # X error
    axes[0,0].plot(df['time'], x_error, 'b-', linewidth=2)
    axes[0,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0,0].fill_between(df['time'], x_error, alpha=0.3, color='blue')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('X Error (m)')
    axes[0,0].set_title('X Position Tracking Error', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    rms_x = np.sqrt(np.mean(x_error**2))
    axes[0,0].text(0.02, 0.98, f'RMS: {rms_x:.4f} m', transform=axes[0,0].transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Y error
    axes[0,1].plot(df['time'], y_error, 'g-', linewidth=2)
    axes[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0,1].fill_between(df['time'], y_error, alpha=0.3, color='green')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Y Error (m)')
    axes[0,1].set_title('Y Position Tracking Error', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    rms_y = np.sqrt(np.mean(y_error**2))
    axes[0,1].text(0.02, 0.98, f'RMS: {rms_y:.4f} m', transform=axes[0,1].transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Z error
    axes[1,0].plot(df['time'], z_error, 'm-', linewidth=2)
    axes[1,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1,0].fill_between(df['time'], z_error, alpha=0.3, color='magenta')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Z Error (m)')
    axes[1,0].set_title('Altitude Tracking Error', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    rms_z = np.sqrt(np.mean(z_error**2))
    axes[1,0].text(0.02, 0.98, f'RMS: {rms_z:.4f} m', transform=axes[1,0].transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
    
    # Yaw error
    axes[1,1].plot(df['time'], yaw_error, 'c-', linewidth=2)
    axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1,1].fill_between(df['time'], yaw_error, alpha=0.3, color='cyan')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Yaw Error (degrees)')
    axes[1,1].set_title('Yaw Angle Tracking Error', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    rms_yaw = np.sqrt(np.mean(yaw_error**2))
    axes[1,1].text(0.02, 0.98, f'RMS: {rms_yaw:.4f}°', transform=axes[1,1].transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    return fig

def print_statistics(df):
    """Print trajectory statistics"""
    print("\n" + "="*60)
    print("BLIMP TRAJECTORY ANALYSIS REPORT")
    print("="*60)
    
    print(f"Total simulation time: {df['time'].iloc[-1]:.2f} seconds")
    print(f"Total data points: {len(df)}")
    print(f"Average sampling rate: {len(df)/df['time'].iloc[-1]:.1f} Hz")
    
    print(f"\nPosition Range:")
    print(f"  X: {df['x'].min():.3f} to {df['x'].max():.3f} m (range: {df['x'].max()-df['x'].min():.3f} m)")
    print(f"  Y: {df['y'].min():.3f} to {df['y'].max():.3f} m (range: {df['y'].max()-df['y'].min():.3f} m)")
    print(f"  Z: {df['z'].min():.3f} to {df['z'].max():.3f} m (range: {df['z'].max()-df['z'].min():.3f} m)")
    
    print(f"\nSpeed Statistics:")
    print(f"  Max speed: {df['current_speed'].max():.3f} m/s")
    print(f"  Average speed: {df['current_speed'].mean():.3f} m/s")
    print(f"  Speed standard deviation: {df['current_speed'].std():.3f} m/s")
    
    # Calculate total distance traveled
    dx = np.diff(df['x'])
    dy = np.diff(df['y'])
    dz = np.diff(df['z'])
    distances = np.sqrt(dx**2 + dy**2 + dz**2)
    total_distance = np.sum(distances)
    print(f"\nTotal distance traveled: {total_distance:.3f} m")
    
    # Tracking error statistics
    x_error = df['setpoint_x'] - df['x']
    y_error = df['setpoint_y'] - df['y']
    z_error = df['setpoint_z'] - df['z']
    yaw_error = np.degrees(df['setpoint_yaw'] - df['yaw'])
    speed_error = df['setpoint_speed'] - df['current_speed']
    
    print(f"\nTracking Performance (RMS Errors):")
    print(f"  X Position: {np.sqrt(np.mean(x_error**2)):.4f} m")
    print(f"  Y Position: {np.sqrt(np.mean(y_error**2)):.4f} m")
    print(f"  Z Position: {np.sqrt(np.mean(z_error**2)):.4f} m")
    print(f"  Yaw Angle: {np.sqrt(np.mean(yaw_error**2)):.4f} degrees")
    print(f"  Speed: {np.sqrt(np.mean(speed_error**2)):.4f} m/s")
    
    # Mode statistics
    print(f"\nController Modes Used:")
    mode_counts = df['mode'].value_counts()
    for mode, count in mode_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {mode}: {percentage:.1f}% ({count} data points)")

def main():
    """Main function to generate all plots"""
    print("Blimp Trajectory Visualization and Analysis")
    print("=" * 50)
    
    # Create output directory
    create_output_dir()
    
    # Load data
    df = load_trajectory_data()
    if df is None:
        return
    
    # Print statistics
    print_statistics(df)
    
    # Generate plots
    print(f"\nGenerating plots in: {OUTPUT_DIR}")
    
    # 3D trajectory
    print("Generating 3D trajectory plot...")
    fig1 = plot_3d_trajectory(df)
    fig1.savefig(os.path.join(OUTPUT_DIR, 'blimp_trajectory_3d.png'), dpi=300, bbox_inches='tight')
    print("✓ 3D trajectory plot saved")
    
    # 2D trajectory
    print("Generating 2D trajectory plot...")
    fig2 = plot_2d_trajectory(df)
    fig2.savefig(os.path.join(OUTPUT_DIR, 'blimp_trajectory_2d.png'), dpi=300, bbox_inches='tight')
    print("✓ 2D trajectory plot saved")
    
    # Position vs time
    print("Generating position vs time plots...")
    fig3 = plot_position_vs_time(df)
    fig3.savefig(os.path.join(OUTPUT_DIR, 'blimp_position_vs_time.png'), dpi=300, bbox_inches='tight')
    print("✓ Position vs time plot saved")
    
    # Speed analysis
    print("Generating speed analysis plots...")
    fig4 = plot_speed_analysis(df)
    fig4.savefig(os.path.join(OUTPUT_DIR, 'blimp_speed_analysis.png'), dpi=300, bbox_inches='tight')
    print("✓ Speed analysis plot saved")
    
    # Tracking errors
    print("Generating tracking error analysis...")
    fig5 = plot_tracking_errors(df)
    fig5.savefig(os.path.join(OUTPUT_DIR, 'blimp_tracking_errors.png'), dpi=300, bbox_inches='tight')
    print("✓ Tracking error analysis plot saved")
    
    print(f"\nAll plots generated successfully in: {os.path.abspath(OUTPUT_DIR)}")
    print("Close the plot windows or press Ctrl+C to exit.")
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()
