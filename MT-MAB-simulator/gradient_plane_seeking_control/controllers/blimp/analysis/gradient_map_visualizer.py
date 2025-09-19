#!/usr/bin/env python3
"""
Gradient Map Visualizer for Blimp Trajectory Analysis

This script creates comprehensive visualizations of the map-based gradient seeking algorithm:
1. 3D trajectory with gradient vectors
2. 2D trajectory with measurement map overlay
3. Gradient field visualization with light intensity heatmap
4. State transitions and performance metrics
5. Animation of trajectory building over time

Usage:
    python gradient_map_visualizer.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
import os
import sys
from pathlib import Path

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GradientMapVisualizer:
    def __init__(self, data_dir="."):
        """Initialize the visualizer with data directory."""
        self.data_dir = Path(data_dir)
        self.trajectory_df = None
        self.map_df = None
        self.gradient_df = None
        self.target_x = 5.1988
        self.target_y = 5.329
        
    def load_data(self):
        """Load all CSV data files."""
        try:
            # Load trajectory data
            trajectory_files = [
                self.data_dir / "history_gradient_trajectory.csv",
                self.data_dir / "../history_gradient_trajectory.csv",
                self.data_dir / "../logs/history_gradient_trajectory.csv",
                Path("history_gradient_trajectory.csv"),
                Path("../history_gradient_trajectory.csv"),
                Path("../logs/history_gradient_trajectory.csv")
            ]
            
            for file in trajectory_files:
                if file.exists():
                    self.trajectory_df = pd.read_csv(file)
                    print(f"[SUCCESS] Loaded trajectory data from {file}")
                    break
            else:
                print("[ERROR] Trajectory data file not found!")
                return False
                
            # Load measurement map data
            map_files = [
                self.data_dir / "measurement_map.csv",
                self.data_dir / "../measurement_map.csv",
                self.data_dir / "../logs/measurement_map.csv",
                Path("measurement_map.csv"),
                Path("../measurement_map.csv"),
                Path("../logs/measurement_map.csv")
            ]
            
            for file in map_files:
                if file.exists():
                    self.map_df = pd.read_csv(file)
                    print(f"[SUCCESS] Loaded map data from {file} ({len(self.map_df)} measurements)")
                    break
            else:
                print("[WARNING] Map data file not found - skipping map visualization")
                
            # Load gradient field data
            gradient_files = [
                self.data_dir / "gradient_field.csv",
                self.data_dir / "../gradient_field.csv",
                self.data_dir / "../logs/gradient_field.csv",
                Path("gradient_field.csv"),
                Path("../gradient_field.csv"),
                Path("../logs/gradient_field.csv")
            ]
            
            for file in gradient_files:
                if file.exists():
                    self.gradient_df = pd.read_csv(file)
                    valid_gradients = self.gradient_df[self.gradient_df['valid'] == 1]
                    print(f"[SUCCESS] Loaded gradient field from {file} ({len(valid_gradients)}/{len(self.gradient_df)} valid)")
                    break
            else:
                print("[WARNING] Gradient field file not found - skipping gradient visualization")
                
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading data: {e}")
            return False
            
    def create_3d_trajectory_plot(self):
        """Create 3D trajectory plot with gradient vectors."""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color map for different modes
        mode_colors = {'FORWARD': 'gray', 'EXPLORING': 'blue', 'SEEKING': 'red'}
        
        # Plot trajectory colored by mode
        for mode in self.trajectory_df['mode'].unique():
            mode_data = self.trajectory_df[self.trajectory_df['mode'] == mode]
            ax.plot(mode_data['x'], mode_data['y'], mode_data['z'], 
                   color=mode_colors.get(mode, 'black'), label=f'{mode} Mode',
                   linewidth=2, alpha=0.8)
        
        # Plot measurement points if available
        if self.map_df is not None:
            scatter = ax.scatter(self.map_df['x'], self.map_df['y'], 2.0,  # Assume constant altitude
                               c=self.map_df['light_intensity'], cmap='hot', 
                               s=50, alpha=0.7, label='Map Measurements')
            plt.colorbar(scatter, ax=ax, label='Light Intensity', shrink=0.6)
        
        # Plot target location
        ax.scatter([self.target_x], [self.target_y], [2.0], 
                  color='gold', s=200, marker='*', 
                  label='Target Light Source', edgecolors='black', linewidth=2)
        
        # Plot start and end points
        start_point = self.trajectory_df.iloc[0]
        end_point = self.trajectory_df.iloc[-1]
        ax.scatter([start_point['x']], [start_point['y']], [start_point['z']], 
                  color='green', s=100, marker='o', label='Start')
        ax.scatter([end_point['x']], [end_point['y']], [end_point['z']], 
                  color='red', s=100, marker='s', label='End')
        
        # Add gradient vectors at key points
        if self.gradient_df is not None:
            valid_gradients = self.gradient_df[self.gradient_df['valid'] == 1]
            # Sample every 5th point to avoid clutter
            sample_gradients = valid_gradients[::5]
            
            for _, row in sample_gradients.iterrows():
                # Scale gradient vectors for visibility
                scale = 0.5
                ax.quiver(row['x'], row['y'], 2.0,
                         row['grad_x'] * scale, row['grad_y'] * scale, 0,
                         color='purple', alpha=0.6, arrow_length_ratio=0.1)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('3D Trajectory with Gradient Field\nMap-Based Gradient Seeking', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('./plots/gradient_trajectory_3d.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved 3D trajectory plot")
        
    def create_2d_trajectory_with_map(self):
        """Create 2D top-down view with measurement map overlay."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Trajectory with map measurements
        if self.map_df is not None:
            # Create heatmap of light intensity
            scatter1 = ax1.scatter(self.map_df['x'], self.map_df['y'], 
                                 c=self.map_df['light_intensity'], cmap='hot',
                                 s=100, alpha=0.7, label='Map Measurements')
            plt.colorbar(scatter1, ax=ax1, label='Light Intensity')
        
        # Plot trajectory colored by mode
        mode_colors = {'FORWARD': 'gray', 'EXPLORING': 'blue', 'SEEKING': 'red'}
        for mode in self.trajectory_df['mode'].unique():
            mode_data = self.trajectory_df[self.trajectory_df['mode'] == mode]
            ax1.plot(mode_data['x'], mode_data['y'], 
                    color=mode_colors.get(mode, 'black'), label=f'{mode} Mode',
                    linewidth=3, alpha=0.8)
        
        # Add direction arrows
        traj_sample = self.trajectory_df[::20]  # Sample every 20th point
        for _, row in traj_sample.iterrows():
            yaw_rad = row['yaw']
            dx = 0.3 * np.cos(yaw_rad)
            dy = 0.3 * np.sin(yaw_rad)
            ax1.arrow(row['x'], row['y'], dx, dy, head_width=0.1, 
                     head_length=0.1, fc='black', ec='black', alpha=0.5)
        
        # Plot target and start/end points
        ax1.scatter([self.target_x], [self.target_y], color='gold', s=300, 
                   marker='*', label='Target', edgecolors='black', linewidth=2)
        start_point = self.trajectory_df.iloc[0]
        end_point = self.trajectory_df.iloc[-1]
        ax1.scatter([start_point['x']], [start_point['y']], color='green', 
                   s=150, marker='o', label='Start', edgecolors='black')
        ax1.scatter([end_point['x']], [end_point['y']], color='red', 
                   s=150, marker='s', label='End', edgecolors='black')
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('2D Trajectory with Measurement Map', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Right plot: Gradient field visualization
        if self.gradient_df is not None:
            valid_gradients = self.gradient_df[self.gradient_df['valid'] == 1]
            
            # Create gradient magnitude heatmap
            scatter2 = ax2.scatter(valid_gradients['x'], valid_gradients['y'],
                                 c=valid_gradients['magnitude'], cmap='viridis',
                                 s=50, alpha=0.8)
            plt.colorbar(scatter2, ax=ax2, label='Gradient Magnitude')
            
            # Add gradient vector field
            sample_grad = valid_gradients[::3]  # Sample for clarity
            scale = 0.3
            ax2.quiver(sample_grad['x'], sample_grad['y'],
                      sample_grad['grad_x'] * scale, sample_grad['grad_y'] * scale,
                      alpha=0.7, scale=1, scale_units='xy', angles='xy',
                      color='white', width=0.003)
            
            # Overlay trajectory
            ax2.plot(self.trajectory_df['x'], self.trajectory_df['y'], 
                    'r-', linewidth=2, alpha=0.8, label='Trajectory')
            
            # Plot target
            ax2.scatter([self.target_x], [self.target_y], color='gold', s=300,
                       marker='*', label='Target', edgecolors='black', linewidth=2)
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Gradient Field Visualization', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('./plots/gradient_trajectory_2d.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved 2D trajectory and gradient field plots")
        
    def create_performance_analysis(self):
        """Create performance analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Distance to target over time
        ax1.plot(self.trajectory_df['time'], self.trajectory_df['distance_to_target'], 
                'b-', linewidth=2)
        ax1.axhline(y=0.5, color='r', linestyle='--', label='Proximity Threshold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Distance to Target (m)')
        ax1.set_title('Distance to Target Over Time', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Light intensity and gradient magnitude
        ax2_twin = ax2.twinx()
        ax2.plot(self.trajectory_df['time'], self.trajectory_df['light_intensity'], 
                'g-', linewidth=2, label='Light Intensity')
        ax2_twin.plot(self.trajectory_df['time'], self.trajectory_df['gradient_magnitude'], 
                     'r-', linewidth=2, label='Gradient Magnitude', alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Light Intensity', color='g')
        ax2_twin.set_ylabel('Gradient Magnitude', color='r')
        ax2.set_title('Light Signal and Gradient Tracking', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add legend combining both y-axes
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot 3: State transitions
        mode_numeric = self.trajectory_df['mode'].map({'FORWARD': 0, 'EXPLORING': 1, 'SEEKING': 2})
        ax3.plot(self.trajectory_df['time'], mode_numeric, 'o-', markersize=3, linewidth=1)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Control Mode')
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['FORWARD', 'EXPLORING', 'SEEKING'])
        ax3.set_title('State Transitions Over Time', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Speed analysis
        ax4.plot(self.trajectory_df['time'], self.trajectory_df['setpoint_speed'], 
                'b--', linewidth=2, label='Setpoint Speed', alpha=0.7)
        ax4.plot(self.trajectory_df['time'], self.trajectory_df['current_speed'], 
                'r-', linewidth=2, label='Actual Speed')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Speed (m/s)')
        ax4.set_title('Speed Tracking Performance', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./plots/gradient_performance_analysis.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved performance analysis plots")
        
    def create_map_building_analysis(self):
        """Analyze map building process."""
        if self.map_df is None:
            print("[WARNING] No map data available for map building analysis")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Map measurements over time
        ax1.scatter(self.map_df['timestamp'], self.map_df['light_intensity'], 
                   c=self.map_df['light_intensity'], cmap='hot', alpha=0.7)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Light Intensity')
        ax1.set_title('Map Measurements Over Time', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Spatial distribution of measurements
        scatter = ax2.scatter(self.map_df['x'], self.map_df['y'], 
                            c=self.map_df['light_intensity'], cmap='hot', s=100)
        plt.colorbar(scatter, ax=ax2, label='Light Intensity')
        ax2.scatter([self.target_x], [self.target_y], color='gold', s=300, 
                   marker='*', label='Target', edgecolors='black')
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Spatial Distribution of Map Measurements', fontweight='bold')
        ax2.legend()
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Light intensity histogram
        ax3.hist(self.map_df['light_intensity'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('Light Intensity')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Light Intensity Distribution in Map', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Grid cell coverage
        grid_coverage = self.map_df.groupby(['grid_x', 'grid_y']).size().reset_index(name='count')
        scatter4 = ax4.scatter(grid_coverage['grid_x'], grid_coverage['grid_y'], 
                             s=grid_coverage['count']*20, alpha=0.6, c='blue')
        ax4.set_xlabel('Grid X')
        ax4.set_ylabel('Grid Y')
        ax4.set_title(f'Grid Cell Coverage ({len(grid_coverage)} cells)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./plots/gradient_map_analysis.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved map building analysis plots")
        
    def create_summary_report(self):
        """Create a summary report with key metrics."""
        # Calculate key metrics
        total_time = self.trajectory_df['time'].iloc[-1]
        final_distance = self.trajectory_df['distance_to_target'].iloc[-1]
        min_distance = self.trajectory_df['distance_to_target'].min()
        
        # Calculate path length
        positions = self.trajectory_df[['x', 'y']].values
        path_diffs = np.diff(positions, axis=0)
        path_lengths = np.sqrt(np.sum(path_diffs**2, axis=1))
        total_path_length = np.sum(path_lengths)
        
        # Direct distance from start to target
        start_pos = positions[0]
        direct_distance = np.sqrt((start_pos[0] - self.target_x)**2 + (start_pos[1] - self.target_y)**2)
        path_efficiency = direct_distance / total_path_length if total_path_length > 0 else 0
        
        # Mode distribution
        mode_counts = self.trajectory_df['mode'].value_counts()
        mode_percentages = (mode_counts / len(self.trajectory_df) * 100).round(1)
        
        # Map statistics
        map_stats = ""
        if self.map_df is not None:
            map_stats = f"""
Map Building Statistics:
- Total measurements: {len(self.map_df)}
- Light intensity range: {self.map_df['light_intensity'].min():.2f} - {self.map_df['light_intensity'].max():.2f}
- Spatial coverage: {len(self.map_df.groupby(['grid_x', 'grid_y']))} grid cells
- Average light intensity: {self.map_df['light_intensity'].mean():.2f}"""
        
        # Gradient statistics
        gradient_stats = ""
        if self.gradient_df is not None:
            valid_gradients = self.gradient_df[self.gradient_df['valid'] == 1]
            gradient_stats = f"""
Gradient Field Statistics:
- Valid gradient points: {len(valid_gradients)}/{len(self.gradient_df)} ({len(valid_gradients)/len(self.gradient_df)*100:.1f}%)
- Average gradient magnitude: {valid_gradients['magnitude'].mean():.3f}
- Max gradient magnitude: {valid_gradients['magnitude'].max():.3f}"""
        
        report = f"""
=== GRADIENT MAP SEEKING PERFORMANCE REPORT ===

Mission Summary:
- Total mission time: {total_time:.1f} seconds
- Final distance to target: {final_distance:.3f} m
- Minimum distance achieved: {min_distance:.3f} m
- Mission success: {'[SUCCESS] YES' if final_distance <= 0.5 else '[FAILED] NO'} (threshold: 0.5m)

Path Analysis:
- Total path length: {total_path_length:.2f} m
- Direct distance to target: {direct_distance:.2f} m
- Path efficiency: {path_efficiency:.1%}

Control Mode Distribution:
{chr(10).join([f"- {mode}: {count} steps ({pct}%)" for mode, count, pct in zip(mode_counts.index, mode_counts.values, mode_percentages.values)])}
{map_stats}
{gradient_stats}

Performance Insights:
- Average speed: {self.trajectory_df['current_speed'].mean():.3f} m/s
- Speed tracking error (RMS): {np.sqrt(np.mean((self.trajectory_df['setpoint_speed'] - self.trajectory_df['current_speed'])**2)):.3f} m/s
- Light intensity improvement: {self.trajectory_df['light_intensity'].iloc[-1] - self.trajectory_df['light_intensity'].iloc[0]:.2f}
"""
        
        print(report)
        
        # Save report to file
        with open('gradient_mission_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("[SUCCESS] Saved mission report to 'gradient_mission_report.txt'")
        
    def run_full_analysis(self):
        """Run complete visualization and analysis pipeline."""
        print("Starting Gradient Map Visualization Analysis...")
        print("=" * 60)
        
        if not self.load_data():
            return False
            
        print("\n📊 Creating visualizations...")
        
        try:
            self.create_3d_trajectory_plot()
            self.create_2d_trajectory_with_map()
            self.create_performance_analysis()
            self.create_map_building_analysis()
            self.create_summary_report()
            
            print("\n[COMPLETE] Analysis complete! Generated files:")
            print("  * gradient_trajectory_3d.png - 3D trajectory with gradient vectors")
            print("  * gradient_trajectory_2d.png - 2D trajectory with map overlay")
            print("  * gradient_performance_analysis.png - Performance metrics")
            print("  * gradient_map_analysis.png - Map building analysis")
            print("  * gradient_mission_report.txt - Summary report")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function to run the visualization."""
    # Check for data directory argument
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print("*** Gradient Map Trajectory Visualizer ***")
    print("=" * 40)
    
    visualizer = GradientMapVisualizer(data_dir)
    success = visualizer.run_full_analysis()
    
    if success:
        print("\n[SUCCESS] All visualizations completed successfully!")
        print("Check the current directory for generated plots and report.")
    else:
        print("\n[FAILED] Analysis failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
