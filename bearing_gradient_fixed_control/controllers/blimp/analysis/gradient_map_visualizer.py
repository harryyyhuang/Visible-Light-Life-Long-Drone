#!/usr/bin/env python3
"""
Gradient Map Visualizer for Blimp Trajectory Analysis (Universal Version)

Supports both pure gradient controllers and combined bearing+gradient fusion controllers.
Creates comprehensive visualizations:
1. 3D trajectory with gradient vectors
2. 2D trajectory with measurement map overlay
3. Gradient field visualization with light intensity heatmap
4. Performance metrics
5. Summary report

Usage:
    python gradient_map_visualizer.py [data_directory]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
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
            # Try to find trajectory file (any variant)
            trajectory_patterns = [
                "*trajectory*.csv",
                "*fusion*.csv"
            ]
            
            trajectory_file = None
            for pattern in trajectory_patterns:
                files = list(self.data_dir.glob(pattern))
                if files:
                    trajectory_file = files[0]
                    break
            
            if not trajectory_file:
                # Try parent directories
                for pattern in trajectory_patterns:
                    files = list(self.data_dir.parent.glob(pattern))
                    files += list(self.data_dir.parent.glob(f"logs/{pattern}"))
                    if files:
                        trajectory_file = files[0]
                        break
            
            if trajectory_file and trajectory_file.exists():
                self.trajectory_df = pd.read_csv(trajectory_file)
                print(f"[SUCCESS] Loaded trajectory data from {trajectory_file}")
            else:
                print("[ERROR] No trajectory data file found!")
                return False
                
            # Load measurement map
            map_patterns = ["*measurement_map*.csv", "*map*.csv"]
            map_file = None
            for pattern in map_patterns:
                files = list(self.data_dir.glob(pattern))
                files += list(self.data_dir.parent.glob(pattern))
                files += list(self.data_dir.parent.glob(f"logs/{pattern}"))
                if files:
                    map_file = files[0]
                    break
            
            if map_file and map_file.exists():
                self.map_df = pd.read_csv(map_file)
                print(f"[SUCCESS] Loaded map data from {map_file} ({len(self.map_df)} measurements)")
            else:
                print("[WARNING] Map data file not found - skipping map visualization")
                
            # Load gradient field
            grad_patterns = ["*gradient_field*.csv", "*field*.csv"]
            grad_file = None
            for pattern in grad_patterns:
                files = list(self.data_dir.glob(pattern))
                files += list(self.data_dir.parent.glob(pattern))
                files += list(self.data_dir.parent.glob(f"logs/{pattern}"))
                if files:
                    grad_file = files[0]
                    break
            
            if grad_file and grad_file.exists():
                self.gradient_df = pd.read_csv(grad_file)
                valid_gradients = self.gradient_df[self.gradient_df['valid'] == 1]
                print(f"[SUCCESS] Loaded gradient field from {grad_file} ({len(valid_gradients)}/{len(self.gradient_df)} valid)")
            else:
                print("[WARNING] Gradient field file not found - skipping gradient visualization")
                
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def create_3d_trajectory_plot(self):
        """Create 3D trajectory plot with gradient vectors."""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Universal mode colors (handles any mode name)
        mode_colors = {
            'FORWARD': 'gray', 'EXPLORING': 'blue', 'SEEKING': 'red',
            'IDLE': 'gray', 'BEARING_SEEK': 'green', 'BEARING_ONLY': 'green',
            'FUSED': 'orange', 'GRADIENT_EXPLORE': 'blue', 'GRADIENT_SEEK': 'red',
            'NO_SIGNAL': 'gray'
        }
        
        # Plot trajectory (colored by mode if available)
        if 'mode' in self.trajectory_df.columns:
            for mode in self.trajectory_df['mode'].unique():
                mode_data = self.trajectory_df[self.trajectory_df['mode'] == mode]
                ax.plot(mode_data['x'], mode_data['y'], mode_data['z'], 
                       color=mode_colors.get(mode, 'black'), label=f'{mode}',
                       linewidth=2, alpha=0.8)
        else:
            ax.plot(self.trajectory_df['x'], self.trajectory_df['y'], self.trajectory_df['z'], 
                   color='blue', label='Trajectory', linewidth=2, alpha=0.8)
        
        # Plot measurement points
        if self.map_df is not None:
            scatter = ax.scatter(self.map_df['x'], self.map_df['y'], 2.0,
                               c=self.map_df['light_intensity'], cmap='hot', 
                               s=50, alpha=0.7, label='Map Measurements')
            plt.colorbar(scatter, ax=ax, label='Light Intensity', shrink=0.6)
        
        # Plot target
        ax.scatter([self.target_x], [self.target_y], [2.0], 
                  color='gold', s=200, marker='*', 
                  label='Target', edgecolors='black', linewidth=2)
        
        # Plot start/end
        start = self.trajectory_df.iloc[0]
        end = self.trajectory_df.iloc[-1]
        ax.scatter([start['x']], [start['y']], [start['z']], 
                  color='green', s=100, marker='o', label='Start')
        ax.scatter([end['x']], [end['y']], [end['z']], 
                  color='red', s=100, marker='s', label='End')
        
        # Add gradient vectors
        if self.gradient_df is not None:
            valid_grads = self.gradient_df[self.gradient_df['valid'] == 1][::5]
            for _, row in valid_grads.iterrows():
                scale = 0.5
                ax.quiver(row['x'], row['y'], 2.0,
                         row['grad_x'] * scale, row['grad_y'] * scale, 0,
                         color='purple', alpha=0.6, arrow_length_ratio=0.1)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('3D Trajectory with Gradient Field', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        plt.savefig('./plots/gradient_trajectory_3d.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved 3D trajectory plot")
        
    def create_2d_trajectory_with_map(self):
        """Create 2D top-down view with measurement map overlay."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Trajectory with map
        if self.map_df is not None:
            scatter1 = ax1.scatter(self.map_df['x'], self.map_df['y'], 
                                 c=self.map_df['light_intensity'], cmap='hot',
                                 s=100, alpha=0.7, label='Map Measurements')
            plt.colorbar(scatter1, ax=ax1, label='Light Intensity')
        
        # Plot trajectory
        mode_colors = {
            'FORWARD': 'gray', 'EXPLORING': 'blue', 'SEEKING': 'red',
            'IDLE': 'gray', 'BEARING_SEEK': 'green', 'BEARING_ONLY': 'green',
            'FUSED': 'orange', 'GRADIENT_EXPLORE': 'blue', 'GRADIENT_SEEK': 'red',
            'NO_SIGNAL': 'gray'
        }
        
        if 'mode' in self.trajectory_df.columns:
            for mode in self.trajectory_df['mode'].unique():
                mode_data = self.trajectory_df[self.trajectory_df['mode'] == mode]
                ax1.plot(mode_data['x'], mode_data['y'], 
                        color=mode_colors.get(mode, 'black'), label=f'{mode}',
                        linewidth=3, alpha=0.8)
        else:
            ax1.plot(self.trajectory_df['x'], self.trajectory_df['y'], 
                    color='blue', label='Trajectory', linewidth=3, alpha=0.8)
        
        # Add target and markers
        ax1.scatter([self.target_x], [self.target_y], color='gold', s=300, 
                   marker='*', label='Target', edgecolors='black', zorder=5)
        start = self.trajectory_df.iloc[0]
        end = self.trajectory_df.iloc[-1]
        ax1.scatter([start['x']], [start['y']], color='green', s=150, 
                   marker='o', label='Start', zorder=5)
        ax1.scatter([end['x']], [end['y']], color='red', s=150, 
                   marker='s', label='End', zorder=5)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('2D Trajectory with Map Overlay', fontweight='bold')
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Gradient field
        if self.gradient_df is not None:
            valid_grads = self.gradient_df[self.gradient_df['valid'] == 1]
            ax2.quiver(valid_grads['x'], valid_grads['y'], 
                      valid_grads['grad_x'], valid_grads['grad_y'],
                      valid_grads['magnitude'], cmap='viridis', scale=50)
            ax2.scatter([self.target_x], [self.target_y], color='gold', s=300, 
                       marker='*', label='Target', edgecolors='black')
            ax2.set_xlabel('X Position (m)')
            ax2.set_ylabel('Y Position (m)')
            ax2.set_title('Gradient Vector Field', fontweight='bold')
            ax2.legend()
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Gradient field not available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig('./plots/gradient_trajectory_2d.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved 2D trajectory plots")
        
    def create_performance_analysis(self):
        """Create performance analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Distance to target
        ax1.plot(self.trajectory_df['time'], self.trajectory_df['dist_to_target'], 
                'b-', linewidth=2)
        ax1.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Distance to Target (m)')
        ax1.set_title('Distance to Target Over Time', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Speed (use whatever column is available)
        speed_col = None
        if 'current_speed' in self.trajectory_df.columns:
            speed_col = 'current_speed'
        elif 'forward_speed' in self.trajectory_df.columns:
            speed_col = 'forward_speed'
        
        if speed_col:
            ax2.plot(self.trajectory_df['time'], self.trajectory_df[speed_col], 
                    'g-', linewidth=2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Speed (m/s)')
            ax2.set_title('Speed Over Time', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Speed data not available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # Plot 3: Light intensity
        light_col = None
        if 'light_intensity' in self.trajectory_df.columns:
            light_col = 'light_intensity'
        elif 'total_light' in self.trajectory_df.columns:
            light_col = 'total_light'
        
        if light_col:
            ax3.plot(self.trajectory_df['time'], self.trajectory_df[light_col], 
                    'orange', linewidth=2)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Light Intensity')
            ax3.set_title('Light Intensity Over Time', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Light intensity data not available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Gradient magnitude
        grad_col = None
        if 'gradient_magnitude' in self.trajectory_df.columns:
            grad_col = 'gradient_magnitude'
        elif 'grad_mag' in self.trajectory_df.columns:
            grad_col = 'grad_mag'
        
        if grad_col:
            ax4.plot(self.trajectory_df['time'], self.trajectory_df[grad_col], 
                    'purple', linewidth=2)
            ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Gradient Magnitude')
            ax4.set_title('Gradient Magnitude Over Time', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Gradient magnitude data not available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.savefig('./plots/gradient_performance_analysis.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved performance analysis plots")
        
    def create_summary_report(self):
        """Create a summary report with key metrics."""
        total_time = self.trajectory_df['time'].iloc[-1]
        final_distance = self.trajectory_df['dist_to_target'].iloc[-1]
        min_distance = self.trajectory_df['dist_to_target'].min()
        
        # Path length
        positions = self.trajectory_df[['x', 'y']].values
        path_diffs = np.diff(positions, axis=0)
        path_lengths = np.sqrt(np.sum(path_diffs**2, axis=1))
        total_path_length = np.sum(path_lengths)
        
        start_pos = positions[0]
        direct_distance = np.sqrt((start_pos[0] - self.target_x)**2 + 
                                 (start_pos[1] - self.target_y)**2)
        path_efficiency = direct_distance / total_path_length if total_path_length > 0 else 0
        
        # Mode distribution (if available)
        mode_stats = ""
        if 'mode' in self.trajectory_df.columns:
            mode_counts = self.trajectory_df['mode'].value_counts()
            mode_percentages = (mode_counts / len(self.trajectory_df) * 100).round(1)
            mode_stats = "\nControl Mode Distribution:\n"
            mode_stats += "\n".join([f"- {mode}: {count} steps ({pct}%)" 
                                    for mode, count, pct in zip(mode_counts.index, 
                                                                mode_counts.values, 
                                                                mode_percentages.values)])
        
        # Map stats
        map_stats = ""
        if self.map_df is not None:
            map_stats = f"""
Map Statistics:
- Total measurements: {len(self.map_df)}
- Light intensity range: {self.map_df['light_intensity'].min():.2f} - {self.map_df['light_intensity'].max():.2f}
- Grid cells covered: {len(self.map_df.groupby(['grid_x', 'grid_y']))}"""
        
        # Gradient stats
        grad_stats = ""
        if self.gradient_df is not None:
            valid = self.gradient_df[self.gradient_df['valid'] == 1]
            grad_stats = f"""
Gradient Field Statistics:
- Valid points: {len(valid)}/{len(self.gradient_df)} ({len(valid)/len(self.gradient_df)*100:.1f}%)
- Average magnitude: {valid['magnitude'].mean():.3f}
- Max magnitude: {valid['magnitude'].max():.3f}"""
        
        # Speed column
        speed_col = 'current_speed' if 'current_speed' in self.trajectory_df.columns else 'forward_speed'
        avg_speed = self.trajectory_df[speed_col].mean() if speed_col in self.trajectory_df.columns else 0.0
        
        # Light column
        light_col = 'light_intensity' if 'light_intensity' in self.trajectory_df.columns else 'total_light'
        light_improvement = 0.0
        if light_col in self.trajectory_df.columns:
            light_improvement = self.trajectory_df[light_col].iloc[-1] - self.trajectory_df[light_col].iloc[0]
        
        report = f"""
=== BLIMP LIGHT SEEKING PERFORMANCE REPORT ===

Mission Summary:
- Total time: {total_time:.1f} seconds
- Final distance to target: {final_distance:.3f} m
- Minimum distance achieved: {min_distance:.3f} m
- Mission success: {'YES ✓' if final_distance <= 2.0 else 'NO ✗'} (threshold: 2.0m)

Path Analysis:
- Total path length: {total_path_length:.2f} m
- Direct distance: {direct_distance:.2f} m
- Path efficiency: {path_efficiency:.1%}
{mode_stats}
{map_stats}
{grad_stats}

Performance Insights:
- Average speed: {avg_speed:.3f} m/s
- Light intensity improvement: {light_improvement:.2f}
"""
        
        print(report)
        
        with open('plots/mission_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("[SUCCESS] Saved report to 'mission_report.txt'")
        
    def run_full_analysis(self):
        """Run complete visualization pipeline."""
        print("Starting Blimp Trajectory Analysis...")
        print("=" * 60)
        
        if not self.load_data():
            return False
            
        print("\n📊 Creating visualizations...")
        
        try:
            self.create_3d_trajectory_plot()
            self.create_2d_trajectory_with_map()
            self.create_performance_analysis()
            self.create_summary_report()
            
            print("\n✓ Analysis complete! Generated files:")
            print("  * plots/gradient_trajectory_3d.png")
            print("  * plots/gradient_trajectory_2d.png")
            print("  * plots/gradient_performance_analysis.png")
            print("  * mission_report.txt")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function."""
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print("*** Blimp Trajectory Visualizer ***")
    print("=" * 40)
    
    visualizer = GradientMapVisualizer(data_dir)
    success = visualizer.run_full_analysis()
    
    if success:
        print("\n[SUCCESS] All visualizations completed!")
    else:
        print("\n[FAILED] Analysis failed. Check errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())