#!/usr/bin/env python3
"""
Precomputed Map Sensor Analysis Visualizer

Analyzes the individual sensor readings from the precomputed map CSV and creates
3D visualizations of sensor statistics across the grid:
- Minimum sensor value per location
- Maximum sensor value per location
- Mean sensor value per location
- Delta (max - min) per location

Missing grid cells are filled with zeros for visualization.

Usage:
    python sensor_analysis_visualizer.py [precomputed_map.csv]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

class SensorAnalysisVisualizer:
    def __init__(self, csv_file="precomputed_map.csv"):
        """Initialize with the precomputed map CSV file."""
        self.csv_file = csv_file
        self.df = None
        self.grid_data = None
        
    def load_data(self):
        """Load and parse the CSV file."""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"[SUCCESS] Loaded {len(self.df)} measurements from {self.csv_file}")
            
            # Check if sensor columns exist
            sensor_cols = [f'sensor_{i}' for i in range(16)]
            if not all(col in self.df.columns for col in sensor_cols):
                print("[ERROR] CSV does not contain individual sensor data (sensor_0...sensor_15)")
                print("Available columns:", list(self.df.columns))
                return False
            
            print(f"[SUCCESS] Found individual sensor data for all 16 sensors")
            return True
            
        except FileNotFoundError:
            print(f"[ERROR] File not found: {self.csv_file}")
            print("Make sure you've run the map_probe_supervisor first!")
            return False
        except Exception as e:
            print(f"[ERROR] Error loading data: {e}")
            return False
    
    def compute_sensor_statistics(self):
        """Compute min, max, mean, delta, and total for each grid location."""
        print("\n[INFO] Computing sensor statistics for each grid location...")
        
        # Extract sensor columns
        sensor_cols = [f'sensor_{i}' for i in range(16)]
        
        # Compute statistics per row
        self.df['sensor_min'] = self.df[sensor_cols].min(axis=1)
        self.df['sensor_max'] = self.df[sensor_cols].max(axis=1)
        self.df['sensor_mean'] = self.df[sensor_cols].mean(axis=1)
        self.df['sensor_delta'] = self.df['sensor_max'] - self.df['sensor_min']
        self.df['sensor_total'] = self.df[sensor_cols].sum(axis=1)  # Sum of all 16 sensors
        
        # Find brightest sensor index per location
        self.df['brightest_sensor'] = self.df[sensor_cols].idxmax(axis=1).str.replace('sensor_', '').astype(int)
        
        print(f"[SUCCESS] Statistics computed:")
        print(f"  Min range: {self.df['sensor_min'].min():.3f} - {self.df['sensor_min'].max():.3f}")
        print(f"  Max range: {self.df['sensor_max'].min():.3f} - {self.df['sensor_max'].max():.3f}")
        print(f"  Mean range: {self.df['sensor_mean'].min():.3f} - {self.df['sensor_mean'].max():.3f}")
        print(f"  Delta range: {self.df['sensor_delta'].min():.3f} - {self.df['sensor_delta'].max():.3f}")
        print(f"  Total range: {self.df['sensor_total'].min():.3f} - {self.df['sensor_total'].max():.3f}")
        
    def create_full_grid(self):
        """Create a complete grid filling missing cells with zeros."""
        print("\n[INFO] Creating complete grid with zero-fill for missing cells...")
        
        # Determine grid bounds
        grid_x_min = int(self.df['grid_x'].min())
        grid_x_max = int(self.df['grid_x'].max())
        grid_y_min = int(self.df['grid_y'].min())
        grid_y_max = int(self.df['grid_y'].max())
        
        # Estimate grid resolution from actual data
        x_values = self.df['x'].values
        y_values = self.df['y'].values
        x_diffs = np.diff(np.sort(np.unique(x_values)))
        y_diffs = np.diff(np.sort(np.unique(y_values)))
        
        # Use the most common difference as the resolution
        est_x_res = np.median(x_diffs[x_diffs > 0]) if len(x_diffs) > 0 else 0.3
        est_y_res = np.median(y_diffs[y_diffs > 0]) if len(y_diffs) > 0 else 0.3
        grid_resolution = (est_x_res + est_y_res) / 2.0
        
        print(f"  Grid bounds: X[{grid_x_min}, {grid_x_max}], Y[{grid_y_min}, {grid_y_max}]")
        print(f"  Estimated grid resolution: {grid_resolution:.3f}m")
        
        # Create full grid coordinates
        grid_x_range = range(grid_x_min, grid_x_max + 1)
        grid_y_range = range(grid_y_min, grid_y_max + 1)
        
        # Build a dictionary for fast lookup
        data_dict = {}
        for _, row in self.df.iterrows():
            key = (int(row['grid_x']), int(row['grid_y']))
            data_dict[key] = {
                'grid_x': int(row['grid_x']),
                'grid_y': int(row['grid_y']),
                'x': row['x'],
                'y': row['y'],
                'sensor_min': row['sensor_min'],
                'sensor_max': row['sensor_max'],
                'sensor_mean': row['sensor_mean'],
                'sensor_delta': row['sensor_delta'],
                'sensor_total': row['sensor_total'],
                'brightest_sensor': row['brightest_sensor']
            }
        
        # Create complete grid with zero-fill
        grid_data = []
        missing_count = 0
        
        for gx in grid_x_range:
            for gy in grid_y_range:
                key = (gx, gy)
                if key in data_dict:
                    grid_data.append(data_dict[key])
                else:
                    # Missing cell - fill with zeros
                    # Calculate x,y from grid coordinates using detected resolution
                    approx_x = gx * grid_resolution
                    approx_y = gy * grid_resolution
                    grid_data.append({
                        'grid_x': gx,
                        'grid_y': gy,
                        'x': approx_x,
                        'y': approx_y,
                        'sensor_min': 0.0,
                        'sensor_max': 0.0,
                        'sensor_mean': 0.0,
                        'sensor_delta': 0.0,
                        'sensor_total': 0.0,
                        'brightest_sensor': 0
                    })
                    missing_count += 1
        
        self.grid_data = pd.DataFrame(grid_data)
        
        print(f"[SUCCESS] Full grid created:")
        print(f"  Total cells: {len(self.grid_data)}")
        print(f"  Measured cells: {len(self.df)}")
        print(f"  Zero-filled cells: {missing_count}")
        
        # Report if there are complete rows/columns of zeros
        measured_gx = set(self.df['grid_x'].unique())
        measured_gy = set(self.df['grid_y'].unique())
        all_gx = set(grid_x_range)
        all_gy = set(grid_y_range)
        
        missing_gx = sorted(all_gx - measured_gx)
        missing_gy = sorted(all_gy - measured_gy)
        
        if missing_gx:
            print(f"  WARNING: Complete columns missing at grid_x: {missing_gx}")
            print(f"           These columns are entirely zero (likely inside obstacles)")
        if missing_gy:
            print(f"  WARNING: Complete rows missing at grid_y: {missing_gy}")
            print(f"           These rows are entirely zero (likely inside obstacles)")
        
    def create_3d_surface_plots(self):
        """Create 3D surface plots for min, max, mean, delta, and total."""
        print("\n[INFO] Creating 3D surface plots...")
        
        # Work with grid coordinates directly
        grid_x_vals = sorted(self.grid_data['grid_x'].unique()) if 'grid_x' in self.grid_data.columns else []
        grid_y_vals = sorted(self.grid_data['grid_y'].unique()) if 'grid_y' in self.grid_data.columns else []
        
        # If grid coordinates not available, fall back to approximation
        if not grid_x_vals or not grid_y_vals:
            print("[WARNING] Grid coordinates not in data, using x/y positions")
            x_unique = np.sort(self.grid_data['x'].unique())
            y_unique = np.sort(self.grid_data['y'].unique())
        else:
            # Convert grid indices to physical positions
            # Group by grid coordinates to get average physical position
            x_map = self.grid_data.groupby('grid_x')['x'].mean().to_dict()
            y_map = self.grid_data.groupby('grid_y')['y'].mean().to_dict()
            
            x_unique = np.array([x_map.get(gx, gx * 0.3) for gx in grid_x_vals])
            y_unique = np.array([y_map.get(gy, gy * 0.3) for gy in grid_y_vals])
        
        X, Y = np.meshgrid(x_unique, y_unique)
        
        # Create 2D arrays for each metric (now includes total)
        metrics = ['sensor_min', 'sensor_max', 'sensor_mean', 'sensor_delta', 'sensor_total']
        metric_grids = {}
        
        for metric in metrics:
            Z = np.zeros_like(X)
            
            # Fill Z values using grid coordinates
            for i, gy_idx in enumerate(grid_y_vals if grid_y_vals else range(len(y_unique))):
                for j, gx_idx in enumerate(grid_x_vals if grid_x_vals else range(len(x_unique))):
                    if grid_x_vals and grid_y_vals:
                        # Use grid coordinates
                        mask = (self.grid_data['grid_x'] == gx_idx) & (self.grid_data['grid_y'] == gy_idx)
                    else:
                        # Fallback to position-based
                        x_val = x_unique[j]
                        y_val = y_unique[i]
                        mask = (self.grid_data['x'] == x_val) & (self.grid_data['y'] == y_val)
                    
                    if mask.any():
                        Z[i, j] = self.grid_data.loc[mask, metric].values[0]
                    else:
                        Z[i, j] = 0.0
                        
            metric_grids[metric] = Z
        
        # Create figure with 5 subplots (2x3 layout, last position empty)
        fig = plt.figure(figsize=(20, 12))
        
        titles = {
            'sensor_min': 'Minimum Sensor Value per Location',
            'sensor_max': 'Maximum Sensor Value per Location',
            'sensor_mean': 'Mean Sensor Value per Location',
            'sensor_delta': 'Sensor Delta (Max - Min) per Location',
            'sensor_total': 'Total Light (Sum of All 16 Sensors) per Location'
        }
        
        colormaps = {
            'sensor_min': 'viridis',
            'sensor_max': 'plasma',
            'sensor_mean': 'inferno',
            'sensor_delta': 'coolwarm',
            'sensor_total': 'hot'
        }
        
        # Subplot positions in 2x3 grid
        positions = [1, 2, 3, 4, 6]  # Skip position 5 (bottom middle)
        
        for idx, metric in enumerate(metrics):
            ax = fig.add_subplot(2, 3, positions[idx], projection='3d')
            
            Z = metric_grids[metric]
            
            # Plot surface
            surf = ax.plot_surface(X, Y, Z, cmap=colormaps[metric], 
                                  alpha=0.8, edgecolor='none')
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, 
                        label=metric.replace('_', ' ').title())
            
            # Labels
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_zlabel(metric.replace('_', ' ').title())
            ax.set_title(titles[metric], fontweight='bold', pad=20)
            
            # Set view angle
            ax.view_init(elev=25, azim=45)
        
        plt.tight_layout()
        plt.savefig('plots/sensor_statistics_3d.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved 3D surface plots to 'sensor_statistics_3d.png'")
        
    def create_2d_heatmaps(self):
        """Create 2D heatmap views of the same metrics."""
        print("\n[INFO] Creating 2D heatmap plots...")
        
        # Work with grid coordinates directly
        grid_x_vals = sorted(self.grid_data['grid_x'].unique()) if 'grid_x' in self.grid_data.columns else []
        grid_y_vals = sorted(self.grid_data['grid_y'].unique()) if 'grid_y' in self.grid_data.columns else []
        
        # If grid coordinates not available, fall back to approximation
        if not grid_x_vals or not grid_y_vals:
            x_unique = np.sort(self.grid_data['x'].unique())
            y_unique = np.sort(self.grid_data['y'].unique())
        else:
            # Convert grid indices to physical positions
            x_map = self.grid_data.groupby('grid_x')['x'].mean().to_dict()
            y_map = self.grid_data.groupby('grid_y')['y'].mean().to_dict()
            
            x_unique = np.array([x_map.get(gx, gx * 0.3) for gx in grid_x_vals])
            y_unique = np.array([y_map.get(gy, gy * 0.3) for gy in grid_y_vals])
        
        X, Y = np.meshgrid(x_unique, y_unique)
        
        metrics = ['sensor_min', 'sensor_max', 'sensor_mean', 'sensor_delta', 'sensor_total']
        metric_grids = {}
        
        for metric in metrics:
            Z = np.zeros_like(X)
            
            # Fill Z values using grid coordinates
            for i, gy_idx in enumerate(grid_y_vals if grid_y_vals else range(len(y_unique))):
                for j, gx_idx in enumerate(grid_x_vals if grid_x_vals else range(len(x_unique))):
                    if grid_x_vals and grid_y_vals:
                        # Use grid coordinates
                        mask = (self.grid_data['grid_x'] == gx_idx) & (self.grid_data['grid_y'] == gy_idx)
                    else:
                        # Fallback to position-based
                        x_val = x_unique[j]
                        y_val = y_unique[i]
                        mask = (self.grid_data['x'] == x_val) & (self.grid_data['y'] == y_val)
                    
                    if mask.any():
                        Z[i, j] = self.grid_data.loc[mask, metric].values[0]
                        
            metric_grids[metric] = Z
        
        # Create figure with 5 subplots (2x3, last position empty)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        titles = {
            'sensor_min': 'Minimum Sensor Value',
            'sensor_max': 'Maximum Sensor Value',
            'sensor_mean': 'Mean Sensor Value',
            'sensor_delta': 'Sensor Delta (Max - Min)',
            'sensor_total': 'Total Light (Sum of 16 Sensors)'
        }
        
        colormaps = {
            'sensor_min': 'viridis',
            'sensor_max': 'plasma',
            'sensor_mean': 'inferno',
            'sensor_delta': 'coolwarm',
            'sensor_total': 'hot'
        }
        
        positions = [0, 1, 2, 3, 5]  # Skip position 4 (bottom middle)
        
        for idx, metric in enumerate(metrics):
            ax = axes[positions[idx]]
            Z = metric_grids[metric]
            
            im = ax.imshow(Z, origin='lower', cmap=colormaps[metric], 
                          extent=[x_unique.min(), x_unique.max(), 
                                 y_unique.min(), y_unique.max()],
                          aspect='auto', interpolation='bilinear')
            
            plt.colorbar(im, ax=ax, label=metric.replace('_', ' ').title())
            
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_title(titles[metric], fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide the unused subplot (position 4)
        axes[4].axis('off')
        
        plt.tight_layout()
        plt.savefig('plots/sensor_statistics_2d.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved 2D heatmaps to 'sensor_statistics_2d.png'")
        
    def create_brightest_sensor_map(self):
        """Create a map showing which sensor was brightest at each location."""
        print("\n[INFO] Creating brightest sensor direction map...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Scatter plot colored by brightest sensor
        scatter = ax.scatter(self.df['x'], self.df['y'], 
                           c=self.df['brightest_sensor'], 
                           cmap='hsv', s=50, alpha=0.7, vmin=0, vmax=15)
        
        cbar = plt.colorbar(scatter, ax=ax, label='Brightest Sensor Index (0-15)')
        cbar.set_ticks(range(0, 16, 2))
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Brightest Sensor Index per Location\n(0°=sensor_0, 180°=sensor_8)', 
                    fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add light source marker if known
        target_x, target_y = 5.1988, 5.329
        ax.scatter([target_x], [target_y], color='gold', s=300, marker='*', 
                  edgecolors='black', linewidth=2, label='Light Source', zorder=5)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('plots/brightest_sensor_map.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved brightest sensor map to 'brightest_sensor_map.png'")
        
    def create_summary_report(self):
        """Create a text summary report."""
        print("\n[INFO] Creating summary report...")
        
        report = f"""
=== PRECOMPUTED MAP SENSOR ANALYSIS REPORT ===

Data File: {self.csv_file}
Total Measurements: {len(self.df)}
Grid Coverage: {len(self.grid_data)} cells (including zero-filled)

SENSOR STATISTICS SUMMARY:
--------------------------
Minimum Sensor Values:
  Range: {self.df['sensor_min'].min():.3f} - {self.df['sensor_min'].max():.3f}
  Mean: {self.df['sensor_min'].mean():.3f}
  Std: {self.df['sensor_min'].std():.3f}

Maximum Sensor Values:
  Range: {self.df['sensor_max'].min():.3f} - {self.df['sensor_max'].max():.3f}
  Mean: {self.df['sensor_max'].mean():.3f}
  Std: {self.df['sensor_max'].std():.3f}

Mean Sensor Values:
  Range: {self.df['sensor_mean'].min():.3f} - {self.df['sensor_mean'].max():.3f}
  Mean: {self.df['sensor_mean'].mean():.3f}
  Std: {self.df['sensor_mean'].std():.3f}

Sensor Delta (Max - Min):
  Range: {self.df['sensor_delta'].min():.3f} - {self.df['sensor_delta'].max():.3f}
  Mean: {self.df['sensor_delta'].mean():.3f}
  Std: {self.df['sensor_delta'].std():.3f}

Total Light (Sum of All 16 Sensors):
  Range: {self.df['sensor_total'].min():.3f} - {self.df['sensor_total'].max():.3f}
  Mean: {self.df['sensor_total'].mean():.3f}
  Std: {self.df['sensor_total'].std():.3f}

BRIGHTEST SENSOR DISTRIBUTION:
------------------------------
"""
        
        # Histogram of brightest sensors
        sensor_counts = self.df['brightest_sensor'].value_counts().sort_index()
        for sensor_idx in range(16):
            count = sensor_counts.get(sensor_idx, 0)
            pct = (count / len(self.df) * 100) if len(self.df) > 0 else 0
            angle = sensor_idx * 22.5
            report += f"Sensor {sensor_idx:2d} ({angle:5.1f}°): {count:4d} locations ({pct:5.1f}%)\n"
        
        report += f"""
FILES GENERATED:
----------------
- sensor_statistics_3d.png : 3D surface plots (min, max, mean, delta)
- sensor_statistics_2d.png : 2D heatmap views
- brightest_sensor_map.png : Map showing brightest sensor direction
- sensor_analysis_report.txt : This report

"""
        
        print(report)
        
        with open('plots/sensor_analysis_report.txt', 'w') as f:
            f.write(report)
        
        print("[SUCCESS] Saved report to 'sensor_analysis_report.txt'")
        
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("=" * 60)
        print("Starting Sensor Analysis Visualization...")
        print("=" * 60)
        
        if not self.load_data():
            return False
        
        self.compute_sensor_statistics()
        self.create_full_grid()
        
        try:
            self.create_3d_surface_plots()
            self.create_2d_heatmaps()
            self.create_brightest_sensor_map()
            self.create_summary_report()
            
            print("\n" + "=" * 60)
            print("✓ Analysis Complete!")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Error during visualization: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function."""
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "precomputed_map.csv"
    
    print("*** Precomputed Map Sensor Analysis ***")
    print("=" * 40)
    
    visualizer = SensorAnalysisVisualizer(csv_file)
    success = visualizer.run_full_analysis()
    
    if success:
        print("\n✓ All visualizations created successfully!")
    else:
        print("\n✗ Analysis failed. Check errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())