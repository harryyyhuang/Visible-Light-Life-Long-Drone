#!/usr/bin/env python3
"""
Gradient Estimation Quality Analyzer

Compares real-time gradient estimates from a blimp flight trajectory
against the precomputed "ground truth" gradient map to evaluate
estimation quality, identify problem areas, and quantify navigation accuracy.

Usage:
    python gradient_comparison_analyzer.py [trajectory.csv] [precomputed_map.csv]
    
Outputs:
    - gradient_comparison_plots.png : Visual comparison plots
    - gradient_error_analysis.png : Error distribution and spatial patterns
    - gradient_quality_report.txt : Detailed analysis report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import sys
from pathlib import Path

class GradientComparisonAnalyzer:
    def __init__(self, trajectory_file, precomputed_file):
        """Initialize with trajectory log and precomputed map."""
        self.trajectory_file = trajectory_file
        self.precomputed_file = precomputed_file
        self.traj_df = None
        self.map_df = None
        self.comparison_data = None
        
    def load_data(self):
        """Load both trajectory and precomputed map data."""
        print("=" * 60)
        print("Loading Data...")
        print("=" * 60)
        
        # Load trajectory
        try:
            self.traj_df = pd.read_csv(self.trajectory_file)
            print(f"[SUCCESS] Loaded trajectory: {len(self.traj_df)} timesteps")
            print(f"  Columns: {list(self.traj_df.columns)}")
            
            # Convert polar gradient representation to cartesian if needed
            if 'grad_x' not in self.traj_df.columns and 'grad_angle' in self.traj_df.columns:
                print("[INFO] Converting polar gradient (angle, mag) to cartesian (x, y)...")
                self.traj_df['grad_x'] = self.traj_df['grad_mag'] * np.cos(np.radians(self.traj_df['grad_angle']))
                self.traj_df['grad_y'] = self.traj_df['grad_mag'] * np.sin(np.radians(self.traj_df['grad_angle']))
                print("  Created grad_x and grad_y columns from grad_angle and grad_mag")
                
        except Exception as e:
            print(f"[ERROR] Failed to load trajectory: {e}")
            return False
        
        # Load precomputed map
        try:
            self.map_df = pd.read_csv(self.precomputed_file)
            print(f"[SUCCESS] Loaded precomputed map: {len(self.map_df)} measurements")
            print(f"  Columns: {list(self.map_df.columns)}")
        except Exception as e:
            print(f"[ERROR] Failed to load precomputed map: {e}")
            return False
        
        return True
    
    def compute_map_gradients(self):
        """Compute ground truth gradients from the precomputed map."""
        print("\n[INFO] Computing ground truth gradients from map...")
        
        # Check if map already has gradient info, otherwise compute it
        if 'gradient_x' in self.map_df.columns and 'gradient_y' in self.map_df.columns:
            print("  Map already contains gradient estimates")
            self.map_df['map_grad_x'] = self.map_df['gradient_x']
            self.map_df['map_grad_y'] = self.map_df['gradient_y']
        else:
            # Compute gradients using finite differences
            print("  Computing gradients from total_light field...")
            
            # Build a spatial index
            points = self.map_df[['x', 'y']].values
            values = self.map_df['total_light'].values
            
            gradients_x = []
            gradients_y = []
            
            for idx, row in self.map_df.iterrows():
                x, y = row['x'], row['y']
                
                # Find nearby points
                nearby_mask = (
                    (np.abs(self.map_df['x'] - x) < 1.0) & 
                    (np.abs(self.map_df['y'] - y) < 1.0)
                )
                nearby_points = self.map_df[nearby_mask]
                
                if len(nearby_points) >= 3:
                    # Fit plane using least squares
                    X = nearby_points[['x', 'y']].values
                    Z = nearby_points['total_light'].values
                    
                    # Add constant term for plane fitting
                    A = np.c_[X, np.ones(len(X))]
                    
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
                        grad_x, grad_y = coeffs[0], coeffs[1]
                    except:
                        grad_x, grad_y = 0.0, 0.0
                else:
                    grad_x, grad_y = 0.0, 0.0
                
                gradients_x.append(grad_x)
                gradients_y.append(grad_y)
            
            self.map_df['map_grad_x'] = gradients_x
            self.map_df['map_grad_y'] = gradients_y
        
        self.map_df['map_grad_mag'] = np.sqrt(
            self.map_df['map_grad_x']**2 + self.map_df['map_grad_y']**2
        )
        self.map_df['map_grad_angle'] = np.degrees(np.arctan2(
            self.map_df['map_grad_y'], self.map_df['map_grad_x']
        ))
        
        print(f"[SUCCESS] Ground truth gradients computed")
        print(f"  Magnitude range: {self.map_df['map_grad_mag'].min():.3f} - {self.map_df['map_grad_mag'].max():.3f}")
    
    def match_trajectory_to_map(self):
        """For each trajectory point, find nearest map gradient."""
        print("\n[INFO] Matching trajectory points to map gradients...")
        
        # Build KD-tree for fast nearest neighbor search
        map_points = self.map_df[['x', 'y']].values
        tree = cKDTree(map_points)
        
        # For each trajectory point, find nearest map point
        traj_points = self.traj_df[['x', 'y']].values
        distances, indices = tree.query(traj_points, k=1)
        
        # Extract ground truth gradients at trajectory locations
        self.traj_df['map_grad_x'] = self.map_df.iloc[indices]['map_grad_x'].values
        self.traj_df['map_grad_y'] = self.map_df.iloc[indices]['map_grad_y'].values
        self.traj_df['map_grad_mag'] = self.map_df.iloc[indices]['map_grad_mag'].values
        self.traj_df['map_grad_angle'] = self.map_df.iloc[indices]['map_grad_angle'].values
        self.traj_df['map_distance'] = distances
        
        # Compute errors
        self.traj_df['grad_x_error'] = self.traj_df['grad_x'] - self.traj_df['map_grad_x']
        self.traj_df['grad_y_error'] = self.traj_df['grad_y'] - self.traj_df['map_grad_y']
        self.traj_df['grad_mag_error'] = self.traj_df['grad_mag'] - self.traj_df['map_grad_mag']
        
        # Angle error (handle wraparound)
        angle_diff = self.traj_df['grad_angle'] - self.traj_df['map_grad_angle']
        angle_diff = (angle_diff + 180) % 360 - 180  # Wrap to [-180, 180]
        self.traj_df['grad_angle_error'] = angle_diff
        
        # Vector magnitude error
        self.traj_df['grad_vector_error'] = np.sqrt(
            self.traj_df['grad_x_error']**2 + self.traj_df['grad_y_error']**2
        )
        
        print(f"[SUCCESS] Matched {len(self.traj_df)} trajectory points to map")
        print(f"  Average map distance: {distances.mean():.3f}m")
        print(f"  Max map distance: {distances.max():.3f}m")
    
    def create_comparison_plots(self):
        """Create comprehensive comparison visualizations."""
        print("\n[INFO] Creating comparison plots...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Trajectory with gradient vectors
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Subsample for clarity
        step = max(1, len(self.traj_df) // 50)
        traj_sub = self.traj_df.iloc[::step]
        
        # Plot trajectory
        ax1.plot(self.traj_df['x'], self.traj_df['y'], 'b-', alpha=0.3, linewidth=1, label='Trajectory')
        
        # Plot estimated gradients (red)
        scale = 0.5
        ax1.quiver(traj_sub['x'], traj_sub['y'], 
                  traj_sub['grad_x'], traj_sub['grad_y'],
                  color='red', alpha=0.6, scale=50, width=0.003, label='Estimated')
        
        # Plot ground truth gradients (green)
        ax1.quiver(traj_sub['x'], traj_sub['y'], 
                  traj_sub['map_grad_x'], traj_sub['map_grad_y'],
                  color='green', alpha=0.6, scale=50, width=0.003, label='Ground Truth')
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Gradient Vectors: Estimated vs Ground Truth', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. Gradient magnitude comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.traj_df['time'], self.traj_df['grad_mag'], 'r-', label='Estimated', linewidth=2)
        ax2.plot(self.traj_df['time'], self.traj_df['map_grad_mag'], 'g--', label='Ground Truth', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Gradient Magnitude')
        ax2.set_title('Gradient Magnitude Over Time', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Gradient angle comparison
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.traj_df['time'], self.traj_df['grad_angle'], 'r-', label='Estimated', linewidth=2)
        ax3.plot(self.traj_df['time'], self.traj_df['map_grad_angle'], 'g--', label='Ground Truth', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Gradient Angle (deg)')
        ax3.set_title('Gradient Direction Over Time', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Magnitude error over time
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(self.traj_df['time'], self.traj_df['grad_mag_error'], 'purple', linewidth=2)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.fill_between(self.traj_df['time'], 0, self.traj_df['grad_mag_error'], 
                         alpha=0.3, color='purple')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Magnitude Error')
        ax4.set_title('Gradient Magnitude Error Over Time', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Angle error over time
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(self.traj_df['time'], self.traj_df['grad_angle_error'], 'orange', linewidth=2)
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax5.fill_between(self.traj_df['time'], 0, self.traj_df['grad_angle_error'], 
                         alpha=0.3, color='orange')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Angle Error (deg)')
        ax5.set_title('Gradient Direction Error Over Time', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Vector error magnitude over time
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(self.traj_df['time'], self.traj_df['grad_vector_error'], 'brown', linewidth=2)
        ax6.fill_between(self.traj_df['time'], 0, self.traj_df['grad_vector_error'], 
                         alpha=0.3, color='brown')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Vector Error Magnitude')
        ax6.set_title('Gradient Vector Error Over Time', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. Spatial error map
        ax7 = fig.add_subplot(gs[2, 0])
        scatter = ax7.scatter(self.traj_df['x'], self.traj_df['y'], 
                            c=self.traj_df['grad_vector_error'], 
                            cmap='hot', s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax7, label='Vector Error')
        ax7.set_xlabel('X Position (m)')
        ax7.set_ylabel('Y Position (m)')
        ax7.set_title('Spatial Distribution of Gradient Error', fontweight='bold')
        ax7.set_aspect('equal')
        ax7.grid(True, alpha=0.3)
        
        # 8. Error histogram
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.hist(self.traj_df['grad_angle_error'], bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax8.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect')
        ax8.set_xlabel('Angle Error (deg)')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Distribution of Direction Errors', fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Error vs gradient strength
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.scatter(self.traj_df['map_grad_mag'], self.traj_df['grad_angle_error'], 
                   alpha=0.5, s=20, color='blue')
        ax9.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax9.set_xlabel('Ground Truth Gradient Magnitude')
        ax9.set_ylabel('Angle Error (deg)')
        ax9.set_title('Error vs Gradient Strength', fontweight='bold')
        ax9.grid(True, alpha=0.3)
        
        plt.savefig('plots/gradient_comparison_plots.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved comparison plots to 'gradient_comparison_plots.png'")
    
    def create_error_analysis_plot(self):
        """Create detailed error analysis visualizations."""
        print("\n[INFO] Creating error analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. CDF of angle errors
        ax1 = axes[0, 0]
        sorted_errors = np.sort(np.abs(self.traj_df['grad_angle_error']))
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax1.plot(sorted_errors, cdf * 100, linewidth=2, color='blue')
        ax1.axvline(x=10, color='orange', linestyle='--', label='10° threshold')
        ax1.axvline(x=30, color='red', linestyle='--', label='30° threshold')
        ax1.set_xlabel('Absolute Angle Error (deg)')
        ax1.set_ylabel('Cumulative Percentage (%)')
        ax1.set_title('CDF of Gradient Direction Errors', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Error correlation matrix
        ax2 = axes[0, 1]
        error_metrics = self.traj_df[['grad_mag_error', 'grad_angle_error', 
                                      'grad_vector_error', 'map_grad_mag']].corr()
        im = ax2.imshow(error_metrics, cmap='coolwarm', vmin=-1, vmax=1)
        ax2.set_xticks(range(len(error_metrics.columns)))
        ax2.set_yticks(range(len(error_metrics.columns)))
        ax2.set_xticklabels(['Mag Err', 'Ang Err', 'Vec Err', 'True Mag'], rotation=45, ha='right')
        ax2.set_yticklabels(['Mag Err', 'Ang Err', 'Vec Err', 'True Mag'])
        ax2.set_title('Error Correlation Matrix', fontweight='bold')
        plt.colorbar(im, ax=ax2)
        
        # Add correlation values
        for i in range(len(error_metrics)):
            for j in range(len(error_metrics)):
                text = ax2.text(j, i, f'{error_metrics.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=10)
        
        # 3. Error by mode (if mode column exists)
        ax3 = axes[1, 0]
        if 'mode' in self.traj_df.columns:
            modes = self.traj_df['mode'].unique()
            mode_errors = [self.traj_df[self.traj_df['mode'] == m]['grad_angle_error'].abs().mean() 
                          for m in modes]
            ax3.bar(range(len(modes)), mode_errors, color='steelblue', edgecolor='black')
            ax3.set_xticks(range(len(modes)))
            ax3.set_xticklabels(modes, rotation=45, ha='right')
            ax3.set_ylabel('Mean Absolute Angle Error (deg)')
            ax3.set_title('Error by Navigation Mode', fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
        else:
            ax3.text(0.5, 0.5, 'Mode data not available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Error by Navigation Mode', fontweight='bold')
        
        # 4. Rolling average errors
        ax4 = axes[1, 1]
        window = min(50, len(self.traj_df) // 10)
        rolling_mag = self.traj_df['grad_mag_error'].abs().rolling(window=window).mean()
        rolling_ang = self.traj_df['grad_angle_error'].abs().rolling(window=window).mean()
        
        ax4.plot(self.traj_df['time'], rolling_mag, label=f'Magnitude (window={window})', linewidth=2)
        ax4.plot(self.traj_df['time'], rolling_ang, label=f'Angle (window={window})', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Rolling Mean Absolute Error')
        ax4.set_title('Smoothed Error Trends', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/gradient_error_analysis.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved error analysis to 'gradient_error_analysis.png'")
    
    def generate_report(self):
        """Generate detailed text report."""
        print("\n[INFO] Generating analysis report...")
        
        # Compute statistics
        mag_rmse = np.sqrt(np.mean(self.traj_df['grad_mag_error']**2))
        mag_mae = np.mean(np.abs(self.traj_df['grad_mag_error']))
        
        angle_rmse = np.sqrt(np.mean(self.traj_df['grad_angle_error']**2))
        angle_mae = np.mean(np.abs(self.traj_df['grad_angle_error']))
        
        vector_rmse = np.sqrt(np.mean(self.traj_df['grad_vector_error']**2))
        vector_mae = np.mean(self.traj_df['grad_vector_error'])
        
        # Percentiles
        angle_p50 = np.percentile(np.abs(self.traj_df['grad_angle_error']), 50)
        angle_p90 = np.percentile(np.abs(self.traj_df['grad_angle_error']), 90)
        angle_p95 = np.percentile(np.abs(self.traj_df['grad_angle_error']), 95)
        
        # Accuracy metrics
        within_10deg = (np.abs(self.traj_df['grad_angle_error']) < 10).sum() / len(self.traj_df) * 100
        within_30deg = (np.abs(self.traj_df['grad_angle_error']) < 30).sum() / len(self.traj_df) * 100
        within_45deg = (np.abs(self.traj_df['grad_angle_error']) < 45).sum() / len(self.traj_df) * 100
        
        report = f"""
{'=' * 70}
GRADIENT ESTIMATION QUALITY ANALYSIS REPORT
{'=' * 70}

DATA SOURCES:
-------------
Trajectory File: {self.trajectory_file}
Precomputed Map: {self.precomputed_file}

Flight Duration: {self.traj_df['time'].max():.1f} seconds
Trajectory Points: {len(self.traj_df)}
Map Coverage Points: {len(self.map_df)}

GRADIENT MAGNITUDE ERRORS:
--------------------------
RMSE: {mag_rmse:.4f}
MAE:  {mag_mae:.4f}
Mean Error: {self.traj_df['grad_mag_error'].mean():.4f}
Std Dev: {self.traj_df['grad_mag_error'].std():.4f}
Range: [{self.traj_df['grad_mag_error'].min():.4f}, {self.traj_df['grad_mag_error'].max():.4f}]

GRADIENT DIRECTION ERRORS:
--------------------------
RMSE: {angle_rmse:.2f}°
MAE:  {angle_mae:.2f}°
Median Absolute Error: {angle_p50:.2f}°
90th Percentile: {angle_p90:.2f}°
95th Percentile: {angle_p95:.2f}°

GRADIENT VECTOR ERRORS:
-----------------------
RMSE: {vector_rmse:.4f}
MAE:  {vector_mae:.4f}

ACCURACY METRICS:
-----------------
Estimates within 10° of truth: {within_10deg:.1f}%
Estimates within 30° of truth: {within_30deg:.1f}%
Estimates within 45° of truth: {within_45deg:.1f}%

GROUND TRUTH GRADIENT STATISTICS:
----------------------------------
Mean Magnitude: {self.traj_df['map_grad_mag'].mean():.4f}
Magnitude Range: [{self.traj_df['map_grad_mag'].min():.4f}, {self.traj_df['map_grad_mag'].max():.4f}]

ESTIMATION GRADIENT STATISTICS:
--------------------------------
Mean Magnitude: {self.traj_df['grad_mag'].mean():.4f}
Magnitude Range: [{self.traj_df['grad_mag'].min():.4f}, {self.traj_df['grad_mag'].max():.4f}]

"""
        
        # Add mode-specific analysis if available
        if 'mode' in self.traj_df.columns:
            report += "\nERROR BY NAVIGATION MODE:\n"
            report += "--------------------------\n"
            for mode in sorted(self.traj_df['mode'].unique()):
                mode_data = self.traj_df[self.traj_df['mode'] == mode]
                mode_mae = np.mean(np.abs(mode_data['grad_angle_error']))
                mode_pct = len(mode_data) / len(self.traj_df) * 100
                report += f"{mode:20s}: {mode_mae:6.2f}° MAE ({mode_pct:5.1f}% of flight)\n"
        
        report += f"""
INTERPRETATION:
---------------
"""
        
        if angle_mae < 10:
            report += "✓ EXCELLENT: Very low average direction error (<10°)\n"
        elif angle_mae < 20:
            report += "✓ GOOD: Low average direction error (<20°)\n"
        elif angle_mae < 30:
            report += "⚠ FAIR: Moderate direction error (20-30°)\n"
        else:
            report += "✗ POOR: High direction error (>30°)\n"
        
        if within_30deg > 90:
            report += "✓ EXCELLENT: >90% of estimates within 30° of truth\n"
        elif within_30deg > 75:
            report += "✓ GOOD: >75% of estimates within 30° of truth\n"
        else:
            report += "⚠ Significant proportion of estimates have large errors\n"
        
        report += f"""
FILES GENERATED:
----------------
- gradient_comparison_plots.png : Visual comparison of estimates vs truth
- gradient_error_analysis.png : Detailed error analysis
- gradient_quality_report.txt : This report

{'=' * 70}
"""
        
        print(report)
        
        with open('plots/gradient_quality_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("[SUCCESS] Saved report to 'gradient_quality_report.txt'")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        if not self.load_data():
            return False
        
        self.compute_map_gradients()
        self.match_trajectory_to_map()
        
        try:
            self.create_comparison_plots()
            self.create_error_analysis_plot()
            self.generate_report()
            
            print("\n" + "=" * 60)
            print("✓ Analysis Complete!")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python gradient_comparison_analyzer.py <trajectory.csv> <precomputed_map.csv>")
        print("\nSearching for files...")
        
        # Auto-find files
        traj_candidates = list(Path('.').glob('*trajectory*.csv')) + list(Path('logs').glob('*trajectory*.csv'))
        map_candidates = list(Path('.').glob('precomputed_map.csv')) + list(Path('.').glob('*map*.csv'))
        
        if traj_candidates and map_candidates:
            trajectory_file = str(traj_candidates[0])
            map_file = str(map_candidates[0])
            print(f"Found: {trajectory_file}")
            print(f"Found: {map_file}")
        else:
            print("Could not auto-detect files. Please specify them explicitly.")
            return 1
    else:
        trajectory_file = sys.argv[1]
        map_file = sys.argv[2]
    
    print("\n*** Gradient Estimation Quality Analyzer ***\n")
    
    analyzer = GradientComparisonAnalyzer(trajectory_file, map_file)
    success = analyzer.run_full_analysis()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())