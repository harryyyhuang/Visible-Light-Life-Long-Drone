#!/usr/bin/env python3
"""
Gradient Fitting Method Analyzer

Tests and compares different gradient estimation methods on the precomputed map.
Since we know the true light source location, we can evaluate which method
produces the most accurate gradient directions and magnitudes.

Usage:
    python gradient_fitting_analyzer.py [precomputed_map.csv]
    
Outputs:
    - gradient_method_comparison.png : Visual comparison of all methods
    - gradient_method_performance.png : Performance metrics and error analysis
    - gradient_fitting_report.txt : Detailed comparison report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import cKDTree
from scipy.interpolate import Rbf
import sys
from pathlib import Path

# =============================================================================
# CONFIGURATION - Adjust these parameters to test different settings
# =============================================================================

# Neighborhood search parameters
MIN_NEIGHBOR_DISTANCE = 0.1   # meters - exclude points too close (collinear issues)
MAX_NEIGHBOR_DISTANCE = 3.0   # meters - maximum radius to search for neighbors
MIN_NEIGHBOR_COUNT = 3        # minimum points needed for gradient estimation

# Plane fitting quality threshold
MIN_R_SQUARED = 0.2          # minimum R² for accepting a fit (0.0 to 1.0)

# Light source location (ground truth)
LIGHT_SOURCE_X = 5.1988      # meters
LIGHT_SOURCE_Y = 5.329       # meters

# Visualization parameters
SUBSAMPLE_RATE = 5           # Plot every Nth point (to avoid clutter)
ARROW_LENGTH = 0.4           # Fixed arrow length in meters (adjust for visibility)

# =============================================================================
# GRADIENT ESTIMATION METHODS
# =============================================================================

class GradientMethod:
    """Base class for gradient estimation methods."""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def estimate(self, center_x, center_y, neighbor_points):
        """
        Estimate gradient at (center_x, center_y) using neighbor_points.
        
        Args:
            center_x, center_y: Location to estimate gradient
            neighbor_points: DataFrame with columns ['x', 'y', 'total_light']
        
        Returns:
            (grad_x, grad_y, grad_mag, r_squared, success)
        """
        raise NotImplementedError

class LeastSquaresPlaneFit(GradientMethod):
    """Standard least-squares plane fitting: z = ax + by + c"""
    
    def __init__(self):
        super().__init__(
            "Least Squares Plane",
            "Fits plane z=ax+by+c using normal equations (standard method)"
        )
    
    def estimate(self, center_x, center_y, neighbor_points):
        if len(neighbor_points) < MIN_NEIGHBOR_COUNT:
            return 0, 0, 0, 0, False
        
        n = len(neighbor_points)
        x = neighbor_points['x'].values
        y = neighbor_points['y'].values
        z = neighbor_points['total_light'].values
        
        # Normal equations: A * [a, b, c]^T = B
        sx = np.sum(x)
        sy = np.sum(y)
        sz = np.sum(z)
        sxx = np.sum(x * x)
        syy = np.sum(y * y)
        sxy = np.sum(x * y)
        sxz = np.sum(x * z)
        syz = np.sum(y * z)
        
        A = np.array([
            [sxx, sxy, sx],
            [sxy, syy, sy],
            [sx,  sy,  n]
        ])
        B = np.array([sxz, syz, sz])
        
        try:
            # Solve using Cramer's rule (matches C++ implementation)
            det_A = np.linalg.det(A)
            if abs(det_A) < 1e-6:
                return 0, 0, 0, 0, False
            
            # Calculate a and b coefficients
            A_a = A.copy()
            A_a[:, 0] = B
            det_a = np.linalg.det(A_a)
            
            A_b = A.copy()
            A_b[:, 1] = B
            det_b = np.linalg.det(A_b)
            
            grad_x = det_a / det_A
            grad_y = det_b / det_A
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Calculate R²
            c = (sz - grad_x*sx - grad_y*sy) / n
            predictions = grad_x * x + grad_y * y + c
            ss_res = np.sum((z - predictions)**2)
            ss_tot = np.sum((z - np.mean(z))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0
            
            success = r_squared >= MIN_R_SQUARED
            return grad_x, grad_y, grad_mag, r_squared, success
            
        except:
            return 0, 0, 0, 0, False

class WeightedLeastSquares(GradientMethod):
    """Weighted least squares - closer points have more influence"""
    
    def __init__(self):
        super().__init__(
            "Weighted Least Squares",
            "Weights points by inverse distance² from center"
        )
    
    def estimate(self, center_x, center_y, neighbor_points):
        if len(neighbor_points) < MIN_NEIGHBOR_COUNT:
            return 0, 0, 0, 0, False
        
        x = neighbor_points['x'].values
        y = neighbor_points['y'].values
        z = neighbor_points['total_light'].values
        
        # Calculate weights (inverse square of distance)
        dx = x - center_x
        dy = y - center_y
        dist = np.sqrt(dx**2 + dy**2)
        weights = 1.0 / (dist**2 + 0.01)  # Add small epsilon to avoid division by zero
        weights = weights / np.sum(weights)  # Normalize
        
        # Weighted least squares
        W = np.diag(weights)
        A = np.column_stack([x, y, np.ones(len(x))])
        
        try:
            # (A^T W A)^-1 A^T W z
            ATWA = A.T @ W @ A
            ATWz = A.T @ W @ z
            coeffs = np.linalg.solve(ATWA, ATWz)
            
            grad_x, grad_y = coeffs[0], coeffs[1]
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Calculate R²
            predictions = A @ coeffs
            ss_res = np.sum(weights * (z - predictions)**2)
            ss_tot = np.sum(weights * (z - np.mean(z))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0
            
            success = r_squared >= MIN_R_SQUARED
            return grad_x, grad_y, grad_mag, r_squared, success
            
        except:
            return 0, 0, 0, 0, False

class RobustRANSAC(GradientMethod):
    """RANSAC-based plane fitting - robust to outliers"""
    
    def __init__(self):
        super().__init__(
            "RANSAC Plane Fit",
            "Random sample consensus - robust to outlier measurements"
        )
    
    def estimate(self, center_x, center_y, neighbor_points):
        if len(neighbor_points) < MIN_NEIGHBOR_COUNT:
            return 0, 0, 0, 0, False
        
        x = neighbor_points['x'].values
        y = neighbor_points['y'].values
        z = neighbor_points['total_light'].values
        
        n_points = len(x)
        n_iterations = min(50, n_points * 2)
        best_inliers = 0
        best_coeffs = None
        threshold = np.std(z) * 0.5  # Outlier threshold
        
        try:
            for _ in range(n_iterations):
                # Randomly sample 3 points
                if n_points < 3:
                    break
                sample_idx = np.random.choice(n_points, 3, replace=False)
                
                # Fit plane to sample
                A_sample = np.column_stack([x[sample_idx], y[sample_idx], np.ones(3)])
                z_sample = z[sample_idx]
                
                try:
                    coeffs = np.linalg.solve(A_sample, z_sample)
                except:
                    continue
                
                # Count inliers
                A = np.column_stack([x, y, np.ones(n_points)])
                predictions = A @ coeffs
                errors = np.abs(z - predictions)
                inliers = np.sum(errors < threshold)
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_coeffs = coeffs
            
            if best_coeffs is None:
                return 0, 0, 0, 0, False
            
            grad_x, grad_y = best_coeffs[0], best_coeffs[1]
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Calculate R² on all points
            A = np.column_stack([x, y, np.ones(n_points)])
            predictions = A @ best_coeffs
            ss_res = np.sum((z - predictions)**2)
            ss_tot = np.sum((z - np.mean(z))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0
            
            success = r_squared >= MIN_R_SQUARED
            return grad_x, grad_y, grad_mag, r_squared, success
            
        except:
            return 0, 0, 0, 0, False

class LocalPolynomialFit(GradientMethod):
    """2nd order polynomial surface fitting"""
    
    def __init__(self):
        super().__init__(
            "Polynomial Surface",
            "Fits z = ax + by + cx² + dy² + exy + f (captures curvature)"
        )
    
    def estimate(self, center_x, center_y, neighbor_points):
        if len(neighbor_points) < 6:  # Need more points for 2nd order
            return 0, 0, 0, 0, False
        
        x = neighbor_points['x'].values
        y = neighbor_points['y'].values
        z = neighbor_points['total_light'].values
        
        try:
            # Build design matrix for polynomial: z = a*x + b*y + c*x² + d*y² + e*xy + f
            A = np.column_stack([x, y, x**2, y**2, x*y, np.ones(len(x))])
            coeffs = np.linalg.lstsq(A, z, rcond=None)[0]
            
            # Gradient at center point: ∂z/∂x = a + 2c*x + e*y
            #                           ∂z/∂y = b + 2d*y + e*x
            grad_x = coeffs[0] + 2*coeffs[2]*center_x + coeffs[4]*center_y
            grad_y = coeffs[1] + 2*coeffs[3]*center_y + coeffs[4]*center_x
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Calculate R²
            predictions = A @ coeffs
            ss_res = np.sum((z - predictions)**2)
            ss_tot = np.sum((z - np.mean(z))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0
            
            success = r_squared >= MIN_R_SQUARED
            return grad_x, grad_y, grad_mag, r_squared, success
            
        except:
            return 0, 0, 0, 0, False

class FiniteDifferenceGradient(GradientMethod):
    """Simple finite difference approximation"""
    
    def __init__(self):
        super().__init__(
            "Finite Difference",
            "Estimates gradient using nearest neighbors in x and y directions"
        )
    
    def estimate(self, center_x, center_y, neighbor_points):
        if len(neighbor_points) < 4:
            return 0, 0, 0, 0, False
        
        x = neighbor_points['x'].values
        y = neighbor_points['y'].values
        z = neighbor_points['total_light'].values
        
        try:
            # Find points closest to ±x and ±y directions
            dx = x - center_x
            dy = y - center_y
            
            # Points primarily in +x direction
            x_pos_mask = (dx > 0) & (abs(dy) < abs(dx))
            # Points primarily in -x direction
            x_neg_mask = (dx < 0) & (abs(dy) < abs(dx))
            # Points primarily in +y direction
            y_pos_mask = (dy > 0) & (abs(dx) < abs(dy))
            # Points primarily in -y direction
            y_neg_mask = (dy < 0) & (abs(dx) < abs(dy))
            
            grad_x = 0
            grad_y = 0
            
            # Estimate dz/dx
            if np.any(x_pos_mask) and np.any(x_neg_mask):
                z_x_pos = np.mean(z[x_pos_mask])
                z_x_neg = np.mean(z[x_neg_mask])
                dx_avg = np.mean(abs(dx[x_pos_mask])) + np.mean(abs(dx[x_neg_mask]))
                grad_x = (z_x_pos - z_x_neg) / dx_avg if dx_avg > 0 else 0
            
            # Estimate dz/dy
            if np.any(y_pos_mask) and np.any(y_neg_mask):
                z_y_pos = np.mean(z[y_pos_mask])
                z_y_neg = np.mean(z[y_neg_mask])
                dy_avg = np.mean(abs(dy[y_pos_mask])) + np.mean(abs(dy[y_neg_mask]))
                grad_y = (z_y_pos - z_y_neg) / dy_avg if dy_avg > 0 else 0
            
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Rough R² estimate
            r_squared = 0.5  # Finite difference doesn't provide a natural R²
            success = grad_mag > 0
            
            return grad_x, grad_y, grad_mag, r_squared, success
            
        except:
            return 0, 0, 0, 0, False

# =============================================================================
# ANALYZER CLASS
# =============================================================================

class GradientFittingAnalyzer:
    """Compares different gradient estimation methods on precomputed map."""
    
    def __init__(self, map_file):
        self.map_file = map_file
        self.map_df = None
        self.methods = []
        self.results = {}
        
        # Initialize all methods
        self.methods = [
            LeastSquaresPlaneFit(),
            WeightedLeastSquares(),
            RobustRANSAC(),
            LocalPolynomialFit(),
            FiniteDifferenceGradient()
        ]
    
    def load_map(self):
        """Load precomputed map data."""
        print("=" * 70)
        print("Loading Precomputed Map...")
        print("=" * 70)
        
        try:
            self.map_df = pd.read_csv(self.map_file)
            print(f"[SUCCESS] Loaded {len(self.map_df)} map points")
            print(f"  Columns: {list(self.map_df.columns)}")
            
            # Build KD-tree for neighbor search
            self.points = self.map_df[['x', 'y']].values
            self.tree = cKDTree(self.points)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load map: {e}")
            return False
    
    def get_neighbors(self, center_x, center_y):
        """Find neighboring points within distance thresholds."""
        # Query all points within MAX_NEIGHBOR_DISTANCE
        indices = self.tree.query_ball_point([center_x, center_y], MAX_NEIGHBOR_DISTANCE)
        
        if not indices:
            return pd.DataFrame()
        
        neighbors = self.map_df.iloc[indices].copy()
        
        # Filter by MIN_NEIGHBOR_DISTANCE
        dx = neighbors['x'] - center_x
        dy = neighbors['y'] - center_y
        dist = np.sqrt(dx**2 + dy**2)
        
        neighbors = neighbors[dist >= MIN_NEIGHBOR_DISTANCE]
        
        return neighbors
    
    def calculate_true_gradient(self, x, y):
        """Calculate ground truth gradient direction to light source."""
        # Direction vector to light source
        dx = LIGHT_SOURCE_X - x
        dy = LIGHT_SOURCE_Y - y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < 0.01:
            return 0, 0, 0
        
        # Unit vector pointing toward light
        true_grad_x = dx / dist
        true_grad_y = dy / dist
        true_angle = np.degrees(np.arctan2(dy, dx))
        
        return true_grad_x, true_grad_y, true_angle
    
    def test_all_methods(self):
        """Test all gradient estimation methods on sample points."""
        print("\n" + "=" * 70)
        print("Testing Gradient Estimation Methods...")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Min neighbor distance: {MIN_NEIGHBOR_DISTANCE}m")
        print(f"  Max neighbor distance: {MAX_NEIGHBOR_DISTANCE}m")
        print(f"  Min neighbor count: {MIN_NEIGHBOR_COUNT}")
        print(f"  Min R² threshold: {MIN_R_SQUARED}")
        print(f"  Light source: ({LIGHT_SOURCE_X}, {LIGHT_SOURCE_Y})")
        
        # Sample test points from the map
        sample_indices = range(0, len(self.map_df), SUBSAMPLE_RATE)
        sample_points = self.map_df.iloc[sample_indices]
        
        print(f"\nTesting on {len(sample_points)} sample points...")
        
        for method in self.methods:
            print(f"\n  Testing: {method.name}")
            
            gradients_x = []
            gradients_y = []
            magnitudes = []
            r_squareds = []
            angle_errors = []
            success_count = 0
            
            for idx, row in sample_points.iterrows():
                cx, cy = row['x'], row['y']
                
                # Get neighbors
                neighbors = self.get_neighbors(cx, cy)
                
                if len(neighbors) < MIN_NEIGHBOR_COUNT:
                    gradients_x.append(0)
                    gradients_y.append(0)
                    magnitudes.append(0)
                    r_squareds.append(0)
                    angle_errors.append(180)
                    continue
                
                # Estimate gradient
                gx, gy, mag, r2, success = method.estimate(cx, cy, neighbors)
                
                gradients_x.append(gx)
                gradients_y.append(gy)
                magnitudes.append(mag)
                r_squareds.append(r2)
                
                if success:
                    success_count += 1
                    
                    # Calculate error vs ground truth
                    true_gx, true_gy, true_angle = self.calculate_true_gradient(cx, cy)
                    est_angle = np.degrees(np.arctan2(gy, gx))
                    
                    angle_diff = (est_angle - true_angle + 180) % 360 - 180
                    angle_errors.append(abs(angle_diff))
                else:
                    angle_errors.append(180)
            
            # Store results
            self.results[method.name] = {
                'x': sample_points['x'].values,
                'y': sample_points['y'].values,
                'grad_x': np.array(gradients_x),
                'grad_y': np.array(gradients_y),
                'magnitude': np.array(magnitudes),
                'r_squared': np.array(r_squareds),
                'angle_error': np.array(angle_errors),
                'success_rate': success_count / len(sample_points) * 100,
                'mean_angle_error': np.mean([e for e in angle_errors if e < 180]),
                'median_angle_error': np.median([e for e in angle_errors if e < 180])
            }
            
            print(f"    Success rate: {self.results[method.name]['success_rate']:.1f}%")
            print(f"    Mean angle error: {self.results[method.name]['mean_angle_error']:.2f}°")
    
    def create_comparison_plots(self):
        """Create visual comparison of all methods."""
        print("\n[INFO] Creating comparison plots...")
        
        n_methods = len(self.methods)
        fig = plt.figure(figsize=(20, 4 * n_methods))
        
        for idx, method in enumerate(self.methods):
            result = self.results[method.name]
            
            # Vector field plot
            ax = fig.add_subplot(n_methods, 1, idx + 1)
            
            # Plot all map points colored by light intensity
            scatter_bg = ax.scatter(self.map_df['x'], self.map_df['y'], 
                                   c=self.map_df['total_light'], 
                                   cmap='viridis', s=20, alpha=0.5, 
                                   edgecolors='none', label='Map Points',
                                   vmin=0, vmax=self.map_df['total_light'].max())
            
            # Add colorbar for light intensity
            cbar_light = plt.colorbar(scatter_bg, ax=ax, label='Total Light Intensity', 
                                     pad=0.12, aspect=30)
            
            # Plot light source
            ax.scatter([LIGHT_SOURCE_X], [LIGHT_SOURCE_Y], 
                      color='gold', s=500, marker='*', 
                      edgecolors='black', linewidth=2, 
                      label='Light Source', zorder=10)
            
            # Plot gradient vectors (normalized to unit length for visibility)
            mask = result['magnitude'] > 0
            
            # Normalize gradient vectors to unit length
            gx_norm = result['grad_x'][mask] / (result['magnitude'][mask] + 1e-6)
            gy_norm = result['grad_y'][mask] / (result['magnitude'][mask] + 1e-6)
            
            # Use fixed arrow length from configuration
            arrow_len = ARROW_LENGTH  # All arrows are this length for visibility
            
            quiver = ax.quiver(result['x'][mask], result['y'][mask],
                              gx_norm * arrow_len, gy_norm * arrow_len,
                              result['angle_error'][mask],
                              cmap='RdYlGn_r', alpha=0.9, 
                              scale=1, scale_units='xy', angles='xy',
                              width=0.004,
                              clim=[0, 90], zorder=5)
            
            cbar_err = plt.colorbar(quiver, ax=ax, label='Angle Error (deg)', 
                                   aspect=30)
            
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_title(f'{method.name} - {method.description}\n'
                        f'Success: {result["success_rate"]:.1f}% | '
                        f'Mean Error: {result["mean_angle_error"]:.1f}° | '
                        f'Median Error: {result["median_angle_error"]:.1f}°',
                        fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/gradient_method_comparison.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved comparison plots to 'gradient_method_comparison.png'")
    
    def create_performance_plots(self):
        """Create performance analysis plots."""
        print("\n[INFO] Creating performance analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Success rate comparison
        ax1 = axes[0, 0]
        method_names = [m.name for m in self.methods]
        success_rates = [self.results[name]['success_rate'] for name in method_names]
        
        bars = ax1.bar(range(len(method_names)), success_rates, color='steelblue', edgecolor='black')
        ax1.set_xticks(range(len(method_names)))
        ax1.set_xticklabels(method_names, rotation=45, ha='right')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Gradient Estimation Success Rate', fontweight='bold')
        ax1.set_ylim([0, 100])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Angle error distribution
        ax2 = axes[0, 1]
        for method in self.methods:
            result = self.results[method.name]
            valid_errors = [e for e in result['angle_error'] if e < 180]
            if valid_errors:
                ax2.hist(valid_errors, bins=30, alpha=0.5, label=method.name, edgecolor='black')
        
        ax2.set_xlabel('Angle Error (deg)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Gradient Direction Errors', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Mean angle error comparison
        ax3 = axes[1, 0]
        mean_errors = [self.results[name]['mean_angle_error'] for name in method_names]
        
        bars = ax3.bar(range(len(method_names)), mean_errors, color='coral', edgecolor='black')
        ax3.set_xticks(range(len(method_names)))
        ax3.set_xticklabels(method_names, rotation=45, ha='right')
        ax3.set_ylabel('Mean Angle Error (deg)')
        ax3.set_title('Average Gradient Direction Error', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, mean_errors):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}°', ha='center', va='bottom', fontweight='bold')
        
        # 4. CDF comparison
        ax4 = axes[1, 1]
        for method in self.methods:
            result = self.results[method.name]
            valid_errors = sorted([e for e in result['angle_error'] if e < 180])
            if valid_errors:
                cdf = np.arange(1, len(valid_errors) + 1) / len(valid_errors) * 100
                ax4.plot(valid_errors, cdf, linewidth=2, label=method.name)
        
        ax4.axvline(x=10, color='green', linestyle='--', alpha=0.5, label='10° threshold')
        ax4.axvline(x=30, color='orange', linestyle='--', alpha=0.5, label='30° threshold')
        ax4.set_xlabel('Angle Error (deg)')
        ax4.set_ylabel('Cumulative Percentage (%)')
        ax4.set_title('CDF of Gradient Direction Errors', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/gradient_method_performance.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved performance analysis to 'gradient_method_performance.png'")
    
    def generate_report(self):
        """Generate text report comparing methods."""
        print("\n[INFO] Generating comparison report...")
        
        report = f"""
{'=' * 70}
GRADIENT FITTING METHOD COMPARISON REPORT
{'=' * 70}

DATA SOURCE:
------------
Map file: {self.map_file}
Map points: {len(self.map_df)}
Test points: {len(self.results[self.methods[0].name]['x'])}

CONFIGURATION:
--------------
Light source location: ({LIGHT_SOURCE_X}, {LIGHT_SOURCE_Y})
Min neighbor distance: {MIN_NEIGHBOR_DISTANCE}m
Max neighbor distance: {MAX_NEIGHBOR_DISTANCE}m
Min neighbor count: {MIN_NEIGHBOR_COUNT}
Min R² threshold: {MIN_R_SQUARED}

METHODS TESTED:
---------------
"""
        
        for method in self.methods:
            report += f"\n{method.name}:\n  {method.description}\n"
        
        report += f"\n\n{'=' * 70}\nRESULTS SUMMARY\n{'=' * 70}\n\n"
        
        # Create comparison table
        report += f"{'Method':<25} {'Success':<10} {'Mean Err':<12} {'Median Err':<12} {'90th %ile':<10}\n"
        report += "-" * 70 + "\n"
        
        for method in self.methods:
            result = self.results[method.name]
            valid_errors = [e for e in result['angle_error'] if e < 180]
            p90 = np.percentile(valid_errors, 90) if valid_errors else 0
            
            report += f"{method.name:<25} {result['success_rate']:>6.1f}%   "
            report += f"{result['mean_angle_error']:>8.2f}°   "
            report += f"{result['median_angle_error']:>8.2f}°   "
            report += f"{p90:>8.2f}°\n"
        
        report += f"\n\n{'=' * 70}\nDETAILED ANALYSIS\n{'=' * 70}\n"
        
        # Find best method for each metric
        best_success = max(self.methods, key=lambda m: self.results[m.name]['success_rate'])
        best_accuracy = min(self.methods, key=lambda m: self.results[m.name]['mean_angle_error'])
        
        report += f"\nBest Success Rate: {best_success.name} ({self.results[best_success.name]['success_rate']:.1f}%)\n"
        report += f"Best Accuracy: {best_accuracy.name} ({self.results[best_accuracy.name]['mean_angle_error']:.2f}° mean error)\n"
        
        report += f"\n\nRECOMMENDATIONS:\n"
        report += "-" * 70 + "\n"
        
        # Provide recommendations based on results
        if self.results[best_accuracy.name]['mean_angle_error'] < 15:
            report += f"✓ EXCELLENT: {best_accuracy.name} achieves <15° mean error\n"
            report += f"  Recommended for production use\n"
        elif self.results[best_accuracy.name]['mean_angle_error'] < 30:
            report += f"✓ GOOD: {best_accuracy.name} achieves <30° mean error\n"
            report += f"  Suitable for navigation with some inaccuracy\n"
        else:
            report += f"⚠ POOR: Best method still has >{self.results[best_accuracy.name]['mean_angle_error']:.1f}° error\n"
            report += f"  Consider: adjusting parameters, adding more map points, or obstacle handling\n"
        
        report += f"\n\nFILES GENERATED:\n"
        report += "-" * 70 + "\n"
        report += "- gradient_method_comparison.png : Visual comparison of all methods\n"
        report += "- gradient_method_performance.png : Performance metrics\n"
        report += "- gradient_fitting_report.txt : This report\n"
        
        report += f"\n{'=' * 70}\n"
        
        print(report)
        
        with open('plots/gradient_fitting_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("[SUCCESS] Saved report to 'gradient_fitting_report.txt'")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        if not self.load_map():
            return False
        
        try:
            self.test_all_methods()
            self.create_comparison_plots()
            self.create_performance_plots()
            self.generate_report()
            
            print("\n" + "=" * 70)
            print("✓ Analysis Complete!")
            print("=" * 70)
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python gradient_fitting_analyzer.py <precomputed_map.csv>")
        print("\nSearching for precomputed_map.csv...")
        
        candidates = list(Path('.').glob('precomputed_map.csv')) + \
                    list(Path('logs').glob('precomputed_map.csv'))
        
        if candidates:
            map_file = str(candidates[0])
            print(f"Found: {map_file}")
        else:
            print("Could not find precomputed_map.csv")
            return 1
    else:
        map_file = sys.argv[1]
    
    print("\n*** Gradient Fitting Method Analyzer ***\n")
    
    analyzer = GradientFittingAnalyzer(map_file)
    success = analyzer.run_full_analysis()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())