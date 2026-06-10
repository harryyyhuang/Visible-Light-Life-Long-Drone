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

# Obstacle (cylinder) -- must match the Webots world (blimp.wbt: DEF PILLAR_CENTER).
# The light casts shadows, so behind the obstacle the field gradient does NOT point
# straight at the source; it leads around. The error metric below accounts for this.
OBSTACLE_X = 0.8             # meters
OBSTACLE_Y = 1.5             # meters
OBSTACLE_RADIUS = 0.4        # meters (physical cylinder radius -> casts the light shadow)
OBSTACLE_MARGIN = 0.5        # meters (clearance the drone keeps; no map data inside this)
AVOID_RADIUS = OBSTACLE_RADIUS + OBSTACLE_MARGIN  # only used to draw the safety circle

# Visualization parameters
SUBSAMPLE_RATE = 3           # Keep ~1-in-N² points via 2D grid subsampling (scatter in X and Y)
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
    
    def reference_direction(self, x, y):
        """Region-aware "correct" gradient direction at (x, y).

        Returns (true_angle_deg, region, avoid_unit):
          region == 'near'   : within 0.1 m of the source -> skip (true_angle None),
                               because the inverse-square field is so steep there that
                               any tiny numerical error blows up the angle error.
          region == 'free'   : the straight line of sight to the source is clear, so the
                               correct direction is straight to the source (the usual
                               metric is valid here).
          region == 'shadow' : the obstacle blocks the line of sight to the source, so
                               straight-to-source points into the pillar and is NOT the
                               useful direction. true_angle is then the tangent direction
                               that skirts the obstacle while making progress toward the
                               source, and avoid_unit is its unit vector (used to score
                               how well a method steers around the obstacle).

        This replaces the old straight-to-source-everywhere metric, which unfairly
        penalised any method that correctly steered around the obstacle.
        """
        dx, dy = LIGHT_SOURCE_X - x, LIGHT_SOURCE_Y - y
        dist = np.hypot(dx, dy)
        if dist < 0.1:                       # near-source: field too steep to score
            return None, 'near', None

        # Is the straight path from here to the source blocked by the obstacle?
        if not self._segment_hits_obstacle(x, y, LIGHT_SOURCE_X, LIGHT_SOURCE_Y):
            return np.degrees(np.arctan2(dy, dx)), 'free', None

        # Blocked -> the correct direction skirts the obstacle (visibility-graph hop).
        av = self._avoidance_dir(x, y)
        if av is None:                       # inside the obstacle (no map data here)
            return np.degrees(np.arctan2(dy, dx)), 'free', None
        return np.degrees(np.arctan2(av[1], av[0])), 'shadow', av

    @staticmethod
    def _segment_hits_obstacle(px, py, qx, qy):
        """True if segment P->Q passes within OBSTACLE_RADIUS of the obstacle centre."""
        dx, dy = qx - px, qy - py
        L2 = dx * dx + dy * dy
        if L2 < 1e-12:
            return (px - OBSTACLE_X) ** 2 + (py - OBSTACLE_Y) ** 2 <= OBSTACLE_RADIUS ** 2
        t = ((OBSTACLE_X - px) * dx + (OBSTACLE_Y - py) * dy) / L2
        t = max(0.0, min(1.0, t))           # nearest point on the segment to the centre
        nx, ny = px + t * dx, py + t * dy
        return (nx - OBSTACLE_X) ** 2 + (ny - OBSTACLE_Y) ** 2 <= OBSTACLE_RADIUS ** 2

    @staticmethod
    def _avoidance_dir(x, y):
        """Unit tangent direction around the obstacle that heads toward the source.

        From (x, y) there are two tangents to the obstacle circle; we return the one
        whose direction is more aligned with the straight-to-source direction, i.e. the
        short way around.
        """
        cx, cy = OBSTACLE_X - x, OBSTACLE_Y - y      # vector to the obstacle centre
        d = np.hypot(cx, cy)
        if d <= OBSTACLE_RADIUS:
            return None
        ucx, ucy = cx / d, cy / d
        phi = np.arcsin(min(1.0, OBSTACLE_RADIUS / d))   # half-angle to the two tangents
        c, s = np.cos(phi), np.sin(phi)
        t1 = (ucx * c - ucy * s, ucx * s + ucy * c)      # to-centre rotated by +phi
        t2 = (ucx * c + ucy * s, -ucx * s + ucy * c)     # to-centre rotated by -phi
        sx, sy = LIGHT_SOURCE_X - x, LIGHT_SOURCE_Y - y
        sn = np.hypot(sx, sy)
        sx, sy = sx / sn, sy / sn
        return t1 if (t1[0] * sx + t1[1] * sy) >= (t2[0] * sx + t2[1] * sy) else t2
    
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
        
        # Sample test points from the map using 2D grid subsampling.
        # The CSV is written row-by-row in scan order so a plain row-stride
        # (every Nth row) produces horizontal stripes -- all X positions for
        # a handful of Y values.  Instead we bin both X and Y into cells and
        # keep one point per cell so the samples are scattered across the
        # whole field.
        x_vals = self.map_df['x'].values
        y_vals = self.map_df['y'].values
        x_bins = np.arange(x_vals.min(), x_vals.max() + 1e-9,
                           (x_vals.max() - x_vals.min()) / max(1, (x_vals.max() - x_vals.min()) / (SUBSAMPLE_RATE * 0.2)))
        y_bins = np.arange(y_vals.min(), y_vals.max() + 1e-9,
                           (y_vals.max() - y_vals.min()) / max(1, (y_vals.max() - y_vals.min()) / (SUBSAMPLE_RATE * 0.2)))
        x_idx = np.digitize(x_vals, x_bins)
        y_idx = np.digitize(y_vals, y_bins)
        cell_keys = list(zip(x_idx, y_idx))
        seen_cells = {}
        for i, key in enumerate(cell_keys):
            if key not in seen_cells:
                seen_cells[key] = i
        sample_indices = sorted(seen_cells.values())
        sample_points = self.map_df.iloc[sample_indices]
        
        print(f"\nTesting on {len(sample_points)} sample points...")
        
        for method in self.methods:
            print(f"\n  Testing: {method.name}")
            
            gradients_x = []
            gradients_y = []
            magnitudes = []
            r_squareds = []
            angle_errors = []
            regions = []        # 'free' | 'shadow' | 'near' | 'fail'
            avoid_align = []    # shadow only: cos(angle) between estimate and avoidance dir
            success_count = 0

            for idx, row in sample_points.iterrows():
                cx, cy = row['x'], row['y']
                neighbors = self.get_neighbors(cx, cy)

                if len(neighbors) < MIN_NEIGHBOR_COUNT:
                    gradients_x.append(0); gradients_y.append(0)
                    magnitudes.append(0); r_squareds.append(0)
                    angle_errors.append(180); regions.append('fail'); avoid_align.append(np.nan)
                    continue

                gx, gy, mag, r2, success = method.estimate(cx, cy, neighbors)
                gradients_x.append(gx); gradients_y.append(gy)
                magnitudes.append(mag); r_squareds.append(r2)

                if not success:
                    # sentinel -1 so the comparison plot can colour rejects in grey
                    angle_errors.append(-1); regions.append('fail'); avoid_align.append(np.nan)
                    continue

                success_count += 1
                # Region-aware ground truth: straight-to-source in free space, the
                # tangent around the obstacle in its shadow (see reference_direction).
                true_angle, region, av = self.reference_direction(cx, cy)
                regions.append(region)
                if true_angle is None:                 # near-source: drop from the stats
                    angle_errors.append(np.nan); avoid_align.append(np.nan)
                    continue
                est_angle = np.degrees(np.arctan2(gy, gx))
                angle_errors.append(abs((est_angle - true_angle + 180) % 360 - 180))
                if region == 'shadow' and av is not None and mag > 0:
                    avoid_align.append((gx * av[0] + gy * av[1]) / mag)
                else:
                    avoid_align.append(np.nan)

            angle_errors = np.array(angle_errors, dtype=float)
            regions = np.array(regions)
            avoid_align = np.array(avoid_align, dtype=float)

            valid = (angle_errors >= 0) & (angle_errors < 180)
            free_mask = valid & (regions == 'free')
            shadow_mask = valid & (regions == 'shadow')

            def _stat(arr, fn):
                arr = arr[np.isfinite(arr)]
                return float(fn(arr)) if arr.size else float('nan')

            self.results[method.name] = {
                'x': sample_points['x'].values,
                'y': sample_points['y'].values,
                'grad_x': np.array(gradients_x),
                'grad_y': np.array(gradients_y),
                'magnitude': np.array(magnitudes),
                'r_squared': np.array(r_squareds),
                'angle_error': angle_errors,
                'region': regions,
                'avoid_align': avoid_align,
                'success_rate': success_count / len(sample_points) * 100,
                'mean_angle_error': _stat(angle_errors[valid], np.mean),
                'median_angle_error': _stat(angle_errors[valid], np.median),
                'free_mean_error': _stat(angle_errors[free_mask], np.mean),
                'free_median_error': _stat(angle_errors[free_mask], np.median),
                'shadow_mean_error': _stat(angle_errors[shadow_mask], np.mean),
                'shadow_align_mean': _stat(avoid_align[shadow_mask], np.mean),
                'shadow_align_pct': _stat(avoid_align[shadow_mask], lambda a: np.mean(a > 0) * 100),
                'n_free': int(free_mask.sum()),
                'n_shadow': int(shadow_mask.sum()),
            }

            r = self.results[method.name]
            print(f"    Success: {r['success_rate']:.1f}% | "
                  f"free-space err: {r['free_mean_error']:.1f} deg | "
                  f"shadow escape: {r['shadow_align_pct']:.0f}% "
                  f"(n_free={r['n_free']}, n_shadow={r['n_shadow']})")
    
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
            
            # Plot light source
            ax.scatter([LIGHT_SOURCE_X], [LIGHT_SOURCE_Y],
                      color='gold', s=500, marker='*',
                      edgecolors='black', linewidth=2,
                      label='Light Source', zorder=10)

            # Obstacle (physical cylinder) and the safety circle the drone keeps clear
            from matplotlib.patches import Circle
            ax.add_patch(Circle((OBSTACLE_X, OBSTACLE_Y), OBSTACLE_RADIUS,
                                facecolor='dimgray', edgecolor='black', zorder=9,
                                label='Obstacle'))
            ax.add_patch(Circle((OBSTACLE_X, OBSTACLE_Y), AVOID_RADIUS,
                                fill=False, linestyle='--', edgecolor='gray', zorder=9))
            
            # Three arrow categories drawn in separate layers:
            #   1. Valid estimates with measurable angle error  -> RdYlGn_r colourmap
            #   2. Low-R² rejections (sentinel -1)             -> grey arrows
            #   3. Near-source skip (sentinel NaN)             -> not drawn
            ae   = result['angle_error']
            mag  = result['magnitude']
            rx   = result['x']
            ry   = result['y']
            rgx  = result['grad_x']
            rgy  = result['grad_y']

            # Masks
            valid_mask  = np.isfinite(ae) & (ae >= 0) & (mag > 0)
            reject_mask = (ae == -1)      & (mag > 0)

            def _norm_arrows(mask):
                m = mag[mask]
                return (rgx[mask] / (m + 1e-6) * ARROW_LENGTH,
                        rgy[mask] / (m + 1e-6) * ARROW_LENGTH)

            # -- Grey arrows for low-R² rejected estimates --
            if reject_mask.any():
                gx_r, gy_r = _norm_arrows(reject_mask)
                ax.quiver(rx[reject_mask], ry[reject_mask], gx_r, gy_r,
                          color='#888888', alpha=0.45,
                          scale=1, scale_units='xy', angles='xy',
                          width=0.003, zorder=4,
                          label=f'Low R² rejected ({reject_mask.sum()})')

            # -- Coloured arrows for valid estimates --
            quiver = None
            if valid_mask.any():
                gx_v, gy_v = _norm_arrows(valid_mask)
                quiver = ax.quiver(rx[valid_mask], ry[valid_mask], gx_v, gy_v,
                                   ae[valid_mask],
                                   cmap='RdYlGn_r', alpha=0.9,
                                   scale=1, scale_units='xy', angles='xy',
                                   width=0.004, clim=[0, 90], zorder=5)
            
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_title(f'{method.name} - {method.description}\n'
                        f'Success: {result["success_rate"]:.1f}% | '
                        f'Free-space error: {result["free_mean_error"]:.1f}° | '
                        f'Shadow escape: {result["shadow_align_pct"]:.0f}%',
                        fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Colorbars
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax_light = divider.append_axes("right", size="2%")
            plt.colorbar(scatter_bg, cax=cax_light, label='Total Light Intensity')

            # Angle error colourbar -- only when there are valid estimates
            if quiver is not None:
                cax_error = divider.append_axes("left", size="2%")
                cbar_error = plt.colorbar(quiver, cax=cax_error,
                                          label='Angle Error (deg) — grey = low R²')
                cax_error.yaxis.set_ticks_position('left')
                cax_error.yaxis.set_label_position('left')

        
        plt.tight_layout()
        plt.savefig('plots/gradient_method_comparison.png', dpi=300, bbox_inches='tight')
        print("[SUCCESS] Saved comparison plots to 'gradient_method_comparison.png'")
    
    def create_performance_plots(self):
        """Region-conditioned performance plots.

        Free-space accuracy (where straight-to-source is the correct answer) is the
        valid ranking metric; the shadow panel reports how well each method steers
        AROUND the obstacle, which is what the controller actually needs there.
        """
        print("\n[INFO] Creating performance analysis plots...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        names = [m.name for m in self.methods]

        def _bars(ax, vals, color, ylabel, title, fmt, ylim=None):
            bars = ax.bar(range(len(names)), vals, color=color, edgecolor='black')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight='bold')
            if ylim:
                ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3, axis='y')
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2., b.get_height(),
                        fmt.format(v), ha='center', va='bottom', fontweight='bold')

        # 1. Success rate
        _bars(axes[0, 0], [self.results[n]['success_rate'] for n in names],
              'steelblue', 'Success Rate (%)', 'Gradient Estimation Success Rate',
              '{:.1f}%', ylim=[0, 100])

        # 2. Free-space accuracy -- the valid metric for ranking
        _bars(axes[0, 1], [self.results[n]['free_mean_error'] for n in names],
              'coral', 'Mean Angle Error (deg)',
              'Accuracy in Free Space\n(line of sight to source clear)', '{:.1f}°')

        # 3. Obstacle avoidance behind the pillar
        _bars(axes[1, 0], [self.results[n]['shadow_align_pct'] for n in names],
              'seagreen', '% of shadow estimates pointing around',
              'Obstacle Avoidance in the Shadow\n(estimate aligned with the way around)',
              '{:.0f}%', ylim=[0, 100])

        # 4. CDF of FREE-SPACE errors only (where the metric is valid)
        ax4 = axes[1, 1]
        for method in self.methods:
            r = self.results[method.name]
            e = np.sort(r['angle_error'][(r['region'] == 'free') &
                                         (r['angle_error'] >= 0) & (r['angle_error'] < 180)])
            if e.size:
                ax4.plot(e, np.arange(1, e.size + 1) / e.size * 100, linewidth=2, label=method.name)
        ax4.axvline(x=10, color='green', linestyle='--', alpha=0.5, label='10° threshold')
        ax4.axvline(x=30, color='orange', linestyle='--', alpha=0.5, label='30° threshold')
        ax4.set_xlabel('Free-space Angle Error (deg)')
        ax4.set_ylabel('Cumulative Percentage (%)')
        ax4.set_title('CDF of Free-space Direction Errors', fontweight='bold')
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
        
        # Region-conditioned comparison table
        report += (f"{'Method':<24} {'Success':<9} {'Free err':<10} "
                   f"{'Free med':<10} {'Shadow esc':<12} {'Shadow err':<10}\n")
        report += "-" * 76 + "\n"
        for method in self.methods:
            r = self.results[method.name]
            report += (f"{method.name:<24} {r['success_rate']:>6.1f}%  "
                       f"{r['free_mean_error']:>7.1f}°  "
                       f"{r['free_median_error']:>7.1f}°  "
                       f"{r['shadow_align_pct']:>9.0f}%   "
                       f"{r['shadow_mean_error']:>7.1f}°\n")
        report += ("\nFree err / Free med : mean / median angle error in free space, "
                   "where straight-to-source is the correct answer.\n"
                   "Shadow esc          : %% of behind-obstacle estimates that point around "
                   "the obstacle toward the source.\n"
                   "Shadow err          : mean angle error behind the obstacle against the "
                   "tangent (around) direction.\n").replace('%%', '%')

        report += f"\n\n{'=' * 70}\nDETAILED ANALYSIS\n{'=' * 70}\n"

        # Rank on the valid metrics: free-space accuracy and obstacle escape
        best_success = max(self.methods, key=lambda m: self.results[m.name]['success_rate'])
        best_accuracy = min(self.methods, key=lambda m: self.results[m.name]['free_mean_error'])
        best_escape = max(self.methods, key=lambda m: self.results[m.name]['shadow_align_pct'])

        report += f"\nBest Success Rate:        {best_success.name} ({self.results[best_success.name]['success_rate']:.1f}%)\n"
        report += f"Best Free-space Accuracy: {best_accuracy.name} ({self.results[best_accuracy.name]['free_mean_error']:.1f}° mean)\n"
        report += f"Best Obstacle Escape:     {best_escape.name} ({self.results[best_escape.name]['shadow_align_pct']:.0f}% point around)\n"
        
        report += f"\n\nRECOMMENDATIONS:\n"
        report += "-" * 70 + "\n"
        
        # Provide recommendations based on results
        if self.results[best_accuracy.name]['free_mean_error'] < 15:
            report += f"✓ EXCELLENT: {best_accuracy.name} achieves <15° mean error\n"
            report += f"  Recommended for production use\n"
        elif self.results[best_accuracy.name]['free_mean_error'] < 30:
            report += f"✓ GOOD: {best_accuracy.name} achieves <30° mean error\n"
            report += f"  Suitable for navigation with some inaccuracy\n"
        else:
            report += f"⚠ POOR: Best method still has >{self.results[best_accuracy.name]['free_mean_error']:.1f}° error\n"
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