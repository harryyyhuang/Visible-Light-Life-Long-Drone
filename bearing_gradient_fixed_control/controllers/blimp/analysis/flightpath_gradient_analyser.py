#!/usr/bin/env python3
"""
Flightpath Gradient Coverage Analyser
======================================
Analyses how gradient estimation quality degrades when only the points
visited along the blimp's actual flight path are available, compared to
having the full precomputed grid map.

The core question: at each position along the flight, if you could only use
the map cells the blimp has visited so far, how accurate would the gradient
estimate be vs the ground truth from the full probe scan?

Inputs
------
  precomputed_map.csv   -- Full grid scan from map_supervisor_probe.cpp
                           Columns: grid_x, grid_y, x, y, total_light,
                                    timestamp, yaw, sensor_0..sensor_15

  trajectory CSV        -- Flight log from blimp.cpp (fixed_fusion_trajectory.csv)
                           Columns: time, x, y, z, yaw_deg, cmd_yaw_deg,
                                    forward_speed, bearing_angle, bearing_valid,
                                    bearing_weight, grad_angle, grad_mag,
                                    grad_valid, grad_weight, map_size,
                                    total_light, dist_to_target

Usage
-----
  python flightpath_gradient_analyser.py precomputed_map.csv trajectory.csv
  python flightpath_gradient_analyser.py                        # auto-detect files

Outputs (all written to plots/ subfolder)
---------
  flightpath_coverage_map.png
      4-panel spatial overview:
        [TL] Full probe map intensity (background) + full trajectory +
             ground-truth gradient arrows from full map
        [TR] Points available at each flight step (cumulative path coverage) +
             how neighbourhood fills in over time
        [BL] Ground-truth gradient arrows vs flight-path-only gradient arrows
             at each trajectory position (coloured by angle error)
        [BR] Map coverage fraction along flight: how many of the full grid
             cells within each neighbourhood are actually available from path

  flightpath_gradient_timeline.png
      5-panel time series:
        1. Gradient angle: full-map estimate vs path-only estimate vs
           blimp's own real-time estimate (from trajectory log)
        2. Gradient magnitude: full-map vs path-only
        3. Plane fit R² quality: full-map vs path-only
        4. Angle error of path-only estimate vs full-map ground truth,
           coloured by whether blimp was in BEARING_ONLY or FUSED phase
        5. Number of neighbours available (path-only vs full-map) +
           map_size from trajectory log

  flightpath_gradient_quality.png
      Distribution analysis:
        - Histogram + CDF of angle errors (path-only vs ground truth)
        - Split by flight phase (BEARING_ONLY / FUSED)
        - Split by neighbourhood coverage fraction

  flightpath_report.txt
      Quantitative summary

Configuration
-------------
  Adjust the constants in the CONFIGURATION block below to match your setup.
  Estimation method is controlled by the GRADIENT_METHOD constant -- change
  that one line to swap between LeastSquaresPlaneFit, WeightedLeastSquares,
  RobustRANSAC, LocalPolynomialFit, or FiniteDifferenceGradient.
  All five methods are ported directly from gradient_fitting_analyzer.py.
  Default: WeightedLeastSquares (best accuracy in prior comparison).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.spatial import cKDTree
import sys
from pathlib import Path

# =============================================================================
# CONFIGURATION  --  keep in sync with gradient_fitting_analyzer.py
# =============================================================================

# Neighbourhood search (must match blimp.cpp / gradient_fitting_analyzer.py)
MIN_NEIGHBOR_DISTANCE = 0.1   # metres - exclude collinear/same-cell points
MAX_NEIGHBOR_DISTANCE = 3.0   # metres - local neighbourhood radius
MIN_NEIGHBOR_COUNT    = 3     # minimum points to attempt a fit

# Plane fit quality threshold
MIN_R_SQUARED = 0.2           # minimum R² to accept a gradient estimate

# Ground-truth light source location (used for angle error reference)
LIGHT_SOURCE_X = 5.1988
LIGHT_SOURCE_Y = 5.329

# Blimp.cpp gradient readiness thresholds (for phase colouring)
GRADIENT_THRESHOLD = 0.5      # grad_mag threshold in blimp.cpp
MIN_MAP_POINTS     = 3        # MIN_MAP_POINTS in blimp.cpp

# Grid resolution of the probe scan (must match map_supervisor_probe.cpp)
GRID_RESOLUTION = 0.2         # metres per cell

# Visualization
ARROW_LENGTH   = 0.5         # fixed arrow length for gradient vectors (metres)
SUBSAMPLE_ARROWS = 25          # draw arrow every Nth trajectory point

OUTPUT_DIR = "plots"

# =============================================================================
# COLOURS
# =============================================================================

COL_FULL    = "#2196F3"   # blue   -- full map estimate
COL_PATH    = "#4CAF50"   # green  -- path-only estimate (valid)
COL_REJECT  = "#FF9800"   # amber  -- path-only estimate (R² rejected)
COL_BLIMP   = "#9C27B0"   # purple -- blimp's own real-time estimate
COL_BEARING = "#F44336"   # red    -- bearing-only phase
COL_FUSED   = "#4CAF50"   # green  -- fused phase
COL_SOURCE  = "#FFD700"   # gold   -- light source

# =============================================================================
# GRADIENT ESTIMATION METHODS
# Ported directly from gradient_fitting_analyzer.py so this analyser can use
# any of the same methods.  Change GRADIENT_METHOD below to swap.
#
# The interface is adapted slightly: estimate() here accepts numpy arrays
# directly rather than a DataFrame, avoiding the overhead of creating a
# temporary DataFrame for every neighbourhood lookup.
# =============================================================================

class GradientMethod:
    """Base class -- mirrors gradient_fitting_analyzer.py GradientMethod."""
    def __init__(self, name, description):
        self.name        = name
        self.description = description

    def estimate(self, center_x, center_y, x, y, z):
        """
        Estimate gradient at (center_x, center_y) using neighbour arrays.
        x, y, z are 1-D numpy arrays of already-filtered neighbour points.
        Returns (grad_x, grad_y, grad_mag, r_squared, success).
        """
        raise NotImplementedError


class LeastSquaresPlaneFit(GradientMethod):
    """Standard least-squares plane fit z=ax+by+c via Cramer's rule.
    Matches blimp.cpp and gradient_fitting_analyzer.py LeastSquaresPlaneFit."""

    def __init__(self):
        super().__init__(
            "Least Squares Plane",
            "Fits plane z=ax+by+c using normal equations / Cramer's rule"
        )

    def estimate(self, center_x, center_y, x, y, z):
        if len(x) < MIN_NEIGHBOR_COUNT:
            return 0, 0, 0, 0, False
        n = len(x)
        sx  = x.sum();  sy  = y.sum();  sz  = z.sum()
        sxx = (x*x).sum(); syy = (y*y).sum(); sxy = (x*y).sum()
        sxz = (x*z).sum(); syz = (y*z).sum()
        A = np.array([[sxx, sxy, sx],
                      [sxy, syy, sy],
                      [sx,  sy,  float(n)]])
        B = np.array([sxz, syz, sz])
        try:
            det_A = np.linalg.det(A)
            if abs(det_A) < 1e-6:
                return 0, 0, 0, 0, False
            Aa = A.copy(); Aa[:, 0] = B
            Ab = A.copy(); Ab[:, 1] = B
            gx  = np.linalg.det(Aa) / det_A
            gy  = np.linalg.det(Ab) / det_A
            mag = np.sqrt(gx**2 + gy**2)
            c      = (sz - gx*sx - gy*sy) / n
            pred   = gx*x + gy*y + c
            ss_res = ((z - pred)**2).sum()
            ss_tot = ((z - z.mean())**2).sum()
            r2     = 1 - ss_res/ss_tot if ss_tot > 1e-10 else 0
            return gx, gy, mag, r2, r2 >= MIN_R_SQUARED
        except Exception:
            return 0, 0, 0, 0, False


class WeightedLeastSquares(GradientMethod):
    """Weighted least squares -- closer points have more influence.
    Weights = 1 / (dist² + 0.01), normalised.
    Matches gradient_fitting_analyzer.py WeightedLeastSquares exactly."""

    def __init__(self):
        super().__init__(
            "Weighted Least Squares",
            "Weights points by inverse distance² from center"
        )

    def estimate(self, center_x, center_y, x, y, z):
        if len(x) < MIN_NEIGHBOR_COUNT:
            return 0, 0, 0, 0, False
        dx      = x - center_x
        dy      = y - center_y
        dist    = np.sqrt(dx**2 + dy**2)
        weights = 1.0 / (dist**2 + 0.01)
        weights = weights / weights.sum()
        W    = np.diag(weights)
        A    = np.column_stack([x, y, np.ones(len(x))])
        try:
            ATWA   = A.T @ W @ A
            ATWz   = A.T @ W @ z
            coeffs = np.linalg.solve(ATWA, ATWz)
            gx, gy = coeffs[0], coeffs[1]
            mag    = np.sqrt(gx**2 + gy**2)
            pred   = A @ coeffs
            ss_res = np.sum(weights * (z - pred)**2)
            ss_tot = np.sum(weights * (z - np.mean(z))**2)
            r2     = 1 - ss_res/ss_tot if ss_tot > 1e-10 else 0
            return gx, gy, mag, r2, r2 >= MIN_R_SQUARED
        except Exception:
            return 0, 0, 0, 0, False


class RobustRANSAC(GradientMethod):
    """RANSAC plane fit -- robust to outlier measurements.
    Matches gradient_fitting_analyzer.py RobustRANSAC."""

    def __init__(self):
        super().__init__(
            "RANSAC Plane Fit",
            "Random sample consensus -- robust to outlier measurements"
        )

    def estimate(self, center_x, center_y, x, y, z):
        if len(x) < MIN_NEIGHBOR_COUNT:
            return 0, 0, 0, 0, False
        n_pts       = len(x)
        n_iter      = min(50, n_pts * 2)
        threshold   = np.std(z) * 0.5
        best_n      = 0
        best_coeffs = None
        try:
            for _ in range(n_iter):
                idx    = np.random.choice(n_pts, 3, replace=False)
                A_s    = np.column_stack([x[idx], y[idx], np.ones(3)])
                try:
                    c = np.linalg.solve(A_s, z[idx])
                except Exception:
                    continue
                A_all  = np.column_stack([x, y, np.ones(n_pts)])
                errs   = np.abs(z - A_all @ c)
                n_in   = int((errs < threshold).sum())
                if n_in > best_n:
                    best_n, best_coeffs = n_in, c
            if best_coeffs is None:
                return 0, 0, 0, 0, False
            gx, gy = best_coeffs[0], best_coeffs[1]
            mag    = np.sqrt(gx**2 + gy**2)
            A_all  = np.column_stack([x, y, np.ones(n_pts)])
            pred   = A_all @ best_coeffs
            ss_res = ((z - pred)**2).sum()
            ss_tot = ((z - z.mean())**2).sum()
            r2     = 1 - ss_res/ss_tot if ss_tot > 1e-10 else 0
            return gx, gy, mag, r2, r2 >= MIN_R_SQUARED
        except Exception:
            return 0, 0, 0, 0, False


class LocalPolynomialFit(GradientMethod):
    """2nd-order polynomial surface fit.
    Matches gradient_fitting_analyzer.py LocalPolynomialFit."""

    def __init__(self):
        super().__init__(
            "Polynomial Surface",
            "Fits z=ax+by+cx²+dy²+exy+f -- captures field curvature"
        )

    def estimate(self, center_x, center_y, x, y, z):
        if len(x) < 6:
            return 0, 0, 0, 0, False
        try:
            A      = np.column_stack([x, y, x**2, y**2, x*y, np.ones(len(x))])
            coeffs = np.linalg.lstsq(A, z, rcond=None)[0]
            gx     = coeffs[0] + 2*coeffs[2]*center_x + coeffs[4]*center_y
            gy     = coeffs[1] + 2*coeffs[3]*center_y + coeffs[4]*center_x
            mag    = np.sqrt(gx**2 + gy**2)
            pred   = A @ coeffs
            ss_res = ((z - pred)**2).sum()
            ss_tot = ((z - z.mean())**2).sum()
            r2     = 1 - ss_res/ss_tot if ss_tot > 1e-10 else 0
            return gx, gy, mag, r2, r2 >= MIN_R_SQUARED
        except Exception:
            return 0, 0, 0, 0, False


class FiniteDifferenceGradient(GradientMethod):
    """Finite-difference approximation using axially-aligned neighbours.
    Matches gradient_fitting_analyzer.py FiniteDifferenceGradient."""

    def __init__(self):
        super().__init__(
            "Finite Difference",
            "dz/dx and dz/dy from nearest neighbours in each axis direction"
        )

    def estimate(self, center_x, center_y, x, y, z):
        if len(x) < 4:
            return 0, 0, 0, 0, False
        try:
            dx = x - center_x
            dy = y - center_y
            xp = (dx > 0) & (np.abs(dy) < np.abs(dx))
            xn = (dx < 0) & (np.abs(dy) < np.abs(dx))
            yp = (dy > 0) & (np.abs(dx) < np.abs(dy))
            yn = (dy < 0) & (np.abs(dx) < np.abs(dy))
            gx = gy = 0.0
            if xp.any() and xn.any():
                dxavg = np.abs(dx[xp]).mean() + np.abs(dx[xn]).mean()
                gx = (z[xp].mean() - z[xn].mean()) / dxavg if dxavg > 0 else 0
            if yp.any() and yn.any():
                dyavg = np.abs(dy[yp]).mean() + np.abs(dy[yn]).mean()
                gy = (z[yp].mean() - z[yn].mean()) / dyavg if dyavg > 0 else 0
            mag = np.sqrt(gx**2 + gy**2)
            return gx, gy, mag, 0.5, mag > 0   # finite diff has no natural R²
        except Exception:
            return 0, 0, 0, 0, False


# ---------------------------------------------------------------------------
# Active method -- change this one line to swap the estimation algorithm.
# Options: LeastSquaresPlaneFit  WeightedLeastSquares  RobustRANSAC
#          LocalPolynomialFit    FiniteDifferenceGradient
# ---------------------------------------------------------------------------
GRADIENT_METHOD = WeightedLeastSquares()


def estimate_at_point(cx, cy, pts_x, pts_y, pts_z):
    """
    Estimate gradient at (cx, cy) using the given point cloud.
    Applies neighbourhood radius and minimum-distance filter, then delegates
    to GRADIENT_METHOD so the algorithm can be swapped from one place.
    Returns (gx, gy, mag, r2, success, n_used).
    """
    if len(pts_x) == 0:
        return 0.0, 0.0, 0.0, 0.0, False, 0

    dists = np.sqrt((pts_x - cx)**2 + (pts_y - cy)**2)
    mask  = (dists >= MIN_NEIGHBOR_DISTANCE) & (dists <= MAX_NEIGHBOR_DISTANCE)
    n_used = int(mask.sum())

    if n_used < MIN_NEIGHBOR_COUNT:
        return 0.0, 0.0, 0.0, 0.0, False, n_used

    gx, gy, mag, r2, ok = GRADIENT_METHOD.estimate(
        cx, cy, pts_x[mask], pts_y[mask], pts_z[mask])
    return gx, gy, mag, r2, ok, n_used

# =============================================================================
# ANGLE HELPERS
# =============================================================================

def wrap_error(est_deg, ref_deg):
    """Signed angle difference wrapped to [-180, 180]."""
    return (est_deg - ref_deg + 180.0) % 360.0 - 180.0


def angle_from_grad(gx, gy):
    return np.degrees(np.arctan2(gy, gx))

# =============================================================================
# DATA LOADING
# =============================================================================

def load_map(map_file):
    """Load precomputed_map.csv from map_supervisor_probe.cpp."""
    print(f"[INFO] Loading precomputed map: {map_file}")
    df = pd.read_csv(map_file)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Normalise column names -- probe writes total_light, fitting analyzer uses total_light
    if "total_light" not in df.columns and "light_intensity" in df.columns:
        df["total_light"] = df["light_intensity"]

    required = ["x", "y", "total_light"]
    for col in required:
        if col not in df.columns:
            print(f"[ERROR] Map file missing required column: '{col}'")
            sys.exit(1)

    df = df.dropna(subset=required)
    print(f"  Valid rows after dropna: {len(df)}")
    return df


def load_trajectory(traj_file):
    """Load fixed_fusion_trajectory.csv from blimp.cpp."""
    print(f"[INFO] Loading trajectory: {traj_file}")
    df = pd.read_csv(traj_file)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    required = ["time", "x", "y"]
    for col in required:
        if col not in df.columns:
            print(f"[ERROR] Trajectory file missing required column: '{col}'")
            sys.exit(1)

    df = df.dropna(subset=required).sort_values("time").reset_index(drop=True)
    print(f"  Time range: {df['time'].min():.2f}s -- {df['time'].max():.2f}s")
    return df

# =============================================================================
# CORE ANALYSIS
# =============================================================================

def run_analysis(map_df, traj_df):
    """
    For each point along the trajectory, estimate the gradient two ways:

      full_map  -- using ALL points in precomputed_map within the neighbourhood
      path_only -- using only the map cells the blimp has visited so far
                   (i.e. the subset of precomputed_map cells that fall within
                    0.5*GRID_RESOLUTION of any previously visited trajectory point)

    The path-only set grows step-by-step as the blimp moves, mimicking exactly
    what the blimp's real-time map contains.

    Returns a dict of per-step result arrays.
    """
    print("\n[INFO] Running gradient analysis along trajectory...")

    n = len(traj_df)
    map_x = map_df["x"].values
    map_y = map_df["y"].values
    map_z = map_df["total_light"].values

    # KD-tree for full map lookups
    map_tree = cKDTree(np.column_stack([map_x, map_y]))

    # Results arrays
    res = {
        # Full-map estimates
        "full_gx":  np.zeros(n), "full_gy":  np.zeros(n),
        "full_mag": np.zeros(n), "full_r2":  np.zeros(n),
        "full_ok":  np.zeros(n, dtype=bool), "full_n": np.zeros(n, dtype=int),
        "full_angle": np.zeros(n),

        # Path-only estimates
        "path_gx":  np.zeros(n), "path_gy":  np.zeros(n),
        "path_mag": np.zeros(n), "path_r2":  np.zeros(n),
        "path_ok":  np.zeros(n, dtype=bool), "path_n": np.zeros(n, dtype=int),
        "path_angle": np.zeros(n),

        # Coverage fraction: path_n / full_n at each step
        "coverage_frac": np.zeros(n),

        # Angle errors vs full-map ground truth
        "path_angle_err": np.full(n, np.nan),   # path_only vs full_map
        "blimp_angle_err": np.full(n, np.nan),  # blimp real-time vs full_map

        # Blimp phase (0=bearing-only, 1=fused) -- derived from trajectory log
        "phase": np.zeros(n, dtype=int),
    }

    # Track which map cells are "seen" so far along the path
    seen_cell_indices = set()
    snap_radius = GRID_RESOLUTION * 0.71  # diagonal of one grid cell

    for i, row in traj_df.iterrows():
        cx, cy = row["x"], row["y"]

        # ── Accumulate path-only map: add map cells near current position ──
        new_idxs = map_tree.query_ball_point([cx, cy], snap_radius)
        seen_cell_indices.update(new_idxs)

        # ── Full-map gradient estimate ────────────────────────────────────
        f_gx, f_gy, f_mag, f_r2, f_ok, f_n = estimate_at_point(
            cx, cy, map_x, map_y, map_z)
        res["full_gx"][i]    = f_gx
        res["full_gy"][i]    = f_gy
        res["full_mag"][i]   = f_mag
        res["full_r2"][i]    = f_r2
        res["full_ok"][i]    = f_ok
        res["full_n"][i]     = f_n
        res["full_angle"][i] = angle_from_grad(f_gx, f_gy) if f_ok else np.nan

        # ── Path-only gradient estimate ───────────────────────────────────
        if len(seen_cell_indices) >= MIN_NEIGHBOR_COUNT:
            idx = np.array(sorted(seen_cell_indices))
            p_gx, p_gy, p_mag, p_r2, p_ok, p_n = estimate_at_point(
                cx, cy,
                map_x[idx], map_y[idx], map_z[idx])
        else:
            p_gx = p_gy = p_mag = p_r2 = 0.0
            p_ok = False
            p_n  = len(seen_cell_indices)

        res["path_gx"][i]    = p_gx
        res["path_gy"][i]    = p_gy
        res["path_mag"][i]   = p_mag
        res["path_r2"][i]    = p_r2
        res["path_ok"][i]    = p_ok
        res["path_n"][i]     = p_n
        res["path_angle"][i] = angle_from_grad(p_gx, p_gy) if p_ok else np.nan

        # ── Coverage fraction ─────────────────────────────────────────────
        res["coverage_frac"][i] = p_n / f_n if f_n > 0 else 0.0

        # ── Angle errors vs full-map ground truth ─────────────────────────
        if f_ok and p_ok:
            res["path_angle_err"][i] = wrap_error(
                angle_from_grad(p_gx, p_gy),
                angle_from_grad(f_gx, f_gy))

        # Blimp's own real-time estimate vs full-map ground truth
        if "grad_angle" in traj_df.columns and "grad_valid" in traj_df.columns:
            gv = traj_df.at[i, "grad_valid"]
            if gv and f_ok:
                res["blimp_angle_err"][i] = wrap_error(
                    traj_df.at[i, "grad_angle"],
                    angle_from_grad(f_gx, f_gy))

        # ── Phase from trajectory log ─────────────────────────────────────
        if "grad_weight" in traj_df.columns:
            res["phase"][i] = 1 if traj_df.at[i, "grad_weight"] > 0 else 0
        elif "map_size" in traj_df.columns and "grad_mag" in traj_df.columns:
            ms = traj_df.at[i, "map_size"]
            gm = traj_df.at[i, "grad_mag"]
            res["phase"][i] = 1 if (ms >= MIN_MAP_POINTS and gm >= GRADIENT_THRESHOLD) else 0

        if (i + 1) % 100 == 0:
            print(f"  Step {i+1}/{n}  seen={len(seen_cell_indices)} cells  "
                  f"path_ok={res['path_ok'][i]}  full_ok={res['full_ok'][i]}")

    print(f"[INFO] Analysis complete.")
    print(f"  Full-map estimates valid:  "
          f"{res['full_ok'].sum()}/{n} ({res['full_ok'].mean()*100:.1f}%)")
    print(f"  Path-only estimates valid: "
          f"{res['path_ok'].sum()}/{n} ({res['path_ok'].mean()*100:.1f}%)")
    print(f"  Mean coverage fraction:    "
          f"{res['coverage_frac'].mean()*100:.1f}%")

    return res

# =============================================================================
# PLOT 1 -- Spatial coverage overview (4 panels)
# =============================================================================

def plot_coverage_map(map_df, traj_df, res, save=True):
    print("\n[INFO] Generating coverage map plot...")

    n  = len(traj_df)
    tx = traj_df["x"].values
    ty = traj_df["y"].values
    t  = traj_df["time"].values

    map_x = map_df["x"].values
    map_y = map_df["y"].values
    map_z = map_df["total_light"].values

    fig = plt.figure(figsize=(18, 16))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28,
                            left=0.06, right=0.97, top=0.92, bottom=0.05)

    # ── Panel TL: Full probe map + trajectory ─────────────────────────────
    ax = fig.add_subplot(gs[0, 0])

    sc = ax.scatter(map_x, map_y, c=map_z, cmap="YlOrRd",
                    s=8, alpha=0.7, zorder=2)
    plt.colorbar(sc, ax=ax, label="Light Intensity (total_light)", fraction=0.035)

    ax.plot(tx, ty, color="black", lw=1.2, alpha=0.6, zorder=3, label="Flightpath")
    ax.scatter(tx[0],  ty[0],  color="green",  s=80, marker="^", zorder=5, label="Start")
    ax.scatter(tx[-1], ty[-1], color="red",    s=80, marker="s", zorder=5, label="End")
    ax.scatter(LIGHT_SOURCE_X, LIGHT_SOURCE_Y,
               color=COL_SOURCE, s=200, marker="*",
               edgecolors="black", lw=0.8, zorder=6, label="Light source")

    # Full-map gradient arrows
    step = max(1, n // (n // SUBSAMPLE_ARROWS + 1))
    for i in range(0, n, step):
        if res["full_ok"][i]:
            fa = res["full_angle"][i]
            ax.annotate("",
                xy=(tx[i] + ARROW_LENGTH*np.cos(np.radians(fa)),
                    ty[i] + ARROW_LENGTH*np.sin(np.radians(fa))),
                xytext=(tx[i], ty[i]),
                arrowprops=dict(arrowstyle="-|>", color=COL_FULL,
                                lw=1.2, mutation_scale=8), zorder=4)

    ax.set_title("Full Probe Map + Ground-Truth Gradient (blue arrows)",
                 fontweight="bold")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85)

    # ── Panel TR: Path-only coverage ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])

    # Background: full map (faint)
    ax.scatter(map_x, map_y, c="lightgrey", s=6, alpha=0.4, zorder=1)

    # Colour trajectory by coverage fraction
    cmap_cov = plt.cm.RdYlGn
    norm_cov  = Normalize(vmin=0, vmax=1)
    for i in range(n - 1):
        ax.plot([tx[i], tx[i+1]], [ty[i], ty[i+1]],
                color=cmap_cov(norm_cov(res["coverage_frac"][i])),
                lw=2.0, alpha=0.85, zorder=3)

    sm = ScalarMappable(cmap=cmap_cov, norm=norm_cov)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Coverage fraction (path neighbours / full neighbours)",
                 fraction=0.035)

    ax.scatter(LIGHT_SOURCE_X, LIGHT_SOURCE_Y,
               color=COL_SOURCE, s=200, marker="*",
               edgecolors="black", lw=0.8, zorder=6)
    ax.set_title("Neighbourhood Coverage Fraction Along Path\n"
                 "(green = full coverage, red = sparse coverage)",
                 fontweight="bold")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

    # ── Panel BL: Path-only gradient vs full-map gradient ─────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(map_x, map_y, c=map_z, cmap="YlOrRd", s=6, alpha=0.35, zorder=1)
    ax.plot(tx, ty, color="grey", lw=0.8, alpha=0.5, zorder=2)

    # Colour arrows by absolute angle error
    abs_err = np.abs(res["path_angle_err"])
    err_norm = Normalize(vmin=0, vmax=90)
    cmap_err = plt.cm.RdYlGn_r

    for i in range(0, n, step):
        # Full-map arrow (blue, always)
        if res["full_ok"][i]:
            fa = res["full_angle"][i]
            ax.annotate("",
                xy=(tx[i] + ARROW_LENGTH*np.cos(np.radians(fa)),
                    ty[i] + ARROW_LENGTH*np.sin(np.radians(fa))),
                xytext=(tx[i], ty[i]),
                arrowprops=dict(arrowstyle="-|>", color=COL_FULL,
                                lw=1.2, mutation_scale=7, alpha=0.7), zorder=4)

        # Path-only arrow (coloured by error)
        if res["path_ok"][i] and not np.isnan(res["path_angle_err"][i]):
            pa  = res["path_angle"][i]
            col = cmap_err(err_norm(abs_err[i]))
            ax.annotate("",
                xy=(tx[i] + ARROW_LENGTH*np.cos(np.radians(pa)),
                    ty[i] + ARROW_LENGTH*np.sin(np.radians(pa))),
                xytext=(tx[i], ty[i]),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=1.4, mutation_scale=8), zorder=5)
        elif res["path_ok"][i]:
            pa = res["path_angle"][i]
            ax.annotate("",
                xy=(tx[i] + ARROW_LENGTH*np.cos(np.radians(pa)),
                    ty[i] + ARROW_LENGTH*np.sin(np.radians(pa))),
                xytext=(tx[i], ty[i]),
                arrowprops=dict(arrowstyle="-|>", color=COL_PATH,
                                lw=1.4, mutation_scale=8), zorder=5)

    sm2 = ScalarMappable(cmap=cmap_err, norm=err_norm)
    sm2.set_array([])
    plt.colorbar(sm2, ax=ax, label="|Angle error| vs full-map (°)", fraction=0.035)

    ax.scatter(LIGHT_SOURCE_X, LIGHT_SOURCE_Y,
               color=COL_SOURCE, s=200, marker="*",
               edgecolors="black", lw=0.8, zorder=6)

    extra = [Line2D([0],[0], color=COL_FULL, lw=2, label="Full-map gradient"),
             Line2D([0],[0], color=COL_PATH, lw=2, label="Path-only gradient (coloured by error)")]
    ax.legend(handles=extra, fontsize=7.5, loc="upper left", framealpha=0.85)
    ax.set_title("Path-Only Gradient vs Full-Map Ground Truth\n"
                 "(path arrow coloured green=accurate, red=large error)",
                 fontweight="bold")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

    # ── Panel BR: Neighbour counts ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.fill_between(t, res["full_n"], 0, alpha=0.2, color=COL_FULL,
                    label="Full-map neighbours")
    ax.plot(t, res["full_n"], color=COL_FULL, lw=2.0, label="Full-map neighbours")
    ax.fill_between(t, res["path_n"], 0, alpha=0.2, color=COL_PATH,
                    label="Path-only neighbours")
    ax.plot(t, res["path_n"], color=COL_PATH, lw=2.0, label="Path-only neighbours")
    ax.axhline(MIN_NEIGHBOR_COUNT, color="orange", lw=1.5, ls="--",
               label=f"Min neighbours ({MIN_NEIGHBOR_COUNT})")

    if "map_size" in traj_df.columns:
        ax.plot(t, traj_df["map_size"].values, color=COL_BLIMP, lw=1.4,
                ls="-.", alpha=0.8, label="Blimp map_size (from log)")

    ax.set_xlabel("Time (s)"); ax.set_ylabel("Neighbour count in neighbourhood")
    ax.set_title("Available Neighbours in Local Neighbourhood Over Time",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Flightpath Gradient Coverage Analysis  —  "
        f"Path-Only Map vs Full Probe Map  "
        f"[{GRADIENT_METHOD.name}]",
        fontsize=13, fontweight="bold", y=0.97)

    if save:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        out = str(Path(OUTPUT_DIR) / "flightpath_coverage_map.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[SUCCESS] Saved -> {out}")
    return fig

# =============================================================================
# PLOT 2 -- Time-series timeline (5 panels)
# =============================================================================

def plot_timeline(traj_df, res, save=True):
    print("[INFO] Generating timeline plot...")

    n  = len(traj_df)
    t  = traj_df["time"].values

    phase   = res["phase"]
    fused_t = t[phase == 1][0] if (phase == 1).any() else t[-1]

    fig = plt.figure(figsize=(18, 18))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(5, 1, figure=fig, hspace=0.42,
                            left=0.07, right=0.97, top=0.94, bottom=0.04)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    def shade(ax):
        ax.axvspan(t[0],    fused_t, alpha=0.05, color=COL_BEARING, zorder=0)
        ax.axvspan(fused_t, t[-1],   alpha=0.05, color=COL_FUSED,   zorder=0)
        ax.axvline(fused_t, color="black", lw=1.2, ls="--", alpha=0.5)

    # ── Panel 1: Gradient angle traces ────────────────────────────────────
    ax = axes[0]
    ax.plot(t, res["full_angle"],  color=COL_FULL,  lw=2.0,
            label="Full-map gradient (ground truth)", zorder=4)
    ax.plot(t, res["path_angle"],  color=COL_PATH,  lw=1.6, alpha=0.85,
            label="Path-only gradient estimate", zorder=5)

    if "grad_angle" in traj_df.columns and "grad_valid" in traj_df.columns:
        blimp_angle = np.where(traj_df["grad_valid"].astype(bool),
                               traj_df["grad_angle"].values, np.nan)
        ax.plot(t, blimp_angle, color=COL_BLIMP, lw=1.2, alpha=0.7,
                ls="-.", label="Blimp real-time gradient (from log)", zorder=3)

    if "bearing_angle" in traj_df.columns:
        ax.plot(t, traj_df["bearing_angle"].values, color=COL_BEARING,
                lw=1.0, alpha=0.5, ls=":", label="Bearing direction", zorder=2)

    shade(ax)
    ax.set_ylim(-185, 185)
    ax.set_ylabel("Angle (°)", fontsize=10)
    ax.set_title("Gradient Direction: Full Map vs Path-Only vs Blimp Real-Time",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85, ncol=2)
    ax.grid(True, alpha=0.25)
    _phase_labels(ax, t[0], fused_t, t[-1])

    # ── Panel 2: Gradient magnitude ───────────────────────────────────────
    ax = axes[1]
    ax.plot(t, res["full_mag"],  color=COL_FULL,  lw=2.0,
            label="Full-map gradient magnitude")
    ax.plot(t, res["path_mag"],  color=COL_PATH,  lw=1.6, alpha=0.85,
            label="Path-only gradient magnitude")
    if "grad_mag" in traj_df.columns:
        ax.plot(t, traj_df["grad_mag"].values, color=COL_BLIMP,
                lw=1.2, alpha=0.7, ls="-.",
                label="Blimp real-time magnitude (from log)")
    ax.axhline(GRADIENT_THRESHOLD, color="orange", lw=1.3, ls="--",
               label=f"blimp.cpp GRADIENT_THRESHOLD ({GRADIENT_THRESHOLD})")
    shade(ax)
    ax.set_ylabel("Gradient Magnitude", fontsize=10)
    ax.set_title("Gradient Magnitude: Full Map vs Path-Only",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.25)

    # ── Panel 3: R² quality ───────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, res["full_r2"],   color=COL_FULL,  lw=2.0,
            label="Full-map plane fit R²")
    ax.plot(t, res["path_r2"],   color=COL_PATH,  lw=1.6, alpha=0.85,
            label="Path-only plane fit R²")
    ax.axhline(MIN_R_SQUARED, color="orange", lw=1.3, ls="--",
               label=f"MIN_R_SQUARED threshold ({MIN_R_SQUARED})")
    ax.set_ylim(-0.05, 1.1)
    shade(ax)
    ax.set_ylabel("R² (plane fit quality)", fontsize=10)
    ax.set_title("Plane Fit Quality: Full Map vs Path-Only",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
    ax.grid(True, alpha=0.25)

    # ── Panel 4: Angle error of path-only vs ground truth ─────────────────
    ax = axes[3]

    ae = res["path_angle_err"]

    # Colour scatter by phase
    bearing_mask = (phase == 0) & ~np.isnan(ae)
    fused_mask   = (phase == 1) & ~np.isnan(ae)
    ax.scatter(t[bearing_mask], ae[bearing_mask],
               color=COL_BEARING, s=5, alpha=0.5, label="Bearing-only phase", zorder=3)
    ax.scatter(t[fused_mask], ae[fused_mask],
               color=COL_FUSED, s=5, alpha=0.5, label="Fused phase", zorder=3)

    # Rolling mean
    win  = 30
    roll = np.full(n, np.nan)
    abs_ae = np.abs(ae)
    for i in range(win, n):
        w = abs_ae[i-win:i]
        w = w[~np.isnan(w)]
        if len(w): roll[i] = w.mean()
    ax.plot(t, roll, color="black", lw=2.0, zorder=6,
            label="Rolling |error| mean (n=30)")

    ax.axhline( 30, color="orange", lw=1.0, ls="--", alpha=0.5, label="±30°")
    ax.axhline(-30, color="orange", lw=1.0, ls="--", alpha=0.5)
    ax.axhline(  0, color="green",  lw=1.0, ls="--", alpha=0.4)
    ax.set_ylim(-185, 185)
    shade(ax)
    ax.set_ylabel("Angle Error (°)", fontsize=10)
    ax.set_title("Path-Only Gradient Error vs Full-Map Ground Truth\n"
                 "(coloured by blimp phase: red=bearing-only, green=fused)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85, ncol=2)
    ax.grid(True, alpha=0.25)

    # ── Panel 5: Neighbour counts + coverage fraction ─────────────────────
    ax = axes[4]
    ax.plot(t, res["full_n"], color=COL_FULL, lw=2.0,
            label="Full-map neighbours in radius")
    ax.plot(t, res["path_n"], color=COL_PATH, lw=2.0,
            label="Path-only neighbours in radius")
    ax.axhline(MIN_NEIGHBOR_COUNT, color="orange", lw=1.3, ls="--",
               label=f"Minimum required ({MIN_NEIGHBOR_COUNT})")

    ax_r = ax.twinx()
    ax_r.fill_between(t, res["coverage_frac"]*100, 0,
                      alpha=0.15, color="#9C27B0")
    ax_r.plot(t, res["coverage_frac"]*100, color="#9C27B0",
              lw=1.4, label="Coverage %")
    ax_r.set_ylim(-2, 110)
    ax_r.set_ylabel("Coverage fraction (%)", fontsize=9, color="#9C27B0")
    ax_r.tick_params(axis="y", labelcolor="#9C27B0")

    shade(ax)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Neighbours in neighbourhood", fontsize=10)
    ax.set_title("Neighbourhood Fill: How Many of the Full Map's Neighbours "
                 "Are Available on the Path",
                 fontweight="bold")
    l1, b1 = ax.get_legend_handles_labels()
    l2, b2 = ax_r.get_legend_handles_labels()
    ax.legend(l1+l2, b1+b2, fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Flightpath Gradient Timeline  —  Full Map vs Path-Only Coverage  "
        f"[{GRADIENT_METHOD.name}]",
        fontsize=13, fontweight="bold", y=0.97)

    if save:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        out = str(Path(OUTPUT_DIR) / "flightpath_gradient_timeline.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[SUCCESS] Saved -> {out}")
    return fig


def _phase_labels(ax, t0, tt, t1):
    ax.text((t0+tt)/2, 165, "BEARING ONLY", color=COL_BEARING,
            fontsize=8, fontweight="bold", ha="center", alpha=0.7)
    if tt < t1:
        ax.text((tt+t1)/2, 165, "FUSED", color=COL_FUSED,
                fontsize=8, fontweight="bold", ha="center", alpha=0.7)

# =============================================================================
# PLOT 3 -- Distribution analysis
# =============================================================================

def plot_quality(traj_df, res, save=True):
    print("[INFO] Generating quality distribution plot...")

    n     = len(traj_df)
    phase = res["phase"]
    ae    = res["path_angle_err"]
    cov   = res["coverage_frac"]

    valid = ~np.isnan(ae)
    ae_bearing = np.abs(ae[valid & (phase == 0)])
    ae_fused   = np.abs(ae[valid & (phase == 1)])

    # Split by coverage tercile
    cov_vals = cov[valid]
    if len(cov_vals):
        t33 = np.percentile(cov_vals, 33)
        t66 = np.percentile(cov_vals, 66)
        ae_lo  = np.abs(ae[valid & (cov <= t33)])
        ae_mid = np.abs(ae[valid & (cov > t33) & (cov <= t66)])
        ae_hi  = np.abs(ae[valid & (cov > t66)])
    else:
        ae_lo = ae_mid = ae_hi = np.array([])

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("white")
    bins = np.linspace(0, 180, 37)

    # ── Left: Histogram by phase ───────────────────────────────────────────
    ax = axes[0]
    for data, color, label in [
        (ae_bearing, COL_BEARING, f"Bearing-only phase (n={len(ae_bearing)})"),
        (ae_fused,   COL_FUSED,   f"Fused phase (n={len(ae_fused)})"),
    ]:
        if len(data):
            ax.hist(data, bins=bins, alpha=0.6, color=color,
                    label=label, edgecolor="white")
    ax.axvline(30, color="orange", lw=1.5, ls="--", label="30° threshold")
    ax.axvline(10, color="green",  lw=1.5, ls="--", label="10° threshold")
    ax.set_xlabel("|Angle error| vs full map (°)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Error Distribution by Flight Phase", fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3, axis="y")

    def box(data, color, xpos, ax):
        if not len(data): return
        txt = (f"Mean:  {np.mean(data):.1f}°\n"
               f"Median:{np.median(data):.1f}°\n"
               f"<30°:  {(data<30).mean()*100:.0f}%")
        ax.text(xpos, ax.get_ylim()[1]*0.72, txt, fontsize=8, color=color,
                bbox=dict(boxstyle="round", facecolor="white",
                          edgecolor=color, alpha=0.88))

    box(ae_bearing, COL_BEARING, 110, ax)
    box(ae_fused,   COL_FUSED,   60,  ax)

    # ── Middle: Histogram by coverage fraction ─────────────────────────────
    ax = axes[1]
    if len(ae_lo):
        for data, color, label in [
            (ae_lo,  "#F44336", f"Low coverage (<{t33*100:.0f}%, n={len(ae_lo)})"),
            (ae_mid, "#FF9800", f"Mid coverage ({t33*100:.0f}-{t66*100:.0f}%, n={len(ae_mid)})"),
            (ae_hi,  "#4CAF50", f"High coverage (>{t66*100:.0f}%, n={len(ae_hi)})"),
        ]:
            if len(data):
                ax.hist(data, bins=bins, alpha=0.6, color=color,
                        label=label, edgecolor="white")
    ax.axvline(30, color="orange", lw=1.5, ls="--")
    ax.axvline(10, color="green",  lw=1.5, ls="--")
    ax.set_xlabel("|Angle error| vs full map (°)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Error Distribution by Neighbourhood Coverage Fraction",
                 fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3, axis="y")

    # ── Right: CDF ──────────────────────────────────────────────────────────
    ax = axes[2]
    all_valid = np.abs(ae[valid])
    for data, color, label in [
        (ae_bearing, COL_BEARING, "Bearing-only phase"),
        (ae_fused,   COL_FUSED,   "Fused phase"),
        (all_valid,  "black",     "All valid estimates"),
    ]:
        if not len(data): continue
        xs = np.sort(data)
        ys = np.arange(1, len(xs)+1) / len(xs) * 100
        ax.plot(xs, ys, color=color, lw=2.5, label=label)

    ax.axvline(30, color="orange", lw=1.5, ls="--", label="30°")
    ax.axvline(10, color="green",  lw=1.5, ls="--", label="10°")
    ax.axhline(90, color="grey",   lw=1.0, ls=":", alpha=0.6)
    ax.text(32, 91, "90th pctile", fontsize=8, color="grey")
    ax.set_xlabel("|Angle error| vs full map (°)", fontsize=11)
    ax.set_ylabel("Cumulative % of estimates", fontsize=11)
    ax.set_title("CDF of Path-Only Gradient Errors\nvs Full-Map Ground Truth",
                 fontweight="bold")
    ax.set_xlim(0, 180); ax.set_ylim(0, 101)
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Path-Only Gradient Estimation Quality  —  Error vs Full Probe Map Ground Truth  "
        f"[{GRADIENT_METHOD.name}]",
        fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        out = str(Path(OUTPUT_DIR) / "flightpath_gradient_quality.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[SUCCESS] Saved -> {out}")
    return fig

# =============================================================================
# TEXT REPORT
# =============================================================================

def generate_report(map_df, traj_df, res, map_file, traj_file):
    print("[INFO] Generating report...")
    n     = len(traj_df)
    phase = res["phase"]
    ae    = res["path_angle_err"]
    valid = ~np.isnan(ae)

    def stats(data):
        if not len(data): return "    no valid estimates"
        return (f"    n:            {len(data)}\n"
                f"    Mean |error|: {np.mean(data):.2f}°\n"
                f"    Median:       {np.median(data):.2f}°\n"
                f"    90th pctile:  {np.percentile(data, 90):.2f}°\n"
                f"    Within 10°:   {(data<10).mean()*100:.1f}%\n"
                f"    Within 30°:   {(data<30).mean()*100:.1f}%")

    ae_b = np.abs(ae[valid & (phase == 0)])
    ae_f = np.abs(ae[valid & (phase == 1)])

    lines = [
        "=" * 70,
        "FLIGHTPATH GRADIENT COVERAGE ANALYSIS REPORT",
        "=" * 70,
        "",
        "INPUT FILES:",
        f"  Precomputed map  : {map_file}  ({len(map_df)} cells)",
        f"  Trajectory log   : {traj_file}  ({n} timesteps)",
        "",
        "CONFIGURATION:",
        f"  Estimation method    : {GRADIENT_METHOD.name} -- {GRADIENT_METHOD.description}",
        f"  Neighbourhood radius : {MAX_NEIGHBOR_DISTANCE} m",
        f"  Min neighbours       : {MIN_NEIGHBOR_COUNT}",
        f"  Min R²               : {MIN_R_SQUARED}",
        f"  Grid resolution      : {GRID_RESOLUTION} m",
        "",
        "FULL-MAP GRADIENT (ground truth from complete probe scan):",
        f"  Valid estimates : {res['full_ok'].sum()}/{n} "
        f"({res['full_ok'].mean()*100:.1f}%)",
        f"  Mean neighbours : {res['full_n'].mean():.1f}",
        "",
        "PATH-ONLY GRADIENT (only cells visited along flight path):",
        f"  Valid estimates      : {res['path_ok'].sum()}/{n} "
        f"({res['path_ok'].mean()*100:.1f}%)",
        f"  Mean neighbours      : {res['path_n'].mean():.1f}",
        f"  Mean coverage frac   : {res['coverage_frac'].mean()*100:.1f}%",
        f"  Min coverage frac    : {res['coverage_frac'].min()*100:.1f}%",
        f"  Max coverage frac    : {res['coverage_frac'].max()*100:.1f}%",
        "",
        "ANGLE ERROR (path-only vs full-map ground truth):",
        "",
        "  ALL valid estimates:",
        stats(np.abs(ae[valid])),
        "",
        "  BEARING-ONLY phase:",
        stats(ae_b),
        "",
        "  FUSED phase:",
        stats(ae_f),
        "",
        "INTERPRETATION:",
        "-" * 70,
    ]

    if len(ae_b) and len(ae_f):
        imp = np.mean(ae_b) - np.mean(ae_f)
        lines.append(f"  Accuracy change at fusion: {imp:+.1f}° mean error")
        if np.mean(ae_f) < 15:
            lines.append("  EXCELLENT: Path-only gradient is accurate (<15° mean)")
        elif np.mean(ae_f) < 30:
            lines.append("  GOOD: Path-only gradient is usable (<30° mean)")
        else:
            lines.append("  POOR: Path-only gradient has large errors (>30° mean)")
            lines.append("  -> Consider enabling EXPLORE_AMPLITUDE in blimp.cpp")
            lines.append("     to increase spatial spread of the measurement map.")

    cov_at_fusion = res["coverage_frac"][phase == 1].mean() if (phase == 1).any() else 0
    lines += [
        f"  Mean coverage during fused phase: {cov_at_fusion*100:.1f}%",
        "",
        "FILES GENERATED:",
        "  plots/flightpath_coverage_map.png       -- spatial overview",
        "  plots/flightpath_gradient_timeline.png   -- time series",
        "  plots/flightpath_gradient_quality.png    -- error distributions",
        "  plots/flightpath_report.txt              -- this report",
        "=" * 70,
    ]

    report = "\n".join(lines)
    print(report)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    out = str(Path(OUTPUT_DIR) / "flightpath_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[SUCCESS] Saved report -> {out}")

# =============================================================================
# MAIN
# =============================================================================

def find_file(candidates):
    """Return first existing file from a list of candidate paths."""
    for p in candidates:
        if Path(p).exists():
            return str(p)
    return None


def main():
    print("\n" + "=" * 70)
    print("  Flightpath Gradient Coverage Analyser")
    print("=" * 70)

    # ── Resolve input files ───────────────────────────────────────────────
    if len(sys.argv) >= 3:
        map_file  = sys.argv[1]
        traj_file = sys.argv[2]
    else:
        print("\n[INFO] No arguments given -- auto-detecting input files...")

        map_file = find_file([
            "precomputed_map.csv",
            "logs/precomputed_map.csv",
            "fixed_measurement_map.csv",
            "logs/fixed_measurement_map.csv",
        ])
        traj_file = find_file([
            "fixed_fusion_trajectory.csv",
            "logs/fixed_fusion_trajectory.csv",
            "trajectory.csv",
            "logs/trajectory.csv",
            "trajectory_debug.csv",
            "logs/trajectory_debug.csv",
        ])

        if not map_file:
            print("[ERROR] Could not find precomputed_map.csv")
            print("Usage: python flightpath_gradient_analyser.py "
                  "<precomputed_map.csv> <trajectory.csv>")
            return 1
        if not traj_file:
            print("[ERROR] Could not find trajectory CSV")
            print("Usage: python flightpath_gradient_analyser.py "
                  "<precomputed_map.csv> <trajectory.csv>")
            return 1

        print(f"  Map file  : {map_file}")
        print(f"  Traj file : {traj_file}")

    # ── Load data ─────────────────────────────────────────────────────────
    map_df  = load_map(map_file)
    traj_df = load_trajectory(traj_file)

    # ── Run analysis ──────────────────────────────────────────────────────
    res = run_analysis(map_df, traj_df)

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_coverage_map(map_df, traj_df, res)
    plt.close("all")

    plot_timeline(traj_df, res)
    plt.close("all")

    plot_quality(traj_df, res)
    plt.close("all")

    generate_report(map_df, traj_df, res, map_file, traj_file)

    print("\n" + "=" * 70)
    print("  Analysis complete!")
    print(f"  Output: {Path(OUTPUT_DIR).resolve()}/")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    exit(main())