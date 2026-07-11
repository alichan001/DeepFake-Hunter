"""
Eye Movement Analyzer Module
=============================
Measures gaze naturalness using:
  - Spatial entropy of gaze trajectory
  - Velocity statistics (saccades vs fixations)
  - Smoothness index (real eyes have micro-tremors; deepfakes are too smooth)

Landmark indices (dlib 68-point model):
  Left eye  : 36 – 41   (iris center ≈ centroid of these 6 points)
  Right eye : 42 – 47
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from typing import List, Tuple, Dict, Any


# ── tuneable constants ─────────────────────────────────────────────────────────
MIN_ENTROPY_NATURAL   = 1.8   # bits — below this is suspiciously rigid
MIN_VELOCITY_STD      = 1.0   # pixels/frame std — deepfakes tend to be very low
SMOOTHNESS_FAKE_MAX   = 0.15  # smoothness index above this → too smooth (fake)


# ── helpers ────────────────────────────────────────────────────────────────────

def _iris_center(eye_landmarks: np.ndarray) -> np.ndarray:
    """Return the mean (x, y) of 6 eye landmark points."""
    return eye_landmarks.mean(axis=0)


def _displacement_entropy(points: np.ndarray, bins: int = 20) -> float:
    """
    Compute Shannon entropy of the 2-D displacement histogram.
    High entropy → varied natural movement.
    Low entropy  → rigid, unnatural movement.
    """
    if len(points) < 10:
        return 0.0
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    magnitudes = np.sqrt(dx**2 + dy**2)
    counts, _ = np.histogram(magnitudes, bins=bins, density=False)
    counts = counts + 1e-9   # Laplace smoothing
    probs  = counts / counts.sum()
    return float(scipy_entropy(probs, base=2))


def _smoothness_index(points: np.ndarray) -> float:
    """
    Ratio of mean second-derivative magnitude to mean velocity.
    Real eyes have irregular micro-saccades → higher ratio.
    Deepfake eyes are interpolated smoothly   → near 0.
    """
    if len(points) < 4:
        return 0.0
    vel  = np.diff(points, axis=0)
    acc  = np.diff(vel,    axis=0)
    mean_vel = np.linalg.norm(vel, axis=1).mean()
    mean_acc = np.linalg.norm(acc, axis=1).mean()
    if mean_vel < 1e-6:
        return 0.0
    return float(mean_acc / mean_vel)


# ── main class ─────────────────────────────────────────────────────────────────

class EyeMovementAnalyzer:
    """
    Stateful gaze tracker — call `update()` once per frame.

    Example
    -------
    analyzer = EyeMovementAnalyzer()
    for lm in all_landmarks:
        analyzer.update(lm)
    result = analyzer.summary()
    """

    def __init__(self):
        self.gaze_points: List[np.ndarray] = []   # shape (N, 2) over time

    # ── per-frame update ───────────────────────────────────────────────────────

    def update(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Record gaze center for this frame.

        Parameters
        ----------
        landmarks : np.ndarray, shape (68, 2)

        Returns
        -------
        gaze : np.ndarray, shape (2,)  — (x, y) gaze estimate
        """
        left_center  = _iris_center(landmarks[36:42])
        right_center = _iris_center(landmarks[42:48])
        gaze = (left_center + right_center) / 2.0
        self.gaze_points.append(gaze)
        return gaze

    # ── derived metrics ────────────────────────────────────────────────────────

    def entropy_score(self) -> float:
        """Shannon entropy of displacement magnitudes (bits)."""
        if len(self.gaze_points) < 10:
            return 0.0
        pts = np.array(self.gaze_points)
        return round(_displacement_entropy(pts), 4)

    def velocity_std(self) -> float:
        """Std of frame-to-frame gaze velocity (pixels/frame)."""
        if len(self.gaze_points) < 3:
            return 0.0
        pts  = np.array(self.gaze_points)
        vel  = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return round(float(vel.std()), 4)

    def smoothness(self) -> float:
        """
        Smoothness index — near 0 means the gaze path is artificially smooth.
        """
        if len(self.gaze_points) < 4:
            return 0.0
        pts = np.array(self.gaze_points)
        return round(_smoothness_index(pts), 4)

    def gaze_trajectory(self) -> List[List[float]]:
        """Return list of [x, y] gaze positions (for the frontend graph)."""
        return [[float(p[0]), float(p[1])] for p in self.gaze_points]

    def gaze_x_series(self) -> List[float]:
        """Horizontal gaze component over time (for the frontend graph)."""
        if not self.gaze_points:
            return []
        pts = np.array(self.gaze_points)
        # Normalize to zero mean for cleaner visualization
        x = pts[:, 0]
        return [round(float(v), 2) for v in (x - x.mean())]

    def is_suspicious(self) -> Tuple[bool, str]:
        """
        Return (suspicious, reason_string).
        Uses entropy AND smoothness for a more robust decision.
        """
        ent   = self.entropy_score()
        smooth = self.smoothness()
        vstd  = self.velocity_std()

        reasons = []
        if ent < MIN_ENTROPY_NATURAL:
            reasons.append(f"eye movement entropy {ent:.2f} bits is too low (natural ≥ {MIN_ENTROPY_NATURAL})")
        if vstd < MIN_VELOCITY_STD:
            reasons.append(f"velocity std {vstd:.2f} px/frame is too uniform")
        if smooth < SMOOTHNESS_FAKE_MAX and len(self.gaze_points) > 20:
            reasons.append(f"gaze path is suspiciously smooth (index {smooth:.3f})")

        if reasons:
            return True, "Eye movement: " + "; ".join(reasons)
        return False, f"Eye movement entropy {ent:.2f} bits — natural variation present"

    def summary(self) -> Dict[str, Any]:
        """Return a complete summary dict for the combiner."""
        suspicious, reason = self.is_suspicious()
        return {
            "entropy":       self.entropy_score(),
            "velocity_std":  self.velocity_std(),
            "smoothness":    self.smoothness(),
            "gaze_x_series": self.gaze_x_series(),
            "suspicious":    suspicious,
            "reason":        reason,
        }
