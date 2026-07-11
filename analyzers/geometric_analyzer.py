"""
Geometric Consistency Analyzer Module
=======================================
Tracks facial landmark geometry ratios across frames.

The core insight:
  A real face has near-constant geometric proportions regardless of
  expression or lighting — the skull is rigid. Deepfakes often show
  subtle instability in facial geometry because the generation model
  processes each frame semi-independently.

What we measure (8 geometric ratios):
  1. Bilateral symmetry score         — left/right landmark mirror similarity
  2. Inter-pupillary distance (IPD)   — should be near-constant
  3. Nose-to-chin ratio               — stable in real faces
  4. Eye aspect ratio (open/closed)   — variance pattern
  5. Face width-to-height ratio       — overall face shape consistency
  6. Jaw angle stability              — jaw landmark spread variance
  7. Brow height symmetry             — left vs right brow elevation
  8. Philtrum ratio                   — nose-to-lip / lip-to-chin

A high variance in these ratios over time = suspicious = deepfake likely.

Academic reference:
  Li et al. (2018) "Exposing DeepFake Videos By Detecting Face Warping Artifacts"
  CVPR Workshop 2019
"""

import numpy as np
from scipy.stats import variation
from typing import List, Tuple, Dict, Any


# ── landmark index groups (dlib 68-point model) ────────────────────────────────
IDX_LEFT_EYE      = list(range(36, 42))
IDX_RIGHT_EYE     = list(range(42, 48))
IDX_LEFT_BROW     = list(range(17, 22))
IDX_RIGHT_BROW    = list(range(22, 27))
IDX_NOSE_BRIDGE   = list(range(27, 31))
IDX_NOSE_TIP      = list(range(31, 36))
IDX_MOUTH         = list(range(48, 68))
IDX_JAW           = list(range(0, 17))
IDX_LEFT_PUPIL    = 37   # approximate pupil center
IDX_RIGHT_PUPIL   = 44


# ── tuneable constants ─────────────────────────────────────────────────────────
# Coefficient of Variation (std/mean) threshold for each ratio
# Below this = stable = real. Above this = unstable = suspicious.
CV_SUSPICIOUS_THRESHOLD = 0.08   # 8% coefficient of variation
MIN_FRAMES_NEEDED       = 15     # need this many frames for reliable stats


# ── geometry computation ───────────────────────────────────────────────────────

def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(p1 - p2))


def _centroid(pts: np.ndarray) -> np.ndarray:
    """Mean position of a set of landmark points."""
    return pts.mean(axis=0)


def _compute_geometry(landmarks: np.ndarray) -> Dict[str, float]:
    """
    Compute all 8 geometry ratios for one frame.

    Parameters
    ----------
    landmarks : np.ndarray, shape (68, 2)

    Returns
    -------
    dict of ratio_name -> float value
    """
    lm = landmarks

    # ── key points ─────────────────────────────────────────────────────────
    left_eye_center  = _centroid(lm[IDX_LEFT_EYE])
    right_eye_center = _centroid(lm[IDX_RIGHT_EYE])
    left_brow_center = _centroid(lm[IDX_LEFT_BROW])
    right_brow_center= _centroid(lm[IDX_RIGHT_BROW])
    nose_tip         = lm[33]    # tip of nose
    nose_bridge_top  = lm[27]    # top of nose bridge
    left_mouth       = lm[48]    # left mouth corner
    right_mouth      = lm[54]    # right mouth corner
    chin             = lm[8]     # bottom of chin
    face_left        = lm[0]     # leftmost jaw point
    face_right       = lm[16]    # rightmost jaw point

    # ── ratio 1: inter-pupillary distance (normalized by face width) ───────
    ipd        = _dist(left_eye_center, right_eye_center)
    face_width = _dist(face_left, face_right)
    ipd_ratio  = ipd / max(face_width, 1.0)

    # ── ratio 2: face height-to-width ratio ────────────────────────────────
    face_height    = _dist(nose_bridge_top, chin)
    hw_ratio       = face_height / max(face_width, 1.0)

    # ── ratio 3: nose-to-chin / face-height ───────────────────────────────
    nose_chin      = _dist(nose_tip, chin)
    nose_chin_ratio = nose_chin / max(face_height, 1.0)

    # ── ratio 4: mouth width normalized ───────────────────────────────────
    mouth_width    = _dist(left_mouth, right_mouth)
    mouth_ratio    = mouth_width / max(face_width, 1.0)

    # ── ratio 5: bilateral symmetry (left vs right eye distance to center) ─
    face_center_x  = (face_left[0] + face_right[0]) / 2.0
    left_eye_dist  = abs(left_eye_center[0]  - face_center_x)
    right_eye_dist = abs(right_eye_center[0] - face_center_x)
    symmetry_ratio = min(left_eye_dist, right_eye_dist) / max(
                         max(left_eye_dist, right_eye_dist), 1.0)

    # ── ratio 6: brow height symmetry ─────────────────────────────────────
    left_brow_h    = abs(left_brow_center[1]  - left_eye_center[1])
    right_brow_h   = abs(right_brow_center[1] - right_eye_center[1])
    brow_sym_ratio = min(left_brow_h, right_brow_h) / max(
                         max(left_brow_h, right_brow_h), 1.0)

    # ── ratio 7: jaw spread (angle / width) ───────────────────────────────
    jaw_pts        = lm[IDX_JAW]
    jaw_spread     = jaw_pts[:, 0].std() / max(face_width, 1.0)

    # ── ratio 8: philtrum ratio (nose-to-lip / lip-to-chin) ───────────────
    lip_top        = lm[51]    # top of upper lip
    nose_to_lip    = _dist(nose_tip, lip_top)
    lip_to_chin    = _dist(lip_top, chin)
    philtrum_ratio = nose_to_lip / max(lip_to_chin, 1.0)

    return {
        "ipd_ratio":       ipd_ratio,
        "hw_ratio":        hw_ratio,
        "nose_chin_ratio": nose_chin_ratio,
        "mouth_ratio":     mouth_ratio,
        "symmetry_ratio":  symmetry_ratio,
        "brow_sym_ratio":  brow_sym_ratio,
        "jaw_spread":      jaw_spread,
        "philtrum_ratio":  philtrum_ratio,
    }


# ── main class ─────────────────────────────────────────────────────────────────

class GeometricConsistencyAnalyzer:
    """
    Tracks 8 geometric ratios across frames and flags high variance.

    Example
    -------
    ga = GeometricConsistencyAnalyzer()
    for lm in all_landmarks:
        ga.update(lm)
    result = ga.summary()
    """

    def __init__(self):
        self._history: List[Dict[str, float]] = []

    def update(self, landmarks: np.ndarray):
        """
        Record geometry for one frame.

        Parameters
        ----------
        landmarks : np.ndarray, shape (68, 2)
        """
        geom = _compute_geometry(landmarks)
        self._history.append(geom)

    # ── derived metrics ────────────────────────────────────────────────────────

    def _ratio_series(self, key: str) -> np.ndarray:
        return np.array([frame[key] for frame in self._history])

    def coefficient_of_variation(self, key: str) -> float:
        """
        Coefficient of variation (std / mean) for a ratio across all frames.
        Lower = more stable = more real.
        """
        series = self._ratio_series(key)
        mean   = series.mean()
        if abs(mean) < 1e-9:
            return 0.0
        return float(series.std() / abs(mean))

    def all_cvs(self) -> Dict[str, float]:
        """Return CV for all 8 ratios."""
        if len(self._history) < 2:
            return {}
        keys = list(self._history[0].keys())
        return {k: round(self.coefficient_of_variation(k), 4) for k in keys}

    def mean_cv(self) -> float:
        """Mean CV across all 8 ratios — overall geometric stability."""
        cvs = self.all_cvs()
        if not cvs:
            return 0.0
        return round(float(np.mean(list(cvs.values()))), 4)

    def suspicious_ratios(self) -> List[str]:
        """Return names of ratios that exceed the CV threshold."""
        cvs = self.all_cvs()
        return [k for k, v in cvs.items() if v > CV_SUSPICIOUS_THRESHOLD]

    def stability_score(self) -> float:
        """
        Stability score [0, 1].
        1.0 = perfectly stable (very real).
        0.0 = wildly unstable (very fake).
        """
        mean_cv = self.mean_cv()
        # Map: 0 CV → score 1.0, 0.2+ CV → score 0.0
        score = max(0.0, 1.0 - mean_cv / 0.20)
        return round(score, 4)

    def instability_score(self) -> float:
        """Complement of stability_score. Higher = more suspicious."""
        return round(1.0 - self.stability_score(), 4)

    def ratio_timeseries(self, key: str, max_points: int = 100) -> List[float]:
        """Return a time-series for visualization (downsampled)."""
        series = self._ratio_series(key)
        if len(series) > max_points:
            idx    = np.linspace(0, len(series)-1, max_points, dtype=int)
            series = series[idx]
        return [round(float(v), 4) for v in series]

    def is_suspicious(self) -> Tuple[bool, str]:
        """Return (suspicious, reason_string)."""
        if len(self._history) < MIN_FRAMES_NEEDED:
            return False, "Geometric analysis: insufficient frames"

        mean_cv   = self.mean_cv()
        bad_ratios = self.suspicious_ratios()

        if len(bad_ratios) >= 3 or mean_cv > CV_SUSPICIOUS_THRESHOLD:
            bad_str = ", ".join(bad_ratios[:3])
            return True, (
                f"Facial geometry unstable — mean CV: {mean_cv:.3f}, "
                f"unstable ratios: {bad_str}"
            )

        return False, (
            f"Facial geometry consistent across frames — "
            f"mean CV: {mean_cv:.3f} (threshold: {CV_SUSPICIOUS_THRESHOLD})"
        )

    def summary(self) -> Dict[str, Any]:
        """Return complete summary dict for the combiner."""
        suspicious, reason = self.is_suspicious()
        cvs = self.all_cvs()
        return {
            "mean_cv":            self.mean_cv(),
            "stability_score":    self.stability_score(),
            "instability_score":  self.instability_score(),
            "ratio_cvs":          cvs,
            "suspicious_ratios":  self.suspicious_ratios(),
            "frames_analyzed":    len(self._history),
            "ipd_series":         self.ratio_timeseries("ipd_ratio"),
            "symmetry_series":    self.ratio_timeseries("symmetry_ratio"),
            "suspicious":         suspicious,
            "reason":             reason,
        }
