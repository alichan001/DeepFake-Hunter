"""
Fourier Artifact Analyzer Module
==================================
Detects GAN / diffusion model generation artifacts in the frequency domain.

The core insight:
  Real face images have a natural 1/f power spectrum — low frequencies
  dominate and power smoothly decreases as frequency increases.

  GAN-generated faces have a fundamentally different spectrum:
  - Upsampling convolutions leave grid-like spectral peaks
  - Checkerboard patterns appear at specific spatial frequencies
  - The spectrum has abnormal peaks at regular intervals

How it works:
  1. Extract the face ROI from each frame
  2. Convert to grayscale, apply 2D FFT
  3. Shift zero-frequency to center (fftshift)
  4. Compute log-magnitude spectrum
  5. Detect: spectral peaks, grid patterns, 1/f deviation
  6. Aggregate across all frames for a robust score

Academic references:
  - Frank et al. (2020) "Leveraging Frequency Analysis for Deep Fake Image Forgery Detection"
    ICML 2020
  - Durall et al. (2020) "Watch your Up-Convolution: CNN Based Generative Deep Neural
    Networks are Failing to Reproduce Spectral Distributions"
    CVPR 2020

No new dependencies — uses only OpenCV and NumPy which you already have.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


# ── tuneable constants ─────────────────────────────────────────────────────────
GRID_PEAK_THRESHOLD   = 2.5    # a spectral peak N×std above background is "abnormal"
SPECTRUM_DEVIATION_THRESHOLD = 0.35  # normalized deviation from expected 1/f spectrum
MIN_FRAMES_NEEDED     = 10     # minimum frames for reliable analysis
FACE_RESIZE_PX        = 128    # resize face ROI to this before FFT (speed + consistency)


# ── helpers ────────────────────────────────────────────────────────────────────

def _extract_face_roi(frame: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract and resize the face region using bounding box from landmarks.
    All 68 landmarks define the face extent.
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(landmarks[:, 0].min()) - 10)
    y1 = max(0, int(landmarks[:, 1].min()) - 10)
    x2 = min(w, int(landmarks[:, 0].max()) + 10)
    y2 = min(h, int(landmarks[:, 1].max()) + 10)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    # Resize to fixed size for consistent FFT analysis
    roi = cv2.resize(roi, (FACE_RESIZE_PX, FACE_RESIZE_PX))
    return roi


def _compute_2d_fft_spectrum(gray: np.ndarray) -> np.ndarray:
    """
    Compute the log-magnitude 2D FFT spectrum of a grayscale image.

    Returns the centered log-magnitude spectrum (same size as input).
    """
    # Apply Hanning window to reduce edge artifacts
    h, w  = gray.shape
    win_h = np.hanning(h)
    win_w = np.hanning(w)
    window = np.outer(win_h, win_w)

    f32    = gray.astype(np.float32) / 255.0
    windowed = f32 * window

    # 2D FFT + shift zero-frequency to center
    fft    = np.fft.fft2(windowed)
    fft_shift = np.fft.fftshift(fft)

    # Log-magnitude spectrum
    magnitude = np.abs(fft_shift)
    log_mag   = np.log1p(magnitude)   # log1p avoids log(0)

    return log_mag


def _detect_grid_artifacts(spectrum: np.ndarray) -> Tuple[float, int]:
    """
    Detect periodic grid-like spectral peaks characteristic of upsampling convolutions.

    Returns:
        peak_score  : how much the brightest peaks exceed the background
        num_peaks   : number of anomalous spectral peaks detected
    """
    h, w   = spectrum.shape
    center = (h // 2, w // 2)

    # Exclude the DC component (center) from analysis
    mask = np.ones_like(spectrum, dtype=bool)
    mask[center[0]-3:center[0]+4, center[1]-3:center[1]+4] = False

    background = spectrum[mask]
    bg_mean    = background.mean()
    bg_std     = background.std()

    if bg_std < 1e-9:
        return 0.0, 0

    # Threshold for "anomalous" peaks
    threshold = bg_mean + GRID_PEAK_THRESHOLD * bg_std

    # Find peaks
    peaks = spectrum[mask] > threshold
    num_peaks  = int(peaks.sum())
    if num_peaks == 0:
        return 0.0, 0

    # Score = how far the peaks exceed the background (in standard deviations)
    peak_excess = (spectrum[mask][peaks].mean() - bg_mean) / bg_std
    return round(float(peak_excess), 4), num_peaks


def _compute_spectral_1f_deviation(spectrum: np.ndarray) -> float:
    """
    Measure deviation from expected natural 1/f spectrum.

    A natural face image should have log-power decreasing roughly linearly
    with log-frequency (the 1/f or pink noise property).
    GAN faces deviate from this — they have excess high-frequency energy.

    Returns:
        deviation : float [0, 1] — higher = more GAN-like
    """
    h, w     = spectrum.shape
    cy, cx   = h // 2, w // 2
    max_r    = min(cy, cx) - 1

    if max_r < 4:
        return 0.0

    # Compute radially-averaged power profile
    radial_power = []
    for r in range(1, max_r):
        # Thin annular ring at radius r
        y_idx, x_idx = np.ogrid[:h, :w]
        dist = np.sqrt((y_idx - cy)**2 + (x_idx - cx)**2)
        ring = (dist >= r - 0.5) & (dist < r + 0.5)
        if ring.sum() > 0:
            radial_power.append(spectrum[ring].mean())

    if len(radial_power) < 4:
        return 0.0

    radial_power = np.array(radial_power)

    # Expected 1/f shape: log(power) should be linear with log(r)
    # Fit a line in log-log space
    log_r = np.log(np.arange(1, len(radial_power) + 1))
    log_p = np.log1p(radial_power)

    # Linear fit
    coeffs  = np.polyfit(log_r, log_p, 1)
    fitted  = np.polyval(coeffs, log_r)

    # Residual from the linear fit — high residual = non-natural spectrum
    residuals = log_p - fitted
    deviation = float(np.abs(residuals).mean())

    # Normalize to [0, 1] range
    deviation_normalized = min(deviation / 1.0, 1.0)
    return round(deviation_normalized, 4)


# ── main class ─────────────────────────────────────────────────────────────────

class FourierArtifactAnalyzer:
    """
    Frame-by-frame 2D FFT artifact detector.

    Call update() once per frame (only on frames where face was detected).
    Then call summary() for the final result.

    Example
    -------
    fa = FourierArtifactAnalyzer()
    for frame, landmarks in zip(frames, all_landmarks):
        fa.update(frame, landmarks)
    result = fa.summary()
    """

    def __init__(self):
        self.peak_scores:        List[float] = []
        self.peak_counts:        List[int]   = []
        self.deviation_scores:   List[float] = []
        self.frames_processed:   int         = 0
        self._last_spectrum:     Optional[np.ndarray] = None  # for visualization

    def update(self, frame: np.ndarray, landmarks: np.ndarray):
        """
        Analyze one frame for frequency domain artifacts.

        Parameters
        ----------
        frame     : BGR image (from OpenCV)
        landmarks : shape (68, 2) from face detector
        """
        roi = _extract_face_roi(frame, landmarks)
        if roi is None:
            return

        # Convert to grayscale for FFT
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Compute 2D FFT spectrum
        spectrum = _compute_2d_fft_spectrum(gray)
        self._last_spectrum = spectrum

        # Extract metrics
        peak_score, num_peaks = _detect_grid_artifacts(spectrum)
        deviation = _compute_spectral_1f_deviation(spectrum)

        self.peak_scores.append(peak_score)
        self.peak_counts.append(num_peaks)
        self.deviation_scores.append(deviation)
        self.frames_processed += 1

    # ── derived metrics ────────────────────────────────────────────────────────

    def mean_peak_score(self) -> float:
        if not self.peak_scores:
            return 0.0
        return round(float(np.mean(self.peak_scores)), 4)

    def mean_deviation(self) -> float:
        if not self.deviation_scores:
            return 0.0
        return round(float(np.mean(self.deviation_scores)), 4)

    def mean_peak_count(self) -> float:
        if not self.peak_counts:
            return 0.0
        return round(float(np.mean(self.peak_counts)), 2)

    def artifact_score(self) -> float:
        """
        Combined artifact score [0, 1].
        0 = natural (real), 1 = highly artifacted (fake).

        Combines:
          - Spectral peak elevation (40%)
          - 1/f deviation (60%)
        """
        # Normalize peak_score to [0, 1] — raw score is in standard deviations
        peak_norm = min(self.mean_peak_score() / 5.0, 1.0)  # cap at 5 stds

        deviation = self.mean_deviation()   # already [0, 1]

        combined = 0.40 * peak_norm + 0.60 * deviation
        return round(float(combined), 4)

    def is_suspicious(self) -> Tuple[bool, str]:
        """Return (suspicious, reason_string)."""
        if self.frames_processed < MIN_FRAMES_NEEDED:
            return False, "Fourier analysis: insufficient frames"

        score = self.artifact_score()
        dev   = self.mean_deviation()
        peaks = self.mean_peak_score()

        if score > SPECTRUM_DEVIATION_THRESHOLD:
            return True, (
                f"Spectral artifacts detected — 1/f deviation: {dev:.3f}, "
                f"peak elevation: {peaks:.2f}σ — consistent with GAN upsampling"
            )

        return False, (
            f"Frequency spectrum appears natural — 1/f deviation: {dev:.3f}, "
            f"artifact score: {score:.3f}"
        )

    def summary(self) -> Dict[str, Any]:
        """Return complete summary dict for the combiner."""
        suspicious, reason = self.is_suspicious()
        return {
            "artifact_score":    self.artifact_score(),
            "mean_peak_score":   self.mean_peak_score(),
            "mean_peak_count":   self.mean_peak_count(),
            "mean_deviation":    self.mean_deviation(),
            "frames_processed":  self.frames_processed,
            "suspicious":        suspicious,
            "reason":            reason,
        }
