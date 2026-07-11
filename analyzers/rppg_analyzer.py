"""
rPPG Analyzer Module (Remote Photoplethysmography)
===================================================
Detects a heartbeat signal by measuring subtle color changes in facial skin
caused by blood flow, using the green channel of the forehead ROI.

Algorithm:
  1. Extract forehead ROI from landmarks
  2. Compute mean green-channel value per frame
  3. Detrend and bandpass-filter the signal (0.7 – 4.0 Hz → 42 – 240 bpm)
  4. Find dominant frequency via FFT → heart rate in bpm
  5. Decide if a real heartbeat is present based on signal-to-noise ratio

Reference:
  Verkruysse et al. (2008) — "Remote plethysmographic imaging using ambient light"
  de Haan & Jeanne (2013) — "Robust pulse rate from chrominance-based rPPG"

Landmark indices (dlib 68-point model):
  Brow: 17 – 26   (top of face, used for forehead ROI)
"""

import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy.fft import rfft, rfftfreq
from typing import List, Tuple, Dict, Any, Optional


# ── tuneable constants ─────────────────────────────────────────────────────────
HR_LOW_HZ          = 0.7    # 42 bpm  — minimum plausible heart rate
HR_HIGH_HZ         = 4.0    # 240 bpm — maximum plausible heart rate
BUTTER_ORDER       = 3
MIN_FRAMES_FOR_HR  = 90     # need at least 3 s @ 30 fps
SNR_THRESHOLD      = 2.5    # signal-to-noise ratio; below → no heartbeat
FOREHEAD_PAD_UP    = 30     # px above the brow line


# ── helpers ────────────────────────────────────────────────────────────────────

def _forehead_roi(frame: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract the forehead region of interest.
    Uses brow landmarks (17–26) to locate the top of the face.
    Returns None if the ROI is empty or out of frame.
    """
    h, w = frame.shape[:2]
    brow = landmarks[17:27]

    x1 = max(0, int(brow[:, 0].min()))
    x2 = min(w, int(brow[:, 0].max()))
    y1 = max(0, int(brow[:, 1].min()) - FOREHEAD_PAD_UP)
    y2 = max(0, int(brow[:, 1].min()) - 4)   # just above the brows

    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _bandpass(signal: np.ndarray, fps: float) -> np.ndarray:
    """Apply a Butterworth bandpass filter (HR_LOW_HZ – HR_HIGH_HZ)."""
    nyq  = fps / 2.0
    low  = HR_LOW_HZ  / nyq
    high = HR_HIGH_HZ / nyq
    # Clamp to (0, 1) exclusive to avoid ValueError
    low  = max(1e-4, min(low,  0.9999))
    high = max(1e-4, min(high, 0.9999))
    if low >= high:
        return signal
    b, a = butter(BUTTER_ORDER, [low, high], btype="band")
    return filtfilt(b, a, signal)


def _signal_snr(signal: np.ndarray, fps: float) -> Tuple[float, float]:
    """
    Compute SNR around the dominant frequency.

    Returns
    -------
    peak_freq : Hz
    snr       : ratio of peak power to mean background power
    """
    n     = len(signal)
    freqs = rfftfreq(n, d=1.0 / fps)
    power = np.abs(rfft(signal)) ** 2

    valid = (freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ)
    if not valid.any():
        return 0.0, 0.0

    valid_power = power[valid]
    valid_freqs = freqs[valid]

    peak_idx  = np.argmax(valid_power)
    peak_freq = float(valid_freqs[peak_idx])
    peak_pow  = float(valid_power[peak_idx])
    mean_pow  = float(valid_power.mean())

    snr = peak_pow / max(mean_pow, 1e-9)
    return peak_freq, snr


# ── main class ─────────────────────────────────────────────────────────────────

class RPPGAnalyzer:
    """
    Stateful rPPG heartbeat detector — call `update()` once per frame.

    Example
    -------
    analyzer = RPPGAnalyzer()
    for frame, landmarks in zip(frames, all_landmarks):
        analyzer.update(frame, landmarks)
    result = analyzer.summary(fps=30)
    """

    def __init__(self):
        self.green_signal: List[float] = []   # raw mean green value per frame
        self._frames_processed: int    = 0

    # ── per-frame update ───────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, landmarks: np.ndarray) -> Optional[float]:
        """
        Extract and store the mean green-channel value from the forehead ROI.

        Parameters
        ----------
        frame     : np.ndarray — BGR image
        landmarks : np.ndarray — shape (68, 2)

        Returns
        -------
        green_mean : float | None  — None if no ROI could be extracted
        """
        self._frames_processed += 1
        roi = _forehead_roi(frame, landmarks)
        if roi is None or roi.size == 0:
            # Fill with last value (or 0) to keep the signal continuous
            last = self.green_signal[-1] if self.green_signal else 0.0
            self.green_signal.append(last)
            return None

        # BGR → use channel index 1 (green)
        g_mean = float(roi[:, :, 1].mean())
        self.green_signal.append(g_mean)
        return g_mean

    # ── derived metrics ────────────────────────────────────────────────────────

    def _processed_signal(self, fps: float) -> np.ndarray:
        """Detrend + bandpass the raw green signal."""
        raw = np.array(self.green_signal, dtype=np.float64)
        raw = detrend(raw)                    # remove slow drift
        raw -= raw.mean()                     # zero-mean
        raw = _bandpass(raw, fps)             # isolate HR frequencies
        return raw

    def heart_rate(self, fps: float) -> Tuple[Optional[float], bool]:
        """
        Estimate heart rate and decide if a real heartbeat is present.

        Returns
        -------
        bpm           : float | None
        has_heartbeat : bool
        """
        if len(self.green_signal) < MIN_FRAMES_FOR_HR:
            return None, False

        processed = self._processed_signal(fps)
        peak_freq, snr = _signal_snr(processed, fps)

        has_heartbeat = snr >= SNR_THRESHOLD
        bpm = round(peak_freq * 60.0, 1) if has_heartbeat else None
        return bpm, has_heartbeat

    def snr(self, fps: float) -> float:
        """Return the signal-to-noise ratio of the rPPG signal."""
        if len(self.green_signal) < MIN_FRAMES_FOR_HR:
            return 0.0
        processed = self._processed_signal(fps)
        _, snr = _signal_snr(processed, fps)
        return round(float(snr), 3)

    def signal_waveform(self, fps: float, max_points: int = 200) -> List[float]:
        """
        Return the processed rPPG waveform for the frontend graph.
        Down-sampled to *max_points* if longer.
        """
        if len(self.green_signal) < 10:
            return []
        processed = self._processed_signal(fps)
        if len(processed) > max_points:
            indices = np.linspace(0, len(processed) - 1, max_points, dtype=int)
            processed = processed[indices]
        # Normalize to [-1, 1] for clean visualization
        mx = np.abs(processed).max()
        if mx > 1e-9:
            processed = processed / mx
        return [round(float(v), 4) for v in processed]

    def is_suspicious(self, fps: float) -> Tuple[bool, str]:
        """Return (suspicious, reason_string)."""
        bpm, has_heartbeat = self.heart_rate(fps)
        snr_val = self.snr(fps)

        if not has_heartbeat:
            return True, f"No heartbeat detected — rPPG SNR {snr_val:.1f} (threshold: {SNR_THRESHOLD})"
        if bpm and (bpm < 42 or bpm > 200):
            return True, f"Implausible heart rate {bpm} bpm detected"
        return False, f"Heartbeat detected at ~{bpm} bpm (SNR: {snr_val:.1f})"

    def summary(self, fps: float) -> Dict[str, Any]:
        """Return a complete summary dict for the combiner."""
        bpm, has_heartbeat = self.heart_rate(fps)
        suspicious, reason = self.is_suspicious(fps)
        return {
            "heart_rate":       bpm,
            "has_heartbeat":    has_heartbeat,
            "snr":              self.snr(fps),
            "waveform":         self.signal_waveform(fps),
            "frames_processed": self._frames_processed,
            "suspicious":       suspicious,
            "reason":           reason,
        }
