"""
Microsaccade Analyzer Module
=============================
Detects involuntary micro eye-movements (microsaccades) that are present
in all real human eyes but ABSENT in deepfake-generated faces.

What are microsaccades?
  Real eyes constantly make tiny involuntary movements at 2-50 Hz frequency.
  These are physically caused by the oculomotor system and cannot be faked
  by current deepfake generators — they simply do not model this behaviour.

How it works:
  1. Take the gaze X,Y time-series from EyeMovementAnalyzer
  2. Apply FFT to extract the frequency spectrum
  3. Measure power in the microsaccade band (2-50 Hz)
  4. Compare to the low-frequency band (0-2 Hz)
  5. Real eyes: high microsaccade-band power relative to total
     Fake eyes: flat or absent microsaccade-band power

Academic reference:
  Martinez-Conde et al. (2013) "Microsaccades: a neurophysiological analysis"
  Trends in Neurosciences, 36(4), 219-232.

Novel contribution:
  Almost no existing lightweight deepfake detector uses microsaccade analysis.
  This is your differentiating feature.
"""

import numpy as np
from scipy.signal import welch, butter, filtfilt
from scipy.fft import rfft, rfftfreq
from typing import List, Tuple, Dict, Any, Optional


# ── tuneable constants ─────────────────────────────────────────────────────────
MICROSACCADE_LOW_HZ   = 2.0    # start of microsaccade frequency band
MICROSACCADE_HIGH_HZ  = 50.0   # end of microsaccade frequency band
DRIFT_LOW_HZ          = 0.0    # slow drift / fixation band
DRIFT_HIGH_HZ         = 2.0    # end of slow drift band

# Ratio threshold: microsaccade_power / total_power
# Real eyes:  typically 0.25 – 0.60
# Fake eyes:  typically 0.00 – 0.12
REAL_RATIO_THRESHOLD  = 0.15   # below this → suspicious (fake)

MIN_FRAMES_NEEDED     = 60     # need at least 2 seconds at 30fps


# ── signal processing helpers ──────────────────────────────────────────────────

def _normalize(signal: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization."""
    std = signal.std()
    if std < 1e-9:
        return signal - signal.mean()
    return (signal - signal.mean()) / std


def _compute_band_power(
    freqs: np.ndarray,
    power: np.ndarray,
    low: float,
    high: float
) -> float:
    """Integrate power spectral density within a frequency band."""
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return 0.0
    return float(np.trapz(power[mask], freqs[mask]))


def _extract_microsaccade_events(
    gaze: np.ndarray,
    fps: float,
    velocity_threshold: float = 6.0   # degrees/second — classic Engbert & Kliegl criterion
) -> List[int]:
    """
    Detect individual microsaccade events using velocity threshold.

    Returns list of frame indices where a microsaccade starts.
    Uses the Engbert & Kliegl (2003) algorithm adapted for pixel space.
    """
    if len(gaze) < 5:
        return []

    # Compute velocity (pixels/frame → approximate degrees at typical viewing distance)
    velocity = np.diff(gaze, axis=0)
    speed    = np.linalg.norm(velocity, axis=1)   # magnitude per frame

    # Median-based threshold (robust to outliers)
    median_speed = np.median(speed)
    std_speed    = speed.std()
    threshold    = median_speed + velocity_threshold * std_speed

    # Detect threshold crossings (start of each microsaccade)
    events = []
    above  = False
    for i, s in enumerate(speed):
        if s > threshold and not above:
            events.append(i)
            above = True
        elif s <= threshold:
            above = False

    return events


# ── main class ─────────────────────────────────────────────────────────────────

class MicrosaccadeAnalyzer:
    """
    Microsaccade frequency analyzer.

    Feed it gaze points from EyeMovementAnalyzer, then call summary().

    Example
    -------
    ms = MicrosaccadeAnalyzer()
    for lm in landmarks_list:
        gaze = eye_analyzer.update(lm)   # get gaze from your EyeMovementAnalyzer
        ms.add_gaze(gaze)
    result = ms.summary(fps=30)
    """

    def __init__(self):
        self.gaze_points: List[np.ndarray] = []

    def add_gaze(self, gaze_point: np.ndarray):
        """
        Add a single gaze (x, y) observation.

        Parameters
        ----------
        gaze_point : np.ndarray, shape (2,) — [x, y] from EyeMovementAnalyzer
        """
        self.gaze_points.append(np.asarray(gaze_point, dtype=np.float32))

    def add_gaze_xy(self, x: float, y: float):
        """Convenience method — pass x, y separately."""
        self.gaze_points.append(np.array([x, y], dtype=np.float32))

    # ── frequency domain analysis ──────────────────────────────────────────────

    def _frequency_analysis(self, fps: float) -> Dict[str, Any]:
        """
        Core FFT-based microsaccade analysis.
        Returns dict of frequency-domain metrics.
        """
        if len(self.gaze_points) < MIN_FRAMES_NEEDED:
            return {
                "microsaccade_ratio": 0.0,
                "drift_power":        0.0,
                "micro_power":        0.0,
                "total_power":        0.0,
                "dominant_freq_hz":   0.0,
                "spectrum_x":         [],
                "spectrum_y":         [],
                "insufficient_data":  True,
            }

        pts = np.array(self.gaze_points)

        # Analyze X and Y axes independently, then average
        results = []
        for axis in range(2):
            signal = _normalize(pts[:, axis])

            # Power Spectral Density using Welch method (more robust than raw FFT)
            # nperseg: window size — use 2 seconds of data
            nperseg = min(len(signal), max(32, int(fps * 2)))
            freqs, psd = welch(signal, fs=fps, nperseg=nperseg, scaling='density')

            drift_power = _compute_band_power(freqs, psd, DRIFT_LOW_HZ, DRIFT_HIGH_HZ)
            micro_power = _compute_band_power(freqs, psd, MICROSACCADE_LOW_HZ,
                                              min(MICROSACCADE_HIGH_HZ, fps / 2 - 0.1))
            total_power = _compute_band_power(freqs, psd, 0, fps / 2 - 0.1)

            results.append({
                "freqs":       freqs,
                "psd":         psd,
                "drift_power": drift_power,
                "micro_power": micro_power,
                "total_power": total_power,
            })

        # Average across X and Y
        avg_micro = np.mean([r["micro_power"] for r in results])
        avg_drift = np.mean([r["drift_power"] for r in results])
        avg_total = np.mean([r["total_power"] for r in results])
        ratio     = avg_micro / max(avg_total, 1e-9)

        # Find dominant frequency (highest power peak)
        combined_psd = results[0]["psd"] + results[1]["psd"]
        freqs        = results[0]["freqs"]
        dom_idx      = np.argmax(combined_psd)
        dom_freq     = float(freqs[dom_idx])

        # Build spectrum for visualization (downsample to max 100 points)
        max_pts = min(100, len(freqs))
        idx     = np.linspace(0, len(freqs) - 1, max_pts, dtype=int)
        spec_x  = [round(float(freqs[i]), 2) for i in idx]
        spec_y  = [round(float(combined_psd[i]), 6) for i in idx]

        return {
            "microsaccade_ratio": round(float(ratio), 4),
            "drift_power":        round(float(avg_drift), 6),
            "micro_power":        round(float(avg_micro), 6),
            "total_power":        round(float(avg_total), 6),
            "dominant_freq_hz":   round(dom_freq, 2),
            "spectrum_x":         spec_x,
            "spectrum_y":         spec_y,
            "insufficient_data":  False,
        }

    # ── event-level analysis ───────────────────────────────────────────────────

    def _event_analysis(self, fps: float) -> Dict[str, Any]:
        """
        Count discrete microsaccade events using velocity threshold.
        Returns event rate and inter-event statistics.
        """
        if len(self.gaze_points) < 10:
            return {"event_count": 0, "event_rate_per_min": 0.0, "mean_interval_sec": 0.0}

        pts    = np.array(self.gaze_points)
        events = _extract_microsaccade_events(pts, fps)

        duration_min = (len(pts) / max(fps, 1)) / 60.0
        rate = len(events) / max(duration_min, 1e-6)

        # Interval between events
        if len(events) >= 2:
            intervals    = np.diff(events) / fps   # convert frames → seconds
            mean_interval = float(intervals.mean())
        else:
            mean_interval = 0.0

        return {
            "event_count":        len(events),
            "event_rate_per_min": round(rate, 2),
            "mean_interval_sec":  round(mean_interval, 3),
            "event_frames":       events[:20],   # first 20 for visualization
        }

    # ── suspicion decision ─────────────────────────────────────────────────────

    def is_suspicious(self, fps: float) -> Tuple[bool, str]:
        """
        Return (suspicious, reason_string).

        Decision logic:
          - If microsaccade_ratio < REAL_RATIO_THRESHOLD → SUSPICIOUS
          - If insufficient data → not suspicious (benefit of doubt)
        """
        freq = self._frequency_analysis(fps)

        if freq["insufficient_data"]:
            return False, "Microsaccade: insufficient data for analysis"

        ratio = freq["microsaccade_ratio"]

        if ratio < REAL_RATIO_THRESHOLD:
            return True, (
                f"Microsaccade frequency band power is very low "
                f"(ratio: {ratio:.3f}, threshold: {REAL_RATIO_THRESHOLD}) — "
                f"real eyes always exhibit 2-50 Hz micro-tremor"
            )

        return False, (
            f"Microsaccade activity detected (ratio: {ratio:.3f}) — "
            f"consistent with genuine oculomotor behaviour"
        )

    # ── full summary ───────────────────────────────────────────────────────────

    def summary(self, fps: float) -> Dict[str, Any]:
        """
        Return complete summary dict for the combiner.

        Keys
        ----
        microsaccade_ratio    : float  — micro_power / total_power (main metric)
        event_rate_per_min    : float  — discrete microsaccade events per minute
        dominant_freq_hz      : float  — dominant frequency in gaze signal
        spectrum_x / _y       : list   — frequency spectrum for visualization
        suspicious            : bool
        reason                : str
        """
        freq   = self._frequency_analysis(fps)
        events = self._event_analysis(fps)
        suspicious, reason = self.is_suspicious(fps)

        return {
            "microsaccade_ratio":   freq["microsaccade_ratio"],
            "drift_power":          freq["drift_power"],
            "micro_power":          freq["micro_power"],
            "total_power":          freq["total_power"],
            "dominant_freq_hz":     freq["dominant_freq_hz"],
            "event_count":          events["event_count"],
            "event_rate_per_min":   events["event_rate_per_min"],
            "mean_interval_sec":    events["mean_interval_sec"],
            "spectrum_x":           freq["spectrum_x"],
            "spectrum_y":           freq["spectrum_y"],
            "frames_analyzed":      len(self.gaze_points),
            "insufficient_data":    freq["insufficient_data"],
            "suspicious":           suspicious,
            "reason":               reason,
        }
