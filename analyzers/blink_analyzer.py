"""
Blink Analyzer Module
=====================
Detects eye blinks using the Eye Aspect Ratio (EAR) method.

FIX LOG:
  - EAR_THRESHOLD lowered 0.25->0.21 (0.25 too aggressive, causes false positives)
  - Added BLINK_RATE_CAP=30 to prevent absurd values like 60/min
  - Added adaptive threshold: mean_EAR - 1.5*std so it self-calibrates per person
  - CONSEC_FRAMES_MIN raised to 3 to reduce noise
"""

import numpy as np
from scipy.spatial import distance as dist
from typing import List, Tuple, Dict, Any

EAR_THRESHOLD      = 0.21
CONSEC_FRAMES_MIN  = 3
CONSEC_FRAMES_MAX  = 30
NORMAL_BLINK_MIN   = 8.0
NORMAL_BLINK_MAX   = 25.0
BLINK_RATE_CAP     = 30.0
ADAPTIVE_FRAMES    = 60


def _eye_aspect_ratio(eye: np.ndarray) -> float:
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


class BlinkAnalyzer:
    """
    Stateful blink detector with adaptive EAR threshold.
    After ADAPTIVE_FRAMES frames it computes mean_EAR - 1.5*std as the
    personal threshold, handling different face sizes and lighting.
    """

    def __init__(self):
        self.blink_count:    int         = 0
        self._frame_counter: int         = 0
        self.ear_history:    List[float] = []
        self.blink_frames:   List[int]   = []
        self._frame_idx:     int         = 0
        self._threshold:     float       = EAR_THRESHOLD

    def update(self, landmarks: np.ndarray) -> float:
        left_eye  = landmarks[36:42]
        right_eye = landmarks[42:48]
        ear = (_eye_aspect_ratio(left_eye) + _eye_aspect_ratio(right_eye)) / 2.0
        self.ear_history.append(float(ear))

        # Recalculate adaptive threshold every 30 frames after warm-up
        if len(self.ear_history) >= ADAPTIVE_FRAMES and self._frame_idx % 30 == 0:
            arr = np.array(self.ear_history)
            self._threshold = float(np.clip(arr.mean() - 1.5 * arr.std(), 0.15, 0.28))

        if ear < self._threshold:
            self._frame_counter += 1
        else:
            if CONSEC_FRAMES_MIN <= self._frame_counter <= CONSEC_FRAMES_MAX:
                self.blink_count += 1
                self.blink_frames.append(self._frame_idx)
            self._frame_counter = 0

        self._frame_idx += 1
        return ear

    def blink_rate(self, fps: float, total_frames: int) -> float:
        duration_min = (total_frames / max(fps, 1)) / 60.0
        raw = self.blink_count / max(duration_min, 1e-6)
        return round(min(raw, BLINK_RATE_CAP), 2)

    def avg_ear(self) -> float:
        return round(float(np.mean(self.ear_history)), 4) if self.ear_history else 0.0

    def ear_std(self) -> float:
        return round(float(np.std(self.ear_history)), 4) if self.ear_history else 0.0

    def blink_timeline(self, fps: float, window_sec: float = 5.0) -> List[int]:
        if not self.ear_history:
            return []
        window_frames = max(1, int(fps * window_sec))
        total = len(self.ear_history)
        buckets = []
        for start in range(0, total, window_frames):
            end = min(start + window_frames, total)
            count = sum(1 for f in self.blink_frames if start <= f < end)
            buckets.append(count)
        return buckets

    def is_suspicious(self, fps: float, total_frames: int) -> Tuple[bool, str]:
        rate = self.blink_rate(fps, total_frames)
        if rate < NORMAL_BLINK_MIN:
            return True, f"Blink rate {rate:.1f}/min is abnormally low (normal: {NORMAL_BLINK_MIN}–{NORMAL_BLINK_MAX}/min)"
        if rate > NORMAL_BLINK_MAX:
            return True, f"Blink rate {rate:.1f}/min is abnormally high (normal: {NORMAL_BLINK_MIN}–{NORMAL_BLINK_MAX}/min)"
        return False, f"Blink rate {rate:.1f}/min is within the normal range"

    def summary(self, fps: float, total_frames: int) -> Dict[str, Any]:
        rate = self.blink_rate(fps, total_frames)
        suspicious, reason = self.is_suspicious(fps, total_frames)
        return {
            "blink_count":        self.blink_count,
            "blink_rate":         rate,
            "avg_ear":            self.avg_ear(),
            "ear_std":            self.ear_std(),
            "adaptive_threshold": round(self._threshold, 4),
            "timeline":           self.blink_timeline(fps),
            "suspicious":         suspicious,
            "reason":             reason,
        }
