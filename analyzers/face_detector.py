"""
Face Detector Module
====================
Uses dlib's HOG-based frontal face detector and
68-point facial landmark predictor.

Download the landmark model before running:
  wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
Place the .dat file in the project root directory.
"""

import os
import cv2
import dlib
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "shape_predictor_68_face_landmarks.dat"
)

# ── dlib objects (loaded once at import) ──────────────────────────────────────
_detector  = dlib.get_frontal_face_detector()
_predictor = None   # lazy-loaded so import never crashes if .dat is missing


def _load_predictor():
    global _predictor
    if _predictor is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Landmark model not found at:\n  {_MODEL_PATH}\n"
                "Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        _predictor = dlib.shape_predictor(_MODEL_PATH)
    return _predictor


# ── public API ─────────────────────────────────────────────────────────────────

def get_landmarks(frame: np.ndarray):
    """
    Detect the largest face in *frame* and return its 68 landmarks.

    Parameters
    ----------
    frame : np.ndarray
        BGR image from OpenCV.

    Returns
    -------
    face  : dlib.rectangle | None
    landmarks : np.ndarray of shape (68, 2) | None
    """
    predictor = _load_predictor()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _detector(gray, 1)

    if len(faces) == 0:
        return None, None

    # Use the largest detected face
    face = max(faces, key=lambda r: r.area())
    shape = predictor(gray, face)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
    return face, landmarks


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Draw all 68 landmark points on a copy of *frame* (useful for debugging).
    """
    vis = frame.copy()
    for (x, y) in landmarks.astype(int):
        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
    return vis


def get_face_roi(frame: np.ndarray, face, padding: int = 10) -> np.ndarray:
    """
    Crop and return the face region from *frame* with optional padding.
    """
    h, w = frame.shape[:2]
    x1 = max(0, face.left()   - padding)
    y1 = max(0, face.top()    - padding)
    x2 = min(w, face.right()  + padding)
    y2 = min(h, face.bottom() + padding)
    return frame[y1:y2, x1:x2]
