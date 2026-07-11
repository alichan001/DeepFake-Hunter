"""
Signal Combiner v2
==================
Fuses all 6 analyzers into a single verdict.

Weights (must sum to 1.0):
  rPPG heartbeat      → 0.28  (physiological — very hard to fake)
  Blink rate          → 0.18  (behavioural)
  Microsaccade        → 0.20  (NEW — novel oculomotor signal)
  Fourier artifacts   → 0.18  (NEW — frequency domain forgery)
  Geometric stability → 0.10  (NEW — landmark consistency)
  Eye movement        → 0.06  (supplementary)
"""

import math
from typing import Dict, Any, List

W_RPPG    = 0.28
W_BLINK   = 0.18
W_MICRO   = 0.20
W_FOURIER = 0.18
W_GEOM    = 0.10
W_EYE     = 0.06

FAKE_THRESHOLD = 0.45


def _sigmoid_confidence(score: float) -> float:
    dist = abs(score - FAKE_THRESHOLD) / max(FAKE_THRESHOLD, 1e-6)
    conf = 50.0 + 49.0 * (1 - math.exp(-3.5 * dist))
    return round(min(conf, 99.0), 1)


def _build_evidence(
    blink: Dict, eye: Dict, rppg: Dict,
    micro: Dict, fourier: Dict, geom: Dict
) -> List[Dict[str, Any]]:
    rows = []

    # rPPG
    bpm    = rppg.get("heart_rate")
    has_hb = rppg.get("has_heartbeat", False)
    snr    = rppg.get("snr", 0)
    rows.append({
        "signal":  "rPPG heartbeat",
        "value":   f"{bpm} bpm (SNR: {snr:.1f})" if has_hb else f"No signal (SNR: {snr:.1f})",
        "status":  "ok" if has_hb else "danger",
        "message": rppg.get("reason", ""),
    })

    # Blink
    rate = blink.get("blink_rate", 0)
    rows.append({
        "signal":  "Blink rate",
        "value":   f"{rate:.1f} blinks/min",
        "status":  "danger" if rate < 6 else ("warning" if rate < 10 or rate > 25 else "ok"),
        "message": blink.get("reason", ""),
    })

    # Microsaccade
    ms_ratio = micro.get("microsaccade_ratio", 0)
    rows.append({
        "signal":  "Microsaccade activity",
        "value":   f"ratio {ms_ratio:.3f}",
        "status":  "danger" if micro.get("suspicious") else "ok",
        "message": micro.get("reason", ""),
    })

    # Fourier
    art_score = fourier.get("artifact_score", 0)
    rows.append({
        "signal":  "Fourier artifacts",
        "value":   f"score {art_score:.3f}",
        "status":  "danger" if fourier.get("suspicious") else "ok",
        "message": fourier.get("reason", ""),
    })

    # Geometric
    mean_cv = geom.get("mean_cv", 0)
    rows.append({
        "signal":  "Geometric stability",
        "value":   f"CV {mean_cv:.3f}",
        "status":  "warning" if geom.get("suspicious") else "ok",
        "message": geom.get("reason", ""),
    })

    # Eye movement
    entropy = eye.get("entropy", 0)
    rows.append({
        "signal":  "Eye movement",
        "value":   f"{entropy:.2f} bits entropy",
        "status":  "danger" if entropy < 1.0 else ("warning" if entropy < 1.8 else "ok"),
        "message": eye.get("reason", ""),
    })

    return rows


def _build_explanation(evidence: List[Dict]) -> str:
    bad = [r for r in evidence if r["status"] in ("danger", "warning")]
    if not bad:
        return "All 6 physiological and spectral signals appear natural."
    parts = []
    for r in bad:
        msg   = r["message"]
        short = msg.split("—")[0].split("(")[0].strip().rstrip(";").strip()
        if short:
            parts.append(short[0].upper() + short[1:])
    return " + ".join(parts) if parts else bad[0]["message"]


def combine_signals(
    blink:   Dict[str, Any],
    eye:     Dict[str, Any],
    rppg:    Dict[str, Any],
    micro:   Dict[str, Any],
    fourier: Dict[str, Any],
    geom:    Dict[str, Any],
) -> Dict[str, Any]:

    score = 0.0

    # rPPG
    if rppg.get("suspicious", False):
        score += W_RPPG
    elif rppg.get("heart_rate") and (
        rppg["heart_rate"] < 42 or rppg["heart_rate"] > 200
    ):
        score += W_RPPG * 0.5

    # Blink
    if blink.get("suspicious", False):
        rate = blink.get("blink_rate", 15)
        score += W_BLINK if rate < 4 else W_BLINK * 0.65

    # Microsaccade
    if micro.get("suspicious", False):
        score += W_MICRO
    elif not micro.get("insufficient_data", True):
        ratio = micro.get("microsaccade_ratio", 0.3)
        if ratio < 0.20:   # partial credit
            score += W_MICRO * 0.4

    # Fourier
    if fourier.get("suspicious", False):
        score += W_FOURIER
    else:
        art = fourier.get("artifact_score", 0)
        if art > 0.20:     # partial credit
            score += W_FOURIER * (art / 0.35)

    # Geometric
    if geom.get("suspicious", False):
        score += W_GEOM
    else:
        cv = geom.get("mean_cv", 0)
        if cv > 0.05:
            score += W_GEOM * min(cv / 0.08, 1.0)

    # Eye movement
    if eye.get("suspicious", False):
        entropy = eye.get("entropy", 2.0)
        score += W_EYE if entropy < 1.0 else W_EYE * 0.5

    score      = round(min(score, 1.0), 4)
    is_fake    = score >= FAKE_THRESHOLD
    verdict    = "FAKE" if is_fake else "REAL"
    confidence = _sigmoid_confidence(score)
    evidence   = _build_evidence(blink, eye, rppg, micro, fourier, geom)
    explanation= _build_explanation(evidence)

    return {
        "verdict":     verdict,
        "confidence":  confidence,
        "score":       score,
        "explanation": explanation,
        "evidence":    evidence,
        "signals": {
            # existing
            "blink_rate":       blink.get("blink_rate"),
            "blink_timeline":   blink.get("timeline", []),
            "eye_entropy":      eye.get("entropy"),
            "gaze_x_series":    eye.get("gaze_x_series", []),
            "heart_rate":       rppg.get("heart_rate"),
            "has_heartbeat":    rppg.get("has_heartbeat"),
            "rppg_snr":         rppg.get("snr"),
            "rppg_waveform":    rppg.get("waveform", []),
            # new
            "microsaccade_ratio":   micro.get("microsaccade_ratio"),
            "micro_event_rate":     micro.get("event_rate_per_min"),
            "micro_spectrum_x":     micro.get("spectrum_x", []),
            "micro_spectrum_y":     micro.get("spectrum_y", []),
            "fourier_artifact":     fourier.get("artifact_score"),
            "fourier_deviation":    fourier.get("mean_deviation"),
            "geom_mean_cv":         geom.get("mean_cv"),
            "geom_stability":       geom.get("stability_score"),
            "geom_ipd_series":      geom.get("ipd_series", []),
            "geom_symmetry_series": geom.get("symmetry_series", []),
            "geom_ratio_cvs":       geom.get("ratio_cvs", {}),
        },
    }
