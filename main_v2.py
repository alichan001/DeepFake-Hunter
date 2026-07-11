"""
Deepfake Detection Pipeline — main.py (Updated with 3 new analyzers)
=====================================================================
Replace your existing main.py with this file.

New analyzers added:
  - MicrosaccadeAnalyzer   (Feature 1)
  - FourierArtifactAnalyzer (Feature 2)
  - GeometricConsistencyAnalyzer (Feature 3)
"""

import os
import json
import time
import argparse
import tempfile
import traceback

import cv2
import numpy as np
from tqdm import tqdm

from analyzers.face_detector             import get_landmarks
from analyzers.blink_analyzer            import BlinkAnalyzer
from analyzers.eye_movement_analyzer     import EyeMovementAnalyzer
from analyzers.rppg_analyzer             import RPPGAnalyzer
from analyzers.microsaccade_analyzer     import MicrosaccadeAnalyzer
from analyzers.fourier_artifact_analyzer import FourierArtifactAnalyzer
from analyzers.geometric_analyzer        import GeometricConsistencyAnalyzer
from combiner_v2                         import combine_signals
from pdf_report                          import generate_pdf_report

FRAME_SKIP     = 2
MAX_WIDTH      = 640
MAX_TOTAL_SECS = 60


def analyze_video(video_path: str, verbose: bool = True) -> dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames   = min(total_frames, int(fps * MAX_TOTAL_SECS))

    scale = 1.0
    if width > MAX_WIDTH:
        scale = MAX_WIDTH / width

    if verbose:
        print(f"\n{'='*54}")
        print(f"  Video   : {os.path.basename(video_path)}")
        print(f"  FPS     : {fps:.1f}   Frames: {total_frames} → analyzing {max_frames}")
        print(f"  Size    : {width}x{height}  Scale: {scale:.2f}  Skip: every {FRAME_SKIP}")
        print(f"  Analyzers: blink · eye · rPPG · microsaccade · fourier · geometry")
        print(f"{'='*54}\n")

    # ── initialize all analyzers ───────────────────────────────────────────
    blink_analyzer    = BlinkAnalyzer()
    eye_analyzer      = EyeMovementAnalyzer()
    rppg_analyzer     = RPPGAnalyzer()
    micro_analyzer    = MicrosaccadeAnalyzer()
    fourier_analyzer  = FourierArtifactAnalyzer()
    geom_analyzer     = GeometricConsistencyAnalyzer()

    face_detected_count = 0
    start_time          = time.time()

    iterator = tqdm(range(max_frames), desc="Analysing", unit="fr") \
               if verbose else range(max_frames)

    for frame_idx in iterator:
        ret, frame = cap.read()
        if not ret:
            break

        if scale < 1.0:
            small = cv2.resize(frame, (int(width * scale), int(height * scale)))
        else:
            small = frame

        if frame_idx % FRAME_SKIP == 0:
            _, landmarks = get_landmarks(small)

            if landmarks is not None:
                face_detected_count += 1
                landmarks_orig = landmarks / scale if scale < 1.0 else landmarks

                # ── existing analyzers ─────────────────────────────────────
                blink_analyzer.update(landmarks)
                gaze = eye_analyzer.update(landmarks)
                rppg_analyzer.update(frame, landmarks_orig)

                # ── new analyzers ──────────────────────────────────────────
                micro_analyzer.add_gaze(gaze)                   # Feature 1
                fourier_analyzer.update(frame, landmarks_orig)  # Feature 2
                geom_analyzer.update(landmarks_orig)             # Feature 3

    cap.release()
    elapsed  = round(time.time() - start_time, 2)
    eff_fps  = fps / FRAME_SKIP
    ef_total = max_frames // FRAME_SKIP

    # ── compute summaries ──────────────────────────────────────────────────
    blink_summary  = blink_analyzer.summary(eff_fps, ef_total)
    eye_summary    = eye_analyzer.summary()
    rppg_summary   = rppg_analyzer.summary(eff_fps)
    micro_summary  = micro_analyzer.summary(eff_fps)       # Feature 1
    fourier_summary = fourier_analyzer.summary()           # Feature 2
    geom_summary   = geom_analyzer.summary()               # Feature 3

    result = combine_signals(
        blink_summary, eye_summary, rppg_summary,
        micro_summary, fourier_summary, geom_summary
    )

    result["meta"] = {
        "video_path":           video_path,
        "fps":                  fps,
        "total_frames":         total_frames,
        "analyzed_frames":      max_frames,
        "frame_skip":           FRAME_SKIP,
        "face_detected_frames": face_detected_count,
        "face_detection_rate":  round(face_detected_count / max(ef_total, 1), 3),
        "processing_time_sec":  elapsed,
        "adaptive_ear_threshold": blink_summary.get("adaptive_threshold", 0),
    }

    if verbose:
        _print_result(result)

    return result


def _print_result(result: dict):
    v   = result["verdict"]
    c   = result["confidence"]
    exp = result["explanation"]
    bar = "█" * int(c / 2) + "░" * (50 - int(c / 2))
    print(f"\n{'='*54}")
    print(f"  VERDICT    : {'FAKE' if v == 'FAKE' else 'REAL'}")
    print(f"  CONFIDENCE : {c}%")
    print(f"  [{bar}]")
    print(f"\n  EXPLANATION : {exp}")
    print(f"\n  ALL SIGNALS:")
    for row in result.get("evidence", []):
        icon = "x" if row["status"] == "danger" else ("!" if row["status"] == "warning" else "v")
        print(f"    [{icon}] {row['signal']:<28} {row['value']}")
    meta = result.get("meta", {})
    print(f"\n  Analyzed {meta.get('analyzed_frames',0)} frames in {meta.get('processing_time_sec',0)}s")
    print(f"{'='*54}\n")


def create_app():
    from flask import Flask, request, jsonify
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "analyzers": 6})

    @app.route("/analyze", methods=["POST"])
    def analyze():
        if "video" not in request.files:
            return jsonify({"error": "Send multipart/form-data with key 'video'."}), 400
        video_file = request.files["video"]
        if video_file.filename == "":
            return jsonify({"error": "Empty filename."}), 400

        suffix = os.path.splitext(video_file.filename)[-1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            video_file.save(tmp_path)

        try:
            result = analyze_video(tmp_path, verbose=True)
            return jsonify(result)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @app.route("/export", methods=["POST"])
    def export_pdf():
        """
        Generate and return a PDF report from a result JSON.
        Expects JSON body: { "result": {...}, "filename": "video.mp4" }
        """
        data     = request.get_json(silent=True) or {}
        result   = data.get("result")
        filename = data.get("filename", "video.mp4")

        if not result:
            return jsonify({"error": "Missing 'result' in request body."}), 400

        try:
            pdf_bytes = generate_pdf_report(result, filename)
            from flask import Response
            safe_name = filename.rsplit(".", 1)[0].replace(" ", "_") + "_deepguard_report.pdf"
            return Response(
                pdf_bytes,
                mimetype="application/pdf",
                headers={
                    "Content-Disposition": f'attachment; filename="{safe_name}"',
                    "Content-Length":      str(len(pdf_bytes)),
                    "Access-Control-Expose-Headers": "Content-Disposition",
                }
            )
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection — 6 Analyzers")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--video",  type=str,        help="Path to video")
    mode.add_argument("--server", action="store_true", help="Start Flask server")
    parser.add_argument("--port",  type=int,   default=5000)
    parser.add_argument("--host",  type=str,   default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--json",  action="store_true")
    args = parser.parse_args()

    if args.server:
        print(f"\nStarting Deepfake Detection API (6 analyzers)...")
        print(f"  http://{args.host}:{args.port}")
        create_app().run(host=args.host, port=args.port, debug=args.debug)
    else:
        result = analyze_video(args.video, verbose=not args.json)
        if args.json:
            print(json.dumps(result, indent=2))
