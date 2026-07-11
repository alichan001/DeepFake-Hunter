# Deepfake Detection — Setup & Usage

## Project Structure

```
deepfake_detector/
├── main.py                          ← entry point (CLI + Flask server)
├── combiner.py                      ← signal fusion & verdict logic
├── requirements.txt
├── shape_predictor_68_face_landmarks.dat   ← download separately (see below)
└── analyzers/
    ├── __init__.py
    ├── face_detector.py             ← dlib face + landmark detection
    ├── blink_analyzer.py            ← EAR blink rate
    ├── eye_movement_analyzer.py     ← gaze entropy & smoothness
    └── rppg_analyzer.py             ← remote heartbeat (rPPG)
```

---

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on dlib**: dlib requires CMake and a C++ compiler.
> On Windows, install Visual Studio Build Tools first.
> On Ubuntu: `sudo apt-get install build-essential cmake`

---

## 2. Download the landmark model (required)

```bash
# Linux / macOS
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Windows (PowerShell)
curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Then extract with 7-Zip or similar
```

Place `shape_predictor_68_face_landmarks.dat` in the **project root** (same folder as `main.py`).

---

## 3. Run — CLI mode

```bash
# Analyse a single video, print verdict to terminal
python main.py --video path/to/video.mp4

# Output raw JSON (useful for scripting)
python main.py --video path/to/video.mp4 --json
```

---

## 4. Run — API server mode

```bash
python main.py --server
# Server starts at http://localhost:5000
```

### API endpoints

| Method | Endpoint   | Body                          | Returns              |
|--------|------------|-------------------------------|----------------------|
| GET    | /health    | —                             | `{ "status": "ok" }` |
| POST   | /analyze   | `multipart/form-data` key=`video` | Full analysis JSON |

### Example curl call
```bash
curl -X POST http://localhost:5000/analyze \
     -F "video=@/path/to/video.mp4"
```

### Example response
```json
{
  "verdict": "FAKE",
  "confidence": 91.2,
  "score": 0.75,
  "explanation": "Low blink rate (3.1/min) + No heartbeat detected",
  "evidence": [
    { "signal": "Blink rate",    "value": "3.1 blinks/min",       "status": "danger",  "message": "..." },
    { "signal": "Eye movement",  "value": "0.81 bits entropy",     "status": "warning", "message": "..." },
    { "signal": "rPPG heartbeat","value": "No signal (SNR: 1.2)",  "status": "danger",  "message": "..." }
  ],
  "signals": {
    "blink_rate": 3.1,
    "eye_entropy": 0.81,
    "heart_rate": null,
    "has_heartbeat": false,
    "rppg_waveform": [...],
    "blink_timeline": [...],
    "gaze_x_series": [...]
  },
  "meta": {
    "fps": 30.0,
    "total_frames": 900,
    "face_detection_rate": 0.97,
    "processing_time_sec": 14.3
  }
}
```

---

## 5. Connect to the frontend

In your React/HTML frontend, replace the demo mode with:

```javascript
const formData = new FormData();
formData.append('video', videoFile);

const res  = await fetch('http://localhost:5000/analyze', {
    method: 'POST',
    body: formData
});
const data = await res.json();

// data.verdict        → "FAKE" | "REAL"
// data.confidence     → 91.2
// data.explanation    → "Low blink rate + no heartbeat..."
// data.signals.rppg_waveform  → array for the heart signal graph
// data.signals.blink_timeline → array for the blink bar chart
// data.signals.gaze_x_series  → array for the eye movement graph
```

---

## Tuning thresholds

All thresholds are at the top of each analyzer file as constants:

| File                      | Constant            | Default | Meaning                         |
|---------------------------|---------------------|---------|----------------------------------|
| `blink_analyzer.py`       | `EAR_THRESHOLD`     | 0.25    | EAR below this = eye closed     |
| `blink_analyzer.py`       | `NORMAL_BLINK_MIN`  | 10.0    | Min natural blinks/min          |
| `eye_movement_analyzer.py`| `MIN_ENTROPY_NATURAL`| 1.8   | Min natural gaze entropy (bits) |
| `rppg_analyzer.py`        | `SNR_THRESHOLD`     | 2.5     | Min SNR to confirm heartbeat    |
| `combiner.py`             | `W_RPPG`            | 0.40    | Weight of rPPG in final score   |
| `combiner.py`             | `FAKE_THRESHOLD`    | 0.50    | Score ≥ this → FAKE             |

---

## FYP Report notes

- **Face detection**: dlib HOG detector (Dalal & Triggs 2005)
- **Landmarks**: Kazemi & Sullivan (2014) "One Millisecond Face Alignment"
- **Blink detection**: Soukupová & Čech (2016) EAR method
- **rPPG**: Verkruysse et al. (2008); de Haan & Jeanne (2013)
- **Fusion**: Weighted evidence scoring with sigmoid confidence calibration
