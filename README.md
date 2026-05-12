# CV_project

# 🚗 Dashcam Semantic Search

> Open-vocabulary detection, tracking, and counting on dashcam footage.
> Combines **YOLO** (instance segmentation), **ByteTrack / BoT-SORT** (multi-object tracking),
> and **CLIP** (open-vocabulary classification) to identify and count objects defined by
> free-text queries — no model retraining required.

---

## ✨ Features

- **Instance segmentation** with a custom YOLO model (polygon masks for precise crops)
- **Stable multi-object tracking** with ByteTrack or BoT-SORT
- **Open-vocabulary semantic classification** via CLIP (any query works: `"person"`, `"cyclist"`, `"dog"`, …)
- **Crossing detection** based on a composite score of velocity, trajectory, and road-zone position
- **Anti-double-count** system with spatial/temporal cooldown and lightweight Re-ID
- **Visualization** with semi-transparent masks, bounding boxes, and an on-screen counter panel
- **JSON export** with full summary: processed frames, per-track info, and counters

---

## 📁 Project Structure

```
dashcam_search/
│
├── config/
│   └── settings.py           ← ✏️  EDIT THIS FILE to change any hyperparameter
│
├── core/
│   ├── clip_utils.py         ← CLIP encoding (text & image), prompt ensemble
│   ├── geometry.py           ← Geometry: bbox, IoU, distances, crop, mask
│   ├── crossing.py           ← Crossing detection: score, velocity, state machine
│   ├── tracking.py           ← Track lifecycle: creation, Re-ID, cleanup
│   └── semantics.py          ← EMA scores, match validation, counting
│
├── ioutils/
│   ├── video_io.py           ← VideoCapture and VideoWriter setup
│   └── output_writer.py      ← JSON output builder and writer
│
├── visualization/
│   └── drawing.py            ← Rendering: boxes, masks, labels, counters
│
├── models/
│   └── loader.py             ← YOLO + CLIP loader, device selection
│
├── main.py                   ← Entry point: global runtime state + main loop
├── requirements.txt
└── README.md
```


---

## 🚀 Quick Start

1. Place your input video at `videos/input.mp4`
2. Set your queries and parameters in `config/settings.py`
3. Run:

```bash
python main.py
```

Outputs are saved to `outputs/`:
- `output_crossing_open_vocab_seg.mp4` — annotated video
- `output_crossing_open_vocab_seg.json` — full processing summary

---

## 🔧 Configuration (`config/settings.py`)

**This is the only file you need to edit.** All parameters are grouped into labeled sections.

### I/O

```python
INPUT_VIDEO  = "videos/input.mp4"
OUTPUT_VIDEO = "outputs/output_crossing_open_vocab_seg.mp4"
OUTPUT_JSON  = "outputs/output_crossing_open_vocab_seg.json"
```

### Semantic Queries

```python
QUERIES = ["person"]   # Add any free-text query in natural language
```

### Models

| Parameter | Default | Description |
|---|---|---|
| `YOLO_MODEL` | `yolo26m-finetuned-segmentation` | Path to the custom YOLO model |
| `CLIP_MODEL` | `ViT-B-32` | CLIP architecture |
| `CLIP_PRETRAINED` | `laion2b_s34b_b79k` | Pre-trained CLIP checkpoint |
| `TRACKER` | `botsort.yaml` | Tracker config (`botsort.yaml` or `bytetrack.yaml`) |

### YOLO Detection

| Parameter | Default | Description |
|---|---|---|
| `YOLO_CONF` | `0.30` | Minimum detection confidence |
| `YOLO_IOU` | `0.45` | IoU threshold for NMS |
| `IMG_SIZE` | `640` | YOLO input size |

### CLIP Similarity

| Parameter | Default | Description |
|---|---|---|
| `CLIP_SIM_THRESHOLD` | `0.26` | Minimum similarity score to consider a match |
| `CLIP_MARGIN_THRESHOLD` | `0.035` | Minimum margin over the second-best candidate |
| `EMA_KEEP` | `0.85` | EMA memory weight (higher = smoother) |
| `MIN_HITS` | `5` | Positive hits required before confirming a match |
| `CLIP_INTERVAL` | `3` | Update CLIP every N frames |

### Crossing Detection

| Parameter | Default | Description |
|---|---|---|
| `CENTER_BAND_X` | `0.26` | Half-width of the central road band (normalized) |
| `ROAD_Y_MIN/MAX` | `0.22 / 0.92` | Vertical road zone bounds (normalized) |
| `CROSSING_SCORE_ON` | `3.6` | Score threshold to activate the `"crossing"` state |
| `MIN_LATERAL_DISP_BBOX` | `0.40` | Minimum lateral displacement (bbox-normalized) |
| `MIN_VERTICAL_PROGRESS_BBOX` | `0.15` | Minimum vertical progress toward camera |
| `STATE_SWITCH_MIN_FRAMES` | `4` | Frames of consistency required to switch state |

### Tracking & Re-ID

| Parameter | Default | Description |
|---|---|---|
| `MAX_TRACK_AGE` | `150` | Frames before a track is dropped |
| `KEEP_LOST_TRACKS_FOR` | `90` | Frames to keep lost tracks for Re-ID |
| `REID_MAX_CENTER_DIST` | `0.10` | Max normalized center distance for Re-ID |
| `REID_MIN_IOU` | `0.15` | Minimum IoU for Re-ID |

### Visualization

| Parameter | Default | Options |
|---|---|---|
| `VIS_MODE` | `"mask"` | `"box"` / `"mask"` / `"both"` |
| `MASK_ALPHA` | `0.45` | Mask transparency (0–1) |
| `DRAW_ALL` | `False` | If `True`, render all detections regardless of CLIP match |
| `DRAW_COUNTERS` | `True` | Show counter panel on frame |

---

## 🧠 How It Works

```
Video frame
    │
    ▼
YOLO (segment) ──► bbox + polygon mask + track_id
    │
    ▼
ByteTrack / BoT-SORT ──► stable multi-frame tracking
    │
    ├──► Crossing detection ──► composite score (velocity, road zone, trajectory)
    │
    └──► CLIP every N frames ──► similarity with text queries
              │
              ▼
         EMA smoothing ──► confirmed match ──► count with anti-double logic
```

### Crossing Detection

The `"crossing"` state is triggered by a **composite score** that accounts for:
- Presence inside the central road zone
- Lateral velocity normalized to the bounding box width
- Vertical progress toward the camera
- Road zone entry from outside
- Temporal persistence via a hysteresis state machine

### Lightweight Re-ID

When a track disappears and reappears, the system attempts to link it to its previous identity using **IoU + center distance + semantic query**, preventing double-counts for the same subject.

