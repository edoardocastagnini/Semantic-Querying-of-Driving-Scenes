# ── I/O ─────────────────────────────────────────────────────────────────────
INPUT_VIDEO  = "videos/ny_1_trim.mp4"
OUTPUT_VIDEO = "outputs/output_crossing_open_vocab_seg.mp4"
OUTPUT_JSON  = "outputs/output_crossing_open_vocab_seg.json"

# ── Query semantiche CLIP ────────────────────────────────────────────────────
QUERIES = [
    "person",
    "bus",
    "truck"
]

HUMAN_QUERY_PROTOTYPES = [
    "person",
    "pedestrian",
    "human",
    "a person walking",
    "a pedestrian in a road scene",
    "a human in a dashcam image",
]
HUMAN_QUERY_THRESHOLD = 0.85

# ── Modelli ──────────────────────────────────────────────────────────────────
YOLO_MODEL      = "runs/segment/train-2/weights/best.pt"
CLIP_MODEL      = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
TRACKER         = "botsort.yaml"   # or "bytetrack.yaml"

# ── YOLO detection ───────────────────────────────────────────────────────────
ROAD_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
YOLO_CONF    = 0.30
YOLO_IOU     = 0.45
IMG_SIZE     = 640

# ── CLIP similarity ──────────────────────────────────────────────────────────
CLIP_SIM_THRESHOLD    = 0.26
CLIP_MARGIN_THRESHOLD = 0.035
CLIP_INTERVAL         = 3       # compute CLIP features every N frames 
EMA_KEEP              = 0.85    # exponential moving average for smoothing CLIP scores over time
MIN_HITS              = 5       # minimum hits per track to be considered valid

# ── Tracking ─────────────────────────────────────────────────────────────────
MIN_FRAMES_TO_COUNT = 15
MAX_TRACK_AGE       = 150       # frame after which a lost track is deleted
HISTORY_LEN         = 30        # number of past frames to keep in track state for logic and visualization
MIN_TRACK_FRAMES    = 5

# ── Re-ID (light re-identification) ─────────────────────────────────────────
KEEP_LOST_TRACKS_FOR    = 90
REID_MAX_CENTER_DIST    = 0.10
REID_MIN_IOU            = 0.15
REID_REQUIRE_SAME_QUERY = True

# ── Counting / cooldown ──────────────────────────────────────────────────────
COUNT_COOLDOWN_FRAMES         = 180
COUNT_COOLDOWN_DIST           = 0.22
MIN_MEAN_DET_CONF_FOR_COUNT   = 0.35
MIN_LABEL_DOMINANCE_FOR_COUNT = 0.70
CONF_HISTORY_LEN              = 20
QUERY_HISTORY_LEN             = 20

# ── Crop & mask ──────────────────────────────────────────────────────────────
CROP_PAD               = 0.05
MIN_CROP_AREA          = 1600
USE_MASK_FOR_CLIP_CROP = True

# ── Crossing detection ───────────────────────────────────────────────────────
CENTER_BAND_X               = 0.26
ROAD_Y_MIN                  = 0.22
ROAD_Y_MAX                  = 0.92
CROSSING_SCORE_ON           = 3.6
CROSSING_KEEP_FRAMES        = 10
SCORE_IN_ROAD               = 1.0
SCORE_LATERAL_MOVE          = 0.9
SCORE_TOWARD_CAMERA         = 0.5
SCORE_OUTSIDE_ROAD          = -1.0
STATE_SWITCH_MIN_FRAMES     = 4
MIN_LATERAL_DISP_BBOX       = 0.40
MIN_VERTICAL_PROGRESS_BBOX  = 0.15
MIN_ABS_LATERAL_TRAVEL_BBOX = 0.80
MIN_IN_ROAD_FRAMES          = 5

# ── Visualization ──────────────────────────────────────────────────────────
DRAW_ALL                      = False
DRAW_COUNTERS                 = True
BOX_THICKNESS                 = 2
VIS_MODE                      = "mask"   # "box" | "mask" | "both"
MASK_ALPHA                    = 0.45
MASK_BORDER_THICKNESS         = 2
DRAW_MASK_LABELS              = True
VISUAL_PERSIST_FRAMES         = 4
ALLOW_MASK_REUSE_WHEN_MISSING = True
