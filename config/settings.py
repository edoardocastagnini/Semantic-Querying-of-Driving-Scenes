# ── I/O ─────────────────────────────────────────────────────────────────────
INPUT_VIDEO  = "videos/ny_3_cut.mp4"
OUTPUT_VIDEO = "outputs/output_crossing_open_vocab_seg.mp4"
OUTPUT_JSON  = "outputs/output_crossing_open_vocab_seg.json"

# ── CLIP semantic queries ────────────────────────────────────────────────────
QUERIES = [
    "a work truck",
    "a classic new york taxi",
    "a pedestrian"
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

# ── Models ──────────────────────────────────────────────────────────────────
YOLO_MODEL      = "yolo26m-finetuned-segmentation.pt"
TRAFFIC_SIGN_MODEL = "sign_detection_model_finetuned.pt"  
CLIP_MODEL      = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
TRACKER         = "botsort.yaml" 

# ── YOLO detection ───────────────────────────────────────────────────────────
ROAD_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
YOLO_CONF    = 0.26
YOLO_IOU     = 0.45
IMG_SIZE     = 640

# ── Traffic sign YOLO detection ──────────────────────────────────────────────
TRAFFIC_SIGN_CLASS_NAMES = {
    0: "Green Light",
    1: "Red Light",
    2: "Speed Limit 20",
    3: "Speed Limit 30",
    4: "Speed Limit 40",
    5: "Speed Limit 50",
    6: "Speed Limit 60",
    7: "Speed Limit 70",
    8: "Speed Limit 80",
    9: "Speed Limit 90",
    10: "Speed Limit 100",
    11: "Speed Limit 110",
    12: "Speed Limit 120",
    13: "Stop",
    14: "all",
    15: "crosswalk",
    16: "Pedestrians crossing",
    17: "One way",
    18: "Roundabout",
    19: "No entry",
    20: "No parking",
    21: "No stopping",
    22: "Yield",
    23: "Priority road",
    24: "No turn",
    25: "Parking",
}


TRAFFIC_SIGN_CLASSES = []
TRAFFIC_SIGN_CONF    = 0.25
TRAFFIC_SIGN_IOU     = 0.45
TRAFFIC_SIGN_IMG_SIZE = 640
DRAW_TRAFFIC_SIGNS   = True
TRAFFIC_SIGN_SMOOTHING_ENABLED = True
TRAFFIC_SIGN_BOX_EMA_KEEP = 0.65
TRAFFIC_SIGN_MIN_HITS_TO_DRAW = 2
TRAFFIC_SIGN_VISUAL_PERSIST_FRAMES = 6
TRAFFIC_SIGN_MAX_TRACK_AGE = 12
TRAFFIC_SIGN_MATCH_IOU = 0.15
TRAFFIC_SIGN_MATCH_CENTER_DIST = 0.035
TRAFFIC_SIGN_ROUTE_MODE = "clip"  # "clip" | "aliases"
TRAFFIC_SIGN_ROUTE_SIM_THRESHOLD = 0.72
TRAFFIC_SIGN_ROUTE_MARGIN_THRESHOLD = 0.035
TRAFFIC_SIGN_ROUTE_AMBIGUOUS_MARGIN = 0.030
TRAFFIC_SIGN_ROUTE_TOP_K = 3
TRAFFIC_SIGN_ROUTE_GROUP_THRESHOLD = 0.76
TRAFFIC_SIGN_ROUTE_SIGNLIKE_THRESHOLD = 0.64
TRAFFIC_SIGN_ROUTE_GROUP_MARGIN = 0.030
TRAFFIC_SIGN_ROUTE_NEGATIVE_MARGIN = 0.020
TRAFFIC_SIGN_ROUTE_NEGATIVE_QUERIES = [
    "human",
    "person",
    "pedestrian",
    "bus",
    "car",
    "vehicle",
    "truck",
    "motorcycle",
    "bicycle",
    "road",
    "building",
    "tree",
]

# ── CLIP similarity ──────────────────────────────────────────────────────────
CLIP_SIM_THRESHOLD    = 0.26
CLIP_MARGIN_THRESHOLD = 0.04
CLIP_NEGATIVE_MARGIN_THRESHOLD = 0.030
CLIP_NEGATIVE_QUERIES = [
    "road surface",
    "asphalt",
    "shadow",
    "road marking",
    "tree",
    "building",
    "background clutter",
]
CLIP_INTERVAL         = 3       # compute CLIP features every N frames
EMA_KEEP              = 0.0     # no EMA inertia: use the aggregated CLIP score directly
MIN_HITS              = 1      # one valid aggregated temporal decision confirms the label

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
CROP_PAD               = 0.1   # fallback bbox padding used by the single-crop path
MASK_CROP_PAD          = 0.02  # tight padding around the segmentation-mask bbox
CONTEXT_CROP_PAD       = 0.15  # wider padding around the YOLO bbox for CLIP context
MIN_CROP_AREA          = 1   # reject crops smaller than this many pixels before CLIP
USE_MASK_FOR_CLIP_CROP = True
CLIP_MULTI_VIEW_ENABLED = True
CLIP_VIEW_WEIGHTS = {
    "mask": 0.65,
    "clean_box_context": 0.35,
}
CLIP_CONTEXT_NEUTRAL_COLOR = (114, 114, 114)
CLIP_CONTEXT_OCCLUDER_MODE = "blur"  # "blur" | "neutral"
CLIP_CONTEXT_BLUR_KERNEL = 31
CLIP_TEMPORAL_AGGREGATION_ENABLED = True
CLIP_TEMPORAL_WINDOW = 4
CLIP_TEMPORAL_AGGREGATION = "mean"  # "median" | "mean"

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
MASK_BORDER_THICKNESS         = 1
DRAW_MASK_LABELS              = True
VISUAL_PERSIST_FRAMES         = 4
ALLOW_MASK_REUSE_WHEN_MISSING = True
