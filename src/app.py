import cv2
from flask import Flask, Response, request, redirect, url_for
from detect import load_detector
from clip_utils import encode_text, encode_crop
import torch

# ----------------------------
# Device
# ----------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Config
# ----------------------------
VIDEO_PATH = "videos/drive3.mp4"
FRAME_SIZE = (720, 480)
JPEG_QUALITY = 60

SKIP_EVERY_OTHER_FRAME = True
YOLO_CONF_THRESHOLD = 0.40
CLIP_SIM_THRESHOLD = 0.25
EMBEDDING_REFRESH_EVERY = 180
MAX_TRACK_AGE = 360
BBOX_PADDING = 10

# ----------------------------
# Global state
# ----------------------------
queries = ["car"]
text_features = encode_text(queries)

frame_count = 0

track_embeddings = {}
track_last_seen = {}

# ----------------------------
# Model + video
# ----------------------------
model = load_detector()
model.to(device)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Impossibile aprire il video: {VIDEO_PATH}")


import hashlib

def get_color(query):
    h = int(hashlib.md5(query.encode()).hexdigest(), 16)

    r = (h & 255)
    g = (h >> 8) & 255
    b = (h >> 16) & 255

    return (int(b), int(g), int(r))  # OpenCV usa BGR


def clamp_box(x1, y1, x2, y2, width, height, pad=0):
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width - 1, x2 + pad)
    y2 = min(height - 1, y2 + pad)
    return x1, y1, x2, y2


def generate_frames():
    global frame_count, text_features, track_embeddings, track_last_seen, cap

    while True:
        ret, frame = cap.read()

        # loop del video
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, FRAME_SIZE)
        frame_count += 1

        if SKIP_EVERY_OTHER_FRAME and frame_count % 2 != 0:
            continue

        # Tracking YOLO
        # Se vuoi limitare le classi, puoi aggiungere classes=[...]
        results = model.track(frame, persist=True, verbose=False, device=device)
        boxes = results[0].boxes

        annotated = frame.copy()
        frame_h, frame_w = frame.shape[:2]

        if boxes is not None and boxes.id is not None:
            xyxy_list = boxes.xyxy
            id_list = boxes.id
            conf_list = boxes.conf if boxes.conf is not None else None

            for i, (box, track_id_tensor) in enumerate(zip(xyxy_list, id_list)):
                x1, y1, x2, y2 = map(int, box.tolist())
                track_id = int(track_id_tensor.item())

                # confidence filter
                if conf_list is not None:
                    conf = float(conf_list[i].item())
                    if conf < YOLO_CONF_THRESHOLD:
                        continue
                else:
                    conf = 0.0

                # padding box
                x1, y1, x2, y2 = clamp_box(
                    x1, y1, x2, y2,
                    width=frame_w,
                    height=frame_h,
                    pad=BBOX_PADDING
                )

                # aggiorna "last seen"
                track_last_seen[track_id] = frame_count

                # aggiorna embedding:
                # - se il track è nuovo
                # - oppure periodicamente, per correggere eventuali errori
                should_refresh = (
                    track_id not in track_embeddings
                    or frame_count % EMBEDDING_REFRESH_EVERY == 0
                )

                if should_refresh:
                    image_features = encode_crop(frame, (x1, y1, x2, y2))
                    if image_features is not None:
                        track_embeddings[track_id] = image_features

                label = None
                score = None

                # similarity veloce con embedding già salvato
                if track_id in track_embeddings:
                    image_features = track_embeddings[track_id]
                    similarity = (image_features @ text_features.T).squeeze(0)

                    best_idx = similarity.argmax().item()
                    score = float(similarity[best_idx].item())

                    if score > CLIP_SIM_THRESHOLD:
                        label = queries[best_idx]

                if label is not None:
                    color = get_color(label)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"ID {track_id} - {label} ({score:.2f})",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

        # cleanup track vecchi
        to_delete = [
            tid for tid, last_seen in track_last_seen.items()
            if frame_count - last_seen > MAX_TRACK_AGE
        ]

        for tid in to_delete:
            track_embeddings.pop(tid, None)
            track_last_seen.pop(tid, None)

        # overlay info utili
        cv2.putText(
            annotated,
            f"Queries: {', '.join(queries)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # encoding frame per MJPEG
        ok, buffer = cv2.imencode(
            ".jpg",
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        if not ok:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )




@app.route("/get_queries")
def get_queries():
    return ",".join(queries)



@app.route("/add_query", methods=["POST"])
def add_query():
    global queries, text_features

    new_query = request.form.get("query", "").strip()

    if new_query and new_query not in queries:
        queries.append(new_query)
        text_features = encode_text(queries)

    return redirect(url_for("ui"))


@app.route("/remove_query", methods=["POST"])
def remove_query():
    global queries, text_features

    q = request.form.get("query")

    if q in queries:
        queries.remove(q)

    if len(queries) == 0:
        queries = ["car"]

    text_features = encode_text(queries)

    return redirect(url_for("ui"))

@app.route("/ui")
def ui():
    return """
    <html>
        <head>
            <title>CV Demo</title>
            <style>
                body {
                    font-family: Arial;
                    background: #111;
                    color: #eee;
                    margin: 24px;
                }
                input, button {
                    font-size: 16px;
                    padding: 6px;
                }
                img {
                    border: 1px solid #444;
                }
                .query {
                    margin: 4px;
                    padding: 4px 8px;
                    background: #333;
                    display: inline-block;
                }
            </style>
        </head>
        <body>

            <h2>Video stream</h2>
            <img src="/" width="720"><br><br>

            <h3>Aggiungi query</h3>
            <form action="/add_query" method="post">
                <input type="text" name="query" placeholder="es: red car">
                <button type="submit">Add</button>
            </form>

            <h3>Query attive</h3>
            <div id="queries"></div>

            <script>
                function loadQueries() {
                    fetch('/get_queries')
                        .then(res => res.text())
                        .then(data => {
                            const container = document.getElementById("queries");
                            container.innerHTML = "";

                            const list = data.split(",");

                            list.forEach(q => {
                                if(q.trim() === "") return;

                                const div = document.createElement("div");
                                div.className = "query";
                                div.innerHTML = q + 
                                    " <form style='display:inline' action='/remove_query' method='post'>" +
                                    "<input type='hidden' name='query' value='" + q + "'>" +
                                    "<button>x</button></form>";

                                container.appendChild(div);
                            });
                        });
                }

                setInterval(loadQueries, 1000);
                loadQueries();
            </script>

        </body>
    </html>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)