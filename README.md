# Computer Vision and Pattern Recognition - Project Group E 

  

# Semantic Querying of Driving Scenes

  
> Open-vocabulary detection, tracking, and counting on dashcam footage.

> Combines **YOLO** (instance segmentation), **BoT-SORT** (multi-object tracking),

> and **CLIP** (open-vocabulary classification) to identify and count objects defined by free-text queries

---
  ## Poster and Demos

For a more detailed overview of the project, please refer to the project poster: `CV_project_poster.pdf`.

Video demonstrations are available in the following Google Drive folder:  
https://drive.google.com/drive/folders/1Lch155zYHqawX1J1I48choJSvIj1SubM?usp=drive_link

---

 
## Features

  
-  **Instance segmentation** with pretrained YOLO26m model, finetuned for 100 epochs on the *COCO* classes relevant to dashcam driving scenes (e.g. `"car"`, `"person",` `"truck"`, `"traffic light"`, etc...).

-  **Stable multi-object tracking** with BoT-SORT.

-  **Open-vocabulary semantic classification** via CLIP (any query works: `"human being"`, `"vehicle"`, `"SUV car"`, …).

-  **Crossing detection** based on a composite score of velocity, trajectory, and road-zone position, allowing the model to detect pedestrians that are crossing the road.

- **Traffic sign detection** with a small YOLO11n model that runs in parallel when the user inputs queries relevant to traffic signs (e.g `"30km/h limit"`, `"stop sign"`, `"no parking sign"`, etc...), trained on *Mapillary Traffic Sign* dataset.

-  **Anti-double-count** system with spatial/temporal cooldown and lightweight Re-ID., 


---

## 📁 Project Structure

  

```
main/

│

├── config/

│ └── settings.py ← ✏️ EDIT THIS FILE to change any hyperparameter

│

├── core/

│ ├── clip_utils.py ← CLIP encoding (text & image), prompt ensemble

│ ├── geometry.py ← Geometry: bbox, IoU, distances, crop, mask

│ ├── crossing.py ← Crossing detection: score, velocity, state machine

│ ├── tracking.py ← Track lifecycle: creation, Re-ID, cleanup

│ └── semantics.py ← EMA scores, match validation, counting

│

├── ioutils/

│ ├── video_io.py ← VideoCapture and VideoWriter setup

│ └── output_writer.py ← JSON output builder and writer

│

├── visualization/

│ └── drawing.py ← Rendering: boxes, masks, labels, counters

│

├── models/

│ └── loader.py ← YOLO + CLIP loader, device selection

│

├── main.py ← Entry point: global runtime state + main loop

├── requirements.txt

└── README.md

```

  
  

---


## Quick Start

  

1. Put your input video in `videos/` and set the correct path in `config/settings.py` 

2. Set your queries and parameters in `config/settings.py` 

```python

INPUT_VIDEO = "videos/input.mp4"

OUTPUT_VIDEO = "outputs/output_crossing_open_vocab_seg.mp4"

OUTPUT_JSON = "outputs/output_crossing_open_vocab_seg.json"

```

 
### Semantic Queries

 
```python

QUERIES = ["person", "vehicle"] # Add any free-text query in natural language

```


4. Run:

  

```bash

python  main.py

```

  

Outputs are saved to `outputs/`:

-  `output_crossing_open_vocab_seg.mp4` — annotated video

-  `output_crossing_open_vocab_seg.json` — full processing summary

  
 
   
---


