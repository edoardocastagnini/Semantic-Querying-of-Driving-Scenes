# Update Notes

## Multi-View CLIP Crops

CLIP can now evaluate more than one visual patch for each YOLO-tracked object.

The current views are:

- `mask`: the segmented object crop, with pixels outside the segmentation mask removed.
- `clean_box_context`: the padded YOLO bounding-box crop, with a wider context area around the object.

The two CLIP score dictionaries are combined before being passed to the existing semantic logic.

Controlled in `config/settings.py`:

```python
CLIP_MULTI_VIEW_ENABLED = True
CLIP_VIEW_WEIGHTS = {
    "mask": 0.65,
    "clean_box_context": 0.35,
}
MASK_CROP_PAD = 0.02
CONTEXT_CROP_PAD = 0.12
CLIP_CONTEXT_OCCLUDER_MODE = "blur"  # "blur" | "neutral"
CLIP_CONTEXT_BLUR_KERNEL = 31
```

Increase `clean_box_context` weight when object context helps, for example taxi markings or roof lights. Increase `mask` weight when neighboring objects still confuse CLIP.

`CONTEXT_CROP_PAD` controls how much background context CLIP sees. In the `clean_box_context` view, every non-target YOLO segmentation mask that overlaps the padded context crop is suppressed across the whole crop. This prevents CLIP from using nearby detected objects inside the context window as evidence for the target object.

Set `CLIP_CONTEXT_OCCLUDER_MODE = "neutral"` to use the previous neutral-gray suppression behavior.

## Short Temporal Aggregation

CLIP prediction now uses one main temporal mitigation step: aggregate a short sequence of CLIP observations before accepting the semantic decision.

With the current settings, each semantic update uses the mean of 3 combined CLIP observations for the same YOLO track. This reduces the risk that one bad crop, blur, occlusion, or weak mask produces the final prediction.

Controlled in `config/settings.py`:

```python
CLIP_INTERVAL = 3
CLIP_TEMPORAL_AGGREGATION_ENABLED = True
CLIP_TEMPORAL_WINDOW = 3
CLIP_TEMPORAL_AGGREGATION = "mean"  # "median" | "mean"
EMA_KEEP = 0.0
MIN_HITS = 2
```

The workflow is:

```text
YOLO track
-> CLIP crop every CLIP_INTERVAL frames
-> collect CLIP_TEMPORAL_WINDOW score outputs
-> aggregate scores with mean
-> apply CLIP thresholds
-> confirm the label
```

`CLIP_TEMPORAL_WINDOW` counts CLIP observations, not raw video frames. With `CLIP_INTERVAL = 3` and `CLIP_TEMPORAL_WINDOW = 3`, a normal semantic decision uses 3 CLIP observations spread across about 9 video frames. At the start of a new track, CLIP can run immediately until the first temporal window is filled.

`mean` smooths stable scores across the context window. Switch to `median` if one bad or occluded observation is frequently pulling the prediction away from the correct class.

`EMA_KEEP = 0.0` disables score inertia for the decision path. This avoids smoothing the already aggregated CLIP score a second time. `MIN_HITS = 1` is enough because one hit already represents a temporal decision over 3 CLIP observations.

## Related Thresholds

Because each semantic update now already combines multiple views and multiple frames, prediction stability is mainly controlled by:

```python
CLIP_SIM_THRESHOLD = 0.26
CLIP_MARGIN_THRESHOLD = 0.035
CLIP_NEGATIVE_MARGIN_THRESHOLD = 0.030
CLIP_TEMPORAL_WINDOW = 3
CLIP_TEMPORAL_AGGREGATION = "mean"
```

If labels appear too late, reduce `CLIP_TEMPORAL_WINDOW` to `2` or reduce `CLIP_INTERVAL` to `1`. If false positives increase, raise `CLIP_SIM_THRESHOLD`, raise `CLIP_MARGIN_THRESHOLD`, or set `MIN_HITS = 2`.

Counting remains separate from prediction. `MIN_FRAMES_TO_COUNT`, `MIN_LABEL_DOMINANCE_FOR_COUNT`, `MIN_MEAN_DET_CONF_FOR_COUNT`, and cooldown parameters still decide when a confirmed track is counted, not when CLIP predicts its label.

## Debug Output

The JSON output now includes diagnostics for CLIP views and temporal aggregation:

- `last_clip_views_used`
- `last_view_scores`
- `last_combined_scores`
- `last_temporal_scores`
- `last_temporal_window_size`

These fields can be used to compare whether the mask view, the clean context view, or the temporal aggregation is driving the final label.
