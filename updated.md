# Update Notes

## Multi-View CLIP Crops

CLIP can now evaluate more than one visual patch for each YOLO-tracked object.

The current views are:

- `mask`: the segmented object crop, with pixels outside the segmentation mask removed.
- `clean_box_context`: the padded YOLO bounding-box crop, where other segmented objects inside the crop are painted neutral gray.

The two CLIP score dictionaries are combined before being passed to the existing semantic logic.

Controlled in `config/settings.py`:

```python
CLIP_MULTI_VIEW_ENABLED = True
CLIP_VIEW_WEIGHTS = {
    "mask": 0.70,
    "clean_box_context": 0.30,
}
CLIP_CONTEXT_NEUTRAL_COLOR = (114, 114, 114)
```

Increase `clean_box_context` weight when object context helps, for example taxi markings or roof lights. Increase `mask` weight when neighboring objects still confuse CLIP.

## Short Temporal Aggregation

CLIP decisions are now aggregated over a short window before updating the semantic state.

With the default settings, each semantic update uses the median of 3 combined CLIP observations for the same track.

Controlled in `config/settings.py`:

```python
CLIP_TEMPORAL_AGGREGATION_ENABLED = True
CLIP_TEMPORAL_WINDOW = 3
CLIP_TEMPORAL_AGGREGATION = "median"  # "median" | "mean"
```

Use `median` to reduce the effect of one bad or occluded frame. Use `mean` if scores are stable and you want smoother averaging.

## Related Thresholds

Because each semantic update now already combines multiple views and multiple frames, these parameters may need to be less conservative:

```python
CLIP_SIM_THRESHOLD = 0.21
CLIP_MARGIN_THRESHOLD = 0.035
CLIP_NEGATIVE_MARGIN_THRESHOLD = 0.030
MIN_HITS = 5
EMA_KEEP = 0.85
```

Suggested first tuning:

```python
MIN_HITS = 2
EMA_KEEP = 0.75
```

If labels appear too late, reduce `MIN_HITS` or `CLIP_TEMPORAL_WINDOW`. If false positives increase, raise `CLIP_SIM_THRESHOLD`, `CLIP_MARGIN_THRESHOLD`, or `MIN_HITS`.

## Debug Output

The JSON output now includes diagnostics for CLIP views and temporal aggregation:

- `last_clip_views_used`
- `last_view_scores`
- `last_combined_scores`
- `last_temporal_scores`
- `last_temporal_window_size`

These fields can be used to compare whether the mask view, the clean context view, or the temporal aggregation is driving the final label.
