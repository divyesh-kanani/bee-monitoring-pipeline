# Bee detection and tracking

The pipeline uses the existing Ultralytics ByteTrack implementation and keeps
model inference separate from tracking through a shared detection record.
Ultralytics, RF-DETR, OpenCV, PyYAML, NumPy, and `imageio-ffmpeg` are already
listed in the repository's `requirements.txt`.

## Use tracking from the dashboard

Run the existing dashboard with `streamlit run src/hard_negative_app.py`, open
**Detection**, upload the model and video, then enable **Apply object tracking
after detection**. Select ByteTrack, adjust its thresholds and track buffer if
needed, and click **Start detection and tracking**. The dashboard displays and
offers downloads for the tracked MP4, detections CSV (including track IDs), and
tracking summary. Tracking processes every frame in the selected time range.

## Configure and run

Edit [`config.yaml`](config.yaml) to select the model, weights, video, and
detection/tracking/video settings. Relative paths are resolved from the
repository root. The default configuration uses the included YOLO weights and
an existing sample video.

```bash
python tracking/main.py
```

The pipeline writes an H.264 MP4 and a JSON summary under
`tracking/output_videos/`. Put other input clips in `tracking/input_videos/` if
desired, then set `video.input_path` in the configuration. The output FPS
defaults to the source video's FPS.

## Models

- **YOLO:** set `detection.model_type: YOLO` and point `weights` at a `.pt`
  checkpoint.
- **RF-DETR:** set `detection.model_type: RF-DETR` and point `weights` at a
  compatible checkpoint. The adapter follows the `RFDETRLarge.from_checkpoint`
  and `predict` interface used by this repository. RF-DETR weights are not
  included, so provide a downloaded or locally trained checkpoint to run it.
  The adapter's input/output normalization is tested, but full inference with a
  real RF-DETR checkpoint has not been verified in this workspace.

`detection.confidence_threshold` is the minimum score given to ByteTrack.
ByteTrack divides those detections using its high and low thresholds; the
activation threshold controls which unmatched detections can start a new
track. Keep the detector confidence threshold no higher than the low tracking
threshold so low-score candidates remain available. `track_buffer` is measured
in frames. The output summary's duration is
the elapsed time from a track's first to last observed frame. Track IDs are
short-term labels assigned within this video, not biological bee identities.
