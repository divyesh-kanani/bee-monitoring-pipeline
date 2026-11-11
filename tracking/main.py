"""Run frame-by-frame object detection and ByteTrack association."""

import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import yaml

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracking.detector import Detector
from tracking.tracker import Tracker
from tracking.video_utils import draw_tracks, encode_h264, open_video, open_video_writer


TRACKING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACKING_DIR.parent
CONFIG_PATH = TRACKING_DIR / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as error:
        raise OSError(f"Could not read configuration file {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(
            f"Configuration file contains invalid YAML: {path}: {error}"
        ) from error
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    for section in ("detection", "tracking", "video"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping.")
    return config


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    detection_config = config["detection"]
    tracking_config = config["tracking"]
    video_config = config["video"]
    if float(detection_config["confidence_threshold"]) > float(
        tracking_config["low_detection_threshold"]
    ):
        raise ValueError(
            "detection.confidence_threshold must not exceed "
            "tracking.low_detection_threshold."
        )
    input_path = _project_path(str(video_config["input_path"]))
    output_path = _project_path(str(video_config["output_path"]))
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output video paths must be different.")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("video.output_path must use the .mp4 extension.")

    detector = Detector(detection_config, PROJECT_ROOT)
    tracker = Tracker(tracking_config)
    capture, source_fps, width, height = open_video(input_path)
    output_fps = video_config.get("output_fps")
    if output_fps is None:
        output_fps = source_fps
    else:
        output_fps = float(output_fps)
        if not math.isfinite(output_fps) or output_fps <= 0:
            capture.release()
            raise ValueError("video.output_fps must be a positive number or null.")

    try:
        writer, temporary_path = open_video_writer(
            output_path, output_fps, (width, height)
        )
    except OSError:
        capture.release()
        raise
    frame_count = 0
    detection_count = 0
    tracks_by_id: dict[int, dict[str, int]] = {}
    started = time.perf_counter()
    completed = False

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            detections = detector.detect(frame)
            detection_count += len(detections)
            tracks = tracker.update(detections, frame.shape[:2])
            for track in tracks:
                record = tracks_by_id.setdefault(
                    track.tracker_id,
                    {
                        "first_frame": frame_count,
                        "last_frame": frame_count,
                        "frames_tracked": 0,
                    },
                )
                record["last_frame"] = frame_count
                record["frames_tracked"] += 1

            annotated = draw_tracks(frame, tracks)
            writer.write(annotated)
            frame_count += 1

            if bool(video_config.get("display", False)):
                cv2.imshow("Bee detection and tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        completed = True
    finally:
        capture.release()
        writer.release()
        if bool(video_config.get("display", False)):
            cv2.destroyAllWindows()
        if not completed:
            temporary_path.unlink(missing_ok=True)

    if frame_count == 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(f"No frames could be read from input video: {input_path}")

    elapsed = time.perf_counter() - started
    try:
        encode_h264(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    durations = [
        (record["last_frame"] - record["first_frame"] + 1) / source_fps
        for record in tracks_by_id.values()
    ]
    summary = {
        "input_video": str(input_path),
        "output_video": str(output_path),
        "total_unique_track_ids": len(tracks_by_id),
        "detections": detection_count,
        "frames_processed": frame_count,
        "processing_fps": frame_count / elapsed if elapsed > 0 else 0.0,
        "source_fps": source_fps,
        "output_fps": output_fps,
        "track_durations_seconds": {
            "min": min(durations) if durations else 0.0,
            "median": median(durations) if durations else 0.0,
            "mean": mean(durations) if durations else 0.0,
            "max": max(durations) if durations else 0.0,
        },
        "tracks": [
            {
                "track_id": track_id,
                **record,
                "duration_seconds": (
                    record["last_frame"] - record["first_frame"] + 1
                )
                / source_fps,
            }
            for track_id, record in sorted(tracks_by_id.items())
        ],
        "identity_note": (
            "Track IDs are temporary within-video tracker labels; they do not identify "
            "individual bees biologically."
        ),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Tracked video: {output_path}")
    print(f"Summary: {summary_path}")
    return summary


def main() -> int:
    try:
        run_pipeline(load_config())
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
