"""Video input/output helpers and tracked-detection visualization."""

import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from tracking.tracker import TrackedDetection


def open_video(path: Path) -> tuple[cv2.VideoCapture, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open input video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Input video has invalid FPS or dimensions: {path}")
    return capture, fps, width, height


def open_video_writer(
    output_path: Path, fps: float, size: tuple[int, int]
) -> tuple[cv2.VideoWriter, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.temporary.mp4")
    temporary_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        writer.release()
        raise OSError(f"Could not create temporary video: {temporary_path}")
    return writer, temporary_path


def encode_h264(source: Path, destination: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(destination),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        destination.unlink(missing_ok=True)
        details = (
            (error.stderr or "").strip()
            if isinstance(error, subprocess.CalledProcessError)
            else str(error)
        )
        raise OSError(f"Could not encode an H.264 MP4: {details}") from error


def draw_tracks(frame: np.ndarray, tracks: list[TrackedDetection]) -> np.ndarray:
    annotated = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = (int(round(value)) for value in track.xyxy)
        color = _track_color(track.tracker_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{track.class_name} {track.confidence:.2f} ID {track.tracker_id}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_top = max(0, y1 - text_height - 8)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (x1 + text_width, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1, max(text_height, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )
    return annotated


def _track_color(tracker_id: int) -> tuple[int, int, int]:
    hue = (tracker_id * 37) % 180
    pixel = np.uint8([[[hue, 210, 255]]])
    return tuple(
        int(channel) for channel in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    )
