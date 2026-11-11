"""Local Streamlit app for reviewing individual YOLO detections."""

import csv
import colorsys
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2
import imageio_ffmpeg
import numpy as np
import streamlit as st
import yaml
from PIL import Image
from streamlit_drawable_canvas import st_canvas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RFDETRLarge = None
YOLO = None

from hard_negative_mining import DEFAULT_DATA, VIDEO_EXTENSIONS, _read_class_names
from hard_negative_review import (
    annotated_preview_zip,
    create_dataset,
    extract_detections,
    frame_bounds,
    preview_frames,
    read_video_metadata,
    report_csv,
    sampled_frame_numbers,
    zip_directory,
)
SESSION_ROOT = Path(
    os.environ.get(
        "BEE_MINING_TMPDIR",
        os.environ.get(
            "BEE_MINING_DATA_DIR",
            str(Path(tempfile.gettempdir()) / "bee-monitoring-dashboard"),
        ),
    )
).expanduser()
SESSION_MAX_AGE_SECONDS = max(
    60,
    int(float(os.environ.get("BEE_MINING_SESSION_MAX_AGE_HOURS", "24")) * 3600),
)
SESSION_CLEANUP_INTERVAL_SECONDS = max(
    60, int(os.environ.get("BEE_MINING_CLEANUP_INTERVAL_SECONDS", "600"))
)
VIDEO_TYPES = sorted(suffix.lstrip(".") for suffix in VIDEO_EXTENSIONS)
MODEL_BACKENDS = ["Ultralytics YOLO", "RF-DETR"]


def _rfdetr_available() -> bool:
    if RFDETRLarge is not None:
        return True
    try:
        return importlib.util.find_spec("rfdetr") is not None
    except (ImportError, ValueError):
        return False


def _get_rfdetr_class() -> Any:
    global RFDETRLarge
    if RFDETRLarge is not None:
        return RFDETRLarge
    try:
        from rfdetr import RFDETRLarge as RFDETRLargeClass
    except Exception as error:
        raise ValueError(
            f"RF-DETR is not available in the current environment: {error}"
        ) from error
    RFDETRLarge = RFDETRLargeClass
    return RFDETRLarge


def _get_yolo_class() -> Any:
    global YOLO
    if YOLO is None:
        from ultralytics import YOLO as YOLOClass

        YOLO = YOLOClass
    return YOLO


def _supported_weight_extensions(model_type: str) -> list[str]:
    if model_type == "Ultralytics YOLO":
        return ["pt"]
    if model_type == "RF-DETR":
        return ["pth", "pt"]
    raise ValueError(f"Unsupported model backend: {model_type}")


def _session_id() -> str:
    if not st.session_state.get("_bee_dashboard_initialized", False):
        previous_id = st.query_params.get("session")
        if previous_id:
            _clear_session_data(str(previous_id))
        st.session_state["_bee_dashboard_initialized"] = True
        session_id = uuid.uuid4().hex
        st.query_params["session"] = session_id
        return session_id

    value = st.query_params.get("session")
    try:
        session_id = uuid.UUID(str(value)).hex if value else uuid.uuid4().hex
    except ValueError:
        session_id = uuid.uuid4().hex
    st.query_params["session"] = session_id
    return session_id


def _clear_session_data(session_id: str, lock_held: bool = False) -> None:
    try:
        directory = _session_directory(session_id)
    except ValueError:
        return
    if not directory.is_dir() or directory.is_symlink():
        return
    if lock_held:
        shutil.rmtree(directory)
        return
    lock_path = directory / ".lock"
    with lock_path.open("a") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        shutil.rmtree(directory)


def _session_directory(session_id: str) -> Path:
    return SESSION_ROOT / uuid.UUID(session_id).hex


def _touch_session(session_id: str) -> None:
    directory = _session_directory(session_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".activity").touch()


@contextmanager
def _session_access(session_id: str) -> Iterator[None]:
    directory = _session_directory(session_id)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        _touch_session(session_id)
        try:
            yield
        finally:
            if directory.is_dir():
                _touch_session(session_id)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _cleanup_abandoned_sessions(now: float | None = None) -> int:
    if not SESSION_ROOT.is_dir():
        return 0
    current_time = time.time() if now is None else now
    removed = 0
    for directory in SESSION_ROOT.iterdir():
        if not directory.is_dir() or directory.is_symlink():
            continue
        try:
            uuid.UUID(directory.name)
        except ValueError:
            continue
        lock_path = directory / ".lock"
        with lock_path.open("a") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            activity_path = directory / ".activity"
            last_active = activity_path.stat().st_mtime if activity_path.exists() else directory.stat().st_mtime
            if current_time - last_active > SESSION_MAX_AGE_SECONDS:
                shutil.rmtree(directory)
                removed += 1
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return removed


@st.cache_resource
def _start_cleanup_worker() -> threading.Thread:
    def run() -> None:
        while True:
            _cleanup_abandoned_sessions()
            time.sleep(SESSION_CLEANUP_INTERVAL_SECONDS)

    worker = threading.Thread(target=run, name="bee-session-cleanup", daemon=True)
    worker.start()
    return worker


def _read_then_delete(path: Path, session_id: str) -> bytes:
    directory = _session_directory(session_id).resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(directory):
        raise ValueError("Download path is outside the current temporary session.")
    with _session_access(session_id):
        try:
            return resolved_path.read_bytes()
        finally:
            resolved_path.unlink(missing_ok=True)


def _delete_session_artifact(session_id: str, filename: str) -> None:
    if Path(filename).name != filename:
        raise ValueError("Invalid temporary artifact name.")
    (_session_directory(session_id) / filename).unlink(missing_ok=True)


def _load_state(session_id: str) -> dict[str, Any]:
    path = _session_directory(session_id) / "review.json"
    if not path.is_file():
        return {"videos": {}, "frames": [], "class_names": [], "exports": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load saved review session: {error}") from error
    if not isinstance(state, dict) or not isinstance(state.get("videos"), dict) or not isinstance(
        state.get("frames"), list
    ):
        raise ValueError(f"Saved review session has an invalid format: {path}")
    return state


def _save_state(session_id: str, state: dict[str, Any]) -> None:
    directory = _session_directory(session_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "review.json"
    temporary_path = directory / "review.json.tmp"
    temporary_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _store_upload(
    session_id: str,
    uploaded_file: Any,
    suffix: str,
    category: str,
    upload_cache: dict[str, Any] | None = None,
) -> tuple[str, Path]:
    directory = _session_directory(session_id) / (
        "detection_preview" if category == "detection" else "uploads"
    )
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(uploaded_file.name).name)
    upload_id = getattr(uploaded_file, "file_id", None)
    cache_key = (
        f"{session_id}:upload:{category}:{upload_id}" if upload_id is not None else None
    )
    cache = st.session_state if upload_cache is None else upload_cache
    cached = cache.get(cache_key) if cache_key else None
    if (
        isinstance(cached, dict)
        and cached.get("name") == uploaded_file.name
        and cached.get("size") == uploaded_file.size
        and cached.get("suffix") == suffix
        and Path(cached.get("path", "")).is_file()
    ):
        return cached["digest"], Path(cached["path"])

    contents = uploaded_file.getbuffer()
    digest = hashlib.sha256(contents).hexdigest()
    if category == "weights":
        path = directory / f"{digest}{suffix}"
    elif category == "detection":
        path = directory / f"{digest[:12]}_{safe_name}"
    else:
        path = directory / f"{digest[:12]}_{safe_name}"
    if not path.is_file():
        path.write_bytes(contents)
    if cache_key:
        cache[cache_key] = {
            "name": uploaded_file.name,
            "size": uploaded_file.size,
            "suffix": suffix,
            "digest": digest,
            "path": str(path),
        }
    return digest, path


def _stage_detection_decision(
    statuses: dict[str, str], detection_ids: set[str], status: str
) -> None:
    if status not in {"true_positive", "false_positive", "uncertain"}:
        raise ValueError(f"Unsupported review status: {status}")
    for detection_id in detection_ids:
        statuses[detection_id] = status


def _review_display_size(image_width: int, image_height: int) -> tuple[int, int]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    scale = min(1.0, 720 / image_width, 640 / image_height)
    return max(1, round(image_width * scale)), max(1, round(image_height * scale))


def _review_frame_images(
    frame: dict[str, Any], cache: dict[str, Any]
) -> dict[str, Any] | None:
    image_path = Path(frame["image_path"])
    if not image_path.is_file():
        return None
    stat = image_path.stat()
    cache_token = (
        frame["frame_id"],
        str(image_path),
        stat.st_mtime_ns,
        stat.st_size,
    )
    cached = cache.get("entry")
    if isinstance(cached, dict) and cached.get("token") == cache_token:
        return cached

    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    display_size = _review_display_size(width, height)
    display_bgr = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA)
    entry = {
        "token": cache_token,
        "image": image,
        "display_bgr": display_bgr,
        "display_rgb": cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB),
    }
    cache["entry"] = entry
    return entry


def _cached_video_previews(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    frames_to_skip: int,
    cache: dict[str, Any],
    cache_key: str,
) -> tuple[np.ndarray, ...]:
    video_stat = video_path.stat()
    cache_token = (
        str(video_path),
        video_stat.st_mtime_ns,
        video_stat.st_size,
        start_frame,
        end_frame,
        frames_to_skip,
    )
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and cached.get("token") == cache_token:
        return cached["frames"]

    thumbnails = []
    for frame in preview_frames(video_path, start_frame, end_frame, frames_to_skip):
        height, width = frame.shape[:2]
        display_size = _review_display_size(width, height)
        thumbnail = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
        thumbnails.append(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
    result = tuple(thumbnails)
    cache[cache_key] = {"token": cache_token, "frames": result}
    return result


def _delete_detections(
    frame: dict[str, Any],
    detection_ids: set[str],
    staged_statuses: dict[str, str],
) -> None:
    frame["detections"] = [
        detection
        for detection in frame["detections"]
        if detection["detection_id"] not in detection_ids
    ]
    for detection_id in detection_ids:
        staged_statuses.pop(detection_id, None)
    frame["annotation_complete"] = False


def _reset_click_canvas(session_id: str, frame_id: str) -> None:
    key = f"{session_id}:{frame_id}:click_revision"
    st.session_state[key] = int(st.session_state.get(key, 0)) + 1


def _detection_at_canvas_point(
    detections: list[dict[str, Any]],
    canvas_x: float,
    canvas_y: float,
    canvas_width: int,
    canvas_height: int,
    image_width: int,
    image_height: int,
) -> str | None:
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("Canvas dimensions must be positive.")
    image_x = canvas_x * image_width / canvas_width
    image_y = canvas_y * image_height / canvas_height
    matches = [
        detection
        for detection in detections
        if detection["x1"] <= image_x <= detection["x2"]
        and detection["y1"] <= image_y <= detection["y2"]
    ]
    if not matches:
        return None
    selected = min(
        matches,
        key=lambda detection: (detection["x2"] - detection["x1"])
        * (detection["y2"] - detection["y1"]),
    )
    return selected["detection_id"]


def _canvas_point_coordinates(shape: dict[str, Any]) -> tuple[float, float] | None:
    if shape.get("type") != "circle":
        return None
    try:
        radius = float(shape["radius"])
        stroke_width = float(shape.get("strokeWidth", 0))
        left = float(shape["left"])
        top = float(shape["top"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(np.isfinite(value) for value in (radius, stroke_width, left, top)):
        return None
    return left + radius + stroke_width / 2, top


def _commit_frame_review(
    frame: dict[str, Any],
    staged_statuses: dict[str, str],
    shapes: list[dict[str, Any]],
    detection_by_stroke: dict[str, str],
    class_by_stroke: dict[str, int],
    scale_x: float,
    scale_y: float,
    class_names: list[str],
    annotation_complete: bool,
) -> None:
    if any(
        staged_statuses.get(detection["detection_id"], detection["status"]) == "unreviewed"
        for detection in frame["detections"]
    ):
        raise ValueError(
            "Some detections are still unreviewed. Mark them, or explicitly confirm "
            "the remaining detections as true positives before saving."
        )

    for detection in frame["detections"]:
        status = staged_statuses.get(detection["detection_id"], detection["status"])
        if status != detection["status"]:
            detection.setdefault("history", []).append(detection["status"])
            detection["status"] = status
    _save_canvas_annotations(
        frame,
        shapes,
        detection_by_stroke,
        class_by_stroke,
        scale_x,
        scale_y,
        class_names,
    )
    frame["annotation_complete"] = bool(annotation_complete)


def _video_signature(
    video: dict[str, Any], weights_hash: str, class_names: list[str], settings: dict[str, Any]
) -> str:
    signature_data = {
        "video_hash": video["sha256"],
        "weights_hash": weights_hash,
        "class_names": class_names,
        **settings,
    }
    return hashlib.sha256(
        json.dumps(signature_data, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _process_videos(
    session_id: str,
    state: dict[str, Any],
    weights_path: Path,
    weights_hash: str,
    class_names: list[str],
    confidence: float,
    iou: float,
    image_size: int,
) -> None:
    model = None
    output_dir = _session_directory(session_id)
    for video_id, video in state["videos"].items():
        settings = {
            "start_seconds": video["start_seconds"],
            "end_seconds": video["end_seconds"],
            "frames_to_skip": video["frames_to_skip"],
            "confidence": confidence,
            "iou": iou,
            "image_size": image_size,
        }
        signature = _video_signature(video, weights_hash, class_names, settings)
        if video.get("processed_signature") == signature:
            continue
        if model is None:
            try:
                model, _ = _load_backend_model(
                    "Ultralytics YOLO", weights_path, class_names, weights_hash
                )
            except Exception as error:
                raise ValueError(
                    f"Could not load the uploaded YOLO weights: {error}"
                ) from error

        old_frames = [frame for frame in state["frames"] if frame["video_id"] == video_id]
        for frame in old_frames:
            Path(frame["image_path"]).unlink(missing_ok=True)
        state["frames"] = [frame for frame in state["frames"] if frame["video_id"] != video_id]

        start_frame, end_frame = frame_bounds(
            video["start_seconds"],
            video["end_seconds"],
            video["metadata"]["duration"],
            video["metadata"]["fps"],
            video["metadata"]["total_frames"],
        )
        sampled_count = len(
            sampled_frame_numbers(start_frame, end_frame, video["frames_to_skip"])
        )
        progress = st.progress(0.0, text=f"Processing {video['filename']}...")
        try:
            new_frames = extract_detections(
                Path(video["path"]),
                video_id,
                video["filename"],
                output_dir,
                model,
                class_names,
                confidence,
                iou,
                image_size,
                video["start_seconds"],
                video["end_seconds"],
                video["frames_to_skip"],
                progress=lambda frame, end, bar=progress, start=start_frame, count=sampled_count, step=video["frames_to_skip"] + 1: bar.progress(
                    min(1.0, ((frame - start) // step + 1) / max(1, count)),
                    text=f"Processing {video['filename']} — frame {frame} of {end}",
                ),
                metadata=video["metadata"],
            )
        except Exception:
            progress.empty()
            raise
        progress.empty()
        state["frames"].extend(new_frames)
        video["processed_signature"] = signature
        video["sampled_frame_count"] = sampled_count
        _save_state(session_id, state)


def _upload_videos(session_id: str, state: dict[str, Any], uploads: list[Any]) -> list[str]:
    errors = []
    changed = False
    for uploaded in uploads:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            errors.append(f"{uploaded.name}: unsupported video format.")
            continue
        try:
            digest, path = _store_upload(session_id, uploaded, suffix, "videos")
            video_id = hashlib.sha256(f"{uploaded.name}:{digest}".encode()).hexdigest()[:16]
            if video_id not in state["videos"]:
                metadata = read_video_metadata(path)
                state["videos"][video_id] = {
                    "filename": Path(uploaded.name).name,
                    "path": str(path),
                    "sha256": digest,
                    "metadata": metadata,
                    "start_seconds": 0.0,
                    "end_seconds": metadata["duration"],
                    "frames_to_skip": 29,
                }
                changed = True
        except (OSError, ValueError) as error:
            errors.append(f"{uploaded.name}: {error}")
    if changed:
        _save_state(session_id, state)
    return errors


def _render_upload_stage(
    session_id: str, state: dict[str, Any], weights_file: Any, data_path: Path
) -> None:
    st.header("A · Upload and configure")
    st.write("Upload your trained Ultralytics `.pt` model and one or more field videos.")
    videos_file = st.file_uploader(
        "Videos", type=VIDEO_TYPES, accept_multiple_files=True, key=f"{session_id}:videos"
    )
    for error in _upload_videos(session_id, state, videos_file or []):
        st.error(error)

    if weights_file is not None:
        if Path(weights_file.name).suffix.lower() != ".pt":
            st.error("Model weights must be an Ultralytics .pt file.")
            weights_file = None
        else:
            st.caption(f"Model selected: {Path(weights_file.name).name}")

    st.subheader("Detection settings")
    col1, col2, col3 = st.columns(3)
    confidence = col1.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01, key=f"{session_id}:conf")
    iou = col2.slider("IoU threshold", 0.0, 1.0, 0.7, 0.01, key=f"{session_id}:iou")
    image_size = col3.number_input(
        "Image size", min_value=32, max_value=4096, value=960, step=32, key=f"{session_id}:imgsz"
    )
    st.caption(f"Class names and IDs are read from `{data_path}`.")

    configs_valid = True
    settings_changed = False
    videos = list(state["videos"].items())
    preview_cache = st.session_state.setdefault(f"{session_id}:video_previews", {})
    for video_id, video in videos:
        with st.expander(f"📹 {video['filename']}", expanded=True):
            metadata = video["metadata"]
            st.write(
                f"Duration: **{metadata['duration']:.2f} s** · FPS: **{metadata['fps']:.3f}** · "
                f"Frames: **{metadata['total_frames']}** · "
                f"Resolution: **{metadata['width']} × {metadata['height']}**"
            )
            start_col, end_col, skip_col = st.columns(3)
            start_seconds = start_col.number_input(
                "Start time (seconds)",
                min_value=0.0,
                max_value=float(metadata["duration"]),
                value=float(video["start_seconds"]),
                step=0.1,
                key=f"{session_id}:{video_id}:start",
            )
            end_seconds = end_col.number_input(
                "End time (seconds)",
                min_value=0.0,
                max_value=float(metadata["duration"]),
                value=float(video["end_seconds"]),
                step=0.1,
                key=f"{session_id}:{video_id}:end",
            )
            frames_to_skip = skip_col.number_input(
                "Frames to skip",
                min_value=0,
                max_value=1_000_000,
                value=int(video["frames_to_skip"]),
                step=1,
                key=f"{session_id}:{video_id}:skip",
                help="Frames skipped between processed frames: 0 processes every frame; 4 processes every fifth frame.",
            )
            updated_settings = {
                "start_seconds": float(start_seconds),
                "end_seconds": float(end_seconds),
                "frames_to_skip": int(frames_to_skip),
            }
            if any(video.get(key) != value for key, value in updated_settings.items()):
                video.update(updated_settings)
                settings_changed = True
            try:
                first, last = frame_bounds(
                    video["start_seconds"],
                    video["end_seconds"],
                    metadata["duration"],
                    metadata["fps"],
                    metadata["total_frames"],
                )
                sample_numbers = sampled_frame_numbers(first, last, video["frames_to_skip"])
                st.caption(
                    f"Selected {video['start_seconds']:.2f}–{video['end_seconds']:.2f} s · "
                    f"one frame every {video['frames_to_skip'] + 1} frames · "
                    f"approximately {len(sample_numbers)} sampled frames"
                )
                preview = _cached_video_previews(
                    Path(video["path"]),
                    first,
                    last,
                    video["frames_to_skip"],
                    preview_cache,
                    video_id,
                )
                if preview:
                    preview_columns = st.columns(len(preview))
                    for column, image in zip(preview_columns, preview):
                        column.image(image, width="stretch")
            except ValueError as error:
                configs_valid = False
                st.error(str(error))

    if settings_changed:
        _save_state(session_id, state)
    if not videos:
        st.info("Upload at least one video to configure its time interval and sampling.")
    if videos and not configs_valid:
        st.warning("Correct the video time settings before starting inference.")
    if st.button(
        "Start detection",
        type="primary",
        disabled=not videos or weights_file is None or not configs_valid,
        key=f"{session_id}:start_processing",
    ):
        try:
            class_names = _read_class_names(data_path)
            weights_hash, weights_path = _store_upload(
                session_id, weights_file, ".pt", "weights"
            )
            state["class_names"] = class_names
            _save_state(session_id, state)
            try:
                _process_videos(
                    session_id,
                    state,
                    weights_path,
                    weights_hash,
                    class_names,
                    confidence,
                    iou,
                    int(image_size),
                )
            finally:
                weights_path.unlink(missing_ok=True)
            if not state["frames"]:
                st.warning("Inference finished, but no detections were found in the selected intervals.")
            else:
                st.success(
                    f"Detection complete. Found {sum(len(frame['detections']) for frame in state['frames'])} "
                    "individual detections."
                )
            st.rerun()
        except Exception as error:
            st.error(f"Could not process videos ({type(error).__name__}): {error}")


def _next_canvas_stroke(used_strokes: set[str]) -> str:
    index = 0
    while True:
        red, green, blue = colorsys.hsv_to_rgb((index * 0.618033988749895) % 1, 0.85, 0.95)
        color = f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"
        if color.lower() not in used_strokes and color.lower() != "#00a6ff":
            used_strokes.add(color.lower())
            return color
        index += 1


def _canvas_drawing(
    frame: dict[str, Any],
    canvas_width: int,
    canvas_height: int,
    selected_detection_id: str | set[str] = "",
) -> tuple[dict[str, Any], dict[str, str], bool]:
    objects = []
    detection_by_stroke = {}
    used_strokes: set[str] = set()
    changed = False
    selected_ids = (
        {selected_detection_id}
        if isinstance(selected_detection_id, str)
        else selected_detection_id
    )
    for detection in frame["detections"]:
        stroke = str(detection.get("canvas_stroke", "")).lower()
        if not stroke or stroke in used_strokes:
            stroke = _next_canvas_stroke(used_strokes)
            detection["canvas_stroke"] = stroke
            changed = True
        else:
            used_strokes.add(stroke)
        detection_by_stroke[stroke] = detection["detection_id"]
        x1, y1, x2, y2 = (float(detection[key]) for key in ("x1", "y1", "x2", "y2"))
        x_scale = canvas_width / frame["width"]
        y_scale = canvas_height / frame["height"]
        objects.append(
            {
                "type": "rect",
                "left": x1 * x_scale,
                "top": y1 * y_scale,
                "width": (x2 - x1) * x_scale,
                "height": (y2 - y1) * y_scale,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "originX": "left",
                "originY": "top",
                "fill": "rgba(0,0,0,0)",
                "stroke": stroke,
                "strokeWidth": 4
                if detection["detection_id"] in selected_ids
                else 1,
                "selectable": True,
            }
        )
    return {"version": "4.4.0", "objects": objects}, detection_by_stroke, changed


def _review_image(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    selected_detection_id: str | set[str],
    staged_statuses: dict[str, str] | None = None,
    coordinate_scale: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    preview = image.copy()
    scale_x, scale_y = coordinate_scale
    selected_ids = (
        {selected_detection_id}
        if isinstance(selected_detection_id, str)
        else selected_detection_id
    )
    staged_statuses = staged_statuses or {}
    colors = {
        "unreviewed": (255, 190, 0),
        "true_positive": (45, 190, 45),
        "false_positive": (45, 45, 225),
        "uncertain": (0, 210, 255),
    }
    label_rectangles: list[tuple[int, int, int, int]] = []
    for number, detection in enumerate(detections, 1):
        x1, y1, x2, y2 = (
            int(round(float(detection[key]) * scale))
            for key, scale in zip(
                ("x1", "y1", "x2", "y2"),
                (scale_x, scale_y, scale_x, scale_y),
            )
        )
        selected = detection["detection_id"] in selected_ids
        status = staged_statuses.get(detection["detection_id"], detection["status"])
        status_color = colors.get(status, (255, 190, 0))
        color = (
            status_color
            if selected
            else tuple(round(channel * 0.55) for channel in status_color)
        )
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 4 if selected else 1)
        label = f"#{number}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        label_width = text_width + 14
        label_height = text_height + baseline + 10
        candidates = [
            (x1, y1 - label_height),
            (x1, y1),
            (x2 - label_width, y1 - label_height),
            (x2 - label_width, y1),
            (x1, y2),
        ]
        label_left, label_top = candidates[0]
        for candidate_left, candidate_top in candidates:
            candidate_left = max(0, min(image.shape[1] - label_width, candidate_left))
            candidate_top = max(0, min(image.shape[0] - label_height, candidate_top))
            candidate = (
                candidate_left,
                candidate_top,
                candidate_left + label_width,
                candidate_top + label_height,
            )
            if not any(
                candidate[0] < other[2]
                and candidate[2] > other[0]
                and candidate[1] < other[3]
                and candidate[3] > other[1]
                for other in label_rectangles
            ):
                label_left, label_top = candidate_left, candidate_top
                break
        label_rectangles.append(
            (label_left, label_top, label_left + label_width, label_top + label_height)
        )
        cv2.rectangle(
            preview,
            (label_left, label_top),
            (label_left + label_width, label_top + label_height),
            (10, 10, 10),
            thickness=-1,
        )
        cv2.rectangle(
            preview,
            (label_left, label_top),
            (label_left + label_width, label_top + label_height),
            color,
            thickness=2,
        )
        cv2.putText(
            preview,
            label,
            (label_left + 7, label_top + text_height + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return preview


def _canvas_box(
    shape: dict[str, Any], scale_x: float, scale_y: float, width: int, height: int
) -> dict[str, float]:
    values = [
        float(shape.get("left", 0)),
        float(shape.get("top", 0)),
        float(shape.get("width", 0)),
        float(shape.get("height", 0)),
        float(shape.get("scaleX", 1)),
        float(shape.get("scaleY", 1)),
        float(shape.get("angle", 0)),
    ]
    if not all(np.isfinite(value) for value in values):
        raise ValueError("A drawn box has non-finite coordinates.")
    left, top, raw_width, raw_height, scale_width, scale_height, angle = values
    box_width = abs(raw_width * scale_width)
    box_height = abs(raw_height * scale_height)
    center_x = left + box_width / 2
    center_y = top + box_height / 2
    radians = np.deg2rad(angle)
    rotated_width = abs(box_width * np.cos(radians)) + abs(box_height * np.sin(radians))
    rotated_height = abs(box_width * np.sin(radians)) + abs(box_height * np.cos(radians))
    x1 = max(0.0, min(float(width), (center_x - rotated_width / 2) * scale_x))
    y1 = max(0.0, min(float(height), (center_y - rotated_height / 2) * scale_y))
    x2 = max(0.0, min(float(width), (center_x + rotated_width / 2) * scale_x))
    y2 = max(0.0, min(float(height), (center_y + rotated_height / 2) * scale_y))
    if x2 - x1 < 1 or y2 - y1 < 1:
        raise ValueError("Every box must be at least one original-image pixel wide and high.")
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _prepare_canvas_component() -> None:
    import streamlit.elements.image as streamlit_image

    if hasattr(streamlit_image, "image_to_url"):
        return

    from streamlit.elements.lib.image_utils import image_to_url
    from streamlit.elements.lib.layout_utils import LayoutConfig

    def image_to_url_compat(
        image: Any,
        width: int,
        clamp: bool,
        channels: str,
        output_format: str,
        image_id: str,
    ) -> str:
        return image_to_url(
            image,
            LayoutConfig(width=width),
            clamp,
            channels,
            output_format,
            image_id,
        )

    streamlit_image.image_to_url = image_to_url_compat


def _save_canvas_annotations(
    frame: dict[str, Any],
    shapes: list[dict[str, Any]],
    detection_by_stroke: dict[str, str],
    class_by_stroke: dict[str, int],
    scale_x: float,
    scale_y: float,
    class_names: list[str],
) -> None:
    live_strokes = set()
    existing = {detection["detection_id"]: detection for detection in frame["detections"]}
    manual_count = max(
        int(frame.get("manual_detection_count", 0)),
        sum(bool(detection.get("manual")) for detection in frame["detections"]),
    )
    for shape in shapes:
        if shape.get("type") != "rect":
            continue
        stroke = str(shape.get("stroke", "")).lower()
        class_id = class_by_stroke.get(stroke)
        if class_id is None or not 0 <= int(class_id) < len(class_names):
            raise ValueError("Select a valid class for every drawn box.")
        box = _canvas_box(shape, scale_x, scale_y, frame["width"], frame["height"])
        detection_id = detection_by_stroke.get(stroke)
        if detection_id:
            detection = existing[detection_id]
            detection.update(box, class_id=int(class_id), class_name=class_names[int(class_id)])
        else:
            manual_count += 1
            detection_id = f"{frame['frame_id']}-manual{manual_count:03d}"
            while detection_id in existing:
                manual_count += 1
                detection_id = f"{frame['frame_id']}-manual{manual_count:03d}"
            detection = {
                "detection_id": detection_id,
                **box,
                "class_id": int(class_id),
                "class_name": class_names[int(class_id)],
                "predicted_class_id": "",
                "predicted_class": "manual annotation",
                "confidence": 0.0,
                "status": "true_positive",
                "manual": True,
                "history": [],
                "canvas_stroke": stroke,
            }
            frame["detections"].append(detection)
            existing[detection_id] = detection
            frame["manual_detection_count"] = manual_count
        live_strokes.add(stroke)

    deleted_manual_ids = set()
    for stroke, detection_id in detection_by_stroke.items():
        if stroke in live_strokes:
            continue
        detection = existing[detection_id]
        if detection.get("manual"):
            deleted_manual_ids.add(detection_id)
        elif detection["status"] != "false_positive":
            detection["history"].append(detection["status"])
            detection["status"] = "false_positive"
    if deleted_manual_ids:
        frame["detections"] = [
            detection
            for detection in frame["detections"]
            if detection["detection_id"] not in deleted_manual_ids
        ]
    frame["annotation_complete"] = False


def _render_review_stage(session_id: str, state: dict[str, Any]) -> None:
    st.header("B · Batch frame review")
    frames = state["frames"]
    if not frames:
        st.info("Run detection first. Only frames containing model detections are saved for review.")
        return

    current_key = f"{session_id}:current_frame"
    frame_options = [frame["frame_id"] for frame in frames]
    if st.session_state.get(current_key) not in frame_options:
        st.session_state[current_key] = frame_options[0]
    frame_index = frame_options.index(st.session_state[current_key])
    reviewed_frames = sum(
        bool(frame["annotation_complete"])
        and all(detection["status"] != "unreviewed" for detection in frame["detections"])
        for frame in frames
    )
    remaining_frames = len(frames) - reviewed_frames
    st.progress(
        reviewed_frames / len(frames),
        text=f"{reviewed_frames}/{len(frames)} frames reviewed · "
        f"{remaining_frames} remaining",
    )

    navigation = st.columns(4)
    if navigation[0].button(
        "← Previous frame", disabled=frame_index == 0, key=f"{session_id}:previous"
    ):
        st.session_state[current_key] = frame_options[frame_index - 1]
        st.rerun()
    if navigation[1].button(
        "Previous unreviewed",
        disabled=not any(
            any(detection["status"] == "unreviewed" for detection in candidate["detections"])
            for candidate in frames[:frame_index]
        ),
        key=f"{session_id}:previous_unreviewed",
    ):
        candidates = [
            index
            for index, candidate in enumerate(frames[:frame_index])
            if any(detection["status"] == "unreviewed" for detection in candidate["detections"])
        ]
        st.session_state[current_key] = frame_options[candidates[-1]]
        st.rerun()
    navigation[2].markdown(f"**Frame {frame_index + 1}/{len(frames)}**")
    if navigation[3].button(
        "Next frame →",
        disabled=frame_index + 1 >= len(frame_options),
        key=f"{session_id}:next",
    ):
        st.session_state[current_key] = frame_options[frame_index + 1]
        st.rerun()
    next_unreviewed_col = st.columns([1, 3])[0]
    if next_unreviewed_col.button(
        "Next unreviewed frame",
        disabled=not any(
            any(detection["status"] == "unreviewed" for detection in candidate["detections"])
            for candidate in frames[frame_index + 1 :]
        ),
        key=f"{session_id}:next_unreviewed",
    ):
        candidates = [
            index
            for index, candidate in enumerate(frames[frame_index + 1 :], frame_index + 1)
            if any(detection["status"] == "unreviewed" for detection in candidate["detections"])
        ]
        st.session_state[current_key] = frame_options[candidates[0]]
        st.rerun()

    frame = frames[frame_index]
    image_cache = st.session_state.setdefault(f"{session_id}:review_image", {})
    cached_images = _review_frame_images(frame, image_cache)
    if cached_images is None:
        st.error(f"Could not read saved frame: {frame['image_path']}")
        return
    image = cached_images["image"]
    display_bgr = cached_images["display_bgr"]
    original_display = cached_images["display_rgb"]

    class_names = state["class_names"]
    frame["width"], frame["height"] = image.shape[1], image.shape[0]
    detections = frame["detections"]
    detection_ids = [detection["detection_id"] for detection in detections]

    status_key = f"{session_id}:{frame['frame_id']}:staged_statuses"
    if status_key not in st.session_state:
        st.session_state[status_key] = {
            detection["detection_id"]: detection["status"] for detection in detections
        }
    staged_statuses: dict[str, str] = st.session_state[status_key]
    for detection in detections:
        staged_statuses.setdefault(detection["detection_id"], detection["status"])
    accepted_count = sum(
        staged_statuses[detection["detection_id"]] == "true_positive"
        for detection in detections
    )
    rejected_count = sum(
        staged_statuses[detection["detection_id"]] == "false_positive"
        for detection in detections
    )
    uncertain_count = sum(
        staged_statuses[detection["detection_id"]] == "uncertain"
        for detection in detections
    )
    manual_count = sum(bool(detection.get("manual")) for detection in detections)
    stats = st.columns(5)
    stats[0].metric("Detections", len(detections))
    stats[1].metric("Accepted", accepted_count)
    stats[2].metric("Rejected", rejected_count)
    stats[3].metric("Uncertain", uncertain_count)
    stats[4].metric("Manual", manual_count)

    selection_key = f"{session_id}:{frame['frame_id']}:selected_detections"
    completeness_key = f"{session_id}:{frame['frame_id']}:complete"
    selected_ids: set[str] = set(st.session_state.get(selection_key, set()))
    selected_ids.intersection_update(detection_ids)
    image_height, image_width = image.shape[:2]
    canvas_height, canvas_width = display_bgr.shape[:2]
    scale_x = image_width / canvas_width
    scale_y = image_height / canvas_height
    review_preview = _review_image(
        display_bgr,
        detections,
        selected_ids,
        staged_statuses,
        (1 / scale_x, 1 / scale_y),
    )
    review_display = Image.fromarray(
        cv2.cvtColor(review_preview, cv2.COLOR_BGR2RGB)
    )
    original_column, annotated_column = st.columns(2)
    original_column.caption(
        f"Original · frame {frame['frame_number']} · "
        f"{frame['timestamp_seconds']:.2f}s"
    )
    original_column.image(original_display, width=canvas_width)
    annotated_column.caption("Detected · all numbered boxes")
    click_revision_key = f"{session_id}:{frame['frame_id']}:click_revision"
    click_revision = int(st.session_state.get(click_revision_key, 0))
    with annotated_column:
        _prepare_canvas_component()
        click_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=0,
            stroke_color="#ffffff",
            background_image=review_display,
            width=canvas_width,
            height=canvas_height,
            drawing_mode="point",
            initial_drawing={"version": "4.4.0", "objects": []},
            update_streamlit=True,
            display_toolbar=False,
            point_display_radius=1,
            key=f"{session_id}:{frame['frame_id']}:click:{click_revision}",
        )
    click_json = getattr(click_result, "json_data", None)
    if isinstance(click_json, dict):
        click_points = [
            point
            for shape in click_json.get("objects", [])
            if isinstance(shape, dict)
            and (point := _canvas_point_coordinates(shape)) is not None
        ]
        if click_points:
            clicked_detection_id = _detection_at_canvas_point(
                detections,
                *click_points[-1],
                canvas_width,
                canvas_height,
                image_width,
                image_height,
            )
            if clicked_detection_id:
                st.session_state[selection_key] = {clicked_detection_id}
            st.session_state[click_revision_key] = click_revision + 1
            st.rerun()
    st.caption("Box colours: 🟠 unreviewed · 🟢 accepted · 🔴 rejected · 🟡 uncertain")
    if detections:
        st.caption(
            "Click detection numbers to select several at once. Selections are staged "
            "until you save the frame."
        )
        number_columns = st.columns(min(8, len(detections)))
        symbols = {
            "unreviewed": "○",
            "true_positive": "✓",
            "false_positive": "×",
            "uncertain": "?",
        }
        for number, detection in enumerate(detections, 1):
            detection_id_for_number = detection["detection_id"]
            status = staged_statuses[detection_id_for_number]
            if number_columns[(number - 1) % len(number_columns)].button(
                f"#{number} {symbols.get(status, '○')}",
                key=f"{session_id}:{frame['frame_id']}:select:{detection_id_for_number}",
                type="primary" if detection_id_for_number in selected_ids else "secondary",
                help=(
                    f"{detection['class_name']} · {detection['confidence']:.3f} · "
                    f"{status.replace('_', ' ')}"
                ),
            ):
                if detection_id_for_number in selected_ids:
                    selected_ids.remove(detection_id_for_number)
                else:
                    selected_ids.add(detection_id_for_number)
                st.session_state[selection_key] = selected_ids
                _reset_click_canvas(session_id, frame["frame_id"])
                st.rerun()

        bulk_actions = st.columns(3)
        if bulk_actions[0].button(
            "Select all as true positive",
            key=f"{session_id}:{frame['frame_id']}:all_true",
        ):
            _stage_detection_decision(
                staged_statuses, set(detection_ids), "true_positive"
            )
            st.session_state[selection_key] = set(detection_ids)
            _reset_click_canvas(session_id, frame["frame_id"])
            st.rerun()
        if bulk_actions[1].button(
            "Select all as false positive",
            key=f"{session_id}:{frame['frame_id']}:all_false",
        ):
            _stage_detection_decision(
                staged_statuses, set(detection_ids), "false_positive"
            )
            st.session_state[selection_key] = set(detection_ids)
            _reset_click_canvas(session_id, frame["frame_id"])
            st.rerun()
        if bulk_actions[2].button(
            "Confirm remaining as true positives",
            key=f"{session_id}:{frame['frame_id']}:confirm_remaining",
            disabled=not any(
                staged_statuses[detection_id] == "unreviewed"
                for detection_id in detection_ids
            ),
        ):
            remaining_ids = {
                detection_id
                for detection_id in detection_ids
                if staged_statuses[detection_id] == "unreviewed"
            }
            _stage_detection_decision(staged_statuses, remaining_ids, "true_positive")
            st.session_state[selection_key] = remaining_ids
            _reset_click_canvas(session_id, frame["frame_id"])
            st.rerun()
        selection_actions = st.columns(4)
        if selection_actions[0].button(
            "Clear selections",
            key=f"{session_id}:{frame['frame_id']}:clear_selection",
            disabled=not selected_ids,
        ):
            st.session_state[selection_key] = set()
            _reset_click_canvas(session_id, frame["frame_id"])
            st.rerun()
        for column, status, label in (
            (selection_actions[1], "true_positive", "Mark selected true"),
            (selection_actions[2], "false_positive", "Mark selected false"),
            (selection_actions[3], "uncertain", "Mark selected uncertain"),
        ):
            if column.button(
                label,
                key=f"{session_id}:{frame['frame_id']}:mark_{status}",
                disabled=not selected_ids,
            ):
                _stage_detection_decision(staged_statuses, selected_ids, status)
                _reset_click_canvas(session_id, frame["frame_id"])
                st.rerun()
        selected_detections = [
            detection for detection in detections if detection["detection_id"] in selected_ids
        ]
        if len(selected_detections) == 1:
            selected = selected_detections[0]
            st.markdown(
                f"**#{detection_ids.index(selected['detection_id']) + 1} · "
                f"{selected.get('predicted_class', selected['class_name'])} · "
                f"{selected['confidence']:.3f} confidence**"
            )
        elif selected_detections:
            st.caption("Selected detections:")
            for selected in selected_detections:
                st.write(
                    f"#{detection_ids.index(selected['detection_id']) + 1} · "
                    f"{selected.get('predicted_class', selected['class_name'])} · "
                    f"{selected['confidence']:.3f} confidence"
                )

    initial_drawing, detection_by_stroke, drawing_changed = _canvas_drawing(
        frame,
        canvas_width,
        canvas_height,
        selected_ids,
    )
    if drawing_changed:
        _save_state(session_id, state)
    display_image = Image.fromarray(original_display)
    drawing_mode_key = f"{session_id}:{frame['frame_id']}:draw_mode"
    if drawing_mode_key not in st.session_state:
        st.session_state[drawing_mode_key] = "edit"
    tool_columns = st.columns([1, 1, 2])
    if tool_columns[0].button("＋ Add insect", key=f"{session_id}:{frame['frame_id']}:add_insect"):
        st.session_state[drawing_mode_key] = "add"
        st.rerun()
    if tool_columns[1].button("Edit boxes", key=f"{session_id}:{frame['frame_id']}:edit_boxes"):
        st.session_state[drawing_mode_key] = "edit"
        st.rerun()
    drawing_mode_label = st.session_state[drawing_mode_key]
    draft_key = f"{session_id}:{frame['frame_id']}:canvas_draft"
    revision_key = f"{session_id}:{frame['frame_id']}:canvas_revision"
    revision = int(st.session_state.get(revision_key, 0))
    previous_drawing = st.session_state.get(draft_key)
    drawing = previous_drawing if isinstance(previous_drawing, dict) else initial_drawing
    for shape in drawing.get("objects", []):
        if not isinstance(shape, dict):
            continue
        shape["strokeWidth"] = (
            4
            if detection_by_stroke.get(str(shape.get("stroke", "")).lower())
            in selected_ids
            else 2
        )
    previous_strokes = {
        str(shape.get("stroke", "")).lower()
        for shape in drawing.get("objects", [])
        if isinstance(shape, dict)
    }

    with st.expander(
        "Add, correct, or delete bounding boxes",
        expanded=drawing_mode_label == "add",
    ):
        _prepare_canvas_component()
        canvas_result = st_canvas(
            fill_color="rgba(0, 166, 255, 0.08)",
            stroke_width=3,
            stroke_color="#00A6FF",
            background_image=display_image,
            width=canvas_width,
            height=canvas_height,
            drawing_mode="rect" if drawing_mode_label == "add" else "transform",
            initial_drawing=drawing,
            update_streamlit=True,
            display_toolbar=False,
            key=f"{session_id}:{frame['frame_id']}:canvas:{revision}",
        )

    result_drawing = getattr(canvas_result, "json_data", None)
    if not isinstance(result_drawing, dict):
        result_drawing = drawing
    result_drawing.setdefault("objects", [])
    assigned_new_id = False
    used_strokes = set(detection_by_stroke) | previous_strokes
    for shape in result_drawing["objects"]:
        if not isinstance(shape, dict) or shape.get("type") != "rect":
            continue
        stroke = str(shape.get("stroke", "")).lower()
        if stroke not in detection_by_stroke and stroke not in previous_strokes:
            shape["stroke"] = _next_canvas_stroke(used_strokes)
            assigned_new_id = True
    st.session_state[draft_key] = result_drawing
    if assigned_new_id:
        st.session_state[revision_key] = revision + 1
        st.rerun()

    shapes = [
        shape
        for shape in result_drawing.get("objects", [])
        if isinstance(shape, dict) and shape.get("type") == "rect"
    ]
    if selected_ids and st.button(
        "Delete selected boxes",
        key=f"{session_id}:{frame['frame_id']}:delete_selected_boxes",
    ):
        deleted_strokes = {
            stroke
            for stroke, selected_detection_id in detection_by_stroke.items()
            if selected_detection_id in selected_ids
        }
        result_drawing["objects"] = [
            shape
            for shape in result_drawing["objects"]
            if not (
                isinstance(shape, dict)
                and str(shape.get("stroke", "")).lower() in deleted_strokes
            )
        ]
        _delete_detections(frame, selected_ids, staged_statuses)
        _save_state(session_id, state)
        st.session_state[draft_key] = result_drawing
        st.session_state[selection_key] = set()
        st.session_state[completeness_key] = False
        st.session_state[revision_key] = revision + 1
        st.rerun()
    class_by_stroke: dict[str, int] = {}
    new_strokes = []
    for shape in shapes:
        stroke = str(shape.get("stroke", "")).lower()
        detection_id = detection_by_stroke.get(stroke)
        if detection_id:
            detection = next(
                item for item in frame["detections"] if item["detection_id"] == detection_id
            )
            class_by_stroke[stroke] = int(detection["class_id"])
        else:
            new_strokes.append(stroke)
    if len(selected_ids) == 1:
        selected_id = next(iter(selected_ids))
        selected_item = next(
            (item for item in frame["detections"] if item["detection_id"] == selected_id),
            None,
        )
        if selected_item is not None:
            selected_class_id = int(
                st.selectbox(
                    "Correct selected class",
                    range(len(class_names)),
                    index=int(selected_item["class_id"]),
                    format_func=lambda value: f"{value}: {class_names[value]}",
                    key=f"{session_id}:{frame['frame_id']}:class_edit:{selected_id}",
                )
            )
            if selected_item.get("canvas_stroke"):
                class_by_stroke[selected_item["canvas_stroke"]] = selected_class_id
    if new_strokes:
        st.caption("Choose a class for each newly drawn insect:")
        for number, stroke in enumerate(new_strokes, 1):
            class_by_stroke[stroke] = int(
                st.selectbox(
                    f"New insect box {number} class",
                    range(len(class_names)),
                    format_func=lambda value: f"{value}: {class_names[value]}",
                    key=f"{session_id}:{frame['frame_id']}:new_class:{stroke}",
                )
            )
    if drawing_mode_label == "add":
        st.caption("Draw one or more rectangles on the original frame; they are saved as true positives.")

    st.checkbox(
        "I have checked the entire frame; all target insects are annotated.",
        value=bool(frame["annotation_complete"]),
        key=completeness_key,
        help="Required before export. This prevents a frame with a missed insect from becoming a false negative.",
    )
    if any(staged_statuses[item["detection_id"]] == "unreviewed" for item in detections):
        st.warning(
            "Unreviewed detections must be classified before saving. Use "
            "'Confirm remaining as true positives' only after checking those boxes."
        )
    if uncertain_count:
        st.warning(
            f"{uncertain_count} uncertain detection(s) will be saved as uncertain and "
            "this frame will remain excluded from dataset export."
        )
    if not st.session_state[completeness_key]:
        st.warning(
            "Confirm that every target insect in the frame is annotated before this "
            "frame can be included in the dataset."
        )
    if st.button(
        "Save Frame & Next",
        type="primary",
        key=f"{session_id}:{frame['frame_id']}:save_frame_next",
    ):
        try:
            for detection in detections:
                status = staged_statuses[detection["detection_id"]]
                if status != "unreviewed":
                    continue
                raise ValueError(
                    "Some detections are still unreviewed. Mark them, or explicitly "
                    "confirm the remaining detections as true positives before saving."
                )
            _commit_frame_review(
                frame,
                staged_statuses,
                result_drawing["objects"],
                detection_by_stroke,
                class_by_stroke,
                scale_x,
                scale_y,
                class_names,
                bool(st.session_state[completeness_key]),
            )
            _save_state(session_id, state)
            st.session_state.pop(status_key, None)
            st.session_state.pop(selection_key, None)
            st.session_state.pop(draft_key, None)
            st.session_state[revision_key] = revision + 1
            st.session_state[drawing_mode_key] = "edit"
            if frame_index + 1 < len(frame_options):
                st.session_state[current_key] = frame_options[frame_index + 1]
            st.rerun()
        except (KeyError, TypeError, ValueError) as error:
            st.error(f"Could not save frame: {error}")


def _render_export_stage(session_id: str, state: dict[str, Any], data_path: Path) -> None:
    st.header("C · Create and download dataset")
    frames = state["frames"]
    complete_frames = sum(
        frame["annotation_complete"]
        and all(
            detection["status"] in {"true_positive", "false_positive"}
            for detection in frame["detections"]
        )
        for frame in frames
    )
    st.write(
        f"{complete_frames} complete frame(s) are currently eligible. Frames with "
        "unreviewed/uncertain detections, or without whole-frame annotation confirmation, are excluded."
    )
    try:
        class_names = _read_class_names(data_path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        st.error(f"Could not read dataset class names: {error}")
        return
    if class_names != state["class_names"]:
        st.error(
            "The selected data.yaml class names differ from those used for inference. "
            "Restore the original configuration before exporting."
        )
        return

    st.download_button(
        "Download detection review CSV",
        data=lambda reviewed_frames=frames: report_csv(reviewed_frames),
        file_name="review_report.csv",
        mime="text/csv",
        key=f"{session_id}:download_csv",
        on_click=_delete_session_artifact,
        args=(session_id, "review_report.csv"),
    )
    if st.button("Prepare optional annotated-preview ZIP", key=f"{session_id}:make_previews"):
        preview_path = _session_directory(session_id) / "annotated_previews.zip"
        try:
            annotated_preview_zip(frames, preview_path)
            state.setdefault("exports", {})["previews_zip"] = str(preview_path)
            _save_state(session_id, state)
            st.rerun()
        except (OSError, ValueError) as error:
            st.error(f"Could not create annotated previews: {error}")
    preview_path = Path(state.get("exports", {}).get("previews_zip", ""))
    if preview_path.is_file():
        st.download_button(
            "Download annotated preview images (.zip)",
            data=lambda path=preview_path, sid=session_id: _read_then_delete(path, sid),
            file_name="annotated_previews.zip",
            mime="application/zip",
            key=f"{session_id}:download_previews",
        )

    if st.button("Generate reviewed YOLO dataset", type="primary", key=f"{session_id}:export"):
        export_root = _session_directory(session_id) / f"dataset_export_{uuid.uuid4().hex[:8]}"
        zip_path = export_root.with_suffix(".zip")
        try:
            dataset_dir, split_counts = create_dataset(
                frames, class_names, export_root, seed=42
            )
            zip_path = zip_directory(dataset_dir, zip_path)
            shutil.rmtree(dataset_dir)
            state["exports"] = {
                "dataset_zip": str(zip_path),
            }
            _save_state(session_id, state)
            st.success(f"Dataset created: {split_counts}")
            st.rerun()
        except (OSError, ValueError, yaml.YAMLError) as error:
            shutil.rmtree(export_root, ignore_errors=True)
            zip_path.unlink(missing_ok=True)
            st.error(f"Could not create dataset: {error}")

    exports = state.get("exports", {})
    if Path(exports.get("dataset_zip", "")).is_file():
        st.download_button(
            "Download complete YOLO dataset (.zip)",
            data=lambda path=Path(exports["dataset_zip"]), sid=session_id: _read_then_delete(path, sid),
            file_name="dataset_export.zip",
            mime="application/zip",
            key=f"{session_id}:download_dataset",
        )


@st.cache_resource(max_entries=1)
def _load_backend_model_cached(
    model_type: str,
    weights_fingerprint: str,
    class_names: tuple[str, ...],
    _weights_path: str,
) -> Any:
    checkpoint_path = Path(_weights_path)
    if model_type == "Ultralytics YOLO":
        model = _get_yolo_class()(str(checkpoint_path))
        try:
            names = list(model.names.values()) if hasattr(model, "names") else class_names
        except Exception:
            names = class_names
        return model, list(names)
    if model_type == "RF-DETR":
        model = _get_rfdetr_class().from_checkpoint(str(checkpoint_path))
        return model, list(class_names)
    raise ValueError(f"Unsupported model backend: {model_type}")


def _load_backend_model(
    model_type: str,
    weights_path: Path,
    class_names: list[str],
    weights_hash: str | None = None,
) -> Any:
    if weights_hash is None:
        try:
            stat = weights_path.stat()
            weights_hash = f"{weights_path}:{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            weights_hash = str(weights_path)
    return _load_backend_model_cached(
        model_type,
        weights_hash,
        tuple(class_names),
        str(weights_path),
    )


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


def _extract_common_predictions(
    raw_predictions: Any, class_names: list[str]
) -> list[dict[str, Any]]:
    if raw_predictions is None:
        return []

    boxes = getattr(raw_predictions, "xyxy", None)
    confidences = getattr(raw_predictions, "confidence", None)
    class_ids = getattr(raw_predictions, "class_id", None)
    data = getattr(raw_predictions, "data", None)
    if isinstance(data, dict):
        boxes = data.get("xyxy", boxes)
        confidences = data.get("confidence", confidences)
        class_ids = data.get("class_id", class_ids)
        predicted_names = data.get("class_name")
    else:
        predicted_names = None

    if boxes is None and hasattr(raw_predictions, "boxes"):
        model_boxes = raw_predictions.boxes
        if model_boxes is not None:
            boxes = getattr(model_boxes, "xyxy", None)
            confidences = getattr(model_boxes, "conf", confidences)
            class_ids = getattr(model_boxes, "cls", class_ids)

    if boxes is None:
        return []
    box_array = _as_array(boxes).reshape(-1, 4)
    if not len(box_array):
        return []
    if confidences is None or class_ids is None:
        raise ValueError("Model predictions are missing confidence or class IDs.")
    confidence_array = _as_array(confidences).reshape(-1)
    class_array = _as_array(class_ids).reshape(-1)
    name_array = _as_array(predicted_names).reshape(-1) if predicted_names is not None else None
    if not (len(box_array) == len(confidence_array) == len(class_array)):
        raise ValueError("Model returned inconsistent box, confidence, and class counts.")

    detections = []
    for index, (box, confidence, raw_class_id) in enumerate(
        zip(box_array, confidence_array, class_array)
    ):
        class_id = int(raw_class_id)
        class_name = (
            str(name_array[index])
            if name_array is not None and index < len(name_array)
            else class_names[class_id]
            if 0 <= class_id < len(class_names)
            else f"class_{class_id}"
        )
        x1, y1, x2, y2 = [float(value) for value in box]
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(confidence),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return detections


def _rfdetr_shape(model: Any, image_size: int) -> tuple[int, int]:
    config = getattr(model, "model_config", None)
    if config is None:
        raise ValueError("RF-DETR checkpoint did not expose its model configuration.")
    block_size = int(config.patch_size) * int(getattr(config, "num_windows", 1))
    side = max(block_size, (int(image_size) // block_size) * block_size)
    return side, side


def _predict_frame(
    model_type: str,
    model: Any,
    frame: np.ndarray,
    class_names: list[str],
    confidence: float,
    iou: float,
    image_size: int,
) -> list[dict[str, Any]]:
    if model_type == "Ultralytics YOLO":
        results = model(frame, conf=confidence, iou=iou, imgsz=image_size, verbose=False)
        return [
            detection
            for result in results
            for detection in _extract_common_predictions(result, class_names)
        ]
    if model_type == "RF-DETR":
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = model.predict(
            rgb_frame,
            threshold=confidence,
            shape=_rfdetr_shape(model, image_size),
            include_source_image=False,
        )
        return _extract_common_predictions(result, class_names)
    raise ValueError(f"Unsupported model backend: {model_type}")


def _draw_detection_overlay(frame: np.ndarray, detections: list[dict[str, Any]], class_names: list[str]) -> np.ndarray:
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in (detection["x1"], detection["y1"], detection["x2"], detection["y2"])]
        class_name = detection["class_name"]
        score = detection["confidence"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - text_h - 8)), (x1 + text_w, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return annotated


def _run_detection_backend(
    session_id: str,
    model_type: str,
    weights_path: Path,
    video_path: Path,
    class_names: list[str],
    start_seconds: float,
    end_seconds: float,
    frames_to_skip: int,
    confidence: float,
    iou: float,
    image_size: int,
    progress: Any,
    preview: Any,
    timestamp_status: Any,
    stop_key: str,
    tracking_config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    video_filename: str | None = None,
    weights_hash: str | None = None,
) -> tuple[Path, Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    if metadata is None:
        metadata = read_video_metadata(video_path)
    start_frame, end_frame = frame_bounds(
        start_seconds,
        end_seconds,
        metadata["duration"],
        metadata["fps"],
        metadata["total_frames"],
    )
    step = 1 if tracking_config is not None else frames_to_skip + 1
    total_selected = max(0, (end_frame - start_frame + step - 1) // step)
    output_dir = _session_directory(session_id) / "detection_exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_name = Path((video_filename or video_path.name).replace("\\", "/")).name
    title = Path(source_name).stem
    video_output = output_dir / (
        f"{title}_tracked.mp4" if tracking_config is not None else f"{title}_annotated.mp4"
    )
    temporary_video = output_dir / f"{title}_temporary.mp4"
    csv_output = output_dir / f"{title}_detections.csv"
    tracking_summary_path = output_dir / f"{title}_tracking.json"
    temporary_video.unlink(missing_ok=True)
    writer = None
    try:
        model, model_names = _load_backend_model(
            model_type, weights_path, class_names, weights_hash
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, first_frame = cap.read()
        if not ok:
            raise ValueError(f"Could not read the selected start frame from {source_name}.")
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(temporary_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(metadata["fps"]),
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError(
                f"Could not create a temporary video for {source_name}; "
                "check OpenCV MP4 codec support."
            )

        columns = (
            "video",
            "frame_number",
            "timestamp_seconds",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ) + (("track_id",) if tracking_config is not None else ())
        processed = 0
        detection_count = 0
        tracked_detection_count = 0
        if tracking_config is not None:
            from tracking.detector import Detection
            from tracking.tracker import Tracker
            from tracking.video_utils import draw_tracks

            tracker = Tracker(tracking_config)
        else:
            tracker = None
        tracks_by_id: dict[int, dict[str, int]] = {}
        tracking_started = time.perf_counter()
        with csv_output.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=columns)
            csv_writer.writeheader()
            frame_counter = start_frame
            frame = first_frame
            while frame_counter < end_frame:
                if st.session_state.get(stop_key, False):
                    raise RuntimeError("Detection stopped by user.")
                if (frame_counter - start_frame) % step == 0:
                    normalized = _predict_frame(
                        model_type,
                        model,
                        frame,
                        model_names,
                        confidence,
                        iou,
                        image_size,
                    )
                    detection_count += len(normalized)
                    if tracker is not None:
                        tracked = tracker.update(
                            [
                                Detection(
                                    xyxy=(
                                        item["x1"],
                                        item["y1"],
                                        item["x2"],
                                        item["y2"],
                                    ),
                                    confidence=item["confidence"],
                                    class_id=item["class_id"],
                                    class_name=item["class_name"],
                                )
                                for item in normalized
                            ],
                            frame.shape[:2],
                        )
                        annotated = draw_tracks(frame, tracked)
                        tracked_detection_count += len(tracked)
                        track_ids_by_detection = {
                            item.detection_index: item.tracker_id for item in tracked
                        }
                        for item in tracked:
                            record = tracks_by_id.setdefault(
                                item.tracker_id,
                                {
                                    "first_frame": frame_counter,
                                    "last_frame": frame_counter,
                                    "frames_tracked": 0,
                                },
                            )
                            record["last_frame"] = frame_counter
                            record["frames_tracked"] += 1
                    else:
                        annotated = _draw_detection_overlay(frame, normalized, model_names)
                        track_ids_by_detection = {}
                    timestamp = frame_counter / metadata["fps"]
                    for detection_index, item in enumerate(normalized):
                        row = {
                            "video": source_name,
                            "frame_number": frame_counter,
                            "timestamp_seconds": timestamp,
                            **item,
                        }
                        if tracker is not None:
                            row["track_id"] = track_ids_by_detection.get(
                                detection_index, ""
                            )
                        csv_writer.writerow(
                            row
                        )
                    processed += 1
                    progress.progress(
                        min(1.0, processed / max(1, total_selected)),
                        text=f"{source_name}: {processed}/{total_selected} sampled frames",
                    )
                    preview.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {frame_counter} · {timestamp:.2f} s",
                        width="stretch",
                    )
                    timestamp_status.caption(
                        f"Current video timestamp: {timestamp:.2f} s"
                    )
                else:
                    annotated = frame
                writer.write(annotated)
                frame_counter += 1
                if frame_counter >= end_frame:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
    except Exception:
        if writer is not None:
            writer.release()
            writer = None
        cap.release()
        temporary_video.unlink(missing_ok=True)
        csv_output.unlink(missing_ok=True)
        tracking_summary_path.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.release()
        cap.release()
    try:
        _transcode_mp4(temporary_video, video_output)
    finally:
        temporary_video.unlink(missing_ok=True)
    if tracking_config is not None:
        elapsed = time.perf_counter() - tracking_started
        track_durations = [
            (record["last_frame"] - record["first_frame"] + 1) / metadata["fps"]
            for record in tracks_by_id.values()
        ]
        tracking_summary = {
            "algorithm": "ByteTrack",
            "total_unique_track_ids": len(tracks_by_id),
            "detections": detection_count,
            "tracked_detections": tracked_detection_count,
            "frames_processed": processed,
            "processing_fps": processed / elapsed if elapsed > 0 else 0.0,
            "track_durations_seconds": {
                "min": min(track_durations) if track_durations else 0.0,
                "max": max(track_durations) if track_durations else 0.0,
            },
            "tracks": [
                {
                    "track_id": track_id,
                    **record,
                    "duration_seconds": (
                        record["last_frame"] - record["first_frame"] + 1
                    )
                    / metadata["fps"],
                }
                for track_id, record in sorted(tracks_by_id.items())
            ],
            "identity_note": (
                "Track IDs are temporary labels within this video, not biological bee identities."
            ),
        }
        tracking_summary_path.write_text(
            json.dumps(tracking_summary, indent=2), encoding="utf-8"
        )
    else:
        tracking_summary_path.unlink(missing_ok=True)
    del model
    return video_output, csv_output


def _transcode_mp4(source: Path, destination: Path) -> None:
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
        details = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) else str(error)
        raise ValueError(f"Could not encode a browser-compatible H.264 MP4: {details}") from error


def _render_detection_mode(session_id: str, data_path: Path) -> None:
    st.header("Detection mode")
    model_type = st.selectbox("Model type", MODEL_BACKENDS, index=0, key=f"{session_id}:detection_model")
    if model_type == "RF-DETR" and not _rfdetr_available():
        st.warning("RF-DETR is not installed in the current environment, so it cannot be tested here.")
    elif model_type == "RF-DETR":
        st.caption("Upload a fine-tuned RFDETRLarge checkpoint (.pth) from this project's RF-DETR training workflow.")
    weight_types = _supported_weight_extensions(model_type)
    weights_file = st.file_uploader(
        "Trained model weights",
        type=weight_types,
        key=f"{session_id}:detection_weights_{model_type}",
        help="Upload a YOLO .pt checkpoint or an RF-DETR .pth checkpoint.",
    )
    videos = st.file_uploader("Videos", type=VIDEO_TYPES, accept_multiple_files=True, key=f"{session_id}:detection_videos")
    if not videos:
        st.info("Upload one or more videos to start detection mode.")
        return

    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01, key=f"{session_id}:detection_conf")
    if model_type == "Ultralytics YOLO":
        iou = st.slider("IoU threshold", 0.0, 1.0, 0.7, 0.01, key=f"{session_id}:detection_iou")
    else:
        iou = 0.7
        st.caption("RF-DETR uses its checkpoint's post-processing; an IoU threshold is not exposed by its inference API.")
    image_size = st.number_input("Image size", min_value=128, max_value=2048, value=960, step=32, key=f"{session_id}:detection_imgsz")
    apply_tracking = st.checkbox(
        "Apply object tracking after detection",
        value=False,
        key=f"{session_id}:apply_tracking",
        help="Add persistent short-term IDs to detections and create a tracked video.",
    )
    skip_frames = st.number_input(
        "Frames to skip between processed frames",
        min_value=0,
        max_value=1000,
        value=29,
        step=1,
        key=f"{session_id}:detection_skip",
        disabled=apply_tracking,
    )
    tracking_config = None
    if apply_tracking:
        st.caption(
            "Tracking runs detection on every frame in the selected time range so "
            "ByteTrack can maintain IDs. This can take longer than detection alone."
        )
        with st.expander("Tracking options", expanded=True):
            algorithm = st.selectbox(
                "Tracking algorithm",
                ["ByteTrack"],
                key=f"{session_id}:tracking_algorithm",
            )
            low_threshold = st.slider(
                "Low detection threshold",
                0.0,
                0.99,
                0.25,
                0.01,
                key=f"{session_id}:tracking_low_threshold",
                help="Minimum score used for ByteTrack's second, low-confidence matching pass.",
            )
            high_threshold = st.slider(
                "High detection threshold",
                0.01,
                1.0,
                0.5,
                0.01,
                key=f"{session_id}:tracking_high_threshold",
                help="Score boundary for ByteTrack's first matching pass.",
            )
            activation_threshold = st.slider(
                "Track activation threshold",
                0.01,
                1.0,
                0.5,
                0.01,
                key=f"{session_id}:tracking_activation_threshold",
                help="Minimum score for starting a new track.",
            )
            track_buffer = st.number_input(
                "Track buffer (frames)",
                min_value=1,
                max_value=1000,
                value=30,
                step=1,
                key=f"{session_id}:tracking_buffer",
                help="How many frames a temporarily missed track is kept alive.",
            )
            matching_threshold = st.slider(
                "Matching threshold",
                0.0,
                1.0,
                0.8,
                0.01,
                key=f"{session_id}:tracking_matching_threshold",
                help="Controls how readily detections are associated with existing tracks.",
            )
        tracking_config = {
            "algorithm": algorithm,
            "low_detection_threshold": low_threshold,
            "high_detection_threshold": high_threshold,
            "track_activation_threshold": activation_threshold,
            "track_buffer": int(track_buffer),
            "matching_threshold": matching_threshold,
        }
        if conf > low_threshold:
            st.info(
                "For ByteTrack's low-confidence recovery, set the detection "
                "confidence threshold no higher than the low detection threshold."
            )
        if low_threshold >= high_threshold:
            st.error("The low detection threshold must be lower than the high threshold.")
        if activation_threshold < high_threshold:
            st.error(
                "The track activation threshold must be at least the high detection threshold."
            )
    class_names = _read_class_names(data_path)
    video_settings: dict[str, dict[str, float]] = {}
    video_paths: dict[str, Path] = {}
    video_metadata: dict[str, dict[str, Any]] = {}
    for uploaded in videos:
        digest, temp_path = _store_upload(
            session_id, uploaded, Path(uploaded.name).suffix.lower(), "detection"
        )
        upload_id = str(getattr(uploaded, "file_id", uploaded.name))
        metadata_key = f"{session_id}:detection_metadata:{digest}"
        metadata = st.session_state.get(metadata_key)
        if metadata is None or not temp_path.is_file():
            metadata = read_video_metadata(temp_path)
            st.session_state[metadata_key] = metadata
        video_paths[upload_id] = temp_path
        video_metadata[upload_id] = metadata
        with st.expander(f"{uploaded.name} settings", expanded=True):
            start_seconds = st.number_input(
                f"Start time ({uploaded.name})",
                min_value=0.0,
                max_value=float(metadata["duration"]),
                value=0.0,
                step=0.1,
                key=f"{session_id}:start_{uploaded.name}",
            )
            end_seconds = st.number_input(
                f"End time ({uploaded.name})",
                min_value=0.0,
                max_value=float(metadata["duration"]),
                value=float(metadata["duration"]),
                step=0.1,
                key=f"{session_id}:end_{uploaded.name}",
            )
            if end_seconds <= start_seconds:
                st.warning("End time must be greater than start time.")
            video_settings[uploaded.name] = {"start": float(start_seconds), "end": float(end_seconds)}
    stop_key = f"{session_id}:stop_detection"
    if "detection_outputs" not in st.session_state:
        st.session_state["detection_outputs"] = {}

    start_label = (
        "Start detection and tracking" if apply_tracking else "Start detection"
    )
    if st.button(start_label, type="primary", key=f"{session_id}:start_detection"):
        if weights_file is None:
            st.error("Please upload a trained model before starting detection.")
        elif Path(weights_file.name).suffix.lower().lstrip(".") not in _supported_weight_extensions(model_type):
            st.error("The selected model backend does not support this checkpoint extension.")
        elif apply_tracking and (
            low_threshold >= high_threshold
            or activation_threshold < high_threshold
        ):
            st.error("Correct the ByteTrack thresholds before starting.")
        else:
            st.session_state[stop_key] = False
            st.session_state["detection_outputs"] = {}
            weights_hash, weights_path = _store_upload(
                session_id,
                weights_file,
                Path(weights_file.name).suffix.lower(),
                "weights",
            )
            for uploaded in videos:
                source_name = Path(uploaded.name.replace("\\", "/")).name
                upload_id = str(getattr(uploaded, "file_id", uploaded.name))
                saved = video_paths[upload_id]
                try:
                    if not saved.is_file() or saved.stat().st_size == 0:
                        raise ValueError(f"Could not save uploaded video: {uploaded.name}")
                    metadata = video_metadata[upload_id]
                    settings = video_settings.get(uploaded.name, {"start": 0.0, "end": float(metadata["duration"])})
                    start_seconds = float(settings.get("start", 0.0))
                    end_seconds = float(settings.get("end", metadata["duration"]))
                    if end_seconds <= start_seconds:
                        raise ValueError("End time must be greater than start time.")
                    video_out, csv_out = _run_detection_backend(
                        session_id,
                        model_type,
                        weights_path,
                        saved,
                        class_names,
                        start_seconds,
                        end_seconds,
                        int(skip_frames),
                        float(conf),
                        float(iou),
                        int(image_size),
                        st.progress(0.0, text=f"Scanning {uploaded.name}..."),
                        st.empty(),
                        st.empty(),
                        stop_key,
                        tracking_config=tracking_config,
                        metadata=metadata,
                        video_filename=source_name,
                        weights_hash=weights_hash,
                    )
                    st.session_state["detection_outputs"][uploaded.name] = {
                        "video": str(video_out),
                        "csv": str(csv_out),
                        "tracking_summary": (
                            str(
                                _session_directory(session_id)
                                / "detection_exports"
                                / f"{Path(source_name).stem}_tracking.json"
                            )
                            if tracking_config is not None
                            else None
                        ),
                    }
                except Exception as error:
                    st.error(f"{uploaded.name}: {error}")
                    continue

    if st.button("Stop detection", key=f"{session_id}:stop_detection_button"):
        st.session_state[stop_key] = True

    outputs = st.session_state.get("detection_outputs", {})
    if outputs:
        for name, payload in outputs.items():
            if Path(payload["video"]).exists():
                st.subheader(name)
                st.video(str(payload["video"]), format="video/mp4")
                video_path = Path(payload["video"])
                summary_value = payload.get("tracking_summary")
                if summary_value and Path(summary_value).is_file():
                    summary_path = Path(summary_value)
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    st.caption(
                        f"ByteTrack · {summary['total_unique_track_ids']} unique track IDs · "
                        f"{summary['frames_processed']} frames processed · "
                        f"{summary['processing_fps']:.2f} processing FPS"
                    )
                    st.caption(summary["identity_note"])
                    st.download_button(
                        f"Download tracking summary for {name}",
                        data=summary_path.read_bytes(),
                        file_name=summary_path.name,
                        mime="application/json",
                        key=f"{session_id}:download_tracking_summary_{name}",
                    )
                st.download_button(
                    f"Download annotated MP4 for {name}",
                    data=lambda path=video_path: path.read_bytes(),
                    file_name=video_path.name,
                    mime="video/mp4",
                    key=f"{session_id}:download_video_{name}",
                )
                if Path(payload["csv"]).exists():
                    csv_path = Path(payload["csv"])
                    st.download_button(
                        f"Download detection CSV for {name}",
                        data=lambda path=csv_path: path.read_bytes(),
                        file_name=csv_path.name,
                        mime="text/csv",
                        key=f"{session_id}:download_csv_{name}",
                    )


def main() -> None:
    st.set_page_config(page_title="Bee monitoring dashboard", page_icon="🐝", layout="wide")
    st.title("🐝 Bee monitoring dashboard")
    st.caption("Unified detection and hard-negative review for bee monitoring videos.")
    _start_cleanup_worker()
    session_id = _session_id()
    with _session_access(session_id):
        _render_dashboard(session_id)


def _render_dashboard(session_id: str) -> None:
    state_key = f"{session_id}:review_state"
    if state_key not in st.session_state:
        try:
            st.session_state[state_key] = _load_state(session_id)
        except ValueError as error:
            st.error(str(error))
            st.stop()
    state = st.session_state[state_key]

    with st.sidebar:
        st.caption(
            "Temporary session data is cleared on refresh or expiry. "
            "Use a private BEE_MINING_TMPDIR on shared/HPC storage."
        )
        if st.button("Clear session data now", key=f"{session_id}:clear_session"):
            _clear_session_data(session_id, lock_held=True)
            st.session_state.pop("detection_outputs", None)
            st.query_params["session"] = uuid.uuid4().hex
            st.rerun()

    mode = st.radio(
        "Choose a mode",
        ["Detection", "Hard-negative mining"],
        index=1,
        horizontal=True,
        key=f"{session_id}:mode",
    )
    default_data = str(DEFAULT_DATA)
    data_config = st.text_input(
        "YOLO class configuration",
        value=st.session_state.get(f"{session_id}:data_config", default_data),
        help="Class names and IDs are preserved from this existing data.yaml.",
        key=f"{session_id}:data_config",
    )
    data_path = Path(data_config).expanduser()

    if mode == "Detection":
        _render_detection_mode(session_id, data_path)
        return

    weights_file = st.file_uploader(
        "Trained YOLO weights (.pt)",
        type=["pt"],
        key=f"{session_id}:weights",
    )
    with st.expander("About saved sessions and video timing"):
        st.write(
            f"Temporary uploads, frames, decisions, and exports are kept in `{SESSION_ROOT}` "
            "for this browser session only."
        )
        st.write(
            "OpenCV uses the video-reported FPS for seeking and timestamps. Variable-frame-rate "
            "files may have approximate timestamps; source frame numbers are retained."
        )

    _render_upload_stage(session_id, state, weights_file, data_path)
    st.divider()
    _render_review_stage(session_id, state)
    st.divider()
    _render_export_stage(session_id, state, data_path)


if __name__ == "__main__":
    main()
