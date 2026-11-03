"""Local Streamlit app for reviewing individual YOLO detections."""

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
import streamlit as st
import yaml
from ultralytics import YOLO

try:
    from rfdetr import RFDETRLarge
except Exception:  # pragma: no cover - optional dependency for RF-DETR mode
    RFDETRLarge = None

from hard_negative_mining import DEFAULT_DATA, VIDEO_EXTENSIONS, _read_class_names
from hard_negative_review import (
    REVIEW_STATUSES,
    annotated_preview_zip,
    create_dataset,
    draw_review_image,
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
        "BEE_MINING_DATA_DIR",
        str(Path(__file__).resolve().parents[1] / "hard_negative_sessions"),
    )
).expanduser()
VIDEO_TYPES = sorted(suffix.lstrip(".") for suffix in VIDEO_EXTENSIONS)
MODEL_BACKENDS = ["Ultralytics YOLO", "RF-DETR"]


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
        _clear_other_session_data()
        _clear_legacy_detection_data()
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


def _clear_other_session_data() -> None:
    if not SESSION_ROOT.is_dir():
        return
    for path in SESSION_ROOT.iterdir():
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            uuid.UUID(path.name)
        except ValueError:
            continue
        shutil.rmtree(path)


def _clear_legacy_detection_data() -> None:
    for path in (
        SESSION_ROOT / "detection_exports",
        SESSION_ROOT / "detection_preview",
    ):
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)


def _clear_session_data(session_id: str) -> None:
    try:
        normalized_id = uuid.UUID(session_id).hex
    except ValueError:
        return
    for path in (
        _session_directory(normalized_id),
        SESSION_ROOT / "detection_exports" / normalized_id,
        SESSION_ROOT / "detection_preview" / normalized_id,
    ):
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)


def _session_directory(session_id: str) -> Path:
    return SESSION_ROOT / session_id


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
    session_id: str, uploaded_file: Any, suffix: str, category: str
) -> tuple[str, Path]:
    directory = _session_directory(session_id) / "uploads"
    directory.mkdir(parents=True, exist_ok=True)
    contents = uploaded_file.getbuffer()
    digest = hashlib.sha256(contents).hexdigest()
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(uploaded_file.name).name)
    path = (
        directory / f"{digest}{suffix}"
        if category == "weights"
        else directory / f"{digest[:12]}_{safe_name}"
    )
    if not path.is_file():
        path.write_bytes(contents)
    return digest, path


def _detection_edits(
    session_id: str, state: dict[str, Any], detection: dict[str, Any]
) -> dict[str, Any]:
    coordinates = {
        key: float(st.session_state[f"{session_id}:{detection['detection_id']}:{key}"])
        for key in ("x1", "y1", "x2", "y2")
    }
    if (
        coordinates["x1"] < 0
        or coordinates["y1"] < 0
        or coordinates["x2"] > st.session_state["current_frame_width"]
        or coordinates["y2"] > st.session_state["current_frame_height"]
        or coordinates["x2"] <= coordinates["x1"]
        or coordinates["y2"] <= coordinates["y1"]
    ):
        raise ValueError("Bounding box must have positive area and stay inside the image.")
    class_id = int(
        st.session_state[f"{session_id}:{detection['detection_id']}:class"]
    )
    return {
        **coordinates,
        "class_id": class_id,
        "class_name": state["class_names"][class_id],
    }


def _set_detection_status(
    session_id: str,
    state: dict[str, Any],
    detection: dict[str, Any],
    status: str,
) -> None:
    edits = _detection_edits(session_id, state, detection)
    detection.update(edits)
    detection["history"].append(detection["status"])
    detection["status"] = status
    _save_state(session_id, state)


def _update_detection_box(
    session_id: str, state: dict[str, Any], detection: dict[str, Any]
) -> None:
    detection.update(_detection_edits(session_id, state, detection))
    _save_state(session_id, state)


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
    try:
        model = YOLO(str(weights_path))
    except Exception as error:
        raise ValueError(f"Could not load the uploaded YOLO weights: {error}") from error

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
        except (OSError, ValueError) as error:
            errors.append(f"{uploaded.name}: {error}")
    if not errors:
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
    videos = list(state["videos"].items())
    for video_id, video in videos:
        with st.expander(f"📹 {video['filename']}", expanded=True):
            metadata = video["metadata"]
            st.write(
                f"Duration: **{metadata['duration']:.2f} s** · FPS: **{metadata['fps']:.3f}** · "
                f"Frames: **{metadata['total_frames']}** · "
                f"Resolution: **{metadata['width']} × {metadata['height']}**"
            )
            start_col, end_col, skip_col = st.columns(3)
            video["start_seconds"] = start_col.number_input(
                "Start time (seconds)",
                min_value=0.0,
                max_value=float(metadata["duration"]),
                value=float(video["start_seconds"]),
                step=0.1,
                key=f"{session_id}:{video_id}:start",
            )
            video["end_seconds"] = end_col.number_input(
                "End time (seconds)",
                min_value=0.0,
                max_value=float(metadata["duration"]),
                value=float(video["end_seconds"]),
                step=0.1,
                key=f"{session_id}:{video_id}:end",
            )
            video["frames_to_skip"] = skip_col.number_input(
                "Frames to skip",
                min_value=0,
                max_value=1_000_000,
                value=int(video["frames_to_skip"]),
                step=1,
                key=f"{session_id}:{video_id}:skip",
                help="Frames skipped between processed frames: 0 processes every frame; 4 processes every fifth frame.",
            )
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
                preview = preview_frames(
                    Path(video["path"]),
                    first,
                    last,
                    video["frames_to_skip"],
                )
                if preview:
                    preview_columns = st.columns(len(preview))
                    for column, image in zip(preview_columns, preview):
                        column.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
            except ValueError as error:
                configs_valid = False
                st.error(str(error))

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


def _render_review_stage(session_id: str, state: dict[str, Any]) -> None:
    st.header("B · Review individual detections")
    frames = state["frames"]
    detections = [detection for frame in frames for detection in frame["detections"]]
    if not detections:
        st.info("Run detection first. Only frames containing model detections are saved for review.")
        return

    status_counts = {
        status: sum(detection["status"] == status for detection in detections)
        for status in REVIEW_STATUSES
    }
    reviewed_count = len(detections) - status_counts["unreviewed"]
    st.progress(
        reviewed_count / len(detections),
        text=f"{reviewed_count} of {len(detections)} detections reviewed",
    )
    metric_columns = st.columns(4)
    for column, status in zip(
        metric_columns, ("unreviewed", "true_positive", "false_positive", "uncertain")
    ):
        column.metric(status.replace("_", " ").title(), status_counts[status])
    filter_status = st.selectbox(
        "Filter frames by detection status",
        ("all", "unreviewed", "true_positive", "false_positive", "uncertain"),
        format_func=lambda value: value.replace("_", " ").title(),
        key=f"{session_id}:review_filter",
    )
    filtered_frames = [
        frame
        for frame in frames
        if filter_status == "all"
        or any(detection["status"] == filter_status for detection in frame["detections"])
    ]
    if not filtered_frames:
        st.info("No frames match this review filter.")
        return

    current_key = f"{session_id}:current_frame"
    frame_options = [frame["frame_id"] for frame in filtered_frames]
    if st.session_state.get(current_key) not in frame_options:
        st.session_state[current_key] = frame_options[0]
    frame_index = frame_options.index(st.session_state[current_key])
    navigation = st.columns([1, 4, 1])
    if navigation[0].button("← Previous", disabled=frame_index == 0, key=f"{session_id}:previous"):
        st.session_state[current_key] = frame_options[frame_index - 1]
        st.rerun()
    navigation[2].markdown(f"**Frame {frame_index + 1} of {len(frame_options)}**")
    if navigation[2].button(
        "Next →", disabled=frame_index + 1 >= len(frame_options), key=f"{session_id}:next"
    ):
        st.session_state[current_key] = frame_options[frame_index + 1]
        st.rerun()
    selected_frame_id = st.selectbox(
        "Frame",
        frame_options,
        index=frame_index,
        format_func=lambda value: next(
            f"{frame['video_filename']} · frame {frame['frame_number']} · "
            f"{frame['timestamp_seconds']:.2f}s"
            for frame in filtered_frames
            if frame["frame_id"] == value
        ),
        key=current_key,
    )
    frame = next(frame for frame in frames if frame["frame_id"] == selected_frame_id)
    image = cv2.imread(frame["image_path"])
    if image is None:
        st.error(f"Could not read saved frame: {frame['image_path']}")
        return

    class_names = state["class_names"]
    frame["width"], frame["height"] = image.shape[1], image.shape[0]
    st.session_state["current_frame_width"] = frame["width"]
    st.session_state["current_frame_height"] = frame["height"]
    if frame["detections"]:
        detection_labels = {
            detection["detection_id"]: (
                f"{detection['detection_id']} · {detection['class_name']} · "
                f"{detection['confidence']:.2f} · {detection['status'].replace('_', ' ')}"
            )
            for detection in frame["detections"]
        }
        detection_id = st.selectbox(
            "Select one detection (the selected box is highlighted)",
            list(detection_labels),
            format_func=lambda value: detection_labels[value],
            key=f"{session_id}:{frame['frame_id']}:detection",
        )
        selected_detection = next(
            detection
            for detection in frame["detections"]
            if detection["detection_id"] == detection_id
        )
    else:
        selected_detection = None

    original_column, annotated_column = st.columns(2)
    original_column.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Original frame",
        width="stretch",
    )
    annotated_image = draw_review_image(
        image, frame["detections"], selected_detection["detection_id"] if selected_detection else ""
    )
    annotated_column.image(
        cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
        caption="Green: accepted · red: rejected · orange: uncertain · gray: unreviewed · cyan: selected",
        width="stretch",
    )

    if selected_detection is not None:
        st.subheader(f"Detection {selected_detection['detection_id']}")
        st.caption(
            f"Predicted: {selected_detection['class_name']} · "
            f"Confidence: {selected_detection['confidence']:.3f}"
        )
        box_columns = st.columns(4)
        for column, coordinate in zip(box_columns, ("x1", "y1", "x2", "y2")):
            column.number_input(
                coordinate,
                min_value=0.0,
                max_value=float(frame["width"] if coordinate in {"x1", "x2"} else frame["height"]),
                value=float(selected_detection[coordinate]),
                step=1.0,
                key=f"{session_id}:{detection_id}:{coordinate}",
            )
        class_index = selected_detection["class_id"]
        st.selectbox(
            "Correct class",
            range(len(class_names)),
            index=class_index,
            format_func=lambda value: f"{value}: {class_names[value]}",
            key=f"{session_id}:{detection_id}:class",
        )
        decision_columns = st.columns(5)
        try:
            if decision_columns[0].button("Keep detection", key=f"{session_id}:{detection_id}:keep"):
                _set_detection_status(session_id, state, selected_detection, "true_positive")
                st.rerun()
            if decision_columns[1].button("Reject detection", key=f"{session_id}:{detection_id}:reject"):
                _set_detection_status(session_id, state, selected_detection, "false_positive")
                st.rerun()
            if decision_columns[2].button("Mark uncertain", key=f"{session_id}:{detection_id}:uncertain"):
                _set_detection_status(session_id, state, selected_detection, "uncertain")
                st.rerun()
            if decision_columns[3].button("Save box correction", key=f"{session_id}:{detection_id}:save"):
                _update_detection_box(session_id, state, selected_detection)
                st.success("Bounding box and class saved.")
            if decision_columns[4].button(
                "Undo decision",
                disabled=not selected_detection["history"],
                key=f"{session_id}:{detection_id}:undo",
            ):
                selected_detection["status"] = selected_detection["history"].pop()
                _save_state(session_id, state)
                st.rerun()
        except (KeyError, ValueError) as error:
            st.error(f"Could not save this detection: {error}")

    with st.expander("Add a missed insect"):
        st.caption("Draw a box by entering pixel coordinates and select the correct class.")
        with st.form(f"{session_id}:{frame['frame_id']}:add_detection"):
            add_columns = st.columns(5)
            x1 = add_columns[0].number_input("x1", 0.0, float(frame["width"]), float(frame["width"] * 0.25))
            y1 = add_columns[1].number_input("y1", 0.0, float(frame["height"]), float(frame["height"] * 0.25))
            x2 = add_columns[2].number_input("x2", 0.0, float(frame["width"]), float(frame["width"] * 0.75))
            y2 = add_columns[3].number_input("y2", 0.0, float(frame["height"]), float(frame["height"] * 0.75))
            class_id = add_columns[4].selectbox(
                "Class",
                range(len(class_names)),
                format_func=lambda value: f"{value}: {class_names[value]}",
                key=f"{session_id}:{frame['frame_id']}:new_class",
            )
            add_submitted = st.form_submit_button("Add missed insect")
        if add_submitted:
            if x2 <= x1 or y2 <= y1:
                st.error("Bounding box must have positive width and height.")
            else:
                detection_id = f"{frame['frame_id']}-manual{sum(bool(d['manual']) for d in frame['detections']) + 1:03d}"
                frame["detections"].append(
                    {
                        "detection_id": detection_id,
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "predicted_class_id": "",
                        "predicted_class": "manual annotation",
                        "confidence": 0.0,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "status": "true_positive",
                        "manual": True,
                        "history": [],
                    }
                )
                frame["annotation_complete"] = False
                _save_state(session_id, state)
                st.rerun()

    completeness_key = f"{session_id}:{frame['frame_id']}:complete"
    st.checkbox(
        "I checked the entire frame; every target insect is annotated.",
        value=bool(frame["annotation_complete"]),
        key=completeness_key,
        help="Required before export. This prevents a frame with a missed insect from becoming a false negative.",
    )
    if st.button("Save frame completeness", key=f"{session_id}:{frame['frame_id']}:save_complete"):
        frame["annotation_complete"] = bool(st.session_state[completeness_key])
        _save_state(session_id, state)
        st.rerun()
    if any(detection["status"] in {"unreviewed", "uncertain"} for detection in frame["detections"]):
        st.warning("This frame cannot be exported until every detection is reviewed; uncertain detections exclude it.")


def _render_export_stage(session_id: str, state: dict[str, Any], data_path: Path) -> None:
    st.header("C · Create and download dataset")
    frames = state["frames"]
    detections = [detection for frame in frames for detection in frame["detections"]]
    complete_frames = sum(
        frame["annotation_complete"]
        and bool(frame["detections"])
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
    if not detections:
        return
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
        data=report_csv(frames),
        file_name="review_report.csv",
        mime="text/csv",
        key=f"{session_id}:download_csv",
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
            data=preview_path.read_bytes(),
            file_name="annotated_previews.zip",
            mime="application/zip",
            key=f"{session_id}:download_previews",
        )

    if st.button("Generate reviewed YOLO dataset", type="primary", key=f"{session_id}:export"):
        export_root = _session_directory(session_id) / f"dataset_export_{uuid.uuid4().hex[:8]}"
        try:
            dataset_dir, split_counts = create_dataset(
                frames, class_names, export_root, seed=42
            )
            zip_path = zip_directory(dataset_dir, export_root.with_suffix(".zip"))
            csv_path = _session_directory(session_id) / "review_report.csv"
            csv_path.write_bytes(report_csv(frames))
            state["exports"] = {
                "dataset_zip": str(zip_path),
                "report_csv": str(csv_path),
                "dataset_dir": str(dataset_dir),
            }
            _save_state(session_id, state)
            st.success(f"Dataset created: {split_counts}")
            st.rerun()
        except (OSError, ValueError, yaml.YAMLError) as error:
            st.error(f"Could not create dataset: {error}")

    exports = state.get("exports", {})
    if Path(exports.get("dataset_zip", "")).is_file():
        st.download_button(
            "Download complete YOLO dataset (.zip)",
            data=Path(exports["dataset_zip"]).read_bytes(),
            file_name="dataset_export.zip",
            mime="application/zip",
            key=f"{session_id}:download_dataset",
        )


def _load_backend_model(model_type: str, weights_path: Path, class_names: list[str]) -> Any:
    if model_type == "Ultralytics YOLO":
        model = YOLO(str(weights_path))
        try:
            names = list(model.names.values()) if hasattr(model, "names") else class_names
        except Exception:
            names = class_names
        return model, names
    if model_type == "RF-DETR":
        if RFDETRLarge is None:
            raise ValueError("RF-DETR is not available in the current environment.")
        model = RFDETRLarge.from_checkpoint(str(weights_path))
        return model, class_names
    raise ValueError(f"Unsupported model backend: {model_type}")


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
) -> tuple[Path, Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    metadata = read_video_metadata(video_path)
    start_frame, end_frame = frame_bounds(
        start_seconds,
        end_seconds,
        metadata["duration"],
        metadata["fps"],
        metadata["total_frames"],
    )
    step = frames_to_skip + 1
    total_selected = max(0, (end_frame - start_frame + step - 1) // step)
    output_dir = _session_directory(session_id) / "detection_exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    title = video_path.stem
    video_output = output_dir / f"{title}_annotated.mp4"
    temporary_video = output_dir / f"{title}_temporary.mp4"
    csv_output = output_dir / f"{title}_detections.csv"
    temporary_video.unlink(missing_ok=True)
    writer = None
    try:
        model, model_names = _load_backend_model(model_type, weights_path, class_names)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, first_frame = cap.read()
        if not ok:
            raise ValueError(f"Could not read the selected start frame from {video_path.name}.")
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(temporary_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(metadata["fps"]),
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError(
                f"Could not create a temporary video for {video_path.name}; "
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
        )
        processed = 0
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
                    annotated = _draw_detection_overlay(frame, normalized, model_names)
                    timestamp = frame_counter / metadata["fps"]
                    for item in normalized:
                        csv_writer.writerow(
                            {
                                "video": video_path.name,
                                "frame_number": frame_counter,
                                "timestamp_seconds": timestamp,
                                **item,
                            }
                        )
                    processed += 1
                    progress.progress(
                        min(1.0, processed / max(1, total_selected)),
                        text=f"{video_path.name}: {processed}/{total_selected} sampled frames",
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
        raise
    finally:
        if writer is not None:
            writer.release()
        cap.release()
    try:
        _transcode_mp4(temporary_video, video_output)
    finally:
        temporary_video.unlink(missing_ok=True)
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
    if model_type == "RF-DETR" and RFDETRLarge is None:
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
    skip_frames = st.number_input("Frames to skip between processed frames", min_value=0, max_value=1000, value=29, step=1, key=f"{session_id}:detection_skip")
    class_names = _read_class_names(data_path)
    video_settings: dict[str, dict[str, float]] = {}
    for uploaded in videos:
        temp_dir = _session_directory(session_id) / "detection_preview"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded.name
        temp_path.write_bytes(uploaded.getvalue())
        metadata = read_video_metadata(temp_path)
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

    if st.button("Start detection", type="primary", key=f"{session_id}:start_detection"):
        if weights_file is None:
            st.error("Please upload a trained model before starting detection.")
        elif Path(weights_file.name).suffix.lower().lstrip(".") not in _supported_weight_extensions(model_type):
            st.error("The selected model backend does not support this checkpoint extension.")
        else:
            st.session_state[stop_key] = False
            st.session_state["detection_outputs"] = {}
            output_dir = _session_directory(session_id) / "uploads"
            output_dir.mkdir(parents=True, exist_ok=True)
            weights_path = output_dir / f"weights{Path(weights_file.name).suffix.lower()}"
            weights_path.write_bytes(weights_file.getvalue())
            for uploaded in videos:
                saved = output_dir / uploaded.name
                try:
                    saved.write_bytes(uploaded.getvalue())
                    if not saved.is_file() or saved.stat().st_size == 0:
                        raise ValueError(f"Could not save uploaded video: {uploaded.name}")
                    metadata = read_video_metadata(saved)
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
                    )
                    st.session_state["detection_outputs"][uploaded.name] = {
                        "video": str(video_out),
                        "csv": str(csv_out),
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

    session_id = _session_id()
    try:
        state = _load_state(session_id)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    with st.sidebar:
        st.caption("Saved app data is cleared on refresh. Use a private BEE_MINING_DATA_DIR on shared/HPC storage.")
        if st.button("Clear session data now", key=f"{session_id}:clear_session"):
            _clear_session_data(session_id)
            _clear_other_session_data()
            _clear_legacy_detection_data()
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
            f"Review decisions are saved locally in `{SESSION_ROOT}` and can be resumed "
            "using this browser URL. Uploaded media and exports are not added to Git."
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
