"""Video sampling, individual-detection review exports, and YOLO packaging."""

import csv
import io
import math
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any

import cv2
import yaml

try:
    from .hard_negative_mining import _validate_yolo_lines
except ImportError:
    from hard_negative_mining import _validate_yolo_lines


REVIEW_STATUSES = ("unreviewed", "true_positive", "false_positive", "uncertain")
CSV_FIELDS = (
    "video_filename",
    "frame_number",
    "timestamp_seconds",
    "frame_filename",
    "detection_id",
    "predicted_class",
    "predicted_class_id",
    "final_class_id",
    "final_class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "review_status",
    "final_annotation_status",
    "frame_annotation_complete",
)


def read_video_metadata(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Video has invalid frame rate, frame count, or resolution: {video_path}")
        return {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": total_frames / fps,
        }
    finally:
        capture.release()


def frame_bounds(
    start_seconds: float, end_seconds: float, duration: float, fps: float, total_frames: int
) -> tuple[int, int]:
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("Start time must be a non-negative number of seconds.")
    if not math.isfinite(end_seconds) or end_seconds <= start_seconds:
        raise ValueError("End time must be greater than start time.")
    if end_seconds > duration:
        raise ValueError(f"End time ({end_seconds:g}s) exceeds video duration ({duration:.3f}s).")
    start_frame = min(total_frames, math.ceil(start_seconds * fps))
    end_frame = min(total_frames, math.ceil(end_seconds * fps))
    if end_frame <= start_frame:
        raise ValueError("Selected time interval contains no video frames.")
    return start_frame, end_frame


def sampled_frame_numbers(start_frame: int, end_frame: int, frames_to_skip: int) -> range:
    if frames_to_skip < 0:
        raise ValueError("Frames to skip must be zero or greater.")
    return range(start_frame, end_frame, frames_to_skip + 1)


def preview_frames(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    frames_to_skip: int,
) -> list[Any]:
    samples = sampled_frame_numbers(start_frame, end_frame, frames_to_skip)
    if not samples:
        return []
    numbers = sorted({samples[0], samples[len(samples) // 2], samples[-1]})
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {video_path}")
    previews = []
    try:
        for frame_number in numbers:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Could not read preview frame {frame_number} from {video_path}.")
            previews.append(frame)
    finally:
        capture.release()
    return previews


def extract_detections(
    video_path: Path,
    video_id: str,
    video_filename: str,
    output_dir: Path,
    model: Any,
    class_names: list[str],
    confidence: float,
    iou: float,
    image_size: int,
    start_seconds: float,
    end_seconds: float,
    frames_to_skip: int,
    progress: Any = None,
) -> list[dict[str, Any]]:
    metadata = read_video_metadata(video_path)
    start_frame, end_frame = frame_bounds(
        start_seconds,
        end_seconds,
        metadata["duration"],
        metadata["fps"],
        metadata["total_frames"],
    )
    sample_numbers = sampled_frame_numbers(start_frame, end_frame, frames_to_skip)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    frame_number = start_frame
    next_sample = start_frame
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            raise ValueError(f"Could not seek to frame {start_frame} in {video_path}.")
        for target_frame in sample_numbers:
            if target_frame < next_sample:
                continue
            while frame_number <= target_frame:
                success, frame = capture.read()
                if not success:
                    raise ValueError(
                        f"Video ended before selected frame {target_frame}: {video_path}"
                    )
                frame_number += 1
            if progress is not None:
                progress(target_frame, end_frame)
            actual_frame = target_frame
            result_list = model(
                frame,
                conf=confidence,
                iou=iou,
                imgsz=image_size,
                verbose=False,
            )
            if not result_list:
                continue
            result = result_list[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            height, width = frame.shape[:2]
            frame_id = f"{video_id}_{actual_frame:09d}"
            frame_relative = Path("frames") / video_id / f"frame_{actual_frame:09d}.jpg"
            frame_path = output_dir / frame_relative
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(frame_path), frame):
                raise OSError(f"Could not save frame: {frame_path}")

            coordinates = boxes.xyxy.tolist()
            class_ids = boxes.cls.tolist()
            scores = boxes.conf.tolist()
            detections = []
            for index, (box, raw_class_id, score) in enumerate(
                zip(coordinates, class_ids, scores), 1
            ):
                class_id = int(raw_class_id)
                if not 0 <= class_id < len(class_names):
                    raise ValueError(
                        f"Model class ID {class_id} is not defined in the selected data.yaml."
                    )
                x1, y1, x2, y2 = (float(value) for value in box)
                x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
                y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"Model returned a bounding box with no area at frame {actual_frame}.")
                detections.append(
                    {
                        "detection_id": f"{frame_id}-d{index:03d}",
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "predicted_class_id": class_id,
                        "predicted_class": class_names[class_id],
                        "confidence": float(score),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "status": "unreviewed",
                        "manual": False,
                        "history": [],
                    }
                )
            frames.append(
                {
                    "frame_id": frame_id,
                    "video_id": video_id,
                    "video_filename": video_filename,
                    "frame_number": actual_frame,
                    "timestamp_seconds": actual_frame / metadata["fps"],
                    "image_path": str(frame_path),
                    "width": width,
                    "height": height,
                    "annotation_complete": False,
                    "detections": detections,
                }
            )
            next_sample = target_frame + frames_to_skip + 1
    finally:
        capture.release()
    return frames


def draw_review_image(frame: Any, detections: list[dict[str, Any]], selected_id: str) -> Any:
    annotated = frame.copy()
    colors = {
        "unreviewed": (180, 180, 180),
        "true_positive": (50, 200, 50),
        "false_positive": (40, 50, 230),
        "uncertain": (0, 165, 255),
    }
    for detection in detections:
        x1, y1, x2, y2 = (int(round(detection[key])) for key in ("x1", "y1", "x2", "y2"))
        selected = detection["detection_id"] == selected_id
        color = (255, 255, 0) if selected else colors[detection["status"]]
        thickness = 4 if selected else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = f"{detection['detection_id']} {detection['class_name']} {detection['confidence']:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def report_csv(frames: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for frame in frames:
        for detection in frame["detections"]:
            status = detection["status"]
            if status == "uncertain":
                final_status = "excluded_uncertain"
            elif status == "unreviewed":
                final_status = "excluded_unreviewed"
            elif not frame["annotation_complete"]:
                final_status = "excluded_incomplete_frame"
            elif status == "false_positive":
                final_status = "excluded_rejected"
            else:
                final_status = "included"
            writer.writerow(
                {
                    "video_filename": frame["video_filename"],
                    "frame_number": frame["frame_number"],
                    "timestamp_seconds": f"{frame['timestamp_seconds']:.6f}",
                    "frame_filename": Path(frame["image_path"]).name,
                    "detection_id": detection["detection_id"],
                    "predicted_class": detection.get("predicted_class", detection["class_name"]),
                    "predicted_class_id": detection.get(
                        "predicted_class_id", detection["class_id"]
                    ),
                    "final_class_id": detection["class_id"],
                    "final_class": detection["class_name"],
                    "confidence": (
                        "" if detection.get("manual") else f"{detection['confidence']:.6f}"
                    ),
                    **{key: f"{detection[key]:.2f}" for key in ("x1", "y1", "x2", "y2")},
                    "review_status": status,
                    "final_annotation_status": final_status,
                    "frame_annotation_complete": frame["annotation_complete"],
                }
            )
    return output.getvalue().encode("utf-8")


def create_dataset(
    frames: list[dict[str, Any]],
    class_names: list[str],
    output_dir: Path,
    seed: int = 42,
) -> tuple[Path, dict[str, int]]:
    eligible = []
    for frame in frames:
        detections = frame["detections"]
        statuses = {detection["status"] for detection in detections}
        if (
            not frame["annotation_complete"]
            or not detections
            or statuses - {"true_positive", "false_positive"}
        ):
            continue
        image_path = Path(frame["image_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read candidate image: {image_path}")
        height, width = image.shape[:2]
        lines = []
        for detection in detections:
            if detection["status"] != "true_positive":
                continue
            x1, y1, x2, y2 = (float(detection[key]) for key in ("x1", "y1", "x2", "y2"))
            if (
                not all(math.isfinite(value) for value in (x1, y1, x2, y2))
                or x1 < 0
                or y1 < 0
                or x2 > width
                or y2 > height
                or x2 <= x1
                or y2 <= y1
            ):
                raise ValueError(f"Invalid bounding box for detection {detection['detection_id']}.")
            box_width, box_height = x2 - x1, y2 - y1
            lines.append(
                f"{int(detection['class_id'])} "
                f"{(x1 + x2) / (2 * width):.6f} "
                f"{(y1 + y2) / (2 * height):.6f} "
                f"{box_width / width:.6f} {box_height / height:.6f}"
            )
        _validate_yolo_lines(lines, len(class_names), frame["frame_id"])
        eligible.append((frame, lines))

    videos = sorted({frame["video_id"] for frame, _ in eligible})
    if len(videos) < 3:
        raise ValueError(
            f"At least three fully reviewed videos with exportable frames are required "
            f"for train, validation, and test splits; found {len(videos)}."
        )
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Dataset output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.Random(seed).shuffle(videos)
    validation_count = max(1, round(len(videos) * 0.1))
    test_count = max(1, round(len(videos) * 0.1))
    validation_count = min(validation_count, len(videos) - 2)
    test_count = min(test_count, len(videos) - validation_count - 1)
    train_count = len(videos) - validation_count - test_count
    split_by_video = {
        video_id: split
        for split, video_ids in (
            ("train", videos[:train_count]),
            ("val", videos[train_count : train_count + validation_count]),
            ("test", videos[train_count + validation_count :]),
        )
        for video_id in video_ids
    }
    counts = {"train": 0, "val": 0, "test": 0}
    for frame, lines in eligible:
        split = split_by_video[frame["video_id"]]
        name = f"{frame['frame_id']}.jpg"
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frame["image_path"], image_dir / name)
        label_path = label_dir / f"{frame['frame_id']}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        _validate_yolo_lines(
            label_path.read_text(encoding="utf-8").splitlines(), len(class_names), label_path
        )
        counts[split] += 1

    config = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    with (output_dir / "data.yaml").open("x", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)
    return output_dir, counts


def zip_directory(directory: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory))
    return zip_path


def annotated_preview_zip(frames: list[dict[str, Any]], zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for frame in frames:
            image = cv2.imread(frame["image_path"])
            if image is None:
                raise ValueError(f"Could not read candidate image: {frame['image_path']}")
            selected = draw_review_image(image, frame["detections"], "")
            success, encoded = cv2.imencode(".jpg", selected)
            if not success:
                raise OSError(f"Could not encode preview for {frame['frame_id']}.")
            archive.writestr(
                f"{frame['video_id']}/{Path(frame['image_path']).name}", encoded.tobytes()
            )
    return zip_path
