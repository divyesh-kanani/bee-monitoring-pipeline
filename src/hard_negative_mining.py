"""Collect and review YOLO detections from field videos."""

import argparse
import csv
import math
import random
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "config" / "YOLOruns.yaml"
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}
REVIEW_STATUSES = {"unreviewed", "true_positive", "false_positive", "uncertain"}
REVIEW_FIELDS = (
    "video",
    "frame",
    "timestamp_seconds",
    "detection_count",
    "image",
    "annotated",
    "label",
    "review_status",
)
PREDICTION_FIELDS = (
    "video",
    "frame",
    "timestamp_seconds",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
)


def _find_videos(source: Path) -> list[tuple[Path, str]]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video format: {source}")
        return [(source, source.name)]
    if not source.is_dir():
        raise FileNotFoundError(f"Video source does not exist: {source}")

    videos = [
        (path, path.relative_to(source).as_posix())
        for path in sorted(source.rglob("*"))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not videos:
        raise ValueError(f"No supported videos found in: {source}")
    return videos


def _comparison_sample(frame: Any) -> Any:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)


def _yolo_labels(boxes: Any, width: int, height: int) -> list[str]:
    labels = []
    coordinates = boxes.xyxy.tolist()
    classes = boxes.cls.tolist()
    for box, class_id in zip(coordinates, classes):
        x1, y1, x2, y2 = box
        x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
        y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
        box_width, box_height = x2 - x1, y2 - y1
        if box_width <= 0 or box_height <= 0:
            raise ValueError("Model returned a bounding box with no area.")
        center_x = (x1 + x2) / (2 * width)
        center_y = (y1 + y2) / (2 * height)
        labels.append(
            f"{int(class_id)} {center_x:.6f} {center_y:.6f} "
            f"{box_width / width:.6f} {box_height / height:.6f}"
        )
    return labels


def run_inference(
    source: Path,
    weights: Path,
    output: Path,
    confidence: float = 0.25,
    iou: float = 0.7,
    image_size: int = 960,
    frame_interval: float = 2.0,
) -> None:
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    if not 0 <= iou <= 1:
        raise ValueError("IoU threshold must be between 0 and 1.")
    if image_size <= 0:
        raise ValueError("Image size must be a positive integer.")
    if not math.isfinite(frame_interval) or frame_interval <= 0:
        raise ValueError("Frame interval must be a positive number of seconds.")

    videos = _find_videos(source)
    weights = weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights do not exist: {weights}")

    output = output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    candidate_count = 0
    prediction_path = output / "predictions.csv"

    with prediction_path.open("x", newline="", encoding="utf-8") as predictions_file:
        prediction_writer = csv.DictWriter(predictions_file, fieldnames=PREDICTION_FIELDS)
        prediction_writer.writeheader()
        review_rows = []

        for video_path, video_name in videos:
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                capture.release()
                raise ValueError(f"Could not open video: {video_path}")

            try:
                fps = capture.get(cv2.CAP_PROP_FPS)
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
                    raise ValueError(f"Video has invalid frame rate or dimensions: {video_path}")
                frame_step = max(1, round(frame_interval * fps))
                next_sample = 0
                frame_number = 0
                decoded_frames = 0
                previous_sample = None
                video_output = PurePosixPath(video_name).with_suffix("")

                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    decoded_frames += 1
                    if frame_number < next_sample:
                        frame_number += 1
                        continue
                    next_sample += frame_step

                    frame_sample = _comparison_sample(frame)
                    if previous_sample is not None and cv2.absdiff(
                        frame_sample, previous_sample
                    ).mean() <= 2:
                        frame_number += 1
                        continue
                    previous_sample = frame_sample

                    results = model(
                        frame,
                        conf=confidence,
                        iou=iou,
                        imgsz=image_size,
                        verbose=False,
                    )
                    if not results:
                        frame_number += 1
                        continue
                    result = results[0]
                    boxes = result.boxes
                    if boxes is None or len(boxes) == 0:
                        frame_number += 1
                        continue

                    frame_name = f"frame_{frame_number:09d}"
                    relative_frame = video_output / frame_name
                    image_relative = Path("frames", *relative_frame.parts).with_suffix(".jpg")
                    annotated_relative = Path("annotated", *relative_frame.parts).with_suffix(".jpg")
                    label_relative = Path("labels", *relative_frame.parts).with_suffix(".txt")
                    image_path = output / image_relative
                    annotated_path = output / annotated_relative
                    label_path = output / label_relative
                    for path in (image_path, annotated_path, label_path):
                        path.parent.mkdir(parents=True, exist_ok=True)

                    if not cv2.imwrite(str(image_path), frame):
                        raise OSError(f"Could not save candidate frame: {image_path}")
                    if not cv2.imwrite(str(annotated_path), result.plot()):
                        raise OSError(f"Could not save annotated frame: {annotated_path}")

                    labels = _yolo_labels(boxes, width, height)
                    label_path.write_text(
                        "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
                    )
                    timestamp = frame_number / fps
                    review_rows.append(
                        {
                            "video": video_name,
                            "frame": frame_number,
                            "timestamp_seconds": f"{timestamp:.3f}",
                            "detection_count": len(boxes),
                            "image": image_relative.as_posix(),
                            "annotated": annotated_relative.as_posix(),
                            "label": label_relative.as_posix(),
                            "review_status": "unreviewed",
                        }
                    )

                    names = result.names
                    coordinates = boxes.xyxy.tolist()
                    classes = boxes.cls.tolist()
                    confidences = boxes.conf.tolist()
                    for class_id, box, score in zip(classes, coordinates, confidences):
                        prediction_writer.writerow(
                            {
                                "video": video_name,
                                "frame": frame_number,
                                "timestamp_seconds": f"{timestamp:.3f}",
                                "class_id": int(class_id),
                                "class_name": names[int(class_id)],
                                "confidence": f"{score:.6f}",
                                "x1": f"{box[0]:.2f}",
                                "y1": f"{box[1]:.2f}",
                                "x2": f"{box[2]:.2f}",
                                "y2": f"{box[3]:.2f}",
                            }
                        )
                    candidate_count += 1
                    frame_number += 1

                if decoded_frames == 0:
                    raise ValueError(f"Video contains no readable frames: {video_path}")
            finally:
                capture.release()

        with (output / "review.csv").open("x", newline="", encoding="utf-8") as review_file:
            writer = csv.DictWriter(review_file, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(review_rows)

    print(
        f"Saved {candidate_count} candidate frames from {len(videos)} video(s) "
        f"to {output}"
    )


def _read_class_names(data_yaml: Path) -> list[str]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset configuration does not exist: {data_yaml}")
    with data_yaml.open(encoding="utf-8") as data_file:
        data = yaml.safe_load(data_file)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset configuration must contain a YAML mapping: {data_yaml}")

    names = data.get("names")
    if isinstance(names, dict):
        try:
            names = [names[index] if index in names else names[str(index)] for index in range(len(names))]
        except KeyError as error:
            raise ValueError("Dataset class IDs in 'names' must be contiguous from zero.") from error
    if not isinstance(names, list) or not names or not all(
        isinstance(name, str) and name.strip() for name in names
    ):
        raise ValueError("Dataset configuration must define class names as a non-empty list or mapping.")
    if data.get("nc", len(names)) != len(names):
        raise ValueError("Dataset 'nc' does not match the number of class names.")
    return names


def _validate_yolo_lines(
    source_lines: list[str], class_count: int, label_path: Path | str
) -> list[str]:
    labels = []
    for line_number, line in enumerate(source_lines, 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected five YOLO label values at {label_path}:{line_number}.")
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"Invalid YOLO label value at {label_path}:{line_number}.") from error
        if class_id < 0 or class_id >= class_count:
            raise ValueError(f"Class ID {class_id} is outside the configured classes at {label_path}:{line_number}.")
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates):
            raise ValueError(f"YOLO coordinates must be finite and between 0 and 1 at {label_path}:{line_number}.")
        if coordinates[2] <= 0 or coordinates[3] <= 0:
            raise ValueError(f"YOLO box width and height must be positive at {label_path}:{line_number}.")
        center_x, center_y, box_width, box_height = coordinates
        if (
            center_x - box_width / 2 < 0
            or center_y - box_height / 2 < 0
            or center_x + box_width / 2 > 1
            or center_y + box_height / 2 > 1
        ):
            raise ValueError(f"YOLO box extends outside the image at {label_path}:{line_number}.")
        labels.append(line)
    return labels


def _validated_labels(label_path: Path, class_count: int) -> list[str]:
    if not label_path.is_file():
        raise FileNotFoundError(f"Candidate label file does not exist: {label_path}")
    return _validate_yolo_lines(
        label_path.read_text(encoding="utf-8").splitlines(), class_count, label_path
    )


def _candidate_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Review path must stay inside the inference output: {relative_path}") from error
    return path


def prepare_dataset(review_csv: Path, output: Path, data_yaml: Path, seed: int = 42) -> None:
    names = _read_class_names(data_yaml.expanduser().resolve())
    review_csv = review_csv.expanduser().resolve()
    if not review_csv.is_file():
        raise FileNotFoundError(f"Review CSV does not exist: {review_csv}")
    source_root = review_csv.parent
    reviewed = []
    seen_frames = set()

    with review_csv.open(newline="", encoding="utf-8") as review_file:
        reader = csv.DictReader(review_file)
        if reader.fieldnames is None or not set(REVIEW_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"Review CSV must contain these columns: {', '.join(REVIEW_FIELDS)}")
        for row in reader:
            status = row["review_status"].strip().lower()
            if status not in REVIEW_STATUSES:
                raise ValueError(
                    f"Invalid review status {row['review_status']!r}; use: "
                    f"{', '.join(sorted(REVIEW_STATUSES))}."
                )
            if status in {"unreviewed", "uncertain"}:
                continue
            video = row["video"].strip()
            frame = row["frame"].strip()
            if not video or not frame:
                raise ValueError("Reviewed candidates must have a video name and frame number.")
            if (video, frame) in seen_frames:
                raise ValueError(f"Duplicate reviewed candidate: {video}, frame {frame}.")
            seen_frames.add((video, frame))
            image_path = _candidate_path(source_root, row["image"])
            label_path = _candidate_path(source_root, row["label"])
            if not image_path.is_file():
                raise FileNotFoundError(f"Candidate image does not exist: {image_path}")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Candidate image cannot be read: {image_path}")
            labels = _validated_labels(label_path, len(names))
            if status == "true_positive" and not labels:
                raise ValueError(f"True-positive candidate has no insect boxes: {label_path}")
            if status == "false_positive" and labels:
                raise ValueError(
                    f"False-positive label file must be empty after verifying the whole frame: {label_path}"
                )
            reviewed.append(
                {
                    "video": video,
                    "image": image_path,
                    "label": label_path,
                }
            )

    if not reviewed:
        raise ValueError("No reviewed true-positive or false-positive candidates were found.")

    videos = sorted({candidate["video"] for candidate in reviewed})
    if len(videos) < 3:
        raise ValueError(
            f"At least three reviewed videos are required for separate train, validation, "
            f"and test splits; found {len(videos)}."
        )
    random.Random(seed).shuffle(videos)
    validation_count = max(1, round(len(videos) * 0.1))
    test_count = max(1, round(len(videos) * 0.1))
    validation_count = min(validation_count, len(videos) - 2)
    test_count = min(test_count, len(videos) - validation_count - 1)
    train_count = len(videos) - validation_count - test_count
    split_by_video = {
        video: split
        for split, split_videos in (
            ("train", videos[:train_count]),
            ("val", videos[train_count : train_count + validation_count]),
            ("test", videos[train_count + validation_count :]),
        )
        for video in split_videos
    }

    output = output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Dataset output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(reviewed):
        split = split_by_video[candidate["video"]]
        image_name = f"reviewed_{index:06d}{candidate['image'].suffix.lower()}"
        shutil.copy2(candidate["image"], output / "images" / split / image_name)
        shutil.copy2(candidate["label"], output / "labels" / split / f"{Path(image_name).stem}.txt")

    dataset_config = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    with (output / "data.yaml").open("x", encoding="utf-8") as config_file:
        yaml.safe_dump(dataset_config, config_file, sort_keys=False)

    print(
        f"Created reviewed dataset at {output} "
        f"({train_count} train, {validation_count} validation, {test_count} test video(s))."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine and review YOLO detections in field videos.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inference = subparsers.add_parser("infer", help="Find and save candidate detections from videos.")
    inference.add_argument("--source", type=Path, required=True, help="A video file or a folder of videos.")
    inference.add_argument("--weights", type=Path, required=True, help="Trained Ultralytics .pt weights.")
    inference.add_argument("--output", type=Path, default=Path("hard_negative_output"))
    inference.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1).")
    inference.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (0-1).")
    inference.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    inference.add_argument(
        "--frame-interval", type=float, default=2.0, help="Sample one frame every N seconds."
    )

    prepare = subparsers.add_parser("prepare", help="Build a YOLO dataset from manually reviewed frames.")
    prepare.add_argument("--review-csv", type=Path, required=True, help="CSV produced by the infer command.")
    prepare.add_argument("--output", type=Path, default=Path("hard_negative_dataset"))
    prepare.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Existing YOLO data.yaml for class names.")
    prepare.add_argument("--seed", type=int, default=42, help="Seed for video-level split assignment.")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "infer":
            run_inference(
                args.source,
                args.weights,
                args.output,
                confidence=args.conf,
                iou=args.iou,
                image_size=args.imgsz,
                frame_interval=args.frame_interval,
            )
        else:
            prepare_dataset(args.review_csv, args.output, args.data, seed=args.seed)
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
