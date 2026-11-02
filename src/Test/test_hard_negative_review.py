import csv
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src.hard_negative_review import (
    create_dataset,
    extract_detections,
    frame_bounds,
    read_video_metadata,
    report_csv,
    sampled_frame_numbers,
    zip_directory,
)


class FakeBoxes:
    xyxy = np.array([[5, 6, 25, 26], [30, 10, 55, 45]], dtype=float)
    cls = np.array([0, 1], dtype=float)
    conf = np.array([0.9, 0.6], dtype=float)

    def __len__(self):
        return len(self.xyxy)


class FakeModel:
    def __init__(self):
        self.calls = 0
        self.last_shape = None
        self.last_kwargs = {}

    def __call__(self, frame, **kwargs):
        self.calls += 1
        self.last_shape = frame.shape
        self.last_kwargs = kwargs
        return [SimpleNamespace(boxes=FakeBoxes())]


def _detection(detection_id, class_id, class_name, status, box=(5, 5, 25, 25)):
    return {
        "detection_id": detection_id,
        "class_id": class_id,
        "class_name": class_name,
        "predicted_class_id": class_id,
        "predicted_class": class_name,
        "confidence": 0.9,
        "x1": box[0],
        "y1": box[1],
        "x2": box[2],
        "y2": box[3],
        "status": status,
        "manual": False,
        "history": [],
    }


class HardNegativeReviewTests(unittest.TestCase):
    def test_frame_bounds_and_sampling(self):
        self.assertEqual(frame_bounds(30, 90, 600, 30, 18000), (900, 2700))
        self.assertEqual(
            list(sampled_frame_numbers(900, 2700, 29)),
            list(range(900, 2700, 30)),
        )
        self.assertEqual(len(sampled_frame_numbers(900, 2700, 29)), 60)
        self.assertEqual(list(sampled_frame_numbers(4, 10, 0)), list(range(4, 10)))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            frame_bounds(-1, 2, 10, 30, 300)
        with self.assertRaisesRegex(ValueError, "greater"):
            frame_bounds(2, 2, 10, 30, 300)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            frame_bounds(2, 11, 10, 30, 300)
        with self.assertRaisesRegex(ValueError, "skip"):
            sampled_frame_numbers(0, 10, -1)

    def test_extraction_respects_time_range_and_skip_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video = root / "sample.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64)
            )
            self.assertTrue(writer.isOpened())
            for frame_number in range(60):
                writer.write(
                    np.full((64, 64, 3), frame_number * 3, dtype=np.uint8)
                )
            writer.release()

            metadata = read_video_metadata(video)
            self.assertAlmostEqual(metadata["fps"], 10, delta=0.1)
            self.assertEqual(metadata["total_frames"], 60)
            model = FakeModel()
            frames = extract_detections(
                video,
                "video-id",
                "sample.mp4",
                root / "session",
                model,
                ["bee", "hoverfly"],
                confidence=0.2,
                iou=0.7,
                image_size=640,
                start_seconds=1,
                end_seconds=3,
                frames_to_skip=4,
            )
            self.assertEqual([frame["frame_number"] for frame in frames], [10, 15, 20, 25])
            self.assertEqual(model.calls, 4)
            self.assertEqual(model.last_shape, (64, 64, 3))
            self.assertEqual(model.last_kwargs["imgsz"], 640)
            self.assertEqual(frames[0]["timestamp_seconds"], 1.0)
            self.assertEqual(frames[-1]["timestamp_seconds"], 2.5)
            self.assertEqual(len(frames[0]["detections"]), 2)
            self.assertTrue(Path(frames[0]["image_path"]).is_file())
            self.assertEqual(frames[0]["detections"][1]["detection_id"], "video-id_000000010-d002")

    def test_mixed_decisions_uncertain_frames_empty_labels_and_download_zip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            def frame(video_id, frame_number, detections):
                image_path = root / f"{video_id}_{frame_number}.jpg"
                self.assertTrue(
                    cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))
                )
                return {
                    "frame_id": f"{video_id}_{frame_number:09d}",
                    "video_id": video_id,
                    "video_filename": f"{video_id}.mp4",
                    "frame_number": frame_number,
                    "timestamp_seconds": frame_number / 10,
                    "image_path": str(image_path),
                    "width": 64,
                    "height": 64,
                    "annotation_complete": True,
                    "detections": detections,
                }

            frames = [
                frame(
                    "video-a",
                    1,
                    [
                        _detection("a1", 0, "bee", "true_positive"),
                        _detection("a2", 1, "hoverfly", "false_positive"),
                    ],
                ),
                frame(
                    "video-a",
                    2,
                    [
                        _detection("a3", 0, "bee", "true_positive"),
                        _detection("a4", 1, "hoverfly", "uncertain"),
                    ],
                ),
                frame(
                    "video-a",
                    3,
                    [_detection("a5", 0, "bee", "true_positive")],
                ),
                frame(
                    "video-b",
                    1,
                    [_detection("b1", 0, "bee", "false_positive")],
                ),
                frame(
                    "video-c",
                    1,
                    [_detection("c1", 1, "hoverfly", "true_positive")],
                ),
            ]
            report = list(csv.DictReader(report_csv(frames).decode("utf-8").splitlines()))
            self.assertEqual(len(report), 7)
            self.assertEqual(report[0]["final_annotation_status"], "included")
            self.assertEqual(report[1]["final_annotation_status"], "excluded_rejected")
            self.assertEqual(report[3]["final_annotation_status"], "excluded_uncertain")
            self.assertIn("timestamp_seconds", report[0])

            dataset_dir, split_counts = create_dataset(
                frames, ["bee", "hoverfly"], root / "dataset", seed=4
            )
            self.assertEqual(sum(split_counts.values()), 4)
            self.assertTrue(all(split_counts[split] > 0 for split in ("train", "val", "test")))
            split_by_video = {}
            for split in ("train", "val", "test"):
                image_dir = dataset_dir / "images" / split
                label_dir = dataset_dir / "labels" / split
                for image_path in image_dir.glob("*.jpg"):
                    frame_id = image_path.stem
                    video_id = frame_id.rsplit("_", 1)[0]
                    self.assertEqual(split_by_video.setdefault(video_id, split), split)
                    label_path = label_dir / f"{frame_id}.txt"
                    self.assertTrue(label_path.is_file())
                    if video_id == "video-b":
                        self.assertEqual(label_path.read_text(encoding="utf-8"), "")
                    else:
                        labels = label_path.read_text(encoding="utf-8").splitlines()
                        self.assertEqual(len(labels), 1)
                        self.assertEqual(labels[0].split()[0], "0" if video_id == "video-a" else "1")
            self.assertFalse(any(dataset_dir.rglob("*video-a_000000002*")))
            self.assertEqual(set(split_by_video), {"video-a", "video-b", "video-c"})

            zip_path = zip_directory(dataset_dir, root / "dataset.zip")
            self.assertTrue(zip_path.is_file())
            with zipfile.ZipFile(zip_path) as archive:
                self.assertIn("data.yaml", archive.namelist())
                self.assertEqual(
                    sum(name.startswith("images/") for name in archive.namelist()), 4
                )
                self.assertEqual(
                    sum(name.startswith("labels/") for name in archive.namelist()), 4
                )


if __name__ == "__main__":
    unittest.main()
