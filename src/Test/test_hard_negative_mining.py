import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import yaml
from ultralytics.data.utils import check_det_dataset

from src.hard_negative_mining import _validated_labels, prepare_dataset, run_inference


class FakeModel:
    names = {0: "bee"}

    def __init__(self, _weights):
        self.calls = 0
        self.weights = _weights

    def __call__(self, frame, **kwargs):
        self.calls += 1
        self.options = kwargs
        boxes = FakeBoxes()
        return [SimpleNamespace(boxes=boxes, names=self.names, plot=lambda: frame.copy())]


class FakeBoxes:
    xyxy = np.array([[10, 12, 30, 32]], dtype=float)
    cls = np.array([0], dtype=float)
    conf = np.array([0.9], dtype=float)

    def __len__(self):
        return len(self.xyxy)


class HardNegativeMiningTests(unittest.TestCase):
    def test_inference_saves_review_files_and_skips_near_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "field.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                2,
                (64, 64),
            )
            self.assertTrue(writer.isOpened())
            for value in (0, 0, 255, 255, 0):
                writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
            writer.release()
            weights = root / "model.pt"
            weights.touch()
            output = root / "results"

            with patch("src.hard_negative_mining.YOLO", FakeModel):
                run_inference(video_path, weights, output, frame_interval=0.5)

            with (output / "review.csv").open(newline="", encoding="utf-8") as review_file:
                rows = list(csv.DictReader(review_file))
            with (output / "predictions.csv").open(newline="", encoding="utf-8") as prediction_file:
                predictions = list(csv.DictReader(prediction_file))

            self.assertEqual(len(rows), 3)
            self.assertEqual([row["frame"] for row in rows], ["0", "2", "4"])
            self.assertTrue(all(row["review_status"] == "unreviewed" for row in rows))
            self.assertEqual(len(predictions), 3)
            self.assertEqual(predictions[0]["class_name"], "bee")
            for row in rows:
                self.assertTrue((output / row["image"]).is_file())
                self.assertTrue((output / row["annotated"]).is_file())
                self.assertEqual(len((output / row["label"]).read_text().split()), 5)

    def test_prepare_dataset_preserves_classes_and_splits_by_video(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "inference"
            (source / "frames").mkdir(parents=True)
            (source / "labels").mkdir()
            data_yaml = root / "source.yaml"
            data_yaml.write_text(
                yaml.safe_dump({"nc": 2, "names": ["bee", "hoverfly"]}), encoding="utf-8"
            )
            review_path = source / "review.csv"
            fieldnames = [
                "video", "frame", "timestamp_seconds", "detection_count",
                "image", "annotated", "label", "review_status",
            ]
            rows = []
            for index, (video, status, label) in enumerate(
                (
                    ("field-a.mp4", "true_positive", "0 0.5 0.5 0.2 0.2\n"),
                    ("field-a.mp4", "true_positive", "1 0.4 0.6 0.1 0.2\n"),
                    ("field-b.mp4", "false_positive", ""),
                    ("field-c.mp4", "true_positive", "1 0.4 0.6 0.1 0.2\n"),
                    ("field-d.mp4", "uncertain", "0 0.5 0.5 0.2 0.2\n"),
                )
            ):
                image_path = source / "frames" / f"frame_{index}.jpg"
                label_path = source / "labels" / f"frame_{index}.txt"
                self.assertTrue(cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8)))
                label_path.write_text(label, encoding="utf-8")
                rows.append(
                    {
                        "video": video,
                        "frame": str(index),
                        "timestamp_seconds": "0.0",
                        "detection_count": "1",
                        "image": image_path.relative_to(source).as_posix(),
                        "annotated": "",
                        "label": label_path.relative_to(source).as_posix(),
                        "review_status": status,
                    }
                )
            with review_path.open("w", newline="", encoding="utf-8") as review_file:
                writer = csv.DictWriter(review_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            output = root / "dataset"
            prepare_dataset(review_path, output, data_yaml, seed=3)

            config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config["names"], ["bee", "hoverfly"])
            self.assertEqual(config["nc"], 2)
            self.assertEqual(check_det_dataset(str(output / "data.yaml"))["nc"], 2)
            split_videos = {"train": set(), "val": set(), "test": set()}
            video_splits = {}
            for index, (video, status) in enumerate(
                (
                    ("field-a.mp4", "true_positive"),
                    ("field-a.mp4", "true_positive"),
                    ("field-b.mp4", "false_positive"),
                    ("field-c.mp4", "true_positive"),
                )
            ):
                split = next(
                    split_name
                    for split_name in split_videos
                    if list((output / "images" / split_name).glob(f"reviewed_{index:06d}.*"))
                )
                split_videos[split].add(video)
                self.assertEqual(video_splits.setdefault(video, split), split)
                image = next((output / "images" / split).glob(f"reviewed_{index:06d}.*"))
                label = output / "labels" / split / f"{image.stem}.txt"
                self.assertTrue(label.is_file())
                if status == "false_positive":
                    self.assertEqual(label.read_text(encoding="utf-8"), "")
                else:
                    self.assertTrue(label.read_text(encoding="utf-8").strip())

            nonempty_splits = [videos for videos in split_videos.values() if videos]
            self.assertEqual(len(nonempty_splits), 3)
            self.assertEqual(sum(map(len, split_videos.values())), 3)

    def test_prepare_rejects_false_positive_with_nonempty_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "inference"
            (source / "frames").mkdir(parents=True)
            (source / "labels").mkdir()
            image = source / "frames" / "frame.jpg"
            label = source / "labels" / "frame.txt"
            self.assertTrue(cv2.imwrite(str(image), np.zeros((32, 32, 3), dtype=np.uint8)))
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            review_csv = source / "review.csv"
            with review_csv.open("w", newline="", encoding="utf-8") as review_file:
                writer = csv.DictWriter(
                    review_file,
                    fieldnames=[
                        "video", "frame", "timestamp_seconds", "detection_count",
                        "image", "annotated", "label", "review_status",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "video": "field.mp4",
                        "frame": "0",
                        "timestamp_seconds": "0",
                        "detection_count": "1",
                        "image": "frames/frame.jpg",
                        "annotated": "",
                        "label": "labels/frame.txt",
                        "review_status": "false_positive",
                    }
                )
            data_yaml = root / "data.yaml"
            data_yaml.write_text(
                yaml.safe_dump({"names": ["bee"], "nc": 1}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "label file must be empty"):
                prepare_dataset(review_csv, root / "dataset", data_yaml)

    def test_label_validation_rejects_boxes_outside_image(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            label = Path(temporary_directory) / "candidate.txt"
            label.write_text("0 0.9 0.5 0.4 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "extends outside the image"):
                _validated_labels(label, 1)


if __name__ == "__main__":
    unittest.main()
