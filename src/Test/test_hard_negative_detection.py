import json
import sys
import subprocess
import csv
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import imageio_ffmpeg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hard_negative_app as app


class FakeRFDETRModel:
    model_config = SimpleNamespace(patch_size=14, num_windows=1)

    def __init__(self):
        self.image = None
        self.kwargs = None

    def predict(self, image, **kwargs):
        self.image = image.copy()
        self.kwargs = kwargs
        return SimpleNamespace(
            xyxy=np.array([[1, 2, 20, 25]], dtype=float),
            confidence=np.array([0.85]),
            class_id=np.array([0]),
            data={"class_name": np.array(["bee"])},
        )


class FakeYOLOModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, _frame, **_kwargs):
        self.calls += 1
        boxes = SimpleNamespace(
            xyxy=np.array([[5, 6, 25, 26]], dtype=float),
            cls=np.array([0], dtype=float),
            conf=np.array([0.9], dtype=float),
        )
        return [SimpleNamespace(boxes=boxes)]


class FakeProgress:
    def __init__(self):
        self.values = []

    def progress(self, value, **_kwargs):
        self.values.append(value)


class FakePreview:
    def __init__(self):
        self.images = []

    def image(self, image, **_kwargs):
        self.images.append(image.copy())


class FakeStatus:
    def __init__(self):
        self.messages = []

    def caption(self, message):
        self.messages.append(message)


class HardNegativeDetectionTests(unittest.TestCase):
    def test_rfdetr_adapter_uses_rgb_and_normalizes_predictions(self):
        model = FakeRFDETRModel()
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        frame[:] = (10, 20, 30)

        detections = app._predict_frame(
            "RF-DETR",
            model,
            frame,
            ["insect"],
            confidence=0.4,
            iou=0.7,
            image_size=960,
        )

        self.assertEqual(model.image[0, 0].tolist(), [30, 20, 10])
        self.assertEqual(model.kwargs["shape"], (952, 952))
        self.assertEqual(detections[0]["class_id"], 0)
        self.assertEqual(detections[0]["class_name"], "bee")
        self.assertEqual(detections[0]["confidence"], 0.85)
        self.assertEqual(
            [detections[0][key] for key in ("x1", "y1", "x2", "y2")],
            [1.0, 2.0, 20.0, 25.0],
        )

    def test_backend_accepts_pth_only_for_rf_detr(self):
        self.assertEqual(app._supported_weight_extensions("RF-DETR"), ["pth", "pt"])
        self.assertEqual(app._supported_weight_extensions("Ultralytics YOLO"), ["pt"])
        checkpoint = object()
        with patch.object(
            app.RFDETRLarge, "from_checkpoint", return_value=checkpoint
        ) as load_checkpoint:
            model, class_names = app._load_backend_model(
                "RF-DETR", Path("fine_tuned.pth"), ["bee", "fly"]
            )
        self.assertIs(model, checkpoint)
        self.assertEqual(class_names, ["bee", "fly"])
        load_checkpoint.assert_called_once_with("fine_tuned.pth")

    def test_session_cleanup_removes_session_uploads_and_exports(self):
        session_id = uuid.uuid4().hex
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            session = root / session_id
            other_session = root / uuid.uuid4().hex
            old_exports = root / "detection_exports" / session_id
            old_previews = root / "detection_preview" / session_id
            legacy_export = root / "detection_exports" / "old-random-output"
            for path in (session, other_session, old_exports, old_previews):
                path.mkdir(parents=True)
                (path / "temporary.bin").write_bytes(b"temporary")
            legacy_export.mkdir()
            (legacy_export / "old-video.mp4").write_bytes(b"temporary")

            with patch.object(app, "SESSION_ROOT", root):
                app._clear_session_data(session_id)
                app._clear_other_session_data()
                app._clear_legacy_detection_data()

            self.assertFalse(session.exists())
            self.assertFalse(other_session.exists())
            self.assertFalse(old_exports.exists())
            self.assertFalse(old_previews.exists())
            self.assertFalse(legacy_export.exists())

    def test_transcoded_mp4_is_h264_and_playable_by_opencv(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "temporary.mp4"
            destination = root / "annotated.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for frame_number in range(5):
                writer.write(
                    np.full((48, 64, 3), frame_number * 30, dtype=np.uint8)
                )
            writer.release()

            app._transcode_mp4(source, destination)

            probe = subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(destination)],
                capture_output=True,
                text=True,
            )
            self.assertIn("Video: h264", probe.stderr)
            capture = cv2.VideoCapture(str(destination))
            self.assertTrue(capture.isOpened())
            frame_count = 0
            while capture.read()[0]:
                frame_count += 1
            capture.release()
            self.assertEqual(frame_count, 5)

    def test_video_runner_keeps_source_timing_and_exports_predictions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "sample.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for frame_number in range(10):
                writer.write(
                    np.full((48, 64, 3), frame_number * 20, dtype=np.uint8)
                )
            writer.release()

            model = FakeYOLOModel()
            progress = FakeProgress()
            preview = FakePreview()
            status = FakeStatus()
            stop_key = "test:stop"
            with (
                patch.object(app, "SESSION_ROOT", root),
                patch.object(app, "_load_backend_model", return_value=(model, ["bee"])),
                patch.object(app.st, "session_state", {stop_key: False}),
            ):
                video_output, csv_output = app._run_detection_backend(
                    "test-session",
                    "Ultralytics YOLO",
                    root / "weights.pt",
                    source,
                    ["bee"],
                    start_seconds=0,
                    end_seconds=1,
                    frames_to_skip=1,
                    confidence=0.25,
                    iou=0.7,
                    image_size=640,
                    progress=progress,
                    preview=preview,
                    timestamp_status=status,
                    stop_key=stop_key,
                )

            self.assertEqual(model.calls, 3)
            self.assertEqual(len(preview.images), 3)
            self.assertEqual(len(status.messages), 3)
            with csv_output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual([int(row["frame_number"]) for row in rows], [0, 2, 4])
            self.assertEqual(
                [float(row["timestamp_seconds"]) for row in rows],
                [0.0, 0.4, 0.8],
            )
            capture = cv2.VideoCapture(str(video_output))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 5)
            capture.release()

    def test_video_runner_applies_bytetrack_to_every_frame(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "sample.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for frame_number in range(10):
                writer.write(
                    np.full((48, 64, 3), frame_number * 20, dtype=np.uint8)
                )
            writer.release()

            model = FakeYOLOModel()
            stop_key = "tracking-test:stop"
            tracking_config = {
                "algorithm": "ByteTrack",
                "low_detection_threshold": 0.25,
                "high_detection_threshold": 0.5,
                "track_activation_threshold": 0.5,
                "track_buffer": 30,
                "matching_threshold": 0.8,
            }
            with (
                patch.object(app, "SESSION_ROOT", root),
                patch.object(app, "_load_backend_model", return_value=(model, ["bee"])),
                patch.object(app.st, "session_state", {stop_key: False}),
            ):
                video_output, csv_output = app._run_detection_backend(
                    "tracking-test",
                    "Ultralytics YOLO",
                    root / "weights.pt",
                    source,
                    ["bee"],
                    start_seconds=0,
                    end_seconds=2,
                    frames_to_skip=4,
                    confidence=0.25,
                    iou=0.7,
                    image_size=640,
                    progress=FakeProgress(),
                    preview=FakePreview(),
                    timestamp_status=FakeStatus(),
                    stop_key=stop_key,
                    tracking_config=tracking_config,
                )

            self.assertEqual(model.calls, 10)
            self.assertTrue(video_output.name.endswith("_tracked.mp4"))
            capture = cv2.VideoCapture(str(video_output))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
            capture.release()
            with csv_output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertIn("track_id", rows[0])
            self.assertEqual(len(rows), 10)
            self.assertTrue(any(row["track_id"] for row in rows))
            summary_path = video_output.with_name("sample_tracking.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["frames_processed"], 10)
            self.assertEqual(summary["total_unique_track_ids"], 1)


if __name__ == "__main__":
    unittest.main()
