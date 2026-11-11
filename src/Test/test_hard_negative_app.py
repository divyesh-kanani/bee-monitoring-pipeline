import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from src.hard_negative_mining import _read_class_names


class HardNegativeAppTests(unittest.TestCase):
    def test_dashboard_imports_when_started_from_its_source_directory(self):
        project_root = Path(__file__).resolve().parents[2]
        app_file = project_root / "src" / "hard_negative_app.py"
        script = (
            "import runpy, sys; "
            "src, app = sys.argv[1:]; "
            "sys.path = [src, *[path for path in sys.path[1:] "
            "if path and path != app.rsplit('/src/', 1)[0]]]; "
            "runpy.run_path(app, run_name='import_probe')"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(app_file.parent), str(app_file)],
            cwd=app_file.parent,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_detection_mode_offers_optional_bytetrack_and_parameters(self):
        project_root = Path(__file__).resolve().parents[2]
        app_file = project_root / "src" / "hard_negative_app.py"
        with tempfile.TemporaryDirectory() as temporary_directory:
            sample_path = Path(temporary_directory) / "sample.mp4"
            writer = cv2.VideoWriter(
                str(sample_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for frame_number in range(5):
                writer.write(
                    np.full((48, 64, 3), frame_number * 30, dtype=np.uint8)
                )
            writer.release()

            with patch.dict(
                os.environ,
                {"BEE_MINING_DATA_DIR": str(Path(temporary_directory) / "sessions")},
            ):
                app = AppTest.from_file(str(app_file)).run(timeout=30)
                app.radio[0].set_value("Detection").run(timeout=30)
                videos = next(
                    uploader for uploader in app.file_uploader if uploader.label == "Videos"
                )
                videos.set_value(
                    ("sample.mp4", sample_path.read_bytes(), "video/mp4")
                ).run(timeout=30)

                self.assertFalse(app.exception)
                tracking_checkbox = next(
                    checkbox
                    for checkbox in app.checkbox
                    if checkbox.label == "Apply object tracking after detection"
                )
                tracking_checkbox.set_value(True).run(timeout=30)

                self.assertFalse(app.exception)
                algorithm = next(
                    selectbox
                    for selectbox in app.selectbox
                    if selectbox.label == "Tracking algorithm"
                )
                self.assertEqual(algorithm.options, ["ByteTrack"])
                self.assertTrue(
                    any(
                        number.label == "Track buffer (frames)"
                        for number in app.number_input
                    )
                )
                self.assertTrue(
                    any(
                        button.label == "Start detection and tracking"
                        for button in app.button
                    )
                )

    def test_browser_reload_clears_the_previous_session_directory(self):
        project_root = Path(__file__).resolve().parents[2]
        app_file = project_root / "src" / "hard_negative_app.py"
        with tempfile.TemporaryDirectory() as temporary_directory:
            session_id = uuid.uuid4().hex
            session_dir = Path(temporary_directory) / session_id
            session_dir.mkdir(parents=True)
            (session_dir / "review.json").write_text("{}", encoding="utf-8")

            with patch.dict(os.environ, {"BEE_MINING_DATA_DIR": temporary_directory}):
                app = AppTest.from_file(str(app_file))
                app.query_params["session"] = session_id
                app.run(timeout=30)

            self.assertFalse(app.exception)
            self.assertFalse(session_dir.exists())
            self.assertNotEqual(app.query_params["session"], session_id)

    def test_review_ui_selects_and_persists_individual_decisions(self):
        project_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary_directory:
            session_id = uuid.uuid4().hex
            session_dir = Path(temporary_directory) / session_id
            image_path = session_dir / "frames" / "frame_000000010.jpg"
            image_path.parent.mkdir(parents=True)
            self.assertTrue(
                cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))
            )
            class_names = _read_class_names(project_root / "config" / "YOLOruns.yaml")
            state = {
                "videos": {},
                "class_names": class_names,
                "exports": {},
                "frames": [
                    {
                        "frame_id": "video-a_000000010",
                        "video_id": "video-a",
                        "video_filename": "field.mp4",
                        "frame_number": 10,
                        "timestamp_seconds": 1.0,
                        "image_path": str(image_path),
                        "width": 64,
                        "height": 64,
                        "annotation_complete": False,
                        "detections": [
                            {
                                "detection_id": "video-a_000000010-d001",
                                "class_id": 0,
                                "class_name": class_names[0],
                                "predicted_class_id": 0,
                                "predicted_class": class_names[0],
                                "confidence": 0.9,
                                "x1": 4.0,
                                "y1": 5.0,
                                "x2": 25.0,
                                "y2": 30.0,
                                "status": "unreviewed",
                                "manual": False,
                                "history": [],
                            },
                            {
                                "detection_id": "video-a_000000010-d002",
                                "class_id": 1,
                                "class_name": class_names[1],
                                "predicted_class_id": 1,
                                "predicted_class": class_names[1],
                                "confidence": 0.7,
                                "x1": 30.0,
                                "y1": 5.0,
                                "x2": 55.0,
                                "y2": 35.0,
                                "status": "unreviewed",
                                "manual": False,
                                "history": [],
                            },
                        ],
                    }
                ],
            }
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "review.json").write_text(json.dumps(state), encoding="utf-8")
            app_file = project_root / "src" / "hard_negative_app.py"

            with patch.dict(os.environ, {"BEE_MINING_DATA_DIR": temporary_directory}):
                app = AppTest.from_file(str(app_file))
                app.session_state["_bee_dashboard_initialized"] = True
                app.query_params["session"] = session_id
                app.run(timeout=30)
                self.assertFalse(app.exception)
                self.assertTrue(any(button.label == "1 ○" for button in app.button))
                self.assertTrue(any(button.label == "2 ○" for button in app.button))
                self.assertTrue(
                    any("CSV" in button.label for button in app.download_button)
                )
                self.assertTrue(
                    any(button.label == "Reject detection" for button in app.button)
                )
                self.assertTrue(
                    any(button.label == "Mark uncertain" for button in app.button)
                )
                self.assertTrue(
                    any(button.label == "Undo decision" for button in app.button)
                )
                self.assertTrue(
                    any(button.label == "← Previous" for button in app.button)
                )
                self.assertTrue(any(button.label == "Next →" for button in app.button))
                self.assertTrue(
                    any(
                        checkbox.label
                        == "I have checked the entire frame; all target insects are annotated."
                        for checkbox in app.checkbox
                    )
                )
                self.assertTrue(any(button.label == "Save boxes" for button in app.button))
                self.assertTrue(
                    any(box.label == "Selected box class" for box in app.selectbox)
                )
                self.assertFalse(any(box.label.startswith("Class —") for box in app.selectbox))

                keep_button = next(
                    button for button in app.button if button.label == "Keep detection"
                )
                keep_button.click().run(timeout=30)

                saved_state = json.loads(
                    (session_dir / "review.json").read_text(encoding="utf-8")
                )
                saved_detections = saved_state["frames"][0]["detections"]
                self.assertEqual(saved_detections[0]["status"], "true_positive")
                self.assertEqual(saved_detections[0]["history"], ["unreviewed"])
                self.assertEqual(saved_detections[1]["status"], "unreviewed")
                self.assertEqual(len(saved_detections), 2)
                self.assertEqual(
                    app.session_state[
                        f"{session_id}:video-a_000000010:selected_detection"
                    ],
                    "video-a_000000010-d002",
                )
                reject_button = next(
                    button for button in app.button if button.label == "Reject detection"
                )
                reject_button.click().run(timeout=30)
                saved_state = json.loads(
                    (session_dir / "review.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    saved_state["frames"][0]["detections"][1]["status"],
                    "false_positive",
                )
                delete_button = next(
                    button for button in app.button if button.label == "Delete box"
                )
                delete_button.click().run(timeout=30)
                saved_state = json.loads(
                    (session_dir / "review.json").read_text(encoding="utf-8")
                )
                self.assertEqual(len(saved_state["frames"][0]["detections"]), 1)
                self.assertFalse(saved_state["frames"][0]["annotation_complete"])


if __name__ == "__main__":
    unittest.main()
