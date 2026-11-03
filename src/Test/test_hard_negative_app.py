import json
import os
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
                detections = next(
                    box
                    for box in app.selectbox
                    if box.label.startswith("Select one detection")
                )
                self.assertEqual(len(detections.options), 2)
                self.assertTrue(
                    any("CSV" in button.label for button in app.download_button)
                )
                self.assertTrue(
                    any(expander.label == "Add a missed insect" for expander in app.expander)
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
                    any(button.label == "Save box correction" for button in app.button)
                )

                keep_button = next(
                    button for button in app.button if button.label == "Keep detection"
                )
                keep_button.click().run(timeout=30)
                box_input = next(
                    number
                    for number in app.number_input
                    if number.key == f"{session_id}:video-a_000000010-d001:x1"
                )
                box_input.set_value(6.0).run(timeout=30)
                save_box = next(
                    button
                    for button in app.button
                    if button.label == "Save box correction"
                )
                save_box.click().run(timeout=30)

                add_button = next(
                    button
                    for button in app.form_submit_button
                    if button.label == "Add missed insect"
                )
                add_button.click().run(timeout=30)

            saved_state = json.loads((session_dir / "review.json").read_text(encoding="utf-8"))
            saved_detections = saved_state["frames"][0]["detections"]
            self.assertEqual(saved_detections[0]["status"], "true_positive")
            self.assertEqual(saved_detections[0]["history"], ["unreviewed"])
            self.assertEqual(saved_detections[0]["x1"], 6.0)
            self.assertEqual(saved_detections[1]["status"], "unreviewed")
            self.assertEqual(len(saved_detections), 3)
            self.assertTrue(saved_detections[2]["manual"])
            self.assertEqual(saved_detections[2]["status"], "true_positive")


if __name__ == "__main__":
    unittest.main()
