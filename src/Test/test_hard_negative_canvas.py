import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hard_negative_app as app
from hard_negative_review import create_dataset


def _detection(
    detection_id: str,
    status: str,
    box: tuple[float, float, float, float],
    stroke: str,
    class_id: int = 0,
) -> dict:
    x1, y1, x2, y2 = box
    return {
        "detection_id": detection_id,
        "class_id": class_id,
        "class_name": ["bee", "fly"][class_id],
        "predicted_class_id": class_id,
        "predicted_class": ["bee", "fly"][class_id],
        "confidence": 0.9,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "status": status,
        "manual": False,
        "history": [],
        "canvas_stroke": stroke,
    }


class HardNegativeCanvasTests(unittest.TestCase):
    def test_decision_selection_advances_to_next_unreviewed_then_wraps(self):
        detections = [
            {"detection_id": "d1", "status": "unreviewed"},
            {"detection_id": "d2", "status": "true_positive"},
            {"detection_id": "d3", "status": "unreviewed"},
        ]

        self.assertEqual(app._next_unreviewed_detection(detections, "d2"), "d3")
        self.assertEqual(app._next_unreviewed_detection(detections, "d3"), "d1")

    def test_initial_canvas_includes_editable_boxes_for_every_review_status(self):
        frame = {
            "width": 200,
            "height": 100,
            "detections": [
                _detection("d1", "unreviewed", (5, 6, 25, 30), "#112233"),
                _detection("d2", "false_positive", (30, 10, 50, 35), "#223344"),
                _detection("d3", "uncertain", (60, 10, 80, 35), ""),
            ]
        }

        drawing, stroke_map, changed = app._canvas_drawing(frame, 100, 50)

        self.assertEqual(len(drawing["objects"]), 3)
        self.assertEqual(set(stroke_map.values()), {"d1", "d2", "d3"})
        self.assertTrue(changed)
        self.assertNotEqual(
            drawing["objects"][0]["stroke"], drawing["objects"][1]["stroke"]
        )
        self.assertEqual(drawing["objects"][0]["left"], 2.5)
        self.assertEqual(drawing["objects"][0]["width"], 10.0)

    def test_numbered_preview_uses_review_status_colours(self):
        image = np.zeros((60, 120, 3), dtype=np.uint8)
        detections = [
            _detection("d1", "true_positive", (5, 5, 30, 30), "#112233"),
            _detection("d2", "false_positive", (40, 5, 65, 30), "#223344"),
            _detection("d3", "uncertain", (75, 5, 100, 30), "#334455"),
        ]

        preview = app._review_image(image, detections, "d2")

        self.assertTrue(np.array_equal(preview[5, 5], (25, 105, 25)))
        self.assertTrue(np.array_equal(preview[5, 40], (45, 45, 225)))
        self.assertTrue(np.array_equal(preview[5, 75], (0, 116, 140)))

    def test_move_resize_add_delete_and_class_correction_map_to_original_pixels(self):
        frame = {
            "frame_id": "video-a_000000001",
            "width": 200,
            "height": 100,
            "annotation_complete": True,
            "detections": [
                _detection("video-a_000000001-d001", "unreviewed", (1, 1, 20, 20), "#112233"),
                _detection("video-a_000000001-d002", "true_positive", (30, 10, 60, 40), "#223344"),
            ],
        }
        stroke_map = {
            "#112233": "video-a_000000001-d001",
            "#223344": "video-a_000000001-d002",
        }
        shapes = [
            {
                "type": "rect",
                "stroke": "#112233",
                "left": 10,
                "top": 5,
                "width": 10,
                "height": 8,
                "scaleX": 2,
                "scaleY": 1.5,
                "angle": 0,
            },
            {
                "type": "rect",
                "stroke": "#aabbcc",
                "left": 50,
                "top": 10,
                "width": 10,
                "height": 5,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
            },
        ]

        app._save_canvas_annotations(
            frame,
            shapes,
            stroke_map,
            {"#112233": 1, "#aabbcc": 0},
            scale_x=2,
            scale_y=2,
            class_names=["bee", "fly"],
        )

        edited = frame["detections"][0]
        self.assertEqual(edited["class_id"], 1)
        self.assertEqual(edited["class_name"], "fly")
        self.assertEqual(
            [edited[key] for key in ("x1", "y1", "x2", "y2")],
            [20.0, 10.0, 60.0, 34.0],
        )
        self.assertEqual(frame["detections"][1]["status"], "false_positive")
        self.assertEqual(frame["detections"][1]["history"], ["true_positive"])
        manual = frame["detections"][2]
        self.assertTrue(manual["manual"])
        self.assertEqual(manual["status"], "true_positive")
        self.assertEqual(manual["class_id"], 0)
        self.assertEqual(
            [manual[key] for key in ("x1", "y1", "x2", "y2")],
            [100.0, 20.0, 120.0, 30.0],
        )
        self.assertFalse(frame["annotation_complete"])

    def test_boxes_are_clamped_and_too_small_or_invalid_classes_are_rejected(self):
        box = app._canvas_box(
            {"left": 45, "top": 20, "width": 10, "height": 10},
            scale_x=2,
            scale_y=2,
            width=100,
            height=60,
        )
        self.assertEqual(box, {"x1": 90.0, "y1": 40.0, "x2": 100.0, "y2": 60.0})
        with self.assertRaisesRegex(ValueError, "at least one"):
            app._canvas_box(
                {"left": 0, "top": 0, "width": 0.1, "height": 1},
                scale_x=0.1,
                scale_y=1,
                width=100,
                height=60,
            )
        with self.assertRaisesRegex(ValueError, "valid class"):
            app._save_canvas_annotations(
                {
                    "frame_id": "frame",
                    "width": 10,
                    "height": 10,
                    "annotation_complete": False,
                    "detections": [],
                },
                [{"type": "rect", "stroke": "#aabbcc", "width": 2, "height": 2}],
                {},
                {"#aabbcc": 2},
                1,
                1,
                ["bee"],
            )

    def test_deleting_manual_box_removes_it_and_marks_frame_incomplete(self):
        frame = {
            "frame_id": "frame",
            "width": 100,
            "height": 50,
            "annotation_complete": True,
            "detections": [
                {
                    **_detection("manual-id", "true_positive", (5, 5, 20, 20), "#112233"),
                    "manual": True,
                }
            ],
        }

        app._save_canvas_annotations(
            frame,
            [],
            {"#112233": "manual-id"},
            {},
            1,
            1,
            ["bee", "fly"],
        )

        self.assertEqual(frame["detections"], [])
        self.assertFalse(frame["annotation_complete"])


    def test_corrected_and_manual_boxes_export_with_valid_yolo_classes_and_coords(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_paths = []
            frames = []
            for video_id in ("video-a", "video-b", "video-c"):
                image_path = root / f"{video_id}.jpg"
                self.assertTrue(
                    cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8))
                )
                image_paths.append(image_path)
                frames.append(
                    {
                        "frame_id": f"{video_id}_000000001",
                        "video_id": video_id,
                        "video_filename": f"{video_id}.mp4",
                        "frame_number": 1,
                        "timestamp_seconds": 0.1,
                        "image_path": str(image_path),
                        "width": 200,
                        "height": 100,
                        "annotation_complete": True,
                        "detections": [],
                    }
                )

            frame = frames[0]
            frame["detections"] = [
                _detection("corrected", "true_positive", (20, 10, 60, 34), "#123456", 1),
                {
                    **_detection("manual", "true_positive", (100, 20, 120, 30), "#abcdef"),
                    "manual": True,
                    "predicted_class_id": "",
                    "predicted_class": "manual annotation",
                },
                _detection("rejected", "false_positive", (70, 10, 90, 30), "#654321"),
            ]
            for other in frames[1:]:
                other["detections"] = [
                    _detection(f"{other['video_id']}-bee", "true_positive", (5, 5, 25, 25), "#112233")
                ]

            dataset_dir, split_counts = create_dataset(
                frames, ["bee", "fly"], root / "export", seed=1
            )

            self.assertEqual(sum(split_counts.values()), 3)
            exported = []
            for label_path in (dataset_dir / "labels").rglob("*.txt"):
                exported.extend(label_path.read_text(encoding="utf-8").splitlines())
            self.assertIn("1 0.200000 0.220000 0.200000 0.240000", exported)
            self.assertIn("0 0.550000 0.250000 0.100000 0.100000", exported)
            self.assertFalse(any(line.startswith("0 0.400000") for line in exported))


if __name__ == "__main__":
    unittest.main()
