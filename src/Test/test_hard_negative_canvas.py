import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    def test_uploaded_file_is_hashed_and_written_only_once_per_upload_id(self):
        class UploadedFile(io.BytesIO):
            name = "field clip.mp4"
            file_id = "unique-upload-id"
            size = len(b"video bytes")

        with tempfile.TemporaryDirectory() as temporary_directory:
            session_id = "e50113b750674d78af70f32b6c4df888"
            cache: dict[str, object] = {}
            uploaded = UploadedFile(b"video bytes")
            with patch.object(app, "SESSION_ROOT", Path(temporary_directory)):
                with patch.object(
                    app.hashlib, "sha256", wraps=hashlib.sha256
                ) as digest:
                    first = app._store_upload(
                        session_id, uploaded, ".mp4", "videos", cache
                    )
                    second = app._store_upload(
                        session_id, uploaded, ".mp4", "videos", cache
                    )

            self.assertEqual(first, second)
            self.assertEqual(digest.call_count, 1)
            self.assertEqual(first[1].read_bytes(), b"video bytes")

    def test_empty_upload_rerun_does_not_save_unchanged_review_state(self):
        state = {"videos": {}}
        with patch.object(app, "_save_state") as save_state:
            self.assertEqual(app._upload_videos("session-id", state, []), [])
        save_state.assert_not_called()

    def test_review_frame_image_cache_decodes_once_and_keeps_source_resolution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "frame.jpg"
            original = np.full((1080, 1920, 3), 80, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), original))
            frame = {"frame_id": "frame-1", "image_path": str(image_path)}
            cache: dict[str, object] = {}

            with patch.object(app.cv2, "imread", wraps=cv2.imread) as imread:
                first = app._review_frame_images(frame, cache)
                second = app._review_frame_images(frame, cache)

            self.assertIs(first, second)
            self.assertEqual(imread.call_count, 1)
            self.assertEqual(first["image"].shape, (1080, 1920, 3))
            self.assertEqual(first["display_bgr"].shape[:2], (405, 720))

    def test_video_preview_cache_reuses_sampled_frames(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "video.mp4"
            video_path.write_bytes(b"video")
            cache: dict[str, object] = {}
            frames = [np.zeros((1080, 1920, 3), dtype=np.uint8)]
            with patch.object(app, "preview_frames", return_value=frames) as decode:
                first = app._cached_video_previews(
                    video_path, 0, 100, 4, cache, "video-1"
                )
                second = app._cached_video_previews(
                    video_path, 0, 100, 4, cache, "video-1"
                )

            self.assertIs(first, second)
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(first[0].shape, (405, 720, 3))

    def test_review_preview_scales_boxes_without_changing_original_coordinates(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        detections = [_detection("d1", "unreviewed", (20, 20, 80, 80), "")]

        preview = app._review_image(
            image, detections, set(), coordinate_scale=(0.5, 0.5)
        )

        self.assertEqual(preview.shape, (50, 50, 3))
        self.assertTrue(np.array_equal(preview[40, 25], (140, 105, 0)))

    def test_review_display_is_larger_and_preserves_aspect_ratio(self):
        self.assertEqual(app._review_display_size(1920, 1080), (720, 405))
        self.assertEqual(app._review_display_size(640, 360), (640, 360))
        self.assertEqual(app._review_display_size(900, 1600), (360, 640))

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

    def test_number_one_selection_highlights_only_its_detection(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        detections = [
            _detection("d1", "unreviewed", (5, 40, 35, 90), ""),
            _detection("d2", "unreviewed", (60, 40, 90, 90), ""),
        ]

        preview = app._review_image(image, detections, {"d1"})

        self.assertTrue(np.array_equal(preview[70, 5], (255, 190, 0)))
        self.assertTrue(np.array_equal(preview[70, 90], (140, 105, 0)))

    def test_canvas_click_selects_the_correct_box_after_scaling(self):
        detections = [
            _detection("d1", "unreviewed", (10, 10, 80, 80), ""),
            _detection("d2", "unreviewed", (120, 20, 200, 100), ""),
            _detection("d3", "unreviewed", (240, 60, 340, 160), ""),
        ]
        point = app._canvas_point_coordinates(
            {"type": "circle", "left": 119, "top": 45, "radius": 1, "strokeWidth": 0}
        )

        self.assertEqual(point, (120, 45))
        self.assertEqual(
            app._detection_at_canvas_point(
                detections, *point, 200, 100, 400, 200
            ),
            "d3",
        )
        self.assertIsNone(
            app._detection_at_canvas_point(detections, 105, 5, 200, 100, 400, 200)
        )

    def test_batch_selection_is_separate_from_status_and_reject_targets_only_id(self):
        statuses = {f"d{number}": "unreviewed" for number in range(1, 11)}
        selected_ids = {"d4", "d5", "d7"}

        app._stage_detection_decision(statuses, {"d5"}, "false_positive")

        self.assertEqual(statuses["d5"], "false_positive")
        self.assertEqual(statuses["d4"], "unreviewed")
        self.assertEqual(statuses["d7"], "unreviewed")
        self.assertEqual(selected_ids, {"d4", "d5", "d7"})

    def test_deleting_selected_detection_removes_only_that_id(self):
        frame = {
            "annotation_complete": True,
            "detections": [
                _detection("d4", "true_positive", (1, 1, 5, 5), ""),
                _detection("d5", "false_positive", (10, 10, 15, 15), ""),
            ],
        }
        statuses = {"d4": "true_positive", "d5": "false_positive"}

        app._delete_detections(frame, {"d5"}, statuses)

        self.assertEqual(
            [detection["detection_id"] for detection in frame["detections"]], ["d4"]
        )
        self.assertEqual(statuses, {"d4": "true_positive"})
        self.assertFalse(frame["annotation_complete"])

    def test_batch_decision_for_ten_detections_is_explicit_and_editable(self):
        detections = [
            _detection(f"d{number}", "unreviewed", (number, 1, number + 1, 3), "")
            for number in range(1, 11)
        ]
        frame = {
            "frame_id": "video-a_000000001",
            "width": 200,
            "height": 100,
            "annotation_complete": False,
            "detections": detections,
        }
        drawing, stroke_map, _ = app._canvas_drawing(frame, 200, 100)
        staged = {detection["detection_id"]: detection["status"] for detection in detections}
        false_positives = {"d1", "d5", "d6", "d7", "d9"}
        app._stage_detection_decision(staged, false_positives, "false_positive")
        app._stage_detection_decision(staged, {"d1"}, "true_positive")
        self.assertEqual(staged["d1"], "true_positive")
        app._stage_detection_decision(
            staged,
            {
                detection["detection_id"]
                for detection in detections
                if staged[detection["detection_id"]] == "unreviewed"
            },
            "true_positive",
        )
        false_positives.remove("d1")

        with self.assertRaisesRegex(ValueError, "still unreviewed"):
            app._commit_frame_review(
                frame,
                {**staged, "d10": "unreviewed"},
                drawing["objects"],
                stroke_map,
                {stroke: 0 for stroke in stroke_map},
                1,
                1,
                ["bee", "fly"],
                True,
            )
        self.assertTrue(all(detection["status"] == "unreviewed" for detection in detections))

        app._commit_frame_review(
            frame,
            staged,
            drawing["objects"],
            stroke_map,
            {stroke: 0 for stroke in stroke_map},
            1,
            1,
            ["bee", "fly"],
            True,
        )
        self.assertEqual(
            [detection["status"] for detection in detections],
            [
                "false_positive"
                if detection["detection_id"] in false_positives
                else "true_positive"
                for detection in detections
            ],
        )
        self.assertTrue(frame["annotation_complete"])

    def test_save_frame_assigns_unique_ids_to_multiple_manual_boxes(self):
        frame = {
            "frame_id": "video-a_000000001",
            "width": 200,
            "height": 100,
            "annotation_complete": False,
            "detections": [
                _detection("d1", "true_positive", (5, 5, 25, 25), "#112233")
            ],
        }
        drawing, stroke_map, _ = app._canvas_drawing(frame, 200, 100)
        shapes = list(drawing["objects"])
        shapes.extend(
            [
                {
                    "type": "rect",
                    "stroke": "#aabbcc",
                    "left": 40,
                    "top": 10,
                    "width": 20,
                    "height": 15,
                    "scaleX": 1,
                    "scaleY": 1,
                    "angle": 0,
                },
                {
                    "type": "rect",
                    "stroke": "#ddeeff",
                    "left": 90,
                    "top": 20,
                    "width": 15,
                    "height": 10,
                    "scaleX": 1,
                    "scaleY": 1,
                    "angle": 0,
                },
            ]
        )

        app._commit_frame_review(
            frame,
            {"d1": "true_positive"},
            shapes,
            stroke_map,
            {"#112233": 0, "#aabbcc": 1, "#ddeeff": 0},
            1,
            1,
            ["bee", "fly"],
            True,
        )

        manual = [detection for detection in frame["detections"] if detection["manual"]]
        self.assertEqual(
            [detection["detection_id"] for detection in manual],
            ["video-a_000000001-manual001", "video-a_000000001-manual002"],
        )
        self.assertEqual([detection["class_id"] for detection in manual], [1, 0])
        self.assertTrue(all(detection["status"] == "true_positive" for detection in manual))

    def test_deleted_all_detections_can_export_only_as_a_confirmed_empty_label(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frames = []
            for video_id in ("video-a", "video-b", "video-c"):
                image_path = root / f"{video_id}.jpg"
                self.assertTrue(
                    cv2.imwrite(
                        str(image_path), np.zeros((40, 80, 3), dtype=np.uint8)
                    )
                )
                frames.append(
                    {
                        "frame_id": f"{video_id}_000000001",
                        "video_id": video_id,
                        "video_filename": f"{video_id}.mp4",
                        "frame_number": 1,
                        "timestamp_seconds": 0.1,
                        "image_path": str(image_path),
                        "width": 80,
                        "height": 40,
                        "annotation_complete": True,
                        "detections": (
                            []
                            if video_id == "video-a"
                            else [
                                _detection(
                                    f"{video_id}-bee",
                                    "true_positive",
                                    (5, 5, 20, 20),
                                    "#112233",
                                )
                            ]
                        ),
                    }
                )

            dataset_dir, split_counts = create_dataset(
                frames, ["bee", "fly"], root / "export", seed=1
            )

            self.assertEqual(sum(split_counts.values()), 3)
            empty_label = next(
                (dataset_dir / "labels").rglob("video-a_000000001.txt")
            )
            self.assertEqual(empty_label.read_text(encoding="utf-8"), "")

    def test_side_by_side_preview_preserves_original_and_frame_dimensions(self):
        image = np.zeros((120, 240, 3), dtype=np.uint8)
        original = image.copy()
        detections = [_detection("d1", "unreviewed", (40, 35, 100, 90), "")]

        preview = app._review_image(image, detections, set())

        self.assertEqual(preview.shape, image.shape)
        self.assertTrue(np.array_equal(image, original))
        self.assertGreater(np.count_nonzero(preview), 0)
        self.assertTrue(np.any(np.all(preview[:, :, :] == (255, 255, 255), axis=2)))

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
        self.assertEqual(edited["detection_id"], "video-a_000000001-d001")
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
