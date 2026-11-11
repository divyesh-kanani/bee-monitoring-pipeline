from types import SimpleNamespace
from typing import Any

import numpy as np

from tracking.detector import Detection, Detector
from tracking.tracker import Tracker


def _tracker() -> Tracker:
    return Tracker(
        {
            "algorithm": "ByteTrack",
            "high_detection_threshold": 0.25,
            "low_detection_threshold": 0.1,
            "track_activation_threshold": 0.25,
            "track_buffer": 3,
            "matching_threshold": 0.8,
        }
    )


def test_bytetrack_matches_low_scores_recovers_gaps_and_separates_tracks() -> None:
    tracker = _tracker()
    outputs = [
        tracker.update([Detection((10, 10, 30, 30), 0.9, 0, "bee")], (100, 100)),
        tracker.update([Detection((11, 10, 31, 30), 0.18, 0, "bee")], (100, 100)),
        tracker.update([], (100, 100)),
        tracker.update(
            [
                Detection((13, 10, 33, 30), 0.88, 0, "bee"),
                Detection((60, 10, 80, 30), 0.92, 0, "bee"),
            ],
            (100, 100),
        ),
        tracker.update(
            [
                Detection((14, 10, 34, 30), 0.9, 0, "bee"),
                Detection((61, 10, 81, 30), 0.91, 0, "bee"),
            ],
            (100, 100),
        ),
    ]

    original_id = outputs[0][0].tracker_id
    assert outputs[1][0].tracker_id == original_id
    assert outputs[3][0].tracker_id == original_id
    assert len({track.tracker_id for track in outputs[4]}) == 2


def test_rfdetr_adapter_normalizes_predictions() -> None:
    class FakeRFDETR:
        model_config = SimpleNamespace(patch_size=14, num_windows=1)

        def predict(self, image: np.ndarray, **kwargs: Any) -> SimpleNamespace:
            assert image[0, 0].tolist() == [30, 20, 10]
            assert kwargs["threshold"] == 0.1
            assert kwargs["shape"] == (952, 952)
            return SimpleNamespace(
                xyxy=np.array([[1, 2, 20, 25]], dtype=float),
                confidence=np.array([0.85]),
                class_id=np.array([0]),
                data={"class_name": np.array(["bee"])},
            )

    detector = Detector.__new__(Detector)
    detector.model_type = "RF-DETR"
    detector.model = FakeRFDETR()
    detector.confidence = 0.1
    detector.image_size = 960
    detector.class_names = None
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    frame[:] = (10, 20, 30)

    assert detector.detect(frame) == [
        Detection((1.0, 2.0, 20.0, 25.0), 0.85, 0, "bee")
    ]
