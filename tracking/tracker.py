"""ByteTrack adapter for the detector's model-independent detections."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from tracking.detector import Detection


@dataclass(frozen=True)
class TrackedDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    tracker_id: int
    detection_index: int


class Tracker:
    def __init__(self, config: dict[str, Any]) -> None:
        algorithm = str(config.get("algorithm", "")).lower()
        if algorithm != "bytetrack":
            raise ValueError("tracking.algorithm must be 'ByteTrack'.")

        high = float(config["high_detection_threshold"])
        low = float(config["low_detection_threshold"])
        activation = float(config["track_activation_threshold"])
        matching = float(config["matching_threshold"])
        buffer = int(config["track_buffer"])
        if not all(0 <= value <= 1 for value in (high, low, activation, matching)):
            raise ValueError("ByteTrack thresholds must be between 0 and 1.")
        if low >= high:
            raise ValueError(
                "tracking.low_detection_threshold must be lower than the high threshold."
            )
        if activation < high:
            raise ValueError(
                "tracking.track_activation_threshold must be at least the high detection threshold."
            )
        if buffer <= 0:
            raise ValueError("tracking.track_buffer must be a positive integer.")

        self.tracker = BYTETracker(
            SimpleNamespace(
                track_high_thresh=high,
                track_low_thresh=low,
                new_track_thresh=activation,
                track_buffer=buffer,
                match_thresh=matching,
                # Score fusion can reject low-confidence detections in ByteTrack's second pass.
                fuse_score=False,
            )
        )

    def update(
        self, detections: list[Detection], frame_shape: tuple[int, int]
    ) -> list[TrackedDetection]:
        boxes = np.empty((len(detections), 6), dtype=np.float32)
        for index, detection in enumerate(detections):
            boxes[index] = (*detection.xyxy, detection.confidence, detection.class_id)

        tracked = self.tracker.update(Boxes(boxes, frame_shape))
        if len(tracked) == 0:
            return []
        if tracked.ndim != 2 or tracked.shape[1] < 8:
            raise RuntimeError("ByteTrack returned an unexpected detection format.")

        output = []
        for row in tracked:
            detection_index = int(row[7])
            if not 0 <= detection_index < len(detections):
                raise RuntimeError("ByteTrack returned an invalid source detection index.")
            detection = detections[detection_index]
            output.append(
                TrackedDetection(
                    xyxy=tuple(float(value) for value in row[:4]),
                    confidence=float(row[5]),
                    class_id=int(row[6]),
                    class_name=detection.class_name,
                    tracker_id=int(row[4]),
                    detection_index=detection_index,
                )
            )
        return output
