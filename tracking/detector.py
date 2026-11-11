"""Model adapters that return detections in a common format."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


class Detector:
    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.config = config
        self.model_type = str(config["model_type"]).upper()
        if self.model_type not in {"YOLO", "RF-DETR"}:
            raise ValueError("detection.model_type must be 'YOLO' or 'RF-DETR'.")
        self.image_size = int(config["image_size"])
        self.confidence = float(config["confidence_threshold"])
        self.iou = float(config["nms_iou_threshold"])
        self.class_names = config.get("class_names")
        if self.class_names is not None and not isinstance(self.class_names, list):
            raise ValueError("detection.class_names must be a list or null.")
        if self.image_size <= 0:
            raise ValueError("detection.image_size must be a positive integer.")
        if not 0 <= self.confidence <= 1 or not 0 <= self.iou <= 1:
            raise ValueError(
                "Detection confidence and NMS IoU thresholds must be between 0 and 1."
            )

        weights_path = Path(config["weights"]).expanduser()
        if not weights_path.is_absolute():
            weights_path = project_root / weights_path
        if not weights_path.is_file():
            raise FileNotFoundError(f"Model weights do not exist: {weights_path}")
        self.weights_path = weights_path.resolve()

        device = self._resolve_device(str(config.get("device", "auto")))
        if self.model_type == "YOLO":
            try:
                from ultralytics import YOLO

                self.model = YOLO(str(self.weights_path))
            except Exception as error:
                raise ValueError(
                    f"Could not load YOLO weights {self.weights_path}: {error}"
                ) from error
            self.device = device.split(":", 1)[1] if device.startswith("cuda:") else device
        elif self.model_type == "RF-DETR":
            try:
                from rfdetr import RFDETRLarge

                self.model = RFDETRLarge.from_checkpoint(
                    str(self.weights_path),
                    device=device,
                )
            except Exception as error:
                raise ValueError(
                    f"Could not load RF-DETR weights {self.weights_path}: {error}"
                ) from error
            self.device = device

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device.lower() == "auto":
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.isdigit():
            return f"cuda:{device}"
        if device.lower() == "cuda":
            return "cuda:0"
        if device.lower() != "cpu" and not device.lower().startswith("cuda:"):
            raise ValueError(
                "detection.device must be 'auto', 'cpu', 'cuda', 'cuda:N', or a device index."
            )
        return device.lower()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.model_type == "YOLO":
            results = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                coordinates = _as_array(boxes.xyxy)
                confidences = _as_array(boxes.conf).reshape(-1)
                class_ids = _as_array(boxes.cls).reshape(-1)
                names = getattr(result, "names", getattr(self.model, "names", {}))
                detections.extend(
                    self._make_detections(coordinates, confidences, class_ids, names)
                )
            return detections

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = self.model.predict(
            rgb_frame,
            threshold=self.confidence,
            shape=self._rfdetr_shape(),
            include_source_image=False,
        )
        if predictions is None:
            return []
        data = getattr(predictions, "data", None)
        predicted_names = data.get("class_name") if isinstance(data, dict) else None
        boxes = getattr(predictions, "xyxy", None)
        confidences = getattr(predictions, "confidence", None)
        class_ids = getattr(predictions, "class_id", None)
        if isinstance(data, dict):
            boxes = data.get("xyxy", boxes)
            confidences = data.get("confidence", confidences)
            class_ids = data.get("class_id", class_ids)
        if boxes is None:
            return []
        if confidences is None or class_ids is None:
            raise ValueError("RF-DETR predictions are missing confidence scores or class IDs.")
        return self._make_detections(
            _as_array(boxes).reshape(-1, 4),
            _as_array(confidences).reshape(-1),
            _as_array(class_ids).reshape(-1),
            {},
            predicted_names,
        )

    def _rfdetr_shape(self) -> tuple[int, int]:
        model_config = getattr(self.model, "model_config", None)
        if model_config is None:
            raise ValueError("RF-DETR checkpoint does not expose its model configuration.")
        block_size = int(model_config.patch_size) * int(
            getattr(model_config, "num_windows", 1)
        )
        side = max(block_size, (self.image_size // block_size) * block_size)
        return side, side

    def _make_detections(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        model_names: Any,
        predicted_names: Any = None,
    ) -> list[Detection]:
        names = _as_array(predicted_names).reshape(-1) if predicted_names is not None else None
        if not (len(boxes) == len(confidences) == len(class_ids)):
            raise ValueError("Model returned inconsistent box, confidence, and class counts.")

        detections = []
        for index, (box, confidence, raw_class_id) in enumerate(
            zip(boxes, confidences, class_ids)
        ):
            class_id = int(raw_class_id)
            if names is not None and index < len(names):
                class_name = str(names[index])
            elif self.class_names is not None and 0 <= class_id < len(self.class_names):
                class_name = str(self.class_names[class_id])
            elif isinstance(model_names, dict):
                class_name = str(model_names.get(class_id, f"class_{class_id}"))
            elif isinstance(model_names, (list, tuple)) and 0 <= class_id < len(model_names):
                class_name = str(model_names[class_id])
            else:
                class_name = f"class_{class_id}"
            coordinates = tuple(float(value) for value in box)
            detections.append(
                Detection(
                    xyxy=coordinates,
                    confidence=float(confidence),
                    class_id=class_id,
                    class_name=class_name,
                )
            )
        return detections
