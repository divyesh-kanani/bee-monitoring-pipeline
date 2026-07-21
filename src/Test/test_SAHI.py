from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import os

# Load YOLO model with SAHI
model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="/home/s54dkana/projects/bee-project/runs/detect/models/bee_yolov11x-2/weights/best.pt",
    confidence_threshold=0.3,
    device="cuda:0",
)

# Video path
video_path = "videos/RIMG0006.MOV"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise Exception(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Time range
start_time = 30
end_time = 60

start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Output
os.makedirs("output", exist_ok=True)

video_name = os.path.splitext(os.path.basename(video_path))[0]
output_path = f"output/{video_name}_output_sahi.mp4"

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    raise Exception("Could not open VideoWriter")

print("Processing video with SAHI...")

while cap.isOpened():

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if current_frame > end_frame:
        break

    ret, frame = cap.read()

    if not ret:
        break

    # SAHI inference
    result = get_sliced_prediction(
        frame,
        model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    annotated_frame = frame.copy()

    # Draw detections
    for obj in result.object_prediction_list:

        x1 = int(obj.bbox.minx)
        y1 = int(obj.bbox.miny)
        x2 = int(obj.bbox.maxx)
        y2 = int(obj.bbox.maxy)

        score = obj.score.value
        label = obj.category.name

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            annotated_frame,
            f"{label} {score:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    writer.write(annotated_frame)

writer.release()
cap.release()

print("Done!")
print(f"Saved to: {output_path}")