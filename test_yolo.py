from ultralytics import YOLO
import cv2
import os

# Load trained model  26x
model = YOLO("/home/s54dkana/projects/bee-project/runs/detect/models/bee_yolov26x/weights/best.pt")

# Input video
video_path = "videos/RIMG0006.MOV"

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise Exception(f"Could not open video: {video_path}")

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Time range (seconds)
start_time = 0
end_time = 12000

start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Output folder
os.makedirs("output", exist_ok=True)

video_name = os.path.splitext(os.path.basename(video_path))[0]
output_path = f"output/{video_name}_output.mp4"

# Video writer
writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    raise Exception("Could not open VideoWriter")

# Process every frame
frame_skip = 1
frame_count = 0

print("Processing video...")

while cap.isOpened():

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if current_frame > end_frame:
        break

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip != 0:
        continue

    # Run inference
    results = model(
        frame,
        imgsz=1536,
        conf=0.75,
        verbose=False,
        augment=True
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # Save frame
    writer.write(annotated_frame)

# Cleanup
writer.release()
cap.release()

print(f"\nDone!")
print(f"Saved output to: {output_path}")