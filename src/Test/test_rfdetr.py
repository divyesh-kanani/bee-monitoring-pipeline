from rfdetr import RFDETRLarge
import cv2
import os

# Load fine-tuned checkpoint
model = RFDETRLarge.from_checkpoint(
    "/home/s54dkana/projects/bee-project/runs/rfdetr_large/checkpoint_best_total.pth"
)

video_path = "videos/RIMG0006.MOV"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs("output", exist_ok=True)

writer = cv2.VideoWriter(
    "output/rfdetr_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detections = model.predict(
        rgb,
        threshold=0.4,
        shape=(704, 704),   # same resolution you trained with
    )

    for box, score, cls in zip(
        detections.xyxy,
        detections.confidence,
        detections.data["class_name"],
    ):
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            frame,
            f"{cls} {score:.2f}",
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2,
        )

    writer.write(frame)

writer.release()
cap.release()

print("Done")