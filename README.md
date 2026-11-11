# 🐝 Bee Monitoring Pipeline for Environmental Observation

A deep learning-based computer vision pipeline for automatic pollinator detection and monitoring from field videos. This project aims to support biodiversity research and environmental observation by detecting and classifying pollinators using state-of-the-art object detection models.

---

## 📌 Project Overview

Monitoring pollinators such as bees is essential for understanding ecosystem health and biodiversity. Manual observation is time-consuming and difficult to scale. This project develops an automated monitoring pipeline capable of detecting pollinators from field recordings using modern deep learning techniques.

The project was developed as part of the **MORO Project** at the **University of Bonn**.

---

## 🎯 Objectives

- Build an end-to-end bee monitoring pipeline
- Train and compare multiple object detection models
- Improve detection accuracy using Hard Negative Mining
- Evaluate model performance using standard detection metrics
- Prepare the pipeline for future deployment on edge devices

---

## 🏗️ Pipeline

```
Field Videos
      │
      ▼
Frame Extraction
      │
      ▼
Dataset Annotation
      │
      ▼
Model Training
      │
      ▼
Object Detection
      │
      ▼
Performance Evaluation
      │
      ▼
Monitoring & Analysis
```

---

## 📂 Dataset

The dataset consists of field recordings containing pollinating insects collected under real-world environmental conditions.

### Classes

- Bee
- Bumblebee
- Hoverfly
- Other Insect
- Moth

The dataset was manually annotated in YOLO format and further improved using Hard Negative Mining to reduce false positive detections.

---
## 🤖 Models

The trained model weights are not included in this repository because they exceed GitHub's file size limit.

You can download the pretrained weights from Google Drive.

| Model | Download |
|-------|----------|
| YOLO11x | https://drive.google.com/file/d/1xV4-iaVbs1vrGKcq5Fk-BA2mhyGmiUT8/view?usp=sharing |
| YOLO26x | https://drive.google.com/file/d/1guemxRUqqwk9p6cDBaxCtWghELpJboAN/view?usp=sharing |
| RF-DETR | https://drive.google.com/file/d/1wIGSBCD4cjMfEY9hkn5lLV-aDT988RE1/view?usp=sharing |

After downloading, place the weights inside the `models/` directory.

```
models/
├── yolov11x.pt
├── yolov26x.pt
└── rfdetr_best.pth
```

---

### Quick start

```bash
cd "/media/data40/vs code/CLG/MORO research/bee-monitoring-pipeline"
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
streamlit run src/hard_negative_app.py
```

Open:

```text
http://127.0.0.1:8501
```

Choose **Detection** for normal inference or **Hard-negative mining** for annotation
review. Upload YOLO `.pt` weights or RF-DETR `.pth` weights, then select your videos.
Detection exports a browser-compatible H.264 MP4 and a prediction CSV.

To include tracking, enable **Apply object tracking after detection** in
Detection mode, select **ByteTrack**, and adjust its thresholds and track buffer
under **Tracking options**. Start **detection and tracking** to view and download
the tracked MP4, a detection CSV with track IDs, and a tracking summary.
Tracking processes every frame in the selected time range; detection-only frame
skipping is disabled to keep track IDs consistent. Track IDs are short-term
labels, not biological bee identities.

The app runs locally on `127.0.0.1` and stores session data under
the operating system's temporary directory, isolated by session. For HPC, use an
SSH tunnel instead of exposing it publicly.

### Interactive hard-negative review app

Upload your trained `.pt` model and one or more videos. For each video, set the
start/end times and frames to skip (0 processes every frame; 29 processes every
30th frame). The app shows video metadata, a sample-count estimate, and preview
frames before inference. Confidence, IoU, and image size are shared detection
settings.

Review each detection individually. Draw rectangles directly on the frame, choose
a class for every box, then switch to select mode to move or resize them. Delete
incorrect boxes, or draw additional boxes for missed insects. **Save boxes**
stores edits for the current session. Then keep, reject, or mark detections
uncertain, and confirm that every target insect is annotated. Unreviewed or
uncertain frames are excluded from export; rejected boxes are omitted.

Download the review CSV at any time. Dataset export creates a ready-to-use YOLO
ZIP containing `images/{train,val,test}`, `labels/{train,val,test}`, and
`data.yaml`; an optional separate ZIP contains annotated preview images. All
frames from one source video stay in one split. At least three videos with
complete, exportable frames are required to populate train, validation, and test.
The app never modifies the original dataset or starts training.

Uploads, extracted frames, annotations, and exports are stored in a unique
temporary directory per session. A browser refresh or new session removes that
session's files; dataset and preview ZIPs are deleted after their download is
prepared. CSV reports are generated in memory. Disconnected sessions are removed
by the periodic cleaner after the configured inactivity lifetime (default 24
hours; set `BEE_MINING_SESSION_MAX_AGE_HOURS` to change it). Cleanup runs on app
startup and every 10 minutes while the app process is running. Streamlit cannot
reliably report browser tab closure or confirm when a download has finished, so
cleanup uses download initiation and inactivity timeouts. Set
`BEE_MINING_TMPDIR` to choose a private temporary location on shared/HPC storage.
OpenCV uses the video-reported FPS for sampling and timestamps; variable-frame-rate
video timestamps may therefore be approximate.

### Command-line inference (optional)

The original command-line workflow is also available:

The inference command accepts one supported video or a folder of videos. It samples
frames at a configurable interval, skips near-identical sampled frames, and saves
only frames with detections. Adjust the confidence, IoU, image-size, and sampling
settings as needed:

```bash
python src/hard_negative_mining.py infer \
  --weights /path/to/your/trained.pt \
  --source videos \
  --output hard_negative_output \
  --conf 0.25 --iou 0.7 --imgsz 960 --frame-interval 2
```

CLI inference creates `review.csv`, `predictions.csv`, and frames grouped by
video/frame. The CLI's frame-level review/export commands are separate from the
interactive app's individual-detection review workflow.

Create a separate reviewed-frame dataset using the class names and IDs from the
existing YOLO configuration (`config/YOLOruns.yaml`):

```bash
python src/hard_negative_mining.py prepare \
  --review-csv hard_negative_output/review.csv \
  --data config/YOLOruns.yaml \
  --output hard_negative_dataset
```

The generated dataset contains `images/{train,val,test}`, matching
`labels/{train,val,test}`, and a ready-to-use `data.yaml`. At least three reviewed
videos are required; all frames from a video are assigned to the same split.
The output is an additive dataset of reviewed frames; the original dataset and
its annotations are not copied or modified. Outputs are not overwritten, so use
a new output directory for each run. Training is not started automatically.

---

### Notes

- The Google Drive links contain the final trained models.
- The repository contains all training and inference scripts required to reproduce the results.
- If you train the models yourself, simply replace the downloaded weights with your own.

## 🤖 Models Evaluated

- YOLO11x
- YOLO26x
- RF-DETR

The models were compared based on:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Inference Performance

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Ultralytics YOLO
- RF-DETR
- OpenCV
- NumPy
- WandB
- Albumentations

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/bee-monitoring-pipeline.git
cd bee-monitoring-pipeline
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Linux/macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Training

Example for YOLO

```bash
python train.py
```

Example for RF-DETR

```bash
python train_rfdetr.py
```

---

## 📈 Evaluation Metrics

The models are evaluated using:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Confusion Matrix
- Precision-Recall Curve

---

## 📸 Example Results

(Add screenshots here)

| Ground Truth | Prediction |
|--------------|------------|
| Image | Image |

---

## 📁 Repository Structure

```
bee-monitoring-pipeline
│
├── configs
├── datasets
├── demo
├── docs
├── models
├── notebooks
├── results
├── scripts
├── src
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🔬 Future Work

- Multi-object tracking
- Edge deployment (Jetson Orin Nano)
- Real-time inference
- Monitoring dashboard
- Pollinator behaviour analysis
- Species tracking across time

---

## 📚 References

- Bjerge et al., *Real-time Insect Tracking and Monitoring with Computer Vision and Deep Learning* (2022)
- Ratnayake et al., *Spatial Monitoring and Insect Behavioural Analysis Using Computer Vision for Precision Pollination* (2022)
- Chong et al., *A Computer Vision Dataset for Pollinator Detection under Real Field Conditions* (2025)

---

## 👨‍💻 Author

**Divyesh Kanani**

Master's Student – Mobile Robotics  
University of Bonn

GitHub: <a href="https://github.com/divyesh-kanani" target="_blank" rel="noopener noreferrer">link</a>

LinkedIn: <a href="https://www.linkedin.com/in/divyesh-kanani" target="_blank" rel="noopener noreferrer">link</a>

---

## 📄 License

This project is released under the MIT License.
