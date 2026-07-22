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

### Running Inference

Example:

```bash
python inference.py --weights models/yolov11x.pt --source demo/input.mp4
```

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

GitHub: [link](https://github.com/divyesh-kanani)

LinkedIn: [link](www.linkedin.com/in/divyesh-kanani)

---

## 📄 License

This project is released under the MIT License.
