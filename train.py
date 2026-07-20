"""
module load Python/3.12.3-GCCcore-13.3.0
Python/3.12.3-GCCcore-13.3.0
cd ~/projects/bee-project
source venv/bin/activate


python training/train.py
"""


import wandb
from ultralytics import YOLO # type: ignore

def main():
    # wandb.init(project="bee_detection", name="yoloV2_training")
    # load model (pretrained)
    model = YOLO("./models/yolo26x.pt")
    print("every things is ok")

    # train the model
    model.train(
        data="/home/s54dkana/bee_dataset/yolo/data.yaml",   # path to your dataset config
        epochs=65,
        imgsz=960,
        batch=8,
        cache=True,
        project="./models",
        name="bee_yolov26x",
        plots=True
    )

if __name__ == "__main__":
    main()