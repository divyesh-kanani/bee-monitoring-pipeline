"""

python training/rfdetr_train.py

"""

from rfdetr import RFDETRLarge
import wandb

DATASET_DIR = "/home/s54dkana/bee_dataset/coco/BuzzSetV2_split"

model = RFDETRLarge()

print("every things is ok")

model.train(
    dataset_dir=DATASET_DIR,
    epochs=100,
    batch_size=16,
    grad_accum_steps=1,
    lr=1e-4,
    output_dir="./runs/rfdetr_large",

    wandb=True,
    project="bee_detection",
    run="rfdetr_large_v1",
)