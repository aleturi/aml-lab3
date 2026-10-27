import os

import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models.custom_net import CustomNet
from utils.evaluation import validate


PROJECT_ROOT = os.path.dirname(__file__)
DATASET_ROOT = os.path.join(
    PROJECT_ROOT, "downloads", "tiny-imagenet", "tiny-imagenet-200"
)
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "custom_net.pt")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tiny_imagenet_dataset_val = ImageFolder(
    root=os.path.join(DATASET_ROOT, "val"),
    transform=transform,
)
val_loader = DataLoader(
    tiny_imagenet_dataset_val,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)


if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. Run train.py first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CustomNet().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    validate(model, val_loader, nn.CrossEntropyLoss(), device)
