import os

import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models.custom_net import CustomNet
from utils.evaluation import validate


DATASET_ROOT = os.path.join(
    os.path.dirname(__file__),
    "downloads",
    "tiny-imagenet",
    "tiny-imagenet-200",
)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tiny_imagenet_dataset_train = ImageFolder(
    root=os.path.join(DATASET_ROOT, "train"),
    transform=transform,
)
tiny_imagenet_dataset_val = ImageFolder(
    root=os.path.join(DATASET_ROOT, "val"),
    transform=transform,
)

print(f"Length of train dataset: {len(tiny_imagenet_dataset_train)}")
print(f"Length of val dataset: {len(tiny_imagenet_dataset_val)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 2 if device == "cuda" else 0

train_loader = DataLoader(
    tiny_imagenet_dataset_train,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
)
val_loader = DataLoader(
    tiny_imagenet_dataset_val,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
)


def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100.0 * correct / total
    print(f"Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%")


if __name__ == "__main__":
    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_acc = 0
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer)
        val_accuracy = validate(model, val_loader, criterion, device)

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            os.makedirs(
                os.path.join(os.path.dirname(__file__), "checkpoints"),
                exist_ok=True,
            )
            torch.save(
                model.state_dict(),
                os.path.join(
                    os.path.dirname(__file__),
                    "checkpoints",
                    "custom_net.pt",
                ),
            )

    print(f"Best validation accuracy: {best_acc:.2f}%")
