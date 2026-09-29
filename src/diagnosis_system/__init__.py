import os
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import _T_co
from torchvision.io import ImageReadMode, decode_image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import v2

from utils.early_stopping import EarlyStopping
from utils.metadata_update import METADATA, DATASET_FOLDER, PROJECT_DIR
PROCESSED_DIR = PROJECT_DIR / "processed"
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
EPOCHS = 200
class DiagnosisDataset(Dataset):

    def __init__(self, split, transform = None, target_transform = None):
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        # Class indices come from train/ so they match across splits (test/ is missing a class)
        self.classes = sorted(d.name for d in (DATASET_FOLDER/"train").iterdir() if d.is_dir())
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        split_folder = DATASET_FOLDER/split
        for label in split_folder.iterdir():
            for sample in label.iterdir():
                if sample.is_file():
                    self.samples.append({"path":sample ,"label":self.class_to_idx[label.name]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = self.samples[index]["path"]
        image = decode_image(str(img_path), mode=ImageReadMode.RGB)
        label = self.samples[index]["label"]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def build_model(num_classes, device):
    model_path = PROCESSED_DIR / "model.pt"
    if model_path.exists():
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        trained_epochs = checkpoint.get("epoch", 0)
        print(f"Loaded model from {model_path} (trained for {trained_epochs} epochs so far)")
    else:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1, progress=True)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
        trained_epochs = 0
    # Backbone frozen; only the classifier's two Linear layers train
    for p in model.parameters():
        p.requires_grad = False
    model.classifier[0].requires_grad_(True)
    model.classifier[3].requires_grad_(True)
    return model.to(device), trained_epochs

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    model.features.eval()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_value, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")


def val_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return val_loss



def main() -> None:
    print("PyTorch:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(METADATA)
    num_classes = metadata.shape[0]

    # Crop scale floored at 0.4 so the crop keeps enough of the plant to be identifiable
    train_transform = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), scale=(0.4, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Matches the IMAGENET1K_V1 eval preprocessing: short side to 256, then center crop
    val_transform = v2.Compose([
        v2.Resize(size=256, antialias=True),
        v2.CenterCrop(size=224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = DiagnosisDataset("train", transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_data = DiagnosisDataset("test", transform=val_transform)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    model, trained_epochs = build_model(num_classes, device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping(6, 0.001)
    total_epochs = 0
    print(f"Let's run {EPOCHS} epochs")
    for epoch in range(trained_epochs + 1, trained_epochs + EPOCHS + 1):
        print(f"Epoch {epoch}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        early_stopping(val_loop(val_dataloader, model, loss_fn, device), model)
        if early_stopping.early_stop:
            total_epochs = epoch
            print("Early stop to avoid overfitting")
            break
        total_epochs = trained_epochs + EPOCHS
    early_stopping.load_best_model(model)
    print(f"Done! Total epochs trained: {total_epochs}")
    model_path = PROCESSED_DIR / "model.pt"
    torch.save({
        "epoch": total_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, model_path)
    print("Saved model to", model_path)
if __name__ == "__main__":
  main()