import os
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import _T_co
from torchvision.io import ImageReadMode, decode_image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import v2
from utils.metadata_update import METADATA, DATASET_FOLDER
class DiagnosisDataset(Dataset):

    def __init__(self, split, transform = None, target_transform = None):
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        split_folder = DATASET_FOLDER/split
        for label in split_folder.iterdir():
            for sample in label.iterdir():
                if sample.is_file():
                    self.samples.append({"path":sample ,"label":label.name})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = self.samples[index]["path"]
        image = decode_image(img_path)
        label = self.samples[index]["label"]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label






def main() -> None:
    train_transform = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = v2.Compose([
        v2.Resize(size=(224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = DiagnosisDataset("train", METADATA, train_transform)

if __name__ == "__main__":
  main()