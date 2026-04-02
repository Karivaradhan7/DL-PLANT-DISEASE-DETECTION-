import os
import random
from glob import glob
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int = 128, transform=None):
        self.root_dir = root_dir
        self.image_files = []
        self.labels = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

        for cls in self.classes:
            pattern = os.path.join(root_dir, cls, '*')
            files = glob(pattern)
            for f in files:
                if f.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_files.append(f)
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        label = self.labels[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def make_dataloaders(data_dir: str, image_size: int, batch_size: int, num_workers: int = 4, seed: int = 42):
    set_seed(seed)

    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = PlantDiseaseDataset(data_dir, image_size=image_size, transform=transform_train)

    n = len(dataset)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)
    test_size = n - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))

    # assign evaluation transform for val/test
    val_ds.dataset.transform = transform_eval
    test_ds.dataset.transform = transform_eval

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, dataset.classes
