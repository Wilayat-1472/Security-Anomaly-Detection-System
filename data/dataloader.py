import os
import torch
from typing import Optional, Callable, Tuple, List
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np


class SecurityDataset(Dataset):
    """
    Surveillance dataset wrapper handling images across threat categories.
    """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculates inverse-frequency class weights for cost-sensitive loss training:
        weight[c] = total_samples / (num_classes * count[c])
        """
        targets = [label for _, label in self.dataset.samples]
        class_counts = np.bincount(targets, minlength=len(self.classes))
        total_samples = len(targets)
        weights = total_samples / (len(self.classes) * np.maximum(class_counts, 1).astype(float))
        return torch.tensor(weights, dtype=torch.float32)


def get_transforms(
    input_size: int = 224,
    augment: bool = False,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
):
    """
    Strong surveillance domain augmentations (lighting, perspective variations, noise)
    to prevent overfitting on compact training splits.
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    input_size: int = 224,
    num_workers: int = 2,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], torch.Tensor]:
    train_transform = get_transforms(input_size, augment=True, mean=mean, std=std)
    val_transform = get_transforms(input_size, augment=False, mean=mean, std=std)

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    test_path = os.path.join(data_dir, "test")

    train_dataset = SecurityDataset(train_path, transform=train_transform)
    val_dataset = SecurityDataset(val_path, transform=val_transform)
    test_dataset = SecurityDataset(test_path, transform=val_transform)

    class_weights = train_dataset.get_class_weights()

    # Drop last only if enough samples; pin_memory for performance
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, train_dataset.classes, class_weights
