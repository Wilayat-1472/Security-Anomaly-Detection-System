import os
import shutil
import random
from typing import Tuple, List
from tqdm import tqdm


def split_dataset(
    data_dir: str,
    output_dir: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
):
    random.seed(seed)
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    splits = {"train": train_split, "val": val_split, "test": test_split}

    for split_name in splits:
        os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_split)
        n_val = int(n * val_split)

        train_imgs = images[:n_train]
        val_imgs = images[n_train : n_train + n_val]
        test_imgs = images[n_train + n_val :]

        for split_name, img_list in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
            dest_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for img in tqdm(img_list, desc=f"{cls} -> {split_name}"):
                src = os.path.join(cls_dir, img)
                dst = os.path.join(dest_dir, img)
                shutil.copy2(src, dst)

    print(f"Dataset split complete. Train: {sum(len(os.listdir(os.path.join(output_dir, 'train', c))) for c in classes)}, "
          f"Val: {sum(len(os.listdir(os.path.join(output_dir, 'val', c))) for c in classes)}, "
          f"Test: {sum(len(os.listdir(os.path.join(output_dir, 'test', c))) for c in classes)}")


def count_images(data_dir: str) -> dict:
    counts = {}
    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len([f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))])
    return counts


def verify_dataset(data_dir: str):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Found {len(classes)} classes: {classes}")
    total = 0
    for cls in classes:
        count = len([f for f in os.listdir(os.path.join(data_dir, cls)) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))])
        total += count
        print(f"  {cls}: {count} images")
    print(f"Total: {total} images")
