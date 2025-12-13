import os
import logging
from typing import Dict, List

import tensorflow as tf
from sklearn.model_selection import train_test_split
import yaml

from utils.helpers import ensure_dir, write_json, count_files_by_class

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

RAW_DIR = os.path.join("data", "raw", "cifar10")
PROCESSED_DIR = os.path.join("data", "processed")


def load_params() -> Dict:
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    image_size = int(params.get("image_size", 32))
    val_split = float(params.get("val_split", 0.1))
    seed = int(params.get("seed", 42))
    augment_in_prepare = bool(params.get("augment_in_prepare", False))

    if not os.path.isdir(RAW_DIR):
        raise FileNotFoundError(
            f"Expected raw dataset at {RAW_DIR}. Please place Kaggle CIFAR-10 folders as data/raw/cifar10/train/<classes>/ and data/raw/cifar10/test/<classes>/"
        )

    train_dir = os.path.join(RAW_DIR, "train")
    test_dir = os.path.join(RAW_DIR, "test")

    logging.info("Counting files per class in train and test...")
    logging.info(f"Train counts: {count_files_by_class(train_dir)}")
    logging.info(f"Test counts:  {count_files_by_class(test_dir)}")

    # Use Keras utility to infer class names and verify structure
    logging.info("Validating and loading file paths via image_dataset_from_directory (no caching, CPU-friendly)...")
    tmp_ds = tf.keras.utils.image_dataset_from_directory(
        directory=train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        shuffle=True,
        seed=seed,
        batch_size=32,  # small tmp batch just to trigger parsing and get class_names
        validation_split=None,
    )
    class_names: List[str] = list(tmp_ds.class_names)
    num_classes = len(class_names)
    logging.info(f"Detected classes ({num_classes}): {class_names}")

    # Build file path lists for train/val
    train_files = []
    train_labels = []
    for idx, cls in enumerate(class_names):
        class_dir = os.path.join(train_dir, cls)
        for base, _, files in os.walk(class_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    train_files.append(os.path.join(base, f))
                    train_labels.append(idx)

    # Split into train/val
    train_paths, val_paths, train_y, val_y = train_test_split(
        train_files, train_labels, test_size=val_split, random_state=seed, stratify=train_labels
    )

    # Build file path lists for test
    test_paths = []
    test_y = []
    for idx, cls in enumerate(class_names):
        class_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        for base, _, files in os.walk(class_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    test_paths.append(os.path.join(base, f))
                    test_y.append(idx)

    # Save processed manifests
    ensure_dir(PROCESSED_DIR)
    write_json(os.path.join(PROCESSED_DIR, "class_names.json"), {"class_names": class_names})

    write_json(os.path.join(PROCESSED_DIR, "train.json"), {"paths": train_paths, "labels": train_y})
    write_json(os.path.join(PROCESSED_DIR, "val.json"), {"paths": val_paths, "labels": val_y})
    write_json(os.path.join(PROCESSED_DIR, "test.json"), {"paths": test_paths, "labels": test_y})

    # Prepare a tiny preview tf.data using the same loader to ensure shapes (not saved)
    _ = tf.keras.utils.image_dataset_from_directory(
        directory=train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        shuffle=True,
        seed=seed,
        batch_size=8,
    )

    logging.info(f"Saved processed manifests to {PROCESSED_DIR}")
    logging.info("Preparation stage complete.")


if __name__ == "__main__":
    main()
