import os
import json
from typing import List, Tuple, Dict

import tensorflow as tf


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(path: str, lines: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def write_json(path: str, data: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_files_by_class(root_dir: str) -> Dict[str, int]:
    counts = {}
    if not os.path.isdir(root_dir):
        return counts
    for cls in sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))):
        class_dir = os.path.join(root_dir, cls)
        n = 0
        for base, _, files in os.walk(class_dir):
            n += sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg", ".png")))
        counts[cls] = n
    return counts


def list_image_files_with_labels(root_dir: str, class_names: List[str]) -> Tuple[List[str], List[int]]:
    paths: List[str] = []
    labels: List[int] = []
    for idx, cls in enumerate(class_names):
        class_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(class_dir):
            # skip silently if class folder missing in split
            continue
        for base, _, files in os.walk(class_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(base, f))
                    labels.append(idx)
    return paths, labels


def decode_and_resize(img_path: tf.Tensor, img_size: int, preprocess: str = "rescale") -> tf.Tensor:
    # img_path is a scalar string tensor
    img = tf.io.read_file(img_path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, (img_size, img_size), antialias=True)
    if preprocess == "mobilenet_v2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img = preprocess_input(img * 255.0)  # function expects [0,255] range
    # else: already [0,1] scaled
    return img


def make_path_label_dataset(paths: List[str], labels: List[int],
                            img_size: int,
                            batch_size: int,
                            shuffle: bool,
                            preprocess: str = "rescale",
                            augment: bool = False,
                            seed: int = 42) -> tf.data.Dataset:
    import numpy as np
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    def _map(path, label):
        img = decode_and_resize(path, img_size, preprocess)
        return img, tf.cast(label, tf.int32)

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
