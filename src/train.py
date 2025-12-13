import os
import logging
from typing import Dict, Tuple

import tensorflow as tf
import yaml

from utils.helpers import read_json, ensure_dir, write_json, make_path_label_dataset

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = os.path.join("models")
METRICS_DIR = os.path.join("metrics")


def load_params() -> Dict:
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="simple_cnn")
    return model


def build_mobilenetv2(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False  # freeze base for CPU-friendly training
    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_head")
    return model


def main():
    params = load_params()
    image_size = int(params.get("image_size", 32))
    batch_size = int(params.get("batch_size", 64))
    lr = float(params.get("learning_rate", 1e-3))
    epochs_cnn = int(params.get("epochs_cnn", 8))
    epochs_tl = int(params.get("epochs_tl", 5))
    seed = int(params.get("seed", 42))

    ensure_dir(MODELS_DIR)
    ensure_dir(METRICS_DIR)

    # Load manifests
    class_names = read_json(os.path.join(PROCESSED_DIR, "class_names.json"))["class_names"]
    num_classes = len(class_names)
    train_manifest = read_json(os.path.join(PROCESSED_DIR, "train.json"))
    val_manifest = read_json(os.path.join(PROCESSED_DIR, "val.json"))

    train_paths, train_labels = train_manifest["paths"], train_manifest["labels"]
    val_paths, val_labels = val_manifest["paths"], val_manifest["labels"]

    # Datasets for CNN (rescaled to [0,1])
    ds_train_cnn = make_path_label_dataset(train_paths, train_labels, image_size, batch_size,
                                           shuffle=True, preprocess="rescale", augment=True, seed=seed)
    ds_val_cnn = make_path_label_dataset(val_paths, val_labels, image_size, batch_size,
                                         shuffle=False, preprocess="rescale", augment=False, seed=seed)

    # Build and train CNN
    logging.info("Training Simple CNN (CPU-friendly)...")
    cnn = build_cnn((image_size, image_size, 3), num_classes)
    cnn.compile(optimizer=tf.keras.optimizers.Adam(lr),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"]) 
    history_cnn = cnn.fit(ds_train_cnn, validation_data=ds_val_cnn, epochs=epochs_cnn, verbose=2)

    cnn_path = os.path.join(MODELS_DIR, "cnn.keras")
    cnn.save(cnn_path)
    logging.info(f"Saved CNN model to {cnn_path}")

    # Datasets for MobileNetV2 (96x96 recommended); preprocess per model
    mobilenet_input = 96 if image_size < 64 else image_size  # bump to 96 by default
    ds_train_tl = make_path_label_dataset(train_paths, train_labels, mobilenet_input, batch_size,
                                          shuffle=True, preprocess="mobilenet_v2", augment=True, seed=seed)
    ds_val_tl = make_path_label_dataset(val_paths, val_labels, mobilenet_input, batch_size,
                                        shuffle=False, preprocess="mobilenet_v2", augment=False, seed=seed)

    logging.info("Training MobileNetV2 head (frozen base, few epochs)...")
    tl = build_mobilenetv2((mobilenet_input, mobilenet_input, 3), num_classes)
    tl.compile(optimizer=tf.keras.optimizers.Adam(lr),
               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
               metrics=["accuracy"]) 
    history_tl = tl.fit(ds_train_tl, validation_data=ds_val_tl, epochs=epochs_tl, verbose=2)

    tl_path = os.path.join(MODELS_DIR, "mobilenetv2.keras")
    tl.save(tl_path)
    logging.info(f"Saved MobileNetV2 model to {tl_path}")

    # Save train metrics (final epoch)
    train_metrics = {
        "class_names": class_names,
        "cnn": {
            "val_accuracy": float(history_cnn.history.get("val_accuracy", [0])[-1]),
            "val_loss": float(history_cnn.history.get("val_loss", [0])[-1]),
            "train_accuracy": float(history_cnn.history.get("accuracy", [0])[-1]),
            "train_loss": float(history_cnn.history.get("loss", [0])[-1]),
            "epochs": epochs_cnn,
        },
        "mobilenetv2": {
            "val_accuracy": float(history_tl.history.get("val_accuracy", [0])[-1]),
            "val_loss": float(history_tl.history.get("val_loss", [0])[-1]),
            "train_accuracy": float(history_tl.history.get("accuracy", [0])[-1]),
            "train_loss": float(history_tl.history.get("loss", [0])[-1]),
            "epochs": epochs_tl,
            "input_size": mobilenet_input,
        },
    }
    write_json(os.path.join(METRICS_DIR, "train.json"), train_metrics)
    logging.info(f"Saved training metrics to {os.path.join(METRICS_DIR, 'train.json')}")

    logging.info("Training stage complete.")


if __name__ == "__main__":
    main()
