import os
import logging
from typing import Dict, List

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import yaml

from utils.helpers import read_json, ensure_dir, make_path_label_dataset

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = os.path.join("models")
METRICS_DIR = os.path.join("metrics")
PLOTS_DIR = os.path.join(METRICS_DIR, "plots")


def load_params() -> Dict:
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True label", xlabel="Predicted label", title="Confusion Matrix")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Labeling the cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    ensure_dir(os.path.dirname(path))
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_misclassified_grid(images: np.ndarray, true_labels: np.ndarray, pred_labels: np.ndarray,
                             class_names: List[str], path: str, max_samples: int = 16) -> None:
    wrong_idx = np.where(true_labels != pred_labels)[0]
    if wrong_idx.size == 0:
        # nothing to plot
        return
    sel = wrong_idx[:max_samples]
    rows = int(np.ceil(len(sel) / 4))
    cols = min(4, len(sel))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(rows, cols)
    for i, idx in enumerate(sel):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        img = images[idx]
        # If image is preprocessed (e.g., MobileNetV2), attempt simple rescale for display
        if img.min() < 0 or img.max() > 1.5:
            img_disp = (img - img.min()) / (img.max() - img.min() + 1e-7)
        else:
            img_disp = img
        ax.imshow(np.clip(img_disp, 0, 1))
        ax.axis('off')
        ax.set_title(f"T: {class_names[true_labels[idx]]}\nP: {class_names[pred_labels[idx]]}")

    # Hide any unused axes
    for j in range(len(sel), rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis('off')

    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    params = load_params()
    model_choice = str(params.get("model_choice", "cnn")).lower()
    image_size = int(params.get("image_size", 32))
    batch_size = int(params.get("batch_size", 64))

    ensure_dir(METRICS_DIR)
    ensure_dir(PLOTS_DIR)

    # Load manifests
    class_names = read_json(os.path.join(PROCESSED_DIR, "class_names.json"))["class_names"]
    num_classes = len(class_names)
    test_manifest = read_json(os.path.join(PROCESSED_DIR, "test.json"))
    test_paths, test_labels = test_manifest["paths"], test_manifest["labels"]

    # Choose model & preprocessing
    if model_choice == "mobilenetv2":
        model_path = os.path.join(MODELS_DIR, "mobilenetv2.keras")
        input_size = 96 if image_size < 64 else image_size
        preprocess = "mobilenet_v2"
    else:
        model_path = os.path.join(MODELS_DIR, "cnn.keras")
        input_size = image_size
        preprocess = "rescale"

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train stage must run before evaluate.")

    logging.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Build dataset (no augmentation)
    ds_test = make_path_label_dataset(test_paths, test_labels, input_size, batch_size,
                                      shuffle=False, preprocess=preprocess, augment=False)

    # Collect predictions and ground truth
    y_true = []
    y_pred = []
    sample_images = []  # for plotting some misclassifications

    for batch_imgs, batch_labels in ds_test:
        preds = model.predict(batch_imgs, verbose=0)
        preds_cls = np.argmax(preds, axis=1)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(preds_cls.tolist())
        # collect some images for plotting
        if len(sample_images) < 64:  # cap memory use
            sample_images.extend(batch_imgs.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_path = os.path.join(PLOTS_DIR, f"confusion_matrix_{model_choice}.png")
    plot_confusion_matrix(cm, class_names, cm_path)

    # Misclassified grid
    wrong_path = os.path.join(PLOTS_DIR, f"misclassified_{model_choice}.png")
    if len(sample_images) > 0:
        save_misclassified_grid(np.array(sample_images), y_true[:len(sample_images)], y_pred[:len(sample_images)],
                                class_names, wrong_path, max_samples=16)

    # Save eval metrics
    eval_metrics = {
        "model": model_choice,
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "confusion_matrix_png": os.path.relpath(cm_path, start="."),
        "misclassified_png": os.path.relpath(wrong_path, start="."),
    }
    ensure_dir(METRICS_DIR)
    from utils.helpers import write_json
    write_json(os.path.join(METRICS_DIR, "eval.json"), eval_metrics)

    logging.info(f"Saved evaluation metrics to {os.path.join(METRICS_DIR, 'eval.json')}")
    logging.info("Evaluation stage complete.")


if __name__ == "__main__":
    main()
