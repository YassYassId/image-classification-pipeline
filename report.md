# Report — CIFAR-10 Image Classification (CPU-Friendly DVC Pipeline)

## Introduction
This report documents a simple, CPU-only machine learning pipeline for classifying images from the CIFAR-10 dataset. The project emphasizes clarity, reproducibility, and fast iteration on machines without a GPU. It uses TensorFlow/Keras for modeling, scikit-learn for metrics/utilities, DVC for pipeline orchestration and tracking of artifacts, and Docker for optional containerized execution.

## Objectives
- Build a complete end-to-end pipeline: prepare → train → evaluate.
- Keep the approach CPU-friendly: lightweight models, small input sizes, few epochs, modest batch sizes.
- Provide clear configuration (`params.yaml`), scripts (`src/*.py`), and artifacts (models, metrics, plots).
- Ensure the pipeline is reproducible with `dvc repro` and portable via Docker.

## Methodology
### Data Preparation
- Input: CIFAR-10 stored as class folders under `data/raw/cifar10/train/` and `data/raw/cifar10/test/`.
- The `prepare` stage validates directory structure and infers class names using `tf.keras.utils.image_dataset_from_directory`.
- It enumerates image file paths and stratifies the training set into train/validation splits using scikit-learn.
- Outputs are lightweight JSON manifests in `data/processed/`:
  - `class_names.json` — ordered class names.
  - `train.json`, `val.json`, `test.json` — lists of image paths with integer labels.

### Models and Training
Two CPU-friendly models are trained:
1. Simple CNN
   - Architecture: `Conv2D(32)` → `MaxPool` → `Conv2D(64)` → `MaxPool` → `Flatten` → `Dense(128)` → `Dropout(0.3)` → `Dense(num_classes, softmax)`.
   - Input size: 32×32.
   - Optimizer: Adam. Loss: Sparse Categorical Crossentropy.
   - Augmentation: lightweight random flip/rotation in input pipeline.

2. MobileNetV2 Transfer Learning
   - Base: `MobileNetV2(include_top=False, weights='imagenet', pooling='avg')`, frozen.
   - Classifier: Dropout(0.2) + Dense(num_classes, softmax).
   - Input size: 96×96 (auto-bumped from smaller `image_size` param).
   - Train only the head for a few epochs to stay CPU-friendly.

Training produces:
- `models/cnn.keras`
- `models/mobilenetv2.keras`
- Final-epoch metrics captured in `metrics/train.json`.

### Evaluation
- Controlled by `params.yaml:model_choice` (`cnn` or `mobilenetv2`).
- Builds a test dataset with appropriate preprocessing.
- Computes accuracy, macro-precision, macro-recall, macro-F1.
- Generates a confusion matrix and a small grid of misclassified samples.
- Saves metrics to `metrics/eval.json` and figures under `metrics/plots/`.

## Results (Example Expectations)
- CNN (few epochs on CPU) typically reaches modest accuracy given short training (e.g., 55–65%).
- MobileNetV2 head with frozen base may outperform the small CNN even with few epochs, due to ImageNet features.
- Actual results depend on number of epochs, batch size, and the exact CIFAR-10 split.

## Conclusion
This pipeline demonstrates a clear, reproducible CPU-only approach to image classification on CIFAR-10 with two lightweight models. It is intentionally simple and easy to extend:
- Adjust hyperparameters in `params.yaml`.
- Add more augmentations or regularization as needed.
- Swap the evaluation `model_choice` and re-run `dvc repro`.

The combination of DVC orchestration and Docker support ensures portability and repeatability across environments.
