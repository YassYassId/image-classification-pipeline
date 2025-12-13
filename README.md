# CIFAR-10 Image Classification — CPU-Friendly DVC Pipeline

A minimal, CPU-only machine learning pipeline for CIFAR-10 image classification using:
- TensorFlow/Keras (tensorflow-cpu)
- scikit-learn
- DVC (Data Version Control)
- Docker (optional)

The pipeline has three stages: prepare → train → evaluate. It trains two lightweight models:
1) Simple CNN (2 Conv blocks) at 32×32
2) MobileNetV2 transfer learning with a small classifier head (frozen base), resized to 96×96

Everything is designed to run reasonably fast on CPU with small epochs.

## Repository Structure
```
project/
├── data/
│   ├── raw/
│   │   └── cifar10/
│   │       ├── train/<classes>/
│   │       └── test/<classes>/
│   └── processed/
├── src/
│   ├── prepare.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils/
│       └── helpers.py
├── models/
├── metrics/
│   ├── eval.json
│   ├── train.json
│   └── plots/
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── Dockerfile
├── README.md
└── report.md
```
Note: `dvc.lock` is generated automatically by DVC after the first successful `dvc repro`.

## Dataset
This project expects the CIFAR-10 dataset (from Kaggle) to be placed locally as folders of images:
```
data/raw/cifar10/
    train/<class_0>/ ... <class_9>/
    test/<class_0>/  ... <class_9>/
```
Only loading from local folders is implemented. No download code is included.

## Parameters
Adjust hyperparameters in `params.yaml`:
- `image_size`: 32 for CNN; MobileNetV2 internally uses 96 if needed
- `val_split`: fraction of training data for validation
- `batch_size`, `learning_rate`
- `epochs_cnn`, `epochs_tl`
- `model_choice`: which model to evaluate (`cnn` or `mobilenetv2`)

## How to Run (Local, CPU)
1) Create and activate a Python 3.10 environment
2) Install requirements
```
pip install -r requirements.txt
```
3) Make sure the dataset folders exist under `data/raw/cifar10/`
4) Run the full pipeline with DVC:
```
dvc repro
```
This executes:
- `src/prepare.py` → writes JSON manifests to `data/processed/`
- `src/train.py`   → trains two models under `models/`, saves metrics to `metrics/train.json`
- `src/evaluate.py`→ evaluates the chosen `params.yaml:model_choice`, writes `metrics/eval.json` and plots

Re-run after changing `params.yaml` or dataset; DVC will skip unchanged steps.

## How to Choose the Model for Evaluation
Set the parameter in `params.yaml`:
```
model_choice: "cnn"         # or "mobilenetv2"
```
Then re-run:
```
dvc repro
```

## Outputs
- Models: `models/cnn.keras`, `models/mobilenetv2.keras`
- Training metrics: `metrics/train.json` (final-epoch train/val accuracy & loss)
- Evaluation metrics: `metrics/eval.json` (accuracy, macro-precision/recall/F1)
- Plots: `metrics/plots/confusion_matrix_<model>.png`, `metrics/plots/misclassified_<model>.png`

## CPU-Friendly Notes
- Small CNN: 2 Conv layers (32, 64), small dense head, modest dropout
- MobileNetV2: base frozen, only classifier trained for a few epochs, 96×96 inputs
- Small batch sizes and few epochs by default

## Docker Usage (CPU-Only)
Build the image:
```
docker build -t cifar10-dvc .
```
Run the pipeline, mounting your local `data/` folder into the container:
- On Linux/macOS:
```
docker run --rm -v $(pwd)/data:/app/data cifar10-dvc dvc repro
```
- On Windows PowerShell:
```
docker run --rm -v ${PWD}\data:/app/data cifar10-dvc dvc repro
```

## Troubleshooting
- Ensure `tensorflow-cpu` is installed, not the GPU build.
- Confirm the dataset directory structure and class names are consistent between `train/` and `test/`.
- If you get out-of-memory on CPU, try reducing `batch_size` and epochs in `params.yaml`.

## License
This project is provided as-is for educational purposes.
