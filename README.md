# EGCA-Net: Frequency and Spatial-Coherence Enhancement for SAR Ship Detection

This repository contains the public, minimal implementation of **EGCA-Net**, a YOLOv8-based detector for SAR ship detection. It adds two components to YOLOv8:

- **ESFF**: entropy-inspired frequency-statistics fusion. A local low/high-frequency split is combined with regional high-frequency mean and standard-deviation gating.
- **DS-SCR**: dual-scale spatial-coherence rectification. A large-kernel depthwise context path and a local smoothing path are fused with parameter-free SimAM refinement.

The repository deliberately excludes datasets, trained weights, private experiment logs, and manuscript material.

## Environment

Tested with Python 3.10, PyTorch 2.7, CUDA 12.8, and Ultralytics 8.3.115.

```bash
pip install -r requirements.txt
git clone --branch v8.3.115 https://github.com/ultralytics/ultralytics.git third_party/ultralytics
python scripts/install_ultralytics_patch.py --ultralytics-root third_party/ultralytics
cd third_party/ultralytics
pip install -e .
```

The patch creates `ultralytics/nn/tasks.py.egca-net.bak` before changing the parser.

## Dataset

Convert SSDD to standard YOLO detection format and create a dataset YAML from [configs/data_ssdd.example.yaml](configs/data_ssdd.example.yaml). The official split used in our experiments contains 928 training images and 232 validation images. Dataset files are not redistributed here; please obtain them from their original source and follow the dataset license.

## Training

```bash
python scripts/train.py --data /path/to/data_ssdd.yaml --model configs/egca_net.yaml
```

The supplied script uses the paper-facing setup: 200 epochs, input size 640, batch size 32, SGD, seed 0, `close_mosaic=15`, `mosaic=0.08`, and `mixup=0.09`.

## Evaluation

```bash
python scripts/val.py --weights /path/to/best.pt --data /path/to/data_ssdd.yaml --imgsz 640 --batch 1
```

## Repository Layout

```text
EGCA-Net/
├── configs/          # Model and dataset-YAML templates
├── egca_modules/     # ESFF and DS-SCR implementations
├── scripts/          # Patch, training, and validation entry points
├── requirements.txt
└── README.md
```

## Notes

- `egca_net.yaml` is a YOLOv8-style architecture with P2--P5 detection outputs.
- The `ESFF` layers operate on the P3/P4 backbone features.
- The `SCR` layers are placed before P2--P5 detection outputs.
- Reported results depend on the checkpoint-selection and evaluator settings. Please state these settings when comparing methods.

## Citation

Add the final bibliographic entry here after acceptance or preprint release.
