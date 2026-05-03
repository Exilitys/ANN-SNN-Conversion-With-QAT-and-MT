# ANN → SNN Conversion with Quantization-Aware Training (QAT) and Multi-Threshold IF Nueron (MT-IF)

Conversion of Artificial Neural Networks (ANNs) to Spiking Neural Networks (SNNs) using QAT and MT-IF Neuron on the **Google QuickDraw** dataset.

---

## Overview

This project implements and benchmarks a pipeline for:

1. **Training quantized ANNs** (2-bit, 3-bit, 4-bit) via QAT on QuickDraw sketch classification
2. **Converting** the trained ANN to an SNN using threshold-based neuron conversion
3. **Fine-tuning** the SNN layer-by-layer using a proxy-ANN mechanism
4. **Evaluating** accuracy, energy efficiency, and firing rates for both ANN and SNN variants

### Supported Architectures

| Architecture           | Arch string                      |
| ---------------------- | -------------------------------- |
| ResNet-18              | `resnet18_quickdraw`             |
| VGG-16                 | `vgg16_quickdraw`                |
| AlexNet                | `alexnet_quickdraw`              |
| EfficientNetV2         | `efficientnetv2_quickdraw`       |
| InceptionNetV4         | `inceptionnetv4_quickdraw`       |
| MobileNetV4-Conv-Small | `mobilenetv4convsmall_quickdraw` |

### Supported Bit-widths

- **2-bit** → `k=3`, `T=3` timesteps
- **3-bit** → `k=7`, `T=7` timesteps
- **4-bit** → `k=15`, `T=15` timesteps

---

## Project Structure

```
Code/
├── resnet_models/
│   ├── resnet.py               # ResNet-18 QAT + SNN model definition
│   ├── quant_layer.py          # QuantConv2d, QuantReLU, APoT quantization
│   ├── main.py                 # ANN QAT training script
│   ├── snn_ft.py               # SNN fine-tuning script
│   └── __init__.py
├── vgg_models/                 # Same structure as resnet_models/
├── alexnet_models/             # Same structure as resnet_models/
├── efficientnet_models/        # Same structure as resnet_models/
├── inceptionnet_models/        # Same structure as resnet_models/
├── mobilenetv4_models/         # Same structure as resnet_models/
├── test
|   |── test.py                 # Unified evaluation script (all models)
|   |── test.txt                # Test command reference (PowerShell)
├── requirements.txt
├── notes.txt                   # Training command log with completion status
```

---

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Conda environment (recommended)

### Installation

```bash
# Create and activate conda environment
conda create -n torch_snn python=3.9
conda activate torch_snn

# Install PyTorch (with CUDA support - adjust to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install numpy pillow
```

### Dataset

Place Google QuickDraw `.npy` files inside the `dataset/` directory. Each file should be named `<classname>.npy` and contain raw 28×28 grayscale images stacked as `(N, 784)` arrays.

The dataset split used throughout:

- **Train**: samples `[0:700]` per class
- **Val**: samples `[700:850]` per class
- **Test**: samples `[850:1000]` per class

---

## Workflow

```
Step 1: Train ANN (main.py)
         ↓
Step 2: Fine-tune SNN (snn_ft.py)
         ↓
Step 3: Evaluate (test.py)
```

---

## Step 1 — Train the QAT-ANN (`main.py`)

### General Usage

```bash
# ResNet-18
python resnet_models/main.py --bit <BIT> --arch resnet18_quickdraw --epochs 30

# VGG-16
python vgg_models/main.py --bit <BIT> --arch vgg16_quickdraw --epochs 30

# AlexNet
python alexnet_models/main.py --bit <BIT> --arch alexnet_quickdraw --epochs 30

# EfficientNetV2
python efficientnet_models/main.py --bit <BIT> --arch efficientnetv2_quickdraw --epochs 30

# InceptionNetV4
python inceptionnet_models/main.py --bit <BIT> --arch inceptionnetv4_quickdraw --epochs 30

# MobileNetV4
python mobilenetv4_models/main.py --bit <BIT> --arch mobilenetv4convsmall_quickdraw --epochs 30
```

### `main.py` Arguments

| Argument                  | Default        | Description                                    |
| ------------------------- | -------------- | ---------------------------------------------- |
| `--arch`                  | model-specific | Architecture name string                       |
| `--bit`                   | `32`           | Quantization bit-width (2, 3, 4, or 32 for FP) |
| `--epochs`                | `300`          | Number of training epochs                      |
| `--batch-size` / `-b`     | `256`          | Mini-batch size                                |
| `--lr`                    | `0.1`          | Initial learning rate                          |
| `--weight-decay` / `--wd` | `1e-4`         | L2 weight decay                                |
| `--resume`                | `''`           | Path to checkpoint to resume from              |
| `--init`                  | `''`           | Path to pre-trained floating-point weights     |
| `--evaluate` / `-e`       | flag           | Run evaluation only (no training)              |
| `--device` / `-id`        | `'0'`          | CUDA device index                              |
| `--print-freq` / `-p`     | `100`          | Print frequency (batches)                      |

**Checkpoints** are saved to:

```
result/<arch>_<bit>bit/
    checkpoint.pth      # latest epoch
    model_best.pth.tar  # best validation accuracy
```

---

## Step 2 — Fine-tune the SNN (`snn_ft.py`)

This script loads the best ANN checkpoint, converts it to an SNN, then fine-tunes it layer-by-layer using a proxy-ANN with `StaircaseActivation`.

### General Usage

```bash
# ResNet-18, 2-bit
python resnet_models/snn_ft.py --bit 2 --k 3 --arch resnet18_quickdraw --num_epochs 3

# VGG-16, 3-bit
python vgg_models/snn_ft.py --bit 3 --k 7 --arch vgg16_quickdraw --num_epochs 3

# AlexNet, 4-bit
python alexnet_models/snn_ft.py --bit 4 --k 15 --arch alexnet_quickdraw --num_epochs 3

# EfficientNetV2, 2-bit
python efficientnet_models/snn_ft.py --bit 2 --k 3 --arch efficientnetv2_quickdraw --num_epochs 3

# InceptionNetV4, 3-bit
python inceptionnet_models/snn_ft.py --bit 3 --k 7 --arch inceptionnetv4_quickdraw --num_epochs 3

# MobileNetV4, 4-bit
python mobilenetv4_models/snn_ft.py --bit 4 --k 15 --arch mobilenetv4convsmall_quickdraw --num_epochs 3
```

### `snn_ft.py` Arguments

| Argument              | Default        | Description                                                    |
| --------------------- | -------------- | -------------------------------------------------------------- |
| `--arch`              | model-specific | Architecture name string                                       |
| `--bit`               | `2`            | Quantization bit-width                                         |
| `-k` / `--k`          | `3`            | Max spike level for SMT-IF neuron (2-bit→3, 3-bit→7, 4-bit→15) |
| `--num_epochs` / `-n` | `1`            | Epochs per layer in Phase 1 fine-tuning                        |
| `--e2e-epochs`        | `10`           | Epochs for Phase 2 end-to-end fine-tuning                      |
| `--start-layer`       | `0`            | Resume Phase 1 from this layer index                           |
| `--force`             | flag           | Always accept fine-tuned weights (skip revert check)           |
| `--evaluate` / `-e`   | flag           | Evaluate the fine-tuned SNN only                               |
| `--batch-size` / `-b` | `128`          | Mini-batch size                                                |
| `--lr`                | `0.1`          | Learning rate                                                  |

**Output checkpoints** are saved to:

```
result/<arch>_<bit>bit_ft/
    layer_checkpoint.pth    # per-layer resume checkpoint
    model_best.pth.tar      # final fine-tuned SNN
```

**Auto-resume**: If `layer_checkpoint.pth` exists, fine-tuning automatically resumes from the last completed layer.

---

## Step 3 — Evaluate (`test.py`)

The unified evaluation script measures ANN accuracy, SNN direct-conversion accuracy, SNN fine-tuned accuracy, energy estimates, and firing rates.

### Basic Usage

```powershell
# PowerShell multi-line syntax
python test.py `
  --arch resnet18_quickdraw --bit 2 `
  --ann-ckpt best_result\resnet18_quickdraw_2bit\model_best-1.pth.tar `
  --ft-ckpt  best_result\resnet18_quickdraw_2bit_ft\model_best-1.pth.tar `
  --run-label run_1 `
  --model-version 1.0
```

### `test.py` Arguments

| Argument          | Description                                                      |
| ----------------- | ---------------------------------------------------------------- |
| `--arch`          | Architecture string (e.g. `resnet18_quickdraw`)                  |
| `--bit`           | Bit-width (2, 3, or 4)                                           |
| `--ann-ckpt`      | Path to the ANN checkpoint (`.pth.tar`)                          |
| `--ft-ckpt`       | Path to the fine-tuned SNN checkpoint                            |
| `--run-label`     | Label string to identify this evaluation run                     |
| `--model-version` | Version tag stored in history (e.g. `1.0`)                       |
| `--show-history`  | Print a formatted table of all past evaluation runs              |
| `--auto`          | Auto-discover all checkpoints in `best_result/` and evaluate all |

### View Evaluation History

```bash
python test.py --show-history
```

### Auto-evaluate All Checkpoints

```bash
python test.py --auto
```

Results are appended to `eval_history.jsonl`

---

## Metrics Computed by `test.py`

| Metric                | Description                                         |
| --------------------- | --------------------------------------------------- |
| `ann_accuracy`        | ANN Top-1 accuracy (%) on test split                |
| `ann_top5_accuracy`   | ANN Top-5 accuracy (%)                              |
| `ops_ann`             | Total MACs per sample (Σ C_in·K²·C_out·H_out·W_out) |
| `e_ann_pJ`            | ANN energy: `4.6 pJ × Ops_ANN` (Horowitz 2014)      |
| `snn_direct_accuracy` | SNN Top-1 accuracy after direct ANN→SNN conversion  |
| `snn_ft_accuracy`     | SNN Top-1 accuracy after fine-tuning                |
| `firing_rate_direct`  | Mean normalized spike amplitude ρ (direct)          |
| `firing_rate_ft`      | Mean normalized spike amplitude ρ (fine-tuned)      |
| `ops_snn_direct`      | T · Σ(ρ_l · MAC_l) for direct SNN                   |
| `ops_snn_ft`          | T · Σ(ρ_l · MAC_l) for fine-tuned SNN               |
| `e_snn_*_pJ`          | SNN energy: `0.9 pJ × Ops_SNN`                      |
| `e_ratio_*`           | Energy ratio: E_SNN / E_ANN                         |

---

## Authors

**Jonathan Carlo**
Binus University
📧 [jonathan.carlo@binus.ac.id](mailto:jonathan.carlo@binus.ac.id)

**Bren Alden**
Binus University
📧 [bren.alden@binus.ac.id](mailto:bren.alden@binus.ac.id)
