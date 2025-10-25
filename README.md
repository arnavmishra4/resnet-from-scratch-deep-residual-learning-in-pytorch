# 🧱 ResNet from Scratch — Deep Residual Learning in PyTorch

This repository is part of my **PyTorch Research Mastery Series** — implementing deep learning architectures from first principles using pure PyTorch.

Features a **ResNet (Residual Network)** built entirely from scratch with manual residual blocks, skip connections, and dataset-agnostic training pipeline.

---

## 🧠 Research Intent

Part of my ongoing effort to master AI architectures from the ground up using raw tensor operations and PyTorch fundamentals instead of high-level library abstractions.

> **Goal:** Build deep understanding of residual learning, gradient flow, and why skip connections solve the vanishing gradient problem in deep networks.

---

## 🧩 Overview

Residual Networks introduced skip connections that allow gradients to flow directly through identity paths — effectively enabling training of very deep CNNs.

This repo includes:
- **ResNet implemented from scratch** (no `torchvision.models`)
- Residual blocks with identity and projection shortcuts
- Dataset-agnostic design — works with any ImageFolder-structured dataset
- Clean, modular training pipeline

---

## ⚙️ Components

| Component | File | Description |
|------------|------|-------------|
| 🧩 `ResidualBlock` | `models/resnet.py` | Residual unit with identity or projection shortcut |
| ⚙️ `ResNet` | `models/resnet.py` | Full ResNet architecture with multiple stages |
| 📦 `data_loader.py` |  | Dataset loading utilities |
| 🧮 `metrics.py` |  | Training accuracy computation |
| ⚙️ `config.py` |  | Hyperparameter configuration |

---

## 🧱 Architecture

The residual block implements:

```
y = F(x, W) + x
```

where:
- `F(x, W)`: output of stacked convolutions
- `x`: identity (skip connection)
- `y`: residual output

When dimensions change, identity is projected via 1×1 convolution.

Each stage doubles feature depth while halving spatial resolution:
```
64 → 128 → 256 → 512
```

---

## 🧠 Training

Configure your dataset in `config.py`:

```python
class Config:
    data_path = "path_to_your_dataset"
    batch_size = 32
    image_size = 224
    num_epochs = 15
    learning_rate = 1e-3
```

Dataset structure:
```
dataset_root/
├── class_1/
│   ├── img001.jpg
│   └── img002.jpg
├── class_2/
│   └── ...
```

Run training:
```bash
python main.py
```

---

## 🧭 Roadmap

| Model | Status |
|-------|--------|
| **ResNet-18** | ✅ Completed |
| **ResNet-34/50 variants** | 🔜 Planned |
| **Preactivation ResNet** | 🧩 Future |

---

## 🧩 Folder Structure

```
resnet-from-scratch/
│
├── main.py
├── config.py
├── data_loader.py
│
├── models/
│   └── resnet.py
│
├── utils/
│   └── metrics.py
│
└── README.md
```

---

## 🧰 Requirements

```
torch
torchvision
numpy
tqdm
```

---

## 🏁 Author

**Arnav Mishra**  
AI Researcher · PyTorch & Deep Learning  
Bhopal, India

---

> *"Understanding ResNet means understanding why deep networks work — not just how to use them."*
