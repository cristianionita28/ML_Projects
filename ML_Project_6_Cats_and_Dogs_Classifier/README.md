# 🐱🐶 Cats vs Dogs Image Classifier (PyTorch)

A binary image classification project using **PyTorch** and **Inception v3** (transfer learning) to distinguish between cats and dogs.

---

## 📌 Project Overview

- Framework: **PyTorch**
- Model: **Inception v3 (pretrained on ImageNet)**
- Task: Binary image classification (Cats vs Dogs)
- Training style: Transfer Learning (custom classifier head)
- Platform tested on: **Windows 11 + NVIDIA GPU**

---

---

## 🧠 Model Details

- Backbone: `torchvision.models.inception_v3(pretrained=True)`
- Frozen layers: all convolutional layers
- Trainable layers:
  - Final fully connected layer (`fc`)
  - Auxiliary classifier (`AuxLogits`)
- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`

---

## 🛠 Requirements

Minimal environment (recommended):

```bash
pip install torch torchvision matplotlib numpy


