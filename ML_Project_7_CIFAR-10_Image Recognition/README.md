# CIFAR-10 Image Classification (CNN + Transfer Learning)

This repo contains two PyTorch notebooks for CIFAR-10 image classification:
1) **Custom CNN (from scratch)** — trains a convolutional neural network end-to-end on CIFAR-10.  
2) **Transfer Learning** — fine-tunes (or trains a new head on top of) a pretrained model (e.g., ResNet/ResNeXt/etc.) for CIFAR-10.

---

## Project Structure

- `CIFAR-10_CNN ver 3.ipynb` — CNN from scratch
- `CIFAR-10_Transfer.ipynb` — transfer learning / fine-tuning
- (optional) `Report_CNN ver 2.html` — exported report / results (static)

---

## Quick Start (Local)

### 1) Create and activate a new environment
---
conda env create -f environment.yaml
conda activate cifar10

pip install -r requirements.txt

Runi in cloud:
https://colab.research.google.com/github/<YOUR_GITHUB_USER>/<YOUR_REPO>/blob/main/CIFAR-10_CNN%20ver%203.ipynb
https://colab.research.google.com/github/<YOUR_GITHUB_USER>/<YOUR_REPO>/blob/main/CIFAR-10_Transfer.ipynb
or:
https://mybinder.org/v2/github/<YOUR_GITHUB_USER>/<YOUR_REPO>/HEAD?filepath=CIFAR-10_CNN%20ver%203.ipynb
https://mybinder.org/v2/github/<YOUR_GITHUB_USER>/<YOUR_REPO>/HEAD?filepath=CIFAR-10_Transfer.ipynb
