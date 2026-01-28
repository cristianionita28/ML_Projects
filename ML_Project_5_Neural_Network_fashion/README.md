# Fashion-MNIST Classification with PyTorch

This project demonstrates a simple **feed-forward neural network (MLP)** built with **PyTorch** to classify images from the **Fashion-MNIST** dataset.  
The notebook covers training, evaluation, GPU (CUDA) usage, and visualization of class probabilities.

---

## 📌 Features
- PyTorch MLP (fully connected neural network)
- Fashion-MNIST dataset
- Training on **CPU or GPU (CUDA)**
- Batch inference and visualization
- Custom helper functions for plotting predictions

---

## 🧠 Model Architecture
- Input: 28×28 images (flattened to 784)
- Hidden layers: 256 → 128 → 64 (ReLU)
- Output: 10 classes (LogSoftmax)
- Loss: Negative Log Likelihood (`NLLLoss`)
- Optimizer: Adam

---

## 🚀 Getting Started

### Option 1: Using `requirements.txt` (pip)
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt
conda env create -f environment.yaml
conda activate fashion-mnist-pytorch

🌐 Run Online
▶ Binder
https://mybinder.org/v2/gh/<your-username>/<your-repo>/main
▶ Google Colab
https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/Fashion-MNIST%20prediction%20with%20neural%20networks.ipynb



