# Unsupervised Learning Mini-Projects  
**Clustering with Hierarchical Methods, DBSCAN and GMM**

This repository contains **three small projects** that illustrate the main families of **unsupervised clustering algorithms**.  
Each project focuses on a different clustering philosophy and shows how to **discover structure in unlabeled data**.

---

## 1️⃣ Hierarchical Clustering Project

**Notebook:** `Hierarchical Clustering Lab [SOLUTION].ipynb`

### What it does
- Applies **Hierarchical (Agglomerative) Clustering**
- Builds a **dendrogram** to explore how observations merge step by step
- Helps decide the **number of clusters visually**, without fixing it in advance

### Unsupervised model used
- **Hierarchical Clustering**
  - Linkage methods (e.g. Ward, complete, single)
  - Distance-based grouping

### Dataset
- A **numerical tabular dataset** (features only, no labels)
- Used to illustrate similarity, distance, and cluster structure

### Key idea
> Explore cluster structure at multiple levels and gain interpretability through dendrograms.

---

## 2️⃣ DBSCAN Clustering Project

**Notebook:** `DBSCAN Notebook [SOLUTION].ipynb`

### What it does
- Applies **DBSCAN**, a density-based clustering algorithm
- Automatically detects:
  - clusters of **arbitrary shape**
  - **noise / outliers**
- Does **not require specifying the number of clusters**

### Unsupervised model used
- **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise)

### Dataset
- A **numerical dataset** where density differences matter
- Suitable for demonstrating noise detection and non-spherical clusters

### Key idea
> Clusters are dense regions of points separated by sparse areas.

---

## 3️⃣ Gaussian Mixture Models (GMM) with Validation

**Notebook:** `GMM_Clustering_Validation_with_Silhouette_Coefficient.ipynb`

### What it does
- Fits **Gaussian Mixture Models (GMM)** to the data
- Performs **soft clustering** (probabilistic cluster membership)
- Uses the **Silhouette Coefficient** to:
  - evaluate clustering quality
  - compare different numbers of components

### Unsupervised model used
- **Gaussian Mixture Models (GMM)**
- Expectation–Maximization (EM) algorithm
- **Silhouette Score** for validation

### Dataset
- A **continuous numerical dataset**
- Well-suited for Gaussian / elliptical cluster assumptions

### Key idea
> Data is modeled as a mixture of Gaussian distributions, not hard cluster boundaries.

---

## Summary Table

| Project | Algorithm | Clustering Type | Needs K? | Handles Noise |
|------|---------|----------------|--------|---------------|
| Hierarchical | Agglomerative | Hard | ❌ No | ❌ Limited |
| DBSCAN | Density-based | Hard | ❌ No | ✅ Yes |
| GMM | Probabilistic | Soft | ✅ Yes | ❌ Sensitive |

---

## Conceptual Takeaway

> **Unsupervised learning discovers structure in X, without using labels (y).**  
Different clustering algorithms reveal **different types of structure** in the same data.

---

## Technologies Used
- Python
- NumPy, pandas
- scikit-learn
- matplotlib / seaborn

---

## One-Sentence Summary
**These projects demonstrate three complementary approaches to clustering: hierarchical structure, density-based grouping, and probabilistic modeling.**

