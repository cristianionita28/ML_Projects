
# PCA Mini Project — Cars Dataset

This project applies **Principal Component Analysis (PCA)** to a cars dataset in order to **reduce dimensionality** and **understand the main sources of variation** in the data.

---

## Project Goal

- Start with **17 original numerical features (X)**
- Use **PCA** to transform them into a **smaller number of components**
- Keep most of the **original variance** while working with fewer variables
- Improve **interpretability and efficiency** for downstream analysis or modeling

> Important: PCA is **unsupervised** — it does **not** use or explain a target variable `y`.

---

## What PCA Does Here

- Standardizes the input features
- Finds new orthogonal axes (**Principal Components**)
- Orders them by how much **variance in X** they explain
- Keeps the first components (e.g. 6) that explain most of the variance

Each principal component is a **linear combination of all original features**, not a subset of columns.

---

## Key Visualizations

- **Scree Plot**  
  Shows how much variance each component explains and helps decide how many components to keep.

- **Feature Weights (Loadings) for Dimensions 1–3**  
  Bar plots showing how strongly each original feature contributes to:
  - Dimension 1 (PC1)
  - Dimension 2 (PC2)
  - Dimension 3 (PC3)

These plots help **interpret what each component represents** (e.g. size, performance, efficiency).

---

## Key Takeaways

- PCA reduces **17 features → a few components**, not by deleting columns but by **combining them**
- The new components explain **variance in X**, not in a response variable
- Feature weights (loadings) explain **how each component is built**
- PCA is useful as:
  - preprocessing for ML models
  - exploratory data analysis
  - noise reduction

---

## Technologies Used

- Python
- pandas, numpy
- scikit-learn (PCA, StandardScaler)
- matplotlib, seaborn

---

## Concept in One Sentence

> **PCA explains most of the variance in the original features using fewer new variables called principal components.**
