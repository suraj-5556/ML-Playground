# 🧠 ML Playground: Interactive Machine Learning Visualization

## 🚀 Overview

The **ML Playground** is an interactive web application built with Streamlit that allows users to visualize and understand the training process of two fundamental machine learning algorithms: **Logistic Regression** (Gradient Descent) and **Pegasos Kernel Support Vector Machines (SVM)**.

It's designed as an **interactive study tool** to bring theoretical concepts—like gradient descent, decision boundaries, support vectors, and loss curves—to life in real-time.



---

## ✨ Features

* **Dual Algorithm Support:** Easily switch between **Logistic Regression** (for linear boundaries) and **Pegasos RBF Kernel SVM** (for non-linear boundaries).
* **Interactive Data Generation:** Generate classic 2D datasets like **Blobs**, **Moons**, and **Circles**, and instantly see how different models tackle separability.
* **Real-Time Boundary Animation:** Animate the training process epoch-by-epoch, watching the decision boundary dynamically shift to find the optimal separation.
* **Parameter Monitoring:** Observe the evolving model parameters ($w_1, w_2, b$ or $\alpha_i$ and $b$) and key metrics (Loss, Accuracy, Support Vectors) at every step.
* **Training History Graphs:** Visualizations for the **Loss Curve** and **Accuracy Curve** to track the overall progress of the algorithm.

---

## 🛠️ Algorithms & Concepts Illustrated

| Algorithm | Loss Function | Boundary Type | Key Visualization |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Binary Cross-Entropy | Linear | Straight Decision Line ($\mathbf{w} \cdot \mathbf{x} + b = 0$) |
| **Pegasos Kernel SVM** | Hinge Loss | Non-Linear (RBF Kernel) | Curved Decision Boundary and **Margin** Visualization |

### Pegasos Kernel SVM Specifics
The implementation of Pegasos with an RBF Kernel demonstrates:
* **Dual Coefficients ($\alpha_i$):** The weights track the dual coefficients. Non-zero $\alpha_i$'s correspond to the **Support Vectors**.
* **Fixed Hyperparameters:** The app uses fixed, illustrative values for regularization ($\lambda=0.03$) and RBF kernel parameter ($\gamma=3.02$).

---

## 💻 Installation and Usage

To run the application locally, you'll need Python and a few common libraries.

### Prerequisites

You need Python 3.8+ installed.

### Step 1: Clone the Repository

```bash
git clone [YOUR_REPO_LINK_HERE]
cd [your_repo_folder]
