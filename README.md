# LinearPy

Welcome to **LinearPy**, a project that implements core linear modeling techniques from scratch using only NumPy. This includes both regression and classification tasks—from simple Linear Regression to more advanced classifiers like and Softmax Regression. Linear modeling is a cornerstone of machine learning and deep learning foundations, and by developing each component manually, we gain strong intuition about how these models operate, optimize, and generalize. Every model is provided as a reusable class with a consistent API and includes support for ElasticNet Regularization and Learning Rate Schedules; classifiers also offer `predict_proba()` to estimate class probabilities.

🌐 **Note:** If you experience issues viewing the project's notebook on GitHub, you can open it online via <a href="https://nbviewer.org/github/hoom4n/LinearPy/blob/main/LinearPy.ipynb">NB Viewer</a>
---

## 📖 Overview

This repo features a comprehensive Jupyter Notebook that walks through linear modeling theory—from definitions and formulations to implementations using a Scikit‑Learn–like structure—then sanity‑checks each model on a synthesized polynomial dataset.

<table>
  <thead>
    <tr>
      <th style="min-width:160px;">Model</th>
      <th style="min-width:120px;">Task</th>
      <th style="min-width:220px;">Features</th>
      <th style="min-width:300px;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Linear Regression (SVD)</td>
      <td>Regression</td>
      <td> - </td>
      <td style="word-break:break-word;">Solves for optimal weights using the Moore-Penrose pseudoinverse (closed‑form solution).</td>
    </tr>
    <tr>
      <td>Linear Regression (Batch GD)</td>
      <td>Regression</td>
      <td>ElasticNet Regularization</td>
      <td style="word-break:break-word;">Iteratively minimizes MSE using full‑batch gradient descent.</td>
    </tr>
    <tr>
      <td>Linear Regression (Mini‑batch GD)</td>
      <td>Regression</td>
      <td>ElasticNet Regularization, Learning Rate Schedule</td>
      <td style="word-break:break-word;">Minimizes MSE on mini‑batches with optional learning rate scheduling.</td>
    </tr>
    <tr>
      <td>Linear Regression (SGD)</td>
      <td>Regression</td>
      <td>ElasticNet Regularization, Learning Rate Schedule</td>
      <td style="word-break:break-word;">Updates weights per sample for fast or online learning.</td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>Binary Classification</td>
      <td>ElasticNet Regularization, Learning Rate Schedule, `predict_proba`, Mini‑batch support</td>
      <td style="word-break:break-word;">Binary classifier trained via gradient descent on log‑loss.</td>
    </tr>
    <tr>
      <td>Softmax Regression</td>
      <td>Multiclass Classification</td>
      <td>ElasticNet Regularization, Learning Rate Schedule, `predict_proba`, Mini‑batch support</td>
      <td style="word-break:break-word;">Multiclass classifier using softmax activation and cross‑entropy loss.</td>
    </tr>
  </tbody>
</table>


Since the project emphasizes end-to-end understanding, a set of lightweight, NumPy-based utilities were implemented to support preprocessing and evaluation workflows:

* `StandardScaler` for feature normalization
* `OneHotEncoder` for categorical targets
* `train_test_split` for dataset partitioning

**Evaluation Metrics**

* Regression: `MSE`, `RMSE`, `MAE`
* Classification: `Accuracy`, `Precision`, `Recall`, `F1 Score`
* A general `ModelEvaluator` class to summarize results

---

## 🔧 Installation

All implementations are available on PyPI and GitHub under **HoomanMLTK**:

```bash
pip install hoomanmltk
````

Or view the source at
[github.com/hoom4n/HoomanMLTK](https://github.com/hoom4n/HoomanMLTK)

**Usage example:**

```python
from hoomanmltk.linear.classification import SoftmaxRegression

model = SoftmaxRegression(alpha=0.5, lambda_=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

```
```
