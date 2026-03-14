# 🌳 Decision Tree Classification

A machine learning project that predicts whether a user will purchase a product based on their **Age** and **Estimated Salary**, using a Decision Tree Classifier trained on social network ad data.

## 📋 Overview

This project implements a **CART (Classification and Regression Trees)** model using the entropy criterion to classify users into buyers vs. non-buyers. The dataset comes from a social network advertising campaign and contains 400 user records.

## 🔬 Methodology

1. **Data Loading** — Import the `Social_Network_Ads.csv` dataset (features: Age, Estimated Salary; target: Purchased)
2. **Train/Test Split** — 75/25 split with a fixed random state for reproducibility
3. **Feature Scaling** — StandardScaler normalization on both features
4. **Model Training** — Decision Tree with `entropy` criterion (information gain)
5. **Evaluation** — Confusion matrix on the test set
6. **Visualization** — Decision boundary plots for both training and test sets

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3 | Primary implementation |
| 📊 R | Alternative implementation |
| 🔢 NumPy | Numerical operations |
| 🐼 Pandas | Data manipulation |
| 📈 Matplotlib | Visualization |
| 🤖 scikit-learn | ML model, preprocessing, evaluation |
| 📦 rpart (R) | Decision tree in R |
| 📦 caTools (R) | Train/test split in R |

## 📦 Dependencies

### Python

```
numpy
pandas
matplotlib
scikit-learn
```

### R

```
caTools
rpart
ElemStatLearn
```

## 🚀 How to Run

### Python

```bash
# Install dependencies
pip install numpy pandas matplotlib scikit-learn

# Run the classifier
python decision_tree_classification.py
```

### R

```r
source("decision_tree_classification.R")
```

> Make sure `Social_Network_Ads.csv` is in the same directory as the script.

## ⚠️ Known Issues

- The R implementation depends on `ElemStatLearn`, which has been archived from CRAN. You may need to install it from an archive or alternative source.
- Visualization uses `plt.show()` which requires a GUI backend — running in a headless environment will need a backend switch (e.g., `matplotlib.use('Agg')`).
- The dataset is small (400 records), so results may vary with different splits.

## 📄 License

See [LICENSE](LICENSE) for details.
