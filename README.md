# Decision Tree Classification

Predict whether a user purchases a product based on age and estimated salary using a Decision Tree classifier with entropy-based splitting.

## Overview

This project applies the CART (Classification and Regression Trees) algorithm to a social network advertising dataset. The classifier learns decision boundaries that split the feature space to maximize information gain (minimize entropy), then visualises those boundaries against training and test data.

Implementations are provided in both Python and R.

## Dataset

**Social_Network_Ads.csv** — 400 records with the following columns:

| Column | Description |
|---|---|
| User ID | Unique identifier (not used as a feature) |
| Gender | Male / Female (not used as a feature) |
| Age | User age |
| EstimatedSalary | Estimated annual salary |
| Purchased | Target label — 0 (no) or 1 (yes) |

Only **Age** and **EstimatedSalary** are used as input features.

## 🛠 Tech Stack

| | Tool | Purpose |
|---|---|---|
| 🐍 | Python 3.8+ | Primary implementation |
| 📊 | scikit-learn | Decision tree model, train/test split, scaling |
| 🔢 | NumPy | Numerical operations |
| 🐼 | pandas | Data loading and manipulation |
| 📈 | matplotlib | Decision boundary visualisation |
| 📉 | R | Alternative implementation |
| 🌲 | rpart | R decision tree library |

## Getting Started

### Python

```bash
pip install numpy pandas matplotlib scikit-learn
python decision_tree_classification.py
```

### R

```r
# Required packages
install.packages(c("caTools", "rpart"))
source("decision_tree_classification.R")
```

## How It Works

1. Load the dataset and select Age + EstimatedSalary as features
2. Split 75/25 into training and test sets
3. Apply standard scaling to normalise features
4. Train a `DecisionTreeClassifier` with `criterion='entropy'`
5. Evaluate with a confusion matrix
6. Plot decision boundaries for both training and test sets

## ⚠️ Known Issues

- The R script depends on `ElemStatLearn`, which has been archived on CRAN. The visualisation section in R will not run without manually installing the package from an archive or replacing it with an alternative like `ggplot2`.
- Decision boundary plots use a fine mesh (`step=0.01`) which can be slow on large feature ranges.

## License

[MIT](LICENSE)
