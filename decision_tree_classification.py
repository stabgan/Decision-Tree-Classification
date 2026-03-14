# Decision Tree Classification

# Importing the libraries
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


def main():
    """Train and evaluate a Decision Tree classifier on Social Network Ads data."""

    # Importing the dataset (resolve path relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "Social_Network_Ads.csv")
    dataset = pd.read_csv(csv_path)
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Decision Tree Classification to the Training set
    classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Visualising the Training set results
    _plot_decision_boundary(
        classifier,
        X_train,
        y_train,
        title="Decision Tree Classification (Training set)",
    )

    # Visualising the Test set results
    _plot_decision_boundary(
        classifier,
        X_test,
        y_test,
        title="Decision Tree Classification (Test set)",
    )


def _plot_decision_boundary(classifier, X_set, y_set, title):
    """Plot the decision boundary for a 2-feature classifier."""
    colors = ("red", "green")
    cmap = ListedColormap(colors)

    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01),
    )
    plt.contourf(
        X1,
        X2,
        classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.75,
        cmap=cmap,
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for idx, label in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == label, 0],
            X_set[y_set == label, 1],
            c=colors[idx],
            label=label,
        )

    plt.title(title)
    plt.xlabel("Age")
    plt.ylabel("Estimated Salary")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
