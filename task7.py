# -----------------------------------------------------------
# Task 7: Support Vector Machines (SVM)
# Complete Code WITH Dataset Included
# -----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------------------------------
# 1. Load Dataset (Breast Cancer Dataset from sklearn)
# -----------------------------------------------------------

data = datasets.load_breast_cancer()
X = data.data
y = data.target

print("Dataset Loaded Successfully!")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("Feature Names:", data.feature_names)


# -----------------------------------------------------------
# Preprocessing: Standard Scaling
# -----------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# -----------------------------------------------------------
# 2. Linear SVM
# -----------------------------------------------------------

svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train, y_train)

y_pred_linear = svm_linear.predict(X_test)

print("\n============================")
print("     LINEAR SVM RESULTS     ")
print("============================")
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))


# -----------------------------------------------------------
# 3. Non-linear SVM (RBF Kernel)
# -----------------------------------------------------------

svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')
svm_rbf.fit(X_train, y_train)

y_pred_rbf = svm_rbf.predict(X_test)

print("\n============================")
print("       RBF SVM RESULTS      ")
print("============================")
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))


# -----------------------------------------------------------
# 4. Hyperparameter Tuning (GridSearchCV)
# -----------------------------------------------------------

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("\n============================")
print("    GridSearchCV Results    ")
print("============================")
print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\n============================")
print("   Tuned SVM Final Results  ")
print("============================")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))


# -----------------------------------------------------------
# 5. Decision Boundary Visualization (Using 2D Synthetic Data)
# -----------------------------------------------------------

# Create 2D circular dataset to show non-linear decision boundary
X_vis, y_vis = datasets.make_circles(n_samples=400, noise=0.08, factor=0.4)

svm_vis = SVC(kernel='rbf', C=1, gamma=2)
svm_vis.fit(X_vis, y_vis)

# Create meshgrid for plot
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 600),
    np.linspace(y_min, y_max, 600)
)

# Predict every point
Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, edgecolors='k')
plt.title("Decision Boundary of SVM (RBF Kernel)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
