# AIML-TASK7
🧠 Task 7: Support Vector Machines (SVM)

This project demonstrates the implementation of Support Vector Machines (SVM) for classification using both Linear and RBF kernels. It includes: dataset loading, preprocessing, model training, hyperparameter tuning, evaluation, and decision boundary visualization.

🚀 Project Overview

This task focuses on understanding how SVMs work for both linearly separable and non-linear datasets. You will learn:

Margin Maximization

Kernel Trick

Hyperparameter Tuning (C, Gamma)

Visualizing Decision Boundaries

Cross-Validation

We use:

Breast Cancer Dataset (built-in from Scikit-learn) for classification

make_circles() synthetic dataset for 2D visualization

📦 Requirements

Install the following Python libraries:

pip install numpy matplotlib scikit-learn
📂 Dataset

This project does not require external downloads. The dataset is loaded directly using:

from sklearn import datasets

Breast Cancer Dataset → For training and testing models

make_circles() → For visualization

🧪 Features Implemented
✔ Load & preprocess dataset
✔ Train SVM (Linear Kernel)
✔ Train SVM (RBF Kernel)
✔ Hyperparameter Tuning (GridSearchCV)
✔ Model evaluation (Accuracy, Classification Report)
✔ Visualize decision boundaries in 2D
📜 Full Code

This README is linked to the Task7.py file containing 100% executable code.

📊 Output Includes

Linear SVM Accuracy

RBF SVM Accuracy

Tuned hyperparameters

Confusion matrix & classification reports

Non-linear decision boundary plot

🧩 Key Concepts
🔹 Support Vectors

Points closest to the decision boundary.

🔹 Margin

Distance between support vectors and hyperplane.

🔹 Kernel Trick

Converts low-dimensional data into high-dimensional space for better separation.

🔹 Hyperparameters

C → Controls margin softness

gamma → Controls influence of single data point

📈 Visualization

The 2D plot generated using make_circles() shows how RBF kernel handles non-linear classification.
