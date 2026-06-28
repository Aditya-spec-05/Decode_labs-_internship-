import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

# ----------------------------------
# Load Dataset
# ----------------------------------
iris = load_iris()
X = iris.data
y = iris.target

# ----------------------------------
# Train, Validation, Test Split
# 70% Train, 15% Validation, 15% Test
# ----------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# ----------------------------------
# Train KNN Model
# ----------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ----------------------------------
# Validation Predictions
# ----------------------------------
y_pred = knn.predict(X_val)
y_prob = knn.predict_proba(X_val)

# ----------------------------------
# Metrics
# ----------------------------------
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')

y_val_bin = label_binarize(y_val, classes=[0, 1, 2])

auc = roc_auc_score(
    y_val_bin,
    y_prob,
    multi_class='ovr'
)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("F1 Score :", f1)
print("AUC Score:", auc)

# ==================================
# Visualization 1: PCA Scatter Plot
# ==================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))

for i, name in enumerate(iris.target_names):
    plt.scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        label=name
    )

plt.title("Iris Dataset Visualization using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

# ==================================
# Visualization 2: Decision Boundary
# ==================================
X_2d = X[:, :2]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_2d,
    y,
    test_size=0.3,
    random_state=42
)

knn2 = KNeighborsClassifier(n_neighbors=5)
knn2.fit(X_train2, y_train2)

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = knn2.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3)

scatter = plt.scatter(
    X_2d[:,0],
    X_2d[:,1],
    c=y,
    edgecolors='k'
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("KNN Decision Boundary")
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=iris.target_names
)
plt.show()

# ==================================
# Visualization 3: Confusion Matrix
# ==================================
cm = confusion_matrix(y_val, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()