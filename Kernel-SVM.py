# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 12:50:47 2022

@author: fxop0
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Lipieza de datos
dataset = pd.read_csv("Social_Network_Ads.csv")

X = dataset.drop(["Purchased", "User ID", "Gender"], axis = 1)
Y = dataset["Purchased"]

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2)

standard_sc = StandardScaler()
X_train = standard_sc.fit_transform(X_train)
X_val = standard_sc.fit_transform(X_val)

# Crear el SVC 
from sklearn.svm import SVC
classifier = SVC(kernel = "rbf", # Gausiano = rbf = radial base function
                 random_state= 0)

# Entrenar el modelo
classifier.fit(X_train, Y_train)

# Predecimos el modelo
y_pred = classifier.predict(X_val)

# Creación de la matriz de confusión
from matplotlib.colors import ListedColormap
cm = confusion_matrix(Y_val, y_pred)

# Creamos una grafica que nos muestra como es la separación
X_set, y_set = X_val, Y_val
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM rbf (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()

X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM rbf (Conjunto de training)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()