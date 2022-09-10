# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:54:51 2022

@author: fxop0
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.drop("Salary", axis = 1)
X = X.drop("Position", axis = 1)
Y = dataset["Salary"]

# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2)

regression = DecisionTreeRegressor()
regression.fit(X, Y)

plt.figure(figsize = (10, 10))
plt.scatter(X, Y, color = "red")
plt.plot(X["Level"], regression.predict(X),"blue")
plt.title("Modelo de regresion lineal")

X_grid = np.arange(min(X["Level"]), max(X["Level"]), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = "red")
plt.plot(X_grid, regression.predict(regression.predict(X_grid)),"blue")
plt.title("Modelo de regresion lineal")

# Predicción de nuestros modelos
regression.predict(polyno_reg.fit_transform([[6.5]])