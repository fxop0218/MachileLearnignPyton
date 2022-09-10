# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 17:12:36 2022

@author: fxop0
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.drop("Salary", axis = 1)
X = X.drop("Position", axis = 1)
Y = dataset["Salary"]

# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2)
X_grid = np.arange(min(X["Level"]), max(X["Level"]), 0.1)
  
# reshape for reshaping the data into a len(X_grid)*1 array, 
# i.e. to make a column out of the X_grid value                  
X_grid = X_grid.reshape((len(X_grid), 1))

regression = RandomForestRegressor(n_estimators = 10, random_state= 0)
regression.fit(X, Y)

plt.figure(figsize = (10, 10))
plt.scatter(X, Y, color = "red")
plt.plot(X["Level"], regression.predict(X),"blue")
plt.title("Modelo de regresion con randomForest")


plt.scatter(X, Y, color = "red")
plt.plot(X_grid, regression.predict(regression.predict(X_grid).reshape(-1, 1)),"blue")
plt.title("Modelo de regresion con randomForest")

# Predicción de nuestros modelos
regression.predict(np.array([[6.5]]))
