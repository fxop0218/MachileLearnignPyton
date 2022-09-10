# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 16:40:39 2022

@author: fxop0
"""

# Regresion lineal polinomica
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.drop("Salary", axis = 1)
X = X.drop("Position", axis = 1)
Y = dataset["Salary"]

# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2)

linear_reg = LinearRegression()
linear_reg.fit(X, Y)

polyno_reg = PolynomialFeatures(degree=7)
X_poly = polyno_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

plt.figure(figsize = (10, 10))
plt.scatter(X, Y, color = "red")
plt.plot(X["Level"], linear_reg.predict(X),"blue")
plt.title("Modelo de regresion lineal")

X_grid = np.arange(min(X["Level"]), max(X["Level"]), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(polyno_reg.fit_transform(X_grid)),"blue")
plt.title("Modelo de regresion lineal")

# Predicción de nuestros modelos
lin_reg_2.predict(polyno_reg.fit_transform([[6.5]])