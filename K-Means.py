# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 14:29:44 2022

@author: fxop0
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

AGE = "Age"
ANNUAL = "Annual Income (k$)"
SPEND = "Spending Score (1-100)"
# Constants 

# Import the dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset[[SPEND, ANNUAL]].values

# Calcular el numero optimo de clusters mediante el método del codo
from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++",
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Mostramos un plot

plt.plot(range(1,11), wcss)
plt.title("Metodo del codo")
plt.xlabel("Numero de clusters")
plt.ylabel("WCSS(k)")
plt.show()

# Aplicar el método de k-means para segmentar el dataset

kmeans = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
kmeans_pred = kmeans.fit_predict(X)

# Visualización del clusters (Unicamente para dos dimensiones)
# s == tamaño de los puntos
plt.scatter(X[kmeans_pred == 0, 0], X[kmeans_pred == 0, 1], s = 100, color = "red", label = "Cluster 1")
plt.scatter(X[kmeans_pred == 1, 0], X[kmeans_pred == 1, 1], s = 100, color = "green", label = "Cluster 2")
plt.scatter(X[kmeans_pred == 2, 0], X[kmeans_pred == 2, 1], s = 100, color = "blue", label = "Cluster 3")
plt.scatter(X[kmeans_pred == 3, 0], X[kmeans_pred == 3, 1], s = 100, color = "cyan", label = "Cluster 4")
plt.scatter(X[kmeans_pred == 4, 0], X[kmeans_pred == 4, 1], s = 100, color = "magenta", label = "Cluster 5")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 200, c = "black", label = "Baricenters")
plt.title("Clusters de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de gastos del 1 al 100")
plt.legend()
plt.show()