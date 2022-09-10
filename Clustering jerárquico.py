# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 17:46:12 2022

@author: fxop0
"""

# Clustering jerárquico

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Constantes
AGE = "Age"
ANNUAL = "Annual Income (k$)"
SPEND = "Spending Score (1-100)"

# Cargar los datos
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset[[SPEND, ANNUAL]].values

# Utilizar el endograma para encontrar el numero correcto de clusters
import scipy.cluster.hierarchy as sch
# Visualización del dendrograma
dendrograma = sch.dendrogram(sch.linkage(X, method = "ward")) # ward minimizar la varianza que hay entre los puntos de los clusters
plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia")
plt.show()

# Ajustar el clustering jerárqucio a nuesro conjunto de datos
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")

hc_predict = hc.fit_predict(X)


# Visualización de los clusters jerárquicos (Unicamente para dos dimensiones)
plt.scatter(X[hc_predict == 0, 0], X[hc_predict == 0, 1], s = 100, color = "red", label = "Cluster 1")
plt.scatter(X[hc_predict == 1, 0], X[hc_predict == 1, 1], s = 100, color = "green", label = "Cluster 2")
plt.scatter(X[hc_predict == 2, 0], X[hc_predict == 2, 1], s = 100, color = "blue", label = "Cluster 3")
plt.scatter(X[hc_predict == 3, 0], X[hc_predict == 3, 1], s = 100, color = "cyan", label = "Cluster 4")
plt.scatter(X[hc_predict == 4, 0], X[hc_predict == 4, 1], s = 100, color = "magenta", label = "Cluster 5")
plt.title("Clusters de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de gastos del 1 al 100")
plt.legend()
plt.show()