# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 19:13:11 2022

@author: fxop0
"""

# Algoritmo de apriori

# Importamos las librerias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importamos el conjunto de datos
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)  # Header = none ya que no tiene cabecera

row_len = dataset.count()[0]
# Creación de la lista que espera el modelo de apriori
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])


from apyori import apriori # Libreria local
rules = apriori(transactions, min_suport = 0.003, min_confidence = 0.2,
                min_lift = 3,min_lenght = 2)

# Visualización de los reusltados
results = list(rules)
