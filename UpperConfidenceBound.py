# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 12:13:32 2022

@author: fxop0
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns 
import math
# Importamos el dataset y preparamos los datos 
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

N = dataset.count()[0]
d = dataset.columns.size
# Random selection (comprobar la probabilidad que hay si se hace de forma totalmente aleatoria) 
"""
ads_selected = [] # orden en el que se ha mostrado los anuncios a cada usuario
total_reward = 0 # Numero de usuarios que han hecho click en el anuncio correcto
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

plt.figure(figsize = (10, 10))
sns.histplot(data = ads_selected)
"""

number_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []

for n in range(0 , N):
    max_upper_bound = 0 # Mejor anuncio mostrado
    ad = 0
    for i in range(0, d):
        if (number_of_selections[i]>0): # Correr el codigo una serie de veces antes de que entre en el codigo, ya que si no divide entre 0
            # Creación de la formula delta
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/number_of_selections[i])
            upper_bound = average_reward + delta_i # Mejor mostrado en la ronda
        else:
            upper_bound = 1e400 # Poner un valor muy elevado, para que todos los anuncios sean como minimo una vez selecionados
        if upper_bound > max_upper_bound: # Comprobar si el actual es mejor al total
            max_upper_bound = upper_bound
            ad = i
    
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
    
# Visualización de los resultados con un histograma
plt.figure(figsize = (7, 5))
plt.xlabel("ID anuncio")
plt.ylabel("Frequencia visualización")
sns.histplot(ads_selected)
plt.show()
