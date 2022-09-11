# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 19:06:13 2022

@author: fxop0
"""

# Muestre thompson

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import random

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

N = dataset.count()[0]
d = dataset.columns.size

number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
ads_selected = []
total_reward = 0

for n in range(0 , N):
    max_random = 0 # Mejor anuncio mostrado
    ad = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_rewards_1[i] + 1,number_of_rewards_0[i] + 1)
        if random_beta > max_random: # Comprobar si el actual es mejor al total
            max_random = random_beta
            ad = i
    
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
    
    
# Visualización de los resultados con un histograma
plt.figure(figsize = (7, 5))
plt.xlabel("ID anuncio")
plt.ylabel("Frequencia visualización")
sns.histplot(ads_selected)
plt.show()
