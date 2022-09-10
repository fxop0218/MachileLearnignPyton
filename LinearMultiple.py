# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 15:16:47 2022

@author: fxop0
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer

dataset = pd.read_csv("50_Startups.csv")

# Separar el resultado a los datos que vamos a utilizar
Y = dataset["Profit"]
X = dataset.drop("Profit", axis = 1)

label_encoder = LabelEncoder()
X["State"] = label_encoder.fit_transform(X["State"])

transformer = make_column_transformer(
    (OneHotEncoder(), ['State']),
    remainder='passthrough')

X = transformer.fit_transform(X)

# Separación de datos
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2)

# Creamos el modelo de regressión
regression = LinearRegression()
regression.fit(X_train, Y_train)

pred = regression.predict(X_val) # Comparar resultado con Y_val

# Construir el modelo de regressión optimo mediante la eliminación hacia atras

import statsmodels.regression.linear_model as sm # Libreria que permite construir el modelo de eliminación hacia atras

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,4,5]].tolist()
SL = 0.05

regression_OLS = sm.OLS(Y, X_opt).fit()
print(regression_OLS.summary())

X_opt = X[:, [0,2,4,5]].tolist()
SL = 0.05

regression_OLS = sm.OLS(Y, X_opt).fit()
print(regression_OLS.summary())

X_opt = X[:, [0,4,5]].tolist()
SL = 0.05

regression_OLS = sm.OLS(Y, X_opt).fit()
print(regression_OLS.summary())

X_opt = X[:, [0,4]].tolist()
SL = 0.05

regression_OLS = sm.OLS(Y, X_opt).fit()
print(regression_OLS.summary())

# Eliminación hacia atrás utilizando solamente p-valores
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)


# Eliminación hacia atrás utilizando  p-valores y el valor de  R Cuadrado Ajustado

def backwardElimination(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)