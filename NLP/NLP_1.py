# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 19:33:54 2022

@author: fxop0
"""

# Procesamiento de lenguaje natural
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Importación y limpiado del dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", sep = "\t", quoting = 3)# Importamos un tsv (tab separated value) quoting = 3 => ignora las comillas dobles
d = len(dataset.columns)
N = dataset.count()[0]
# Limpieza del texto
import re # Importamos la libreria de regular expresión
import nltk # Libreria que permite procesamiento de datos (natural lenguaje toolkit)

nltk.download("stopwords") # descargar palabras que no aportan nada al texto (this etc)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for row in range(0, N): # Pasamos todas las palabras por la limpieza de datos
    review = re.sub("[^a-zA-Z]"," ", dataset["Review"][row]) # Elimina todos los caracteres que no esten comprendidos entre la a-z
    review = review.lower()# Pasamos todos los valores a minuscula
    review = review.split() # Separamos las palabras con separaciones
    ps = PorterStemmer() # Cambia la palabra por la más simple posible (loved = love)
    review = [ps.stem(word) for word in review if not word in stopwords.words("english")] # Quitmaos las palabras relevantes
    review = " ".join(review)# Combinamos las palabras en una sola frase
    corpus.append(review)
    
# Crear un bag de woeds
from sklearn.feature_extraction.text import CountVectorizer # Transformara las palabras en vectores de frequencia
cv = CountVectorizer(max_features = 1500) # De palabra a vector, max_features consigue X palabras más relevantes
X = cv.fit_transform(corpus).toarray() # Creamos la matriz dispersa
Y = dataset["Liked"].values


# Utilizamos naive bayes para la predicción
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2)

standard_sc = StandardScaler()
X_train = standard_sc.fit_transform(X_train)
X_val = standard_sc.fit_transform(X_val)

# Crear el Naive bayes 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

# Entrenar el modelo
classifier.fit(X_train, Y_train)

# Predecimos el modelo
y_pred = classifier.predict(X_val)

# Creación de la matriz de confusión
from matplotlib.colors import ListedColormap
cm = confusion_matrix(Y_val, y_pred)

print(f"Fiabilidad del modelo: {((cm[0][0]+cm[1][1])/200)*100}%")
