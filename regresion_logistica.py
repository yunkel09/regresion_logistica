
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 04:45:22 2022

@author: William Chavarría
@github: https://github.com/yunkel09/regresion_logistica

Regresión Logística
"""

import pandas as pd
import numpy as np
from numpy import log, dot, e, shape
import random
from siuba import *
from siuba.dply.verbs import *
from sklearn.model_selection import train_test_split 
from sklearn.datasets import make_classification, load_iris
# from sklearn.preprocessing import minmax_scale
# from sklearn.preprocessing import MinMaxScaler

# funciones
def split(df, split = 0.7):
 train_size = int(len(df) * split)
 lt = random.sample(range(0, len(df)), train_size)
 train = df.iloc[lt]
 test = df.drop(lt, inplace=False)
 return train, test

def normalize_function(data):
    min = np.amin(data,axis=0)
    max = np.amax(data,axis=0)
    return (data - min)/(max-min)

def standardize(X_tr):
    for i in range(shape(X_tr)[1]):
        X_tr[:,i] = (X_tr[:,i] - np.mean(X_tr[:,i]))/np.std(X_tr[:,i])


# obtener datos
vinos = pd.read_csv("wine.csv")

# dumificar etiqueta
dummy = pd.get_dummies(vinos.quality, drop_first = True)
vinos_d = pd.concat((vinos, dummy), axis = 1)
vinos_d = vinos_d >> \
 select(-_.quality) >> \
 rename(quality = "Good")

y = vinos_d.quality.to_numpy()
X = vinos_d >> \
 select(-_.quality)
X = X.to_numpy()


train, test = split(vinos_d, split = 0.7)

X_tr,X_te,y_tr,y_te = train_test_split(X, y, test_size=0.3)


standardize(X_tr)





def standardize(X_tr):
    for i in range(shape(X_tr)[1]):
        X_tr[:,i] = (X_tr[:,i] - np.mean(X_tr[:,i]))/np.std(X_tr[:,i])

standardize(train)
from sklearn import datasets




