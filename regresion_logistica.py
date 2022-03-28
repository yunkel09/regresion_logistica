
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
import random


# funciones
def train_test_split_pandas(dataset, split = 0.60):
    train_size = int(split * len(dataset))
    # crea un listado de valores random no repetidos
    lt1 = sample(range(0,len(dataset)),train_size)
    # Se utiliza inplace=False para poder obtener los valores sin los datos en lt1
    ftrain = dataset.drop(lt1,axis=0,inplace=False)
    # eliminando de la tabla original los valores ingresados en ftrain
    for i in range(len(dataset)):
        if (i not in lt1):
            dataset.drop(index=i,axis=0,inplace=True)
            

def split(df, split = 0.6):
 train_size = int(len(df) * split)
 lt = random.sample(range(0, len(df)), train_size)
 train = df.iloc[lt]
 test = df.drop(lt, inplace=False)
 return train, test

# obtener datos
vinos = pd.read_csv("wine.csv")


# separar en train y test
train, test = split(vinos, split=0.6)
          






















