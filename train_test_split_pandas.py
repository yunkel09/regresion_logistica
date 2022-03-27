from random import randrange, sample
import pandas as pd
import numpy as np


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
    
    # retornando los dataset, uno con los valores para train y el otro con los restantes
    return ftrain, dataset
    

def main():
    
    # obteniendo valores del dataset
    dwine = pd.read_csv("wine.csv")
    # llamando al procedimiento
    training, restantes = train_test_split_pandas(dwine)
    print(training)
    print(restantes)


if __name__ == "__main__":
    main()
