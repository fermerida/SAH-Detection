

from FileManagement import File
from Logistic_Regression.Model import Model
from Logistic_Regression.Data import Data
from Logistic_Regression import Plotter
import csv
import sys
import numpy as np


def train():
    #Cargando conjuntos de datos
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = File.load_dataset()

    # Convertir imagenes a un solo arreglo
    train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T


    # Vean la diferencia de la conversion
    print('Original: ', train_set_x_orig.shape)
    print('Con reshape: ', train_set_x.shape)


    # Definir los conjuntos de datos
    train_set = Data(train_set_x, train_set_y, 255)
    test_set = Data(test_set_x, test_set_y, 255)

    # Se entrenan los modelos
    model1 = Model(train_set, test_set, reg=False, alpha=0.0001, lam=0)
    model1.training()


    model2 = Model(train_set, test_set, reg=False, alpha=0.001, lam=150) #Baja más quitandole la regularización
    model2.training()

    model2.predict()

    # Se grafican los entrenamientos
    Plotter.show_Model([model1, model2])
    #Plotter.show_Model([model1])
    #Plotter.show_Model([model2])



def menu(): 

    choice = input("""
                      1: Entrenar modelo
                      2: Imprimir Grafica de ultimo modelo
                      3: Verificar imagen

                      Seleccione una de las opciones: """)

    if choice == "1":
        train()
    elif choice == "2":
        graficar()
    elif choice=="3":
        sys.exit
    else:
        print("You must only select either A or B")
        print("Please try again")
        menu()

menu()
