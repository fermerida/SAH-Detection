

from FileManagement import File
from Logistic_Regression.Model import Model
from Logistic_Regression.Data import Data
from Logistic_Regression import Plotter
import csv
import sys
import numpy as np

model1 = None
route_train = ""
route_test=""

def filemanage(train,test):
    global route_train
    route_train=train
    global route_test
    route_test = test
    menu()


def graficar():
    # Se grafican los entrenamientos
    if model1 is None:
        print()
        print("Debe entrenar un modelo antes")
        menu()
    else:
        #Plotter.show_Model([model1, model2])
        Plotter.show_Model([model1])


def train():
    global model1
    #Cargando conjuntos de datos
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = File.load_dataset('datasets/train_catvnoncat.h5','datasets/test_catvnoncat.h5')

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


    #model2 = Model(train_set, test_set, reg=False, alpha=0.001, lam=150) #Baja más quitandole la regularización
    #model2.training()
    menu()


 

def menu(): 

    choice = input("""
                      1: Seleccionar ruta de dataset
                      2: Entrenar modelo
                      3: Imprimir Grafica de ultimo modelo
                      4: Verificar imagen
                      5: Salir

                      Seleccione una de las opciones: """)

    if choice == "1":
        filemanage("","")
    elif choice == "2":
        train()
    elif choice == "3":
        graficar()
    elif choice=="4":
        sys.exit
    elif choice=="5":
        sys.exit
    else:
        print("Debe ingresar alguna de las opciones disponibles")
        print("Intentelo de nuevo")
        menu()

menu()
