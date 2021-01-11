#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import numpy as np
import h5py

source = None

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")



    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # entradas de entrenamiento
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # salidas de entrenamiento



    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # entradas de prueba
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # salidas de prueba


    #Les aplica reshape, convierte al arreglo en un arreglo de areglos
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, ['No Gato', 'Gato']
