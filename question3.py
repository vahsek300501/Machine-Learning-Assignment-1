import os
import numpy as np
import pandas as pd
import sklearn.neighbors
import sklearn.tree
from sklearn.metrics import *
import sklearn.metrics as metrics
import math
import matplotlib.pyplot as plt


class NaiveBayes:
  def __init__(self):
    pass


def readDataset(path):
  os.chdir(path)
  import mnist_reader
  x_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
  x_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')
  return x_train,y_train ,x_test,y_test

def transformData(path):
  x_train,y_train,x_test,y_test = readDataset(path)
  train_data_index = []
  test_data_index = []
  
  i = 0
  while(i < len(y_train)):
    if(y_train[i] == 1) or (y_train[i] == 2):
      train_data_index.append(i)
    i += 1
  
  j = 0
  while(j < len(y_test)):
    if (y_test[j] == 1) or (y_test[j] == 2):
      test_data_index.append(j)
    j += 1
  
  x_train,y_train = x_train[train_data_index],y_train[train_data_index]
  x_test,y_test = x_test[test_data_index],y_test[test_data_index]
  