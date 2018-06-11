from time import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import losses
from keras.callbacks import TensorBoard

class Metrics:
  def __init__(self, predictions, test_labels):
    valid_routes = genfromtxt('valid_routes.csv', delimiter=',')
    self.predictions = predictions
    self.test_labels = test_labels
    self.example_cnt = predictions.shape[0]
    self.optimal_count = 0
    self.valid_count = 0
    unique_route_vecs = dict()
    unique_routes_cnt = len(valid_routes)
    for r in range(unique_routes_cnt):
        route_vec = tuple(valid_routes[r])
        unique_route_vecs[route_vec] = r

    optimal_per_route = [0] * len(valid_routes)
    for i in range(self.example_cnt):
        p = self.predictions[i]
        for vr in range(len(valid_routes)):
            if (valid_routes[vr]==p).all():
                self.valid_count += 1

        test_label = self.test_labels[i]

        if (p==test_label).all():
            self.optimal_count += 1
            indx = unique_route_vecs[tuple(test_label)]
            optimal_per_route[indx] += 1

  def opt(self):
    return self.optimal_count / self.example_cnt

  def corr(self):
    return self.valid_count / self.example_cnt

  def optimal_predictions(self):
    return '{} ({}/{})'.format(self.optimal_count / self.example_cnt, self.optimal_count, self.example_cnt)

  def valid_predictions(self):
    return '{} ({}/{})'.format(self.valid_count / self.example_cnt, self.valid_count, self.example_cnt)

  def optimal_count(self):
    return self.optimal_count

  def valid_count(self):
    return self.valid_count

  def optimal_predictions_distibution_graph(self, predictions):
    pass

  def valid_predictions_distibution_graph(self, predictions):
    pass