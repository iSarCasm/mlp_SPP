import tensorflow as tf
import numpy as np
from scipy.spatial import distance
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, LeakyReLU
from keras import regularizers
from keras import losses
from keras.callbacks import TensorBoard

class MLPModel:
  def __init__(self, optimizer = None, loss = losses.mean_squared_error, hidden_units=32, hidden_layers=2, activation='linear', name = ''):
    self.model = Sequential()

    self.model.add(Dense(hidden_units, input_dim=34))
    self.model.add(LeakyReLU(alpha=0.1))

    for i in range(hidden_layers-1):
      self.model.add(Dense(hidden_units))
      self.model.add(LeakyReLU(alpha=0.1))

    self.model.add(Dense(34))
    self.model.add(Activation('linear'))

    self.model.compile(optimizer=optimizer,
              loss=loss,  metrics=['accuracy', 'binary_accuracy', 'mse', 'binary_crossentropy'])
    self.name = name + 'ex5 Nadam hu{} hl{} {}'.format(hidden_units, hidden_layers, 'leaky relu')
    self.tensorboard = TensorBoard(log_dir='logs/' + self.name)

  def fit(self, train_input, train_labels, validation, epochs=1000, bs=64):
    self.model.fit(train_input, train_labels, validation_data=validation, epochs=epochs, batch_size=bs, verbose=0, callbacks=[self.tensorboard])

  def evaluate(self, x = None, y = None):
    print(self.model.metrics_names)
    return self.model.evaluate(x = x, y = y)

  def predict(self, test_inputs=None, bs=64, valid_vectors = None):
    prediction = self.model.predict(test_inputs)
    prediction = np.round(prediction)

    if valid_vectors:
      closest = valid_vectors[0]
      for vec in valid_vectors:
        if distance.euclidean(prediction, vec) < distance.euclidean(prediction, closest):
          closest = vec
      prediction = closest

    return prediction