import os
import keras
from keras import losses
import numpy as np
from numpy import genfromtxt
from mlp_model import MLPModel
from metrics import Metrics
import csv, sys
from keras.utils import plot_model
from ann_visualizer.visualize import ann_viz;
from scipy.stats import hmean
import time

train_dataset_verysmall = genfromtxt('train_dataset_140.csv', delimiter=',')
train_dataset_small = genfromtxt('train_dataset_1400.csv', delimiter=',')
train_dataset_medium = genfromtxt('train_dataset_14000.csv', delimiter=',')
train_dataset_large = genfromtxt('train_dataset_49000.csv', delimiter=',')
cross_validation_dataset = genfromtxt('validation_dataset2.csv', delimiter=',')
test_dataset = genfromtxt('test_dataset2.csv', delimiter=',')

# Shuffle Train Data
np.random.shuffle(train_dataset_verysmall)
np.random.shuffle(train_dataset_small)
np.random.shuffle(train_dataset_medium)
np.random.shuffle(train_dataset_large)

# Split into Inputs and Labels
train_inputs_verysmall, train_labels_verysmall = np.hsplit(train_dataset_verysmall, 2)
train_inputs_verysmall = np.array(train_inputs_verysmall).reshape(len(train_dataset_verysmall), 34)
train_labels_verysmall = np.array(train_labels_verysmall).reshape(len(train_dataset_verysmall), 34)

train_inputs_small, train_labels_small = np.hsplit(train_dataset_small, 2)
train_inputs_small = np.array(train_inputs_small).reshape(len(train_dataset_small), 34)
train_labels_small = np.array(train_labels_small).reshape(len(train_dataset_small), 34)

train_inputs_medium , train_labels_medium  = np.hsplit(train_dataset_medium, 2)
train_inputs_medium = np.array(train_inputs_medium).reshape(len(train_dataset_medium), 34)
train_labels_medium = np.array(train_labels_medium).reshape(len(train_dataset_medium), 34)

train_inputs_large, train_labels_large = np.hsplit(train_dataset_large, 2)
train_inputs_large = np.array(train_inputs_large).reshape(len(train_dataset_large), 34)
train_labels_large = np.array(train_labels_large).reshape(len(train_dataset_large), 34)

cross_validation_inputs, cross_validation_labels = np.hsplit(cross_validation_dataset, 2)
cross_validation_inputs = np.array(cross_validation_inputs).reshape(len(cross_validation_dataset), 34)
cross_validation_labels = np.array(cross_validation_labels).reshape(len(cross_validation_dataset), 34)

test_inputs, test_labels = np.hsplit(test_dataset, 2)
test_inputs = np.array(test_inputs).reshape(len(test_dataset), 34)
test_labels = np.array(test_labels).reshape(len(test_dataset), 34)

# Normalize Data
gigabit_max_throughput = 125 * 1e6 # 1000 MB/s
gigabit_min_throughput = 1e2       # 100 bytes/s
norm = gigabit_max_throughput - gigabit_min_throughput

train_inputs_verysmall = (train_inputs_verysmall - gigabit_min_throughput) / norm
train_inputs_small = (train_inputs_small - gigabit_min_throughput) / norm
train_inputs_medium = (train_inputs_medium - gigabit_min_throughput) / norm
train_inputs_large = (train_inputs_large - gigabit_min_throughput) / norm
cross_validation_inputs = (cross_validation_inputs - gigabit_min_throughput) / norm
test_inputs = (test_inputs - gigabit_min_throughput) / norm

very_small_data = (train_inputs_verysmall, train_labels_verysmall)
small_data = (train_inputs_small, train_labels_small)
medium_data = (train_inputs_medium, train_labels_medium)
large_data = (train_inputs_large, train_labels_large)

valid_routes = genfromtxt('valid_routes.csv', delimiter=',')

configs = []

for data in [large_data]:
    for loss in [losses.mean_squared_error]:
        for activation in ['relu']:
            for hu in [410]:
                for hl in [2]:
                    c = { 'activation': activation, 'data': data, 'loss': loss, 'hu': hu, 'hl': hl }
                    configs.append(c)

print("{} total configs".format(len(configs)))

table = [['loss', 'activaiton', 'data size', 'hu', 'hl', 'optimal', 'valid']]
fr = open('MLP_results_adam.csv', 'w', newline='')
writer = csv.writer(fr)

progress = 0
current_progress = -1

for config in configs:
    current_progress += 1
    if current_progress < progress:
        print('skip')
        continue

    start = time.time()
    model = MLPModel(
        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=False),
        loss = config['loss'],
        hidden_units = config['hu'],
        hidden_layers = config['hl'],
        activation = config['activation'],
        name = "data{} ".format(config['data'][0].shape[0])
    )

    train_inputs = config['data'][0]
    train_labels = config['data'][1]

    batch = train_labels.shape[0]
    model.fit(train_inputs, train_labels, (cross_validation_inputs, cross_validation_labels), epochs = 100, bs = 64)

    train_prediction = model.predict(train_inputs)
    train_metrcis = Metrics(train_prediction, train_labels)

    crossvalidation_prediction = model.predict(cross_validation_inputs)
    validation_metrcis = Metrics(crossvalidation_prediction, cross_validation_labels)

    test_prediction = model.predict(test_inputs)
    test_metrcis = Metrics(test_prediction, test_labels)

    name = "adam {} d{} hu{} hl{}".format('leaky relu', batch, config['hu'], config['hl'])
    print(name)
    print("{} Training took {}".format(current_progress, time.time() - start))
    print("Optimal predictions: {}".format(train_metrcis.optimal_predictions()))
    print("Valid predictions: {}".format(train_metrcis.valid_predictions()))
    print("H-mean: {}".format(hmean([train_metrcis.opt(), train_metrcis.corr()])))
    print("Validation")
    print("Optimal predictions: {}".format(validation_metrcis.optimal_predictions()))
    print("Valid predictions: {}".format(validation_metrcis.valid_predictions()))
    print("H-mean: {}".format(hmean([validation_metrcis.opt(), validation_metrcis.corr()])))
    print(model.evaluate(x = test_inputs, y = test_labels))
    print("==================================\n")

    table_row = []
    table_row.append(config['loss'])
    table_row.append(config['activation'])
    table_row.append(batch)
    table_row.append(config['hu'])
    table_row.append(config['hl'])
    table_row.append(test_metrcis.optimal_count)
    table_row.append(test_metrcis.valid_count)
    table.append(table_row)

    directory = "MLP/{}".format(name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    f = open(directory + "/predictions", 'w')
    for pi in range(len(test_prediction)):
        f.write(str(test_prediction[pi]))
        f.write("\n")
        f.write(str(test_labels[pi]))
        f.write("\n")
        f.write("\n")
    f.close

    writer.writerow(table_row)


    fr.close
