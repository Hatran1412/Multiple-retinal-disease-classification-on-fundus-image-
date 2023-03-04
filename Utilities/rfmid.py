import numpy as np


def load_data(index):

    if index == 0:
        x_train = np.load('training.npy')
        y_train = np.load('training_labels.npy')
        x_validation= np.load('validation.npy' )
        y_validation=np.load('validation_labels.npy')
    if index == 1:
        x_train = np.load('augmented' + '.npy')
        y_train = np.load('augmented_labels' + '.npy')
        x_validation= np.load('validation.npy' )
        y_validation=np.load('validation_labels.npy')
    if index == 2:
        x_test= np.load('testing.npy')
        y_test=np.load('testing_labels.npy')

    return (x_train, y_train), (x_test, y_test), (x_validation, y_validation)