from __future__ import absolute_import, division, print_function, unicode_literals

import logging.config
import os

import tensorflow as tf
from absl import app
from keras_applications.inception_v3 import keras_utils
from tensorflow.keras.applications import inception_v3
from Utilities.rfmid import load_data
from Utilities.plotting import Plotter
from Utilities.kappa import FinalScore
from Utilities.predictions_writer import Prediction

def main(argv):
    print(tf.version.VERSION)
    image_size = 224
    new_folder = r'output/inceptionv3/testing'

    # load the data
    (x_test, y_test) = load_data(2)

    class_names = ['N','DR','ARMD',"MH",'DN','MYA','TSLN',"ODC","O"]

    # plot data input
    plotter = Plotter(class_names)
    plotter.plot_input_images(x_test, y_test)

    x_test_drawing = x_test

    # normalize input based on model
    x_test = inception_v3.preprocess_input(x_test)

    # load one of the test runs
    model = tf.keras.models.load_model(os.path.join(new_folder , 'model_weights.h5'))
    model.summary()

    # display the content of the model
    baseline_results = model.evaluate(x_test, y_test, verbose=2)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    # test a prediction
    test_predictions_baseline = model.predict(x_test)
    plotter.plot_confusion_matrix_generic(y_test, test_predictions_baseline, new_folder, 0)

    # save the predictions
    prediction_writer = Prediction(test_predictions_baseline, 400)
    prediction_writer.save()
    prediction_writer.save_all(y_test)

    # show the final score
    score = FinalScore(new_folder)
    score.output()

    # plot output results
    plotter.plot_output(test_predictions_baseline, y_test, x_test_drawing)


if __name__ == '__main__':
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('rfmid')
    app.run(main)
