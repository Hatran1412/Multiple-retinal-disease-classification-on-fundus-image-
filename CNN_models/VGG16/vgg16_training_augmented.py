import os
from collections import Sequence

import tensorflow as tf
from model_factory import Factory, ModelTypes
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50, inception_v3, vgg16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import numpy as np
import secrets
from Utilities.rfmid import load_data
from Utilities.plotting import Plotter
from Utilities.kappa import FinalScore
from Utilities.predictions_writer import Prediction
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16

batch_size = 32
num_classes = 9
epochs = 50


class Generator(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, x_set, y_set, batch_size=256):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return np.math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def generator(train, labels):
    while True:
        for i in range(len(train)):
            yield train[i].reshape(1, 224, 224, 3), labels[i].reshape(1, 9)

def generator_validation(test, labels):
    while True:
        for i in range(len(test)):
            yield test[i].reshape(1, 224, 224, 3), labels[i].reshape(1, 9)

token = secrets.token_hex(16)
folder = r'output/vgg16/training_augmented'

newfolder = os.path.join(folder, token)
if not os.path.exists(newfolder):
    os.makedirs(newfolder)

defined_metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

factory = Factory((224, 224, 3), defined_metrics)
model = factory.compile(ModelTypes.vgg16)

(x_train, y_train), (x_validation, y_validation) = load_data(1)

x_validation_drawing = x_validation

x_train = vgg16.preprocess_input(x_train)
x_validation = vgg16.preprocess_input(x_validation)

class_names = ['N','DR','ARMD',"MH",'DN','MYA','TSLN',"ODC","O"]

# plot data input
plotter = Plotter(class_names)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=1)

train_datagen = Generator(x_train, y_train, batch_size)

# With Data Augmentation
history = model.fit_generator(generator=generator(x_train, y_train), steps_per_epoch=len(x_train),
                               epochs=epochs, verbose=1, callbacks=[callback], validation_data=generator_validation(x_validation, y_validation),
                              validation_steps=len(x_validation), shuffle=False )

print("saving")
model.save(os.path.join(newfolder, 'model_weights.h5'))

print("plotting")
plotter.plot_metrics(history, os.path.join(newfolder, 'plot1.png'), 2)

# Hide meanwhile for now
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig(os.path.join(newfolder, 'plot2.png'))
plt.show()


# display the content of the model
baseline_results = model.evaluate(x_validation, y_validation, verbose=2)
for name, value in zip(model.metrics_names, baseline_results):
    print(name, ': ', value)
print()

# test a prediction
validation_predictions_baseline = model.predict(x_validation)
plotter.plot_confusion_matrix_generic(y_validation, validation_predictions_baseline, os.path.join(newfolder, 'plot3.png'), 0)

# save the predictions
prediction_writer = Prediction(validation_predictions_baseline, 400, newfolder)
prediction_writer.save()
prediction_writer.save_all(y_validation)

# show the final score
score = FinalScore(newfolder)
score.output()

# plot output results
plotter.plot_output(validation_predictions_baseline, y_validation, x_validation_drawing, os.path.join(newfolder, 'plot4.png'))