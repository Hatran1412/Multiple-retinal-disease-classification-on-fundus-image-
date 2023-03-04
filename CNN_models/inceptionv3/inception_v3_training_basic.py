import os
import tensorflow as tf
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.python.keras.optimizer_v1 import SGD

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import secrets
from Utilities.rfmid import load_data
from Utilities.plotting import Plotter
from Utilities.kappa import FinalScore
from Utilities.predictions_writer import Prediction
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from sklearn.utils import class_weight
import numpy as np


batch_size = 32
num_classes = 9
epochs = 100
patience = 5


token = secrets.token_hex(16)
folder = r'output/inceptionv3/training'

new_folder = os.path.join(folder, token)

if not os.path.exists(new_folder):
    os.makedirs(new_folder)

base_model = inception_v3.InceptionV3

base_model = base_model(weights='imagenet', include_top=False)

# Comment this out if you want to train all layers
#for layer in base_model.layers:
#    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

tf.keras.utils.plot_model(model, to_file=os.path.join(new_folder, 'model_inception_v3.png'), show_shapes=True, show_layer_names=True)

defined_metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

# Adam Optimizer Example
# model.compile(loss='binary_crossentropy',
#               optimizer=Adam(lr=0.001),
#               metrics=defined_metrics)

# RMSProp Optimizer Example
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=defined_metrics)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
print('Configuration Start -------------------------')
print(sgd.get_config())
print('Configuration End -------------------------')
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=defined_metrics)

(x_train, y_train), (x_validation, y_validation) = load_data(0)

x_validation_drawing = x_validation

x_train = inception_v3.preprocess_input(x_train)
x_validation = inception_v3.preprocess_input(x_validation)

class_names = ['N','DR','ARMD',"MH",'DN','MYA','TSLN',"ODC","O"]

# plot data input
plotter = Plotter(class_names)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1)

#class_weight = class_weight.compute_class_weight('balanced', np.unique(x_train), x_train)

history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True, #class_weight= class_weight,
                    validation_data=(x_validation, y_validation), callbacks=[callback])

print("saving weights")
model.save(os.path.join(new_folder, 'model_weights.h5'))

print("plotting metrics")
plotter.plot_metrics(history, os.path.join(new_folder, 'metrics.png'), 2)

print("plotting accuracy")
plotter.plot_accuracy(history, os.path.join(new_folder, 'accuracy.png'))

print("display the content of the model")
baseline_results = model.evaluate(x_validation, y_validation, verbose=2)
for name, value in zip(model.metrics_names, baseline_results):
    print(name, ': ', value)
print()

# a prediction
validation_predictions_baseline = model.predict(x_validation)
print("plotting confusion matrix")
plotter.plot_confusion_matrix_generic(y_validation, validation_predictions_baseline, os.path.join(new_folder, 'confusion_matrix.png'), 0)

# save the predictions
prediction_writer = Prediction(validation_predictions_baseline, 400, new_folder)
prediction_writer.save()
prediction_writer.save_all(y_validation)

# show the final score
score = FinalScore(new_folder)
score.output()

# plot output results
plotter.plot_output(validation_predictions_baseline, y_validation, x_validation_drawing, os.path.join(new_folder, 'example.png'))
