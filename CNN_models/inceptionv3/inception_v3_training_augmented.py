import os
from collections import Sequence

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import numpy as np
import secrets
from Utilities.rfmid import load_data
#from advanced_pipeline import normalize_vgg16
from Utilities.plotting import Plotter
from Utilities.kappa import FinalScore
from Utilities.predictions_writer import Prediction
import matplotlib.pyplot as plt

batch_size = 32
num_classes = 9
epochs = 30


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
            yield train_a[i].reshape(1, 224, 224, 3), labels[i].reshape(1, 9)

def generator_validation(validation, labels):
    while True:
        for i in range(len(validation)):
            yield validation[i].reshape(1, 224, 224, 3), labels[i].reshape(1, 9)

token = secrets.token_hex(16)
folder = r'output/inceptionv3/training_augmented'

newfolder = os.path.join(folder, token)
if not os.path.exists(newfolder):
    os.makedirs(newfolder)

base_model = inception_v3.InceptionV3

base_model = base_model(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
# model.summary()

tf.keras.utils.plot_model(model, to_file=os.path.join(newfolder, 'model_inception_v3.png'), show_shapes=True, show_layer_names=True)


#for layer in base_model.layers:
#    layer.trainable = False

defined_metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=defined_metrics)

(x_train, y_train), (x_validation, y_validation) = load_data(1)

x_validation_drawing = x_validation

x_train = inception_v3.preprocess_input(x_train)
x_validation = inception_v3.preprocess_input(x_validation)

#print(model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0))
class_names = ['N','DR','ARMD',"MH",'DN','MYA','TSLN',"ODC","O"]

# plot data input
plotter = Plotter(class_names)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)

# history = model.fit(x_train, y_train,
#           epochs=100,
#           batch_size=batch_size,
#           shuffle=True,
#           validation_data=(x_test, y_test), callbacks=[callback])

train_datagen = Generator(x_train, y_train, batch_size)

# With Data Augmentation
history = model.fit_generator(generator=generator(x_train, y_train), steps_per_epoch=len(x_train),
                               epochs=epochs, verbose=1, callbacks=[callback], validation_data=generator_validation(x_validation, y_validation),
                              validation_steps=len(x_validation), shuffle=False )

print("saving")
model.save(os.path.join(newfolder, 'model_weights.h5'))

print("plotting")
plotter.plot_metrics(history, os.path.join(newfolder, 'metrics.png'), 2)

# Hide meanwhile for now
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig(os.path.join(newfolder, 'acc.png'))
plt.show()


# display the content of the model
baseline_results = model.evaluate(x_validation, y_validation, verbose=2)
for name, value in zip(model.metrics_names, baseline_results):
    print(name, ': ', value)
print()

# test a prediction
validation_predictions_baseline = model.predict(x_validation)
plotter.plot_confusion_matrix_generic(y_validation, validation_predictions_baseline, os.path.join(newfolder, 'confusionmatrix.png'), 0)

# save the predictions
prediction_writer = Prediction(validation_predictions_baseline, 400, newfolder)
prediction_writer.save()
prediction_writer.save_all(y_validation)

# show the final score
score = FinalScore(newfolder)
score.output()

# plot output results
plotter.plot_output(validation_predictions_baseline, y_validation, x_validation_drawing, os.path.join(newfolder, 'example.png'))