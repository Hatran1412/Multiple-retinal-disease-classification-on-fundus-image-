import os
import tensorflow as tf
from odir_model_factory import Factory, ModelTypes
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
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
patience = 5

token = secrets.token_hex(16)
folder = r'output/vgg16/training'

new_folder = os.path.join(folder, token)

if not os.path.exists(new_folder):
    os.makedirs(new_folder)

defined_metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

factory = Factory((224, 224, 3), defined_metrics)
model = factory.compile(ModelTypes.vgg16)

(x_train, y_train), (x_validation, y_validation) = load_data(224)

x_validation_drawing = x_validation 

x_train = vgg16.preprocess_input(x_train)
x_validation = vgg16.preprocess_input(x_validation)

class_names = ['N','DR','ARMD',"MH",'DN','MYA','TSLN',"ODC","O"]

# plot data input
plotter = Plotter(class_names)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1)


history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True, #class_weight = class_weight,
                    validation_data=(x_validation, y_validation), callbacks=[callback])

print("saving")
model.save(os.path.join(new_folder, 'model_weights.h5'))

print("plotting")
plotter.plot_metrics(history, os.path.join(new_folder, 'metrics.png'), 2)

# Hide meanwhile for now
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig(os.path.join(new_folder, 'acc.png'))
plt.show()

# display the content of the model
baseline_results = model.evaluate(x_validation, y_validation, verbose=2)
for name, value in zip(model.metrics_names, baseline_results):
    print(name, ': ', value)
print()

# test a prediction
validation_predictions_baseline = model.predict(x_validation)
plotter.plot_confusion_matrix_generic(y_validation, validation_predictions_baseline, os.path.join(new_folder, 'plot3.png'), 0)

# save the predictions
prediction_writer = Prediction(validation_predictions_baseline, 400, new_folder)
prediction_writer.save()
prediction_writer.save_all(y_validation)

# show the final score
score = FinalScore(new_folder)
score.output()

# plot output results
plotter.plot_output(validation_predictions_baseline, y_validation, x_validation_drawing, os.path.join(new_folder, 'plot4.png'))