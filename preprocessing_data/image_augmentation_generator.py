from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataGenerator:
    def data_augmentation(x_train, y_train, augment_size=25000):
        image_generator = ImageDataGenerator(
            rotation_range=10,
            zoom_range=1.1,
            width_shift_range=0.07,
            height_shift_range=0.07,
            brightness_range=[0.2,1.0],
            shear_range=0.25,
            horizontal_flip=False,
            vertical_flip=False,
            data_format="channels_last")
        # fit data for zca whitening
        image_generator.fit(x_train, augment=True)
        # get transformed images
        randidx = np.random.randint(x_train.shape[0], size=augment_size)
        x_augmented = x_train[randidx].copy()
        y_augmented = y_train[randidx].copy()
        x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
                                           batch_size=augment_size, shuffle=False).next()[0]
        # append augmented data to trainset
        x_train2 = np.concatenate((x_train, x_augmented))
        y_train2 = np.concatenate((y_train, y_augmented))
        return x_train2, y_train2