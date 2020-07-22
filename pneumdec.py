import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.get_logger().setLevel('WARNING')

labels = ['PNEUMONIA', 'NORMAL']

train_dir = 'dataset/chest_xray/train'
test_dir = 'dataset/chest_xray/test'

# With data augmentation to prevent overfitting and handling the imbalance in dataset
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.2,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,
    vertical_flip=False
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

train_generator = train_datagen.flow_from_directory(
    train_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    color_mode='grayscale',
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    color_mode='grayscale',
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary'
)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=(150, 150, 1)))
model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same'))
model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    train_generator,
    epochs=12,
    steps_per_epoch=50,
)

model.evaluate(test_generator)
