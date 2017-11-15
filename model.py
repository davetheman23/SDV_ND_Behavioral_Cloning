import csv
import cv2
import numpy as np
import os

from enum import Enum
from keras.models import Sequential
from keras.layers import Activation, Lambda, Dense, Flatten, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# the absolute amount of angle adjustment for left or right images
STEER_ANGLE_ADJUSTMENT = 0.3


class CameraPosition(Enum):
    CENTER = 0
    LEFT = 1
    RIGHT = 2


def read_line(path):
    """A generator to read lines from a csv file"""
    with open(path, 'r') as f:
        for line in csv.reader(f):
            yield line


def correct_image_path(original_path):
    """
    Change the path of a sample image to the path expected

    Note: this assumes the data sample is stored in the same directory as this file.
          For example, if the data originally stored in "c:\windows\drive\data\IMG\some_image.png"
          this function will return "data/IMG/some_image.png"
    """
    path_parts = original_path.split("\\")
    return os.path.join(*path_parts[-3:])


def read_image_from_path(path):
    """Use Open CV to read a image and return an array"""
    if not os.path.exists(path):
        raise ValueError("Path '{}' does not exist!".format(path))
    return cv2.imread(path)


def flip_image(image):
    """
    Flip an image vertically

    Flip an image left-to-right and right-to-left will help augment the data.
    Basically double the amount of data available for training.
    """
    return cv2.flip(image, 1)


def adjust_angle(position, angle):
    """
    Adjust the steering angle based on the position of the camera when the image is taken.

    :param position: the position of the camera
    :param angle: the steering angle that was collected for the center camera
    """
    if position == CameraPosition.CENTER:
        return angle
    elif position == CameraPosition.LEFT:
        return angle + STEER_ANGLE_ADJUSTMENT
    elif position == CameraPosition.RIGHT:
        return angle - STEER_ANGLE_ADJUSTMENT
    else:
        raise ValueError("Position '{}' not supported!".format(position))


def generate_data(samples, batch_size=64):
    """
    Generator for data samples

    This is useful for generating data samples one batch at a time instead of reading all data in the memory,
    which could be very memory intensive.
    """
    # no need to stop, when all the samples are done, it starts from the beginning
    while 1:
        samples = shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset+batch_size]

            # images are inputs and measurements (i.e. steering angles) are outputs
            images = []
            measurements = []
            for line in batch_samples:
                # there are three cameras 0-center, 1-left, 2-right, looping through them to augment
                # the existing data samples for more training/testing data
                for i in range(0, 3):
                    image_path = correct_image_path(line[i])
                    angle = float(line[3])

                    image = read_image_from_path(image_path)
                    adjusted_angle = adjust_angle(CameraPosition(i), angle)

                    images.append(image)
                    measurements.append(adjusted_angle)
                    images.append(flip_image(image))
                    measurements.append(adjusted_angle * (-1.0))

            # turn the data into numpy array
            X_train = np.array(images)
            y_train = np.array(measurements)

            # pipe this batch of data to the caller of this generator
            yield shuffle(X_train, y_train)


def get_model(input_shape):
    # Build a sequential kera model that contains:
    # - 5 convolutional layers (with 5 by 5 filters and varying depths)
    # - 5 fully connected layers with softmax activation except the output layer (because
    #     the output needs both positive and negative values)
    # - loss is mean squared error, optimizer is adam
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100, activation='softmax'))
    model.add(Dense(50, activation='softmax'))
    model.add(Dense(10, activation='softmax'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    return model


def main():
    # ==== Dataset descriptions ==== #
    # data1 - 1 lap of centerline driving with keyboard controls
    # data2 - include 2.5 laps of driving that focus to stay in the middle of the lane
    # data3 and data5 - include a few curves and one lap of driving in the opposite direction
    # data4 - 2 more laps of really careful driving within lane
    # data7 - some curves that the car struggle with and focused on recovery driving
    data_folders = ['data1', 'data2', 'data3', 'data4', 'data5', 'data7']
    LOG_CSV = 'driving_log.csv'

    # collect all data samples from the folders
    data_samples = []
    for data_folder in data_folders:
        for line in read_line(os.path.join(data_folder, LOG_CSV)):
            data_samples.append(line)

    # split into training and validation data samples
    train_samples, validation_samples = train_test_split(data_samples, test_size=0.2)

    # get the data generator for both training and validation data samples
    train_generator = generate_data(train_samples, batch_size=100)
    validation_generator = generate_data(validation_samples, batch_size=100)

    # Figure out the sample of the input image by looking at one of the image from training data
    example_image = read_image_from_path(correct_image_path(train_samples[0][0]))

    # get a model (modeled after the nvida E2E model)
    print("Get and compile a model.")
    model = get_model(input_shape=example_image.shape)

    # fit the model by feeding into the training and validation data generators
    print("Starting modeling training.")
    history_object = model.fit_generator(train_generator,
                                         samples_per_epoch=len(train_samples)*8,
                                         validation_data=validation_generator,
                                         nb_val_samples=len(validation_samples)*8,
                                         nb_epoch=6,
                                         verbose=1)
    print("Finihsed training model.")

    # save the model for later retrivel
    model.save('model_steer03_more_data.h5')
    print("Saved model weights to disk.")


if __name__ == '__main__':
    main()
