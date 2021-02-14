import tensorflow as tf
import numpy as np


def make_generator_net(num_hidden_layer=1,
                       num_hidden_units=100,
                       num_output_units=728):
    """
    Create generator for MNIST data set, simple FCN. Generate images.
    :param num_hidden_layer:
    :param num_hidden_units:
    :param num_output_units:
    :return:
    """

    model = tf.keras.Sequential()
    for i in range(num_hidden_layer):
        model.add(tf.keras.layers.Dense(
            units=num_hidden_units, use_bias=False
        ))
        model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(
        units=num_output_units, activation='tanh'
    ))

    return model


def make_discriminator_net(num_hidden_layer=1,
                           num_hidden_units=100,
                           num_output_units=1):
    """
    Create discriminator for image classification task
    :param num_hidden_layer:
    :param num_hidden_units:
    :param num_output_units:
    :return:
    """
    model = tf.keras.Sequential()
    for i in range(num_hidden_layer):
        model.add(tf.keras.layers.Dense(units=num_hidden_units))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Dense(
        units=num_output_units, activation=None
    ))

    return model
