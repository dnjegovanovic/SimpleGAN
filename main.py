from model.ganmodel import *
import numpy as np


def simple_test_model():
    """
    Test generator and discriminator
    :return:
    """

    img_size = (28, 28)
    z_size = 20
    model_z = 'uniform'

    gen_hidden_layers = 1
    gen_hidden_size = 100
    disc_hidden_layers = 1
    disc_hidden_size = 100

    tf.random.set_seed(1)

    gen_model = make_generator_net(num_hidden_layer=gen_hidden_layers,
                                   num_hidden_units=gen_hidden_size,
                                   num_output_units=np.prod(img_size))

    gen_model.build(input_shape=(None, z_size))
    gen_model.summary()

    disc_model = make_discriminator_net(num_hidden_layer=disc_hidden_layers,
                                       num_hidden_units=disc_hidden_size)

    disc_model.build(input_shape=(None, np.prod(img_size)))
    disc_model.summary()


if __name__ == '__main__':
    simple_test_model()
