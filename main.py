from model.ganmodel import *
from dataprocessing.dataprep import *
import numpy as np
import tensorflow_datasets as tfds


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

    return gen_model, disc_model

def test_data():
    gen_model, disc_model = simple_test_model()
    # define dataset
    mnist_bldr = tfds.builder('mnist')
    mnist_bldr.download_and_prepare()
    mnist = mnist_bldr.as_dataset(shuffle_files=False)

    mnist_trainset = mnist['train']
    mnist_trainset = mnist_trainset.map(data_preprocess)

    # Test passing data
    mnist_trainset = mnist_trainset.batch(32, drop_remainder=True)
    input_z, input_real = next(iter(mnist_trainset))
    print('input_z --shape:{}'.format(input_z.shape))
    print('input_real --shape:{}'.format(input_real.shape))

    g_output = gen_model(input_z)
    print('g_output --shape:{}'.format(g_output.shape))

    d_logits_real = disc_model(input_real)
    d_logits_fake = disc_model(g_output)
    print('d_logits_real --shape:{}'.format(d_logits_real.shape))
    print('d_logits_fake --shape:{}'.format(d_logits_fake.shape))



if __name__ == '__main__':

    #simple_test_model()
    test_data()