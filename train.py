import time
import tensorflow_datasets as tfds
import pickle

from dataprocessing import dataprep as dp
from dataprocessing import visualizeresult as vis

from model.ganmodel import *

if tf.test.is_gpu_available():
    device_name = tf.test.gpu_device_name()

else:
    device_name = 'cpu:0'

print(device_name)


def create_samples(g_model, input_z, btach_size, image_size):
    g_output = g_model(input_z, training=False)
    images = tf.reshape(g_output, (btach_size, *image_size))
    return (images + 1) / 2.0


def train_gan():
    img_size = (28, 28)
    z_size = 20
    model_z = 'uniform'
    batch_size = 64
    num_epoch = 100

    gen_hidden_layers = 1
    gen_hidden_size = 100
    disc_hidden_layers = 1
    disc_hidden_size = 100

    tf.random.set_seed(1)
    np.random.seed(1)

    if model_z == 'uniform':
        fixed_z = tf.random.uniform(shape=(batch_size, z_size), minval=-1, maxval=1)

    elif model_z == 'normal':
        fixed_z = tf.random.normal(shape=(batch_size, z_size))

    # set-up training dataset
    mnist_bldr = tfds.builder('mnist')
    mnist_bldr.download_and_prepare()
    mnist = mnist_bldr.as_dataset(shuffle_files=False)

    mnist_trainset = mnist['train']
    mnist_trainset = mnist_trainset.map(lambda ex: dp.data_preprocess(ex, mode=model_z))

    mnist_trainset = mnist_trainset.shuffle(10000)
    mnist_trainset = mnist_trainset.batch(batch_size, drop_remainder=True)

    # set-up the model
    with tf.device(device_name):
        gen_model = make_generator_net(num_hidden_layer=gen_hidden_layers,
                                       num_hidden_units=gen_hidden_size,
                                       num_output_units=np.prod(img_size))

        gen_model.build(input_shape=(None, z_size))

        disc_model = make_discriminator_net(num_hidden_layer=disc_hidden_layers,
                                            num_hidden_units=disc_hidden_size)

        disc_model.build(input_shape=(None, np.prod(img_size)))

    # Loss function and optimization
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    g_opt = tf.keras.optimizers.Adam()
    d_opt = tf.keras.optimizers.Adam()

    all_losses = []
    all_d_vals = []
    epoch_samples = []

    start_time = time.time()
    for epoch in range(1, num_epoch + 1):
        epoch_losses, epoch_d_vals = [], []

        for i, (input_z, input_real) in enumerate(mnist_trainset):
            # Compute generators loss
            with tf.GradientTape() as g_tape:
                g_output = gen_model(input_z)
                d_logits_fake = disc_model(g_output, training=True)
                labels_real = tf.ones_like(d_logits_fake)
                g_loss = loss_fn(y_true=labels_real, y_pred=d_logits_fake)

            # compute gradients of g_loss
            g_grads = g_tape.gradient(g_loss, gen_model.trainable_variables)

            # Optimization: Applay the gradients
            g_opt.apply_gradients(grads_and_vars=zip(g_grads, gen_model.trainable_variables))

            # Compute discriminator loss
            with tf.GradientTape() as d_tape:
                d_logits_real = disc_model(input_real, training=True)

                d_labels_real = tf.ones_like(d_logits_real)

                d_loss_real = loss_fn(
                    y_true=d_labels_real, y_pred=d_logits_real)

                d_logits_fake = disc_model(g_output, training=True)
                d_labels_fake = tf.zeros_like(d_logits_fake)

                d_loss_fake = loss_fn(
                    y_true=d_labels_fake, y_pred=d_logits_fake)

                d_loss = d_loss_real + d_loss_fake

            # Compute the gradients
            d_grads = d_tape.gradient(d_loss, disc_model.trainable_variables)
            # Optimazie
            d_opt.apply_gradients(grads_and_vars=zip(d_grads, disc_model.trainable_variables))

            epoch_losses.append((g_loss.numpy(), d_loss.numpy(), d_loss_real.numpy(), d_loss_fake.numpy()))

            d_probs_real = tf.reduce_mean(tf.sigmoid(d_logits_real))
            d_probs_fake = tf.reduce_mean(tf.sigmoid(d_logits_fake))

            epoch_d_vals.append((d_probs_real.numpy(), d_probs_fake.numpy()))

        all_losses.append(epoch_losses)
        all_d_vals.append(epoch_d_vals)

        print(
            'Epoch {:03d} | ET {:.2f} min | Avg Losses >>'
            ' G/D {:.4f}/{:.4f} [D-Real: {:.4f} D-Fake: {:.4f}]'
                .format(
                epoch, (time.time() - start_time) / 60,
                *list(np.mean(all_losses[-1], axis=0))))

        epoch_samples.append(create_samples(gen_model, fixed_z, btach_size=batch_size, image_size=img_size).numpy())

    pickle.dump({'all_losses': all_losses,
                 'all_d_vals': all_d_vals,
                 'samples': epoch_samples},
                open('simple_-learning.pkl', 'wb'))

    gen_model.save('simple_gan_gen.h5')
    disc_model.save('simple_gan_disc.h5')

    vis.visuzlize_result(all_losses, all_d_vals, epoch_samples)
