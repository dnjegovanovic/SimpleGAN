import tensorflow as tf


def data_preprocess(ex, mode='uniform', z_size=20):
    """
    Convert image dtype and scale imge in range [-1,1]
    :param z_size:
    :param ex:
    :param mode:
    :return:
    """
    image = ex['image']
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, [-1])
    image = image * 2 - 1.0

    if mode == 'uniform':
        input_z = tf.random.uniform(
            shape=(z_size,), minval=-1.0, maxval=1.0
        )

    elif mode == 'normal':
        input_z = tf.random.normal(shape=(z_size,))

    return input_z, image
