from tensorflow.keras.layers import MaxPool2D, AvgPool2D


def pool(x, pool_type, data_format='channels_last'):
    """Apply pooling operation (via Tensorflow) to input Numpy array x.
    x should be 4-dimensional: N x W x H x C ('channels_last') or N x C x W x H ('channels_first')
    Pooling is applied on W and H dimensions.

    """
    pool_op = {
        'max_4': MaxPool2D(pool_size=(4, 4), strides=(2, 2), data_format=data_format),
        'max_16': MaxPool2D(pool_size=(16, 16), strides=(4, 4), data_format=data_format),
        'avg_4': AvgPool2D(pool_size=(4, 4), strides=(2, 2), data_format=data_format),
        'avg_16': AvgPool2D(pool_size=(16, 16), strides=(4, 4), data_format=data_format),
    }[pool_type]

    return pool_op(x.astype("float32")).numpy()
