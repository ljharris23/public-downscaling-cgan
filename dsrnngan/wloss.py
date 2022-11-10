import tensorflow as tf
from tensorflow.keras import backend as K


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred, axis=-1)


def denormalise(y_in):
    ten = tf.constant(10.0, dtype=tf.float32)
    one = tf.constant(1.0, dtype=tf.float32)
    return tf.subtract(tf.pow(ten, y_in), one)


def sample_crps(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(tf.expand_dims(y_true, axis=0) -
                                tf.expand_dims(y_pred, -1)))  # trailing dim on truth
    ensemble_size = y_pred.shape[0]
    coef = -1/(2*ensemble_size*ensemble_size)
    ens_var = coef * tf.reduce_mean(tf.reduce_sum(tf.abs(tf.expand_dims(y_pred, axis=0) -
                                                         tf.expand_dims(y_pred, axis=1)),
                                                  axis=(0, 1)))
    return mae + ens_var


def sample_crps_phys(y_true, y_pred):
    y_true = denormalise(y_true)
    y_pred = denormalise(y_pred)
    return sample_crps(y_true, y_pred)


def ensmean_MSE(y_true, y_pred):
    pred_mean = tf.reduce_mean(y_pred, axis=0)
    y_true_squ = tf.squeeze(y_true, axis=-1)
    return tf.reduce_mean(tf.math.squared_difference(pred_mean, y_true_squ))


def ensmean_MSE_phys(y_true, y_pred):
    y_true = denormalise(y_true)
    y_pred = denormalise(y_pred)
    return ensmean_MSE(y_true, y_pred)


def CL_chooser(CLtype):
    return {"CRPS": sample_crps,
            "CRPS_phys": sample_crps_phys,
            "ensmeanMSE": ensmean_MSE,
            "ensmeanMSE_phys": ensmean_MSE_phys}[CLtype]
