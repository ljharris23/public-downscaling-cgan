import numpy as np


def mse(a, b):
    return ((a-b)**2).mean()


def mae(a, b):
    return (np.abs(a-b)).mean()


def nn_interp_model(data, upsampling_factor):
    return np.repeat(np.repeat(data, upsampling_factor, axis=-1), upsampling_factor, axis=-2)


def zeros_model(data, upsampling_factor):
    return nn_interp_model(np.zeros(data.shape), upsampling_factor)
