import numpy as np


class NoiseGenerator(object):
    def __init__(self, noise_shapes, batch_size=32, random_seed=None):
        self.noise_shapes = noise_shapes
        self.batch_size = batch_size
        self.prng = np.random.RandomState(seed=random_seed)

    def noise(self, shape, mean, std):
        shape = (self.batch_size,) + shape
        n = self.prng.randn(*shape).astype(np.float32)
        # n = np.zeros(shape, dtype=np.float32)
        if std != 1.0:
            n *= std
        if mean != 0.0:
            n += mean
        return n

    def __call__(self, mean=0.0, std=1.0):
        return self.noise(self.noise_shapes, mean, std)
