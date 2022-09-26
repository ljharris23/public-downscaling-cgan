import numpy as np
from scipy.optimize import root_scalar


class Rootfn:
    def __init__(self, array, freq):
        self.array = array
        self.freq = freq

    def f(self, x):
        return self.freq - np.sum(self.array > x)/self.array.size


def findthresh(array, freq):
    # given array (hi-res or low-res rainfall) and an event frequency,
    # return threshold s.t. rainfall > threshold occurs with approx frequency freq
    rootfn = Rootfn(array, freq)
    return root_scalar(rootfn.f, bracket=[0.0, 100.0], x0=3.0, xtol=0.001)
