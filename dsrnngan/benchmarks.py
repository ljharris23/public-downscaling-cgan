import cv2
import numpy as np
import properscoring as ps
from crps import crps_ensemble


def mse(a, b):
    return ((a-b)**2).mean()


def mae(a, b):
    return (np.abs(a-b)).mean()


def lanczosmodel(indata):
    if type(indata) == dict:
        data = indata['lo_res_inputs']
    else:
        data = indata
    if len(data.shape) == 2:
        return cv2.resize(data,
                          dsize=(data.shape[0]*10, data.shape[1]*10),
                          interpolation=cv2.INTER_LANCZOS4)
    else:
        ans = []
        for i in range(data.shape[0]):
            ans.append(cv2.resize(data[i, ...],
                                  dsize=(data.shape[-2]*10, data.shape[-1]*10),
                                  interpolation=cv2.INTER_LANCZOS4))
        return np.stack(ans, axis=0)


def constantupscalemodel(data):
    reshaped_inputs = np.repeat(np.repeat(data, 10, axis=-1), 10, axis=-2)
    return reshaped_inputs


def zerosmodel(data):
    return constantupscalemodel(np.zeros(data.shape))


def mean_crps(obs, pred):
    answer = 0
    it = np.nditer(obs, flags=['multi_index'])
    for val in it:
        answer += ps.crps_ensemble(val, pred[it.multi_index])
    return answer / obs.size


def mean_crps2(obs, pred):
    return crps_ensemble(obs, pred).mean()


def mean_wasserstein_distance(obs, pred):
    from scipy.stats import wasserstein_distance
    answer = 0
    it = np.nditer(obs, flags=['multi_index'])
    idx = 0
    for val in it:
        answer += wasserstein_distance([val], pred[it.multi_index])
        idx += 1
    return answer / obs.size