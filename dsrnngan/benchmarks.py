import cv2
import numpy as np
import properscoring as ps

import ecpoint
import rainfarm
from crps import crps_ensemble

ecpointCDF = None


def mse(a, b):
    return ((a-b)**2).mean()


def mae(a, b):
    return (np.abs(a-b)).mean()


def rainfarmensemble(indata, ens_size=100):
    ans = []
    for i in range(ens_size):
        ans.append(rainfarmmodel(indata))
    return np.stack(ans, axis=-1)


def rainfarmmodel(indata):
    if type(indata) == dict:
        data = indata['lo_res_inputs']
    else:
        data = indata
    try:
        if len(data.shape) == 2:
            return rainfarm.rainfarm_downscale(data, ds_factor=10)
        else:
            ans = []
            for i in range(data.shape[0]):
                ans.append(rainfarm.rainfarm_downscale(data[i, ...], ds_factor=10))
            return np.stack(ans, axis=0)
    except:  # noqa
        print(f"Largest value in input array was: {data.max()}")
        dta_shape = data.shape
        dta_shape[-1] = 940
        dta_shape[-2] = 940
        return np.zeros(dta_shape)


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


def ecpointmodel(indata, assesslog=False, permute=True, ensemble_size=100, data_format="channels_last"):
    global ecpointCDF
    if type(indata) == dict:
        data = indata['lo_res_inputs']
    else:
        data = indata

    if ecpointCDF is None:
        ecpointCDF = ecpoint.loadcdf(name='CDF3YIFS.txt')
    return ecpoint.predictupscale(raw_inputs=data,
                                  cdf=ecpointCDF,
                                  logout=assesslog,
                                  permute=permute,
                                  ensemble_size=ensemble_size,
                                  data_format=data_format)


def ecpointPDFmodel(indata, assesslog=False, data_format="channels_last"):
    global ecpointCDF
    if type(indata) == dict:
        assert indata['lo_res_inputs'].log_precip is False
        data = indata['lo_res_inputs']
    else:
        data = indata
    if ecpointCDF is None:
        ecpointCDF = ecpoint.loadcdf(name='CDF3YIFS.txt')
    return ecpoint.predictupscalecdf(raw_inputs=data, cdf=ecpointCDF,
                                     logout=assesslog, data_format=data_format)


def ecpointboxensmodel(indata, assesslog=False, permute=True, ensemble_size=100, data_format="channels_last"):
    global ecpointCDF
    if type(indata) == dict:
        assert indata['lo_res_inputs'].log_precip is False
        data = indata['lo_res_inputs']
    else:
        data = indata
    if ecpointCDF is None:
        ecpointCDF = ecpoint.loadcdf(name='CDF3YIFS.txt')
    return ecpoint.predictthenupscale(raw_inputs=data,
                                      cdf=ecpointCDF,
                                      logout=assesslog,
                                      permute=permute,
                                      ensemble_size=ensemble_size,
                                      data_format=data_format)


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


def assessmodels(data_generator, models,
                 era_prec_index=1,
                 loss=mse,
                 verbose=True):
    assert type(models) == list
    error = np.zeros(len(models))
    # for i in range(data_generator.__len__()):
    for i, dta in enumerate(data_generator):
        if verbose and ((i % 10) == 0):
            print(i)
        nim = dta[1]['output']
        for j, model in enumerate(models):
            if model.__name__ == 'ecpointmodel':
                pred = model(dta[0]['lo_res_inputs'])
            else:
                pred = model(dta[0]['lo_res_inputs'][..., era_prec_index])
            error[j] += loss(nim, pred)
    error = error / data_generator.__len__()
    return error


def probablisticassessmodels(data_generator, probablistic_models,
                             loss=mean_crps,
                             era_prec_index=1,
                             verbose=True):
    assert type(probablistic_models) == list
    rawscore = np.zeros(len(probablistic_models))
    # for i in range(data_generator.__len__()):
    for i, dta in enumerate(data_generator):
        if verbose and ((i % 10) == 0):
            print(i)
        # dta = data_generator.__getitem__(i)
        for j, model in enumerate(probablistic_models):
            if model.__name__ == 'ecpointCDFmodel':
                pred = model(dta[0]['lo_res_inputs'])
            else:
                pred = model(dta[0]['lo_res_inputs'][..., era_prec_index])
            scr = loss(dta[1]['output'], pred)
            rawscore[j] += scr.mean()
    rawscore = rawscore / data_generator.__len__()
    return rawscore
