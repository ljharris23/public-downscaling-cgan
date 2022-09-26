import os

import numpy as np

import read_config
from data import ifs_hours, load_ifs_nimrod_batch, get_dates


DEFAULT_FER_BINS = [
        -1.1,
        -0.99,
        -0.75,
        -0.5,
        -0.25,
        0.25,
        0.5,
        0.75,
        1,
        1.5,
        2,
        3,
        5,
        10,
        25,
        50,
        1000,
    ]
DEFAULT_FER_BINS = np.array(DEFAULT_FER_BINS)
bin_widths = np.diff(DEFAULT_FER_BINS)
bin_centres = (DEFAULT_FER_BINS[1:] + DEFAULT_FER_BINS[:-1])/2
bin_centres[0] = -1.0

data_paths = read_config.get_data_paths()

breakpoints_path = data_paths["ECPoint"]["breakpoints_path"]
save_tmp_loc = data_paths["ECPoint"]["save_tmp_loc"]
save_cdf_loc = data_paths["ECPoint"]["save_cdf_loc"]

breakpoints = np.loadtxt(breakpoints_path, skiprows=1, delimiter=",")
breakpoints_ma = np.ma.masked_where(np.logical_or(breakpoints == 9999, breakpoints == -9999), breakpoints)

breakpoints_hourly = breakpoints.copy()
breakpoints_hourly[breakpoints_hourly==9999] = 9999999  # Extend bounds  # noqa: E225
breakpoints_hourly[breakpoints_hourly==-9999] = -9999999  # Extend bounds  # noqa: E225
breakpoints_hourly[:, [3, 4]] = breakpoints_hourly[:, [3, 4]]/12  # Rescale Cape & cdir  # noqa: E225
# breakpoints_hourly[:, [3, 4]] = breakpoints_hourly[:, [3, 4]]
breakpoints_hourly = breakpoints_hourly.astype(np.single)

era_fields = ['prc', 'pr', 'u700', 'v700', 'cape', 'cdir']
ifs_fields = ['cp', 'tp', 'u700', 'v700', 'cape', 'cdir']
nbreak = breakpoints_hourly.shape[0]


def getALLdata(date, hour,
               ifs_fields=['cp', 'tp', 'u700', 'v700', 'cape', 'cdir'],
               tp_axis=1):
    # Load data
    inp, nim = load_ifs_nimrod_batch([date], ifs_fields=ifs_fields,
                                     hour=hour, norm=False,
                                     constants=None, crop=True)
    crs_fct = 10
    nim_rs = nim.reshape((nim.shape[0],
                          nim.shape[1]//crs_fct,
                          crs_fct,
                          nim.shape[2]//crs_fct,
                          crs_fct))
    nim_rs2 = np.moveaxis(nim_rs, -3, -2)
    nim = np.reshape(nim_rs2, (nim_rs2.shape[0], nim_rs2.shape[1],
                               nim_rs2.shape[2], nim_rs2.shape[3]*nim_rs2.shape[4]))
    # Find those non-zero and only output these
    nonzero = (inp[:, :, :, tp_axis] > 0.01)
    filtered_era = remapERA(inp[nonzero, :])
    nimrod_eragrid_nonzero = nim[nonzero, :]
    return filtered_era, nimrod_eragrid_nonzero


def classifyALLerror(date, hour, bpoints=breakpoints_hourly,
                     fields=['cp', 'tp', 'u700', 'v700', 'cape', 'cdir'],
                     priordata=None):
    # Load data
    filtered_era, nimrod_eragrid_nonzero = getALLdata(date, hour, ifs_fields=fields)
    bins = filtbreak(filtered_era, bpoints)
    if priordata is None:
        priordata = [[] for _ in range(bpoints.shape[0])]
    for i in range(bpoints.shape[0]):
        pointset = (bins == i)
        if (pointset.sum() > 0):
            era_prec = np.expand_dims(filtered_era[bins==i, 1], -1)  # noqa: E225
            errors = (nimrod_eragrid_nonzero[bins==i, :] - era_prec)/era_prec  # noqa: E225
            priordata[i] += list(errors.flatten())

    return priordata


def saveclass(dates):
    for i, d in enumerate(dates):
        year = d[:4]
        print(i, d)
        ans = None
        for hour in ifs_hours:
            ans = classifyALLerror(d, hour=hour, bpoints=breakpoints_hourly, priordata=ans)
        for j, dta in enumerate(ans):
            if len(dta) > 0:
                merged = np.array(dta)
                np.savetxt(f"{save_tmp_loc}/{year}/Fer{j}_{i}.txt", merged)
    return


def buildecpoint(year):
    dates = get_dates(year)
    saveclass(dates)
    cdf = generateHRCDF(year)
    savecdf(cdf, name=f"CDF{year}IFS.txt")
    return


def loadgroup(idx, year=[2018]):
    alldata = []
    if type(year) in [str, int]:
        year = [year]
    for yr in year:
        for i in range(400):
            flename = f"{save_tmp_loc}/{yr}/Fer{idx}_{i}.txt"
            if i == 0:
                print(flename)
            if os.path.isfile(flename):
                alldata.append(np.loadtxt(flename))
    if len(alldata) == 0:
        return None
    return np.concatenate(alldata)


def generateHRCDF(year=2018, nBin=100):
    if type(year) in [int, str]:
        year = [year]
    cdfs = np.zeros((nbreak, nBin))
    for i in range(nbreak):
        print(i)
        dta = loadgroup(i, year=year)
        if dta is not None:
            cdfs[i, :] = makecdf(dta)
    return cdfs


def filtbreak(dta, breakp):
    # Assigns a bin based upon "breakpoints"
    nfields = (breakp.shape[1]-1)//2
    ngrid = dta.shape[0]
    bins = -np.ones(ngrid, dtype=int)
    for i in range(ngrid):
        for j in range(breakp.shape[0]):
            ksum = 0
            for k in range(5):
                if breakp[j, 2*k+1] <= dta[i, k] < breakp[j, 2*k+2]:
                    ksum += 1
                else:
                    break
            if ksum == nfields:
                bins[i] = j
                break
    return bins


def remapERA(dta, fields=5):
    # Move from raw ERA to ecPoint variables
    filtered_era = np.zeros(dta.shape[:-1] + (5,), dtype=np.single)
    # Convective fraction
    filtered_era[dta[..., 1]==0, 0] = 0  # noqa: E225
    prgz = dta[..., 1] > 0
    filtered_era[prgz, 0] = dta[prgz, 0]/dta[prgz, 1]
    # Precip, convert to mm
    filtered_era[..., 1] = dta[..., 1]  # *1000
    # Winds, get magnitude
    filtered_era[..., 2] = (dta[..., 2]**2 + dta[..., 3]**2)**0.5
    # Cape
    filtered_era[..., 3] = dta[..., 4]
    # Convert to hourly
    filtered_era[..., 4] = dta[..., 5]/3600
    return filtered_era


def getdata(date, era_fields=['prc', 'pr', 'u700', 'v700', 'cape', 'cdir']):
    # Load data
    assert False, "load_ifs_nimrod doesn't exist, need to fixup"
    dt = load_ifs_nimrod(date=date, era_fields=era_fields, hour=ifs_hours,
                         norm=False, constants=None)
    # Get coinciding nimrod values
    # Long term, this could sample all nearby gridpoints.
    nimrod_eragrid = dt[1][:, ::10, ::10]
    # Find those non-zero and only output these
    nonzero = (dt[0][:, :, :, 1] > 0.01)
    filtered_era = remapERA(dt[0][nonzero, :])
    nimrod_eragrid_nonzero = nimrod_eragrid[nonzero]
    return filtered_era, nimrod_eragrid_nonzero


def getdatadates(dates):
    # Loop over dates
    e = []
    n = []
    for i, d in enumerate(dates):
        et, nt = getdata(d)
        e.append(et)
        n.append(nt)
    return np.concatenate(e, axis=0), np.concatenate(n, axis=0)


def classifyBin(date, bpoints=breakpoints_hourly, era_fields=['prc', 'pr', 'u700', 'v700', 'cape', 'cdir']):
    # Load data
    filtered_era, nimrod_eragrid_nonzero = getdata(date, era_fields=era_fields)

    bins = filtbreak(filtered_era, bpoints)
    binned_bins, tmp = np.histogram(bins, bins=np.arange(bpoints.shape[0]+1))

    histdata = np.zeros((bpoints.shape[0], DEFAULT_FER_BINS.shape[0]-1))
    for binnum in range(bpoints.shape[0]):
        hist, bins_edges = np.histogram((nimrod_eragrid_nonzero[bins==binnum]-  # noqa: E225
                                        filtered_era[bins==binnum, 1])/filtered_era[bins==binnum, 1],  # noqa: E225
                                        bins=DEFAULT_FER_BINS)
        histdata[binnum, :] = hist
    return binned_bins, histdata


def classifyerror(date, bpoints=breakpoints_hourly, era_fields=['prc', 'pr', 'u700', 'v700', 'cape', 'cdir'],
                  priordata=None):
    # Load data
    filtered_era, nimrod_eragrid_nonzero = getdata(date, era_fields=era_fields)

    bins = filtbreak(filtered_era, bpoints)
    if priordata is None:
        priordata = [[] for _ in range(bpoints.shape[0])]
    for i in range(bpoints.shape[0]):
        pointset = (bins == i)
        if (pointset.sum() > 0):
            errors = list((nimrod_eragrid_nonzero[bins==i] -  # noqa: E225
                           filtered_era[bins==i, 1])/filtered_era[bins==i, 1])  # noqa: E225
            priordata[i] += errors

    return priordata


def generateFERs(dates, bpoints=breakpoints_hourly):
    for i, d in enumerate(dates):
        print(i)
        bb_tmp, hs_tmp = classifyBin(d, bpoints=breakpoints_hourly)
        if i == 0:
            bins = bb_tmp
            hs = hs_tmp
        else:
            bins += bb_tmp
            hs += hs_tmp
    fers = hs / np.expand_dims(np.maximum(hs.sum(axis=1), np.ones(hs.shape[0])), -1)
    return bins, fers


def generateCDF(dates, bpoints=breakpoints_hourly, nBin=100):
    dta = None
    for i, d in enumerate(dates):
        print(d)
        dta = classifyerror(d, bpoints=bpoints, priordata=dta)
    cdfs = np.zeros((bpoints.shape[0], nBin))
    for i, sample in enumerate(dta):
        if len(sample) > 100:
            cdfs[i, :] = makecdf(sample, nBin)
    return cdfs


def classifyPred(era, fers, bpoints=breakpoints_hourly):
    bins = filtbreak(era, breakp=bpoints)
    return ecpointpred(era[:, 1], bins, fers)


def ecpointpred(prec, model, pdf, nsamples=100):
    pred = np.zeros((prec.shape[0], nsamples))
    rng = np.arange(pdf.shape[1], dtype=int)
    for i in range(prec.size):
        selection = np.random.choice(rng,
                                     size=nsamples,
                                     replace=True, p=pdf[int(model[i]), :])
        perturbations = bin_centres[selection]
        pred[i, :] = prec[i] * (1 + perturbations)
    return pred


def plotferdist(bb, hs):
    mns = []
    for i in range(breakpoints_ma.shape[0]):
        if bb[i] > 10000:
            pdf = hs[i, :]/(hs[i, :]).sum()
            mn = 1+((bin_centres*pdf)).sum()  # /(DEFAULT_FER_BINS[-1]-DEFAULT_FER_BINS[0])
            mns.append(mn)
    # plt.hist(mns)  # plt not imported?


def makecdf(error, nBin=100):
    import math
    a = np.arange(nBin)
    rep_error = np.zeros((nBin,))
    error_sorted = np.sort(error)
    centre_bin = (((2.0 * a) + 1) / (2.0 * nBin)) * len(error_sorted)
    for k in range(nBin):
        val = centre_bin[k]
        low, up = math.floor(val), math.ceil(val)
        if len(error_sorted) == 0:
            rep_error[k] = -1
            continue
        elif len(error_sorted) == 1:
            low = up = 0
        elif up >= len(error_sorted):
            up = len(error_sorted) - 1
            low = up - 1

        low_val = error_sorted[low]
        up_val = error_sorted[up]
        w_low, w_up = 1 - abs(val - low), 1 - abs(val - up)
        rep_error[k] = ((low_val * w_low) + (up_val * w_up)) / (w_low + w_up)
    error = None
    error_sorted = None
    return rep_error


def savecdf(cdf, name):
    filename = f'{save_cdf_loc}/{name}'
    np.savetxt(filename, cdf)
    return


def loadcdf(name, fixempty=True):
    filename = f'{save_cdf_loc}/{name}'
    if fixempty:
        cdfv2 = np.loadtxt(filename).astype(np.single)
        zeros = (cdfv2[:, -1] == 0)
        nonzeros = (cdfv2[:, -1] != 0)
        cdfv3 = cdfv2.copy()
        av = cdfv2[nonzeros, :].mean(axis=0)
        cdfv3[zeros, :] = av
        return cdfv3
    else:
        return np.loadtxt(filename).astype(np.single)


def predict(raw_inputs=None, cdf=None, logout=False):
    assert raw_inputs is not None
    output = raw_inputs[:, :, :, 1]
    proc_inputs = remapERA(raw_inputs.reshape((-1, raw_inputs.shape[-1])))
    bins = filtbreak(proc_inputs, breakpoints_hourly).astype(int)
    selection = np.random.choice(100,
                                 size=proc_inputs.shape[0],
                                 replace=True).astype(int)
    pred = cdf[bins, selection]
    output *= (1 + pred)
    if logout:
        return np.log10(1+output)
    else:
        return output


def predictcdf(raw_inputs=None,
               cdf=None, logout=False):
    assert raw_inputs is not None
    output = raw_inputs[:, :, :, 1]
    inputs = raw_inputs.reshape((-1, raw_inputs.shape[-1]))
    proc_inputs = remapERA(inputs)
    bins = filtbreak(proc_inputs, breakpoints_hourly).astype(int)
    pred = cdf[bins, :].reshape(output.shape+(100,))
    output = np.repeat(output[..., np.newaxis], 100, axis=-1)
    output *= (1 + pred)
    if logout:
        return np.log10(1+output)
    else:
        return output


def predictupscale(raw_inputs=None,
                   cdf=None,
                   logout=False,
                   permute=True,
                   ensemble_size=100,
                   data_format='channels_last'):
    assert raw_inputs is not None
    assert data_format in ("channels_first", "channels_last")
    if permute:
        assert ensemble_size == 100

    from numpy.random import default_rng
    rng = default_rng()

    # raw_inputs is batch_size x H x W x n_fields (6)
    proc_input = remapERA(raw_inputs[:, :, :, :])  # returns batch_size x H x W x 5

    # filtbreak expects pixels to be unrolled
    bins = filtbreak(np.reshape(proc_input, (-1, 5)), breakpoints_hourly)
    bins = np.reshape(bins, raw_inputs.shape[:3])
    # scale IFS precip up to 940x940
    imgout = np.repeat(np.repeat(raw_inputs[:, :, :, 1], 10, axis=1), 10, axis=2)
    binslarge = np.repeat(np.repeat(bins, 10, axis=1), 10, axis=2)

    # add ensemble dimension
    if data_format == 'channels_first':
        output = np.repeat(imgout[:, np.newaxis, :, :], ensemble_size, axis=1)
        # no need to repeat binslarge; numpy broadcasting will handle it
        binslarge[:, np.newaxis, :, :]
        if permute:
            selection = np.zeros(output.shape, dtype=np.int64)
            for ii in range(imgout.shape[0]):
                for jj in range(imgout.shape[1]):
                    for kk in range(imgout.shape[2]):
                        selection[ii, :, jj, kk] = rng.permutation(100)
        else:
            selection = rng.integers(low=0, high=100, size=output.shape, endpoint=False)

    elif data_format == 'channels_last':
        output = np.repeat(imgout[:, :, :, np.newaxis], ensemble_size, axis=-1)
        # no need to repeat binslarge; numpy broadcasting will handle it
        binslarge = binslarge[:, :, :, np.newaxis]

        if permute:
            selection = np.zeros(output.shape, dtype=np.int64)
            for ii in range(imgout.shape[0]):
                for jj in range(imgout.shape[1]):
                    for kk in range(imgout.shape[2]):
                        selection[ii, jj, kk, :] = rng.permutation(100)
        else:
            selection = rng.integers(low=0, high=100, size=output.shape, endpoint=False)

    pred = cdf[binslarge, selection]
    output *= (pred + 1.0)
    if logout:
        return np.log10(1+output)
    else:
        return output


def predictupscalecdf(raw_inputs=None,
                      cdf=None, logout=False,
                      data_format='channels_last'):
    assert data_format in ("channels_first", "channels_last")

    ans = predictcdf(raw_inputs=raw_inputs, cdf=cdf, logout=logout)
    # ans is batch_size x 94 x 94 x 100
    if data_format == "channels_first":
        ans = np.moveaxis(ans, -1, 1)  # move to batch x 100 x 94 x 94
        upscaled_ans = np.repeat(np.repeat(ans, 10, axis=-1), 10, axis=-2)
    elif data_format == "channels_last":
        upscaled_ans = np.repeat(np.repeat(ans, 10, axis=-2), 10, axis=-3)
    # upscaled_ans = upscaled_ans
    return upscaled_ans  # [:,4:-5:,4:-5,:]


def predictthenupscale(raw_inputs=None,
                       cdf=None,
                       logout=False,
                       permute=True,
                       ensemble_size=100,
                       data_format='channels_last'):
    assert data_format in ("channels_first", "channels_last")
    if permute:
        assert ensemble_size == 100

    ans = predictcdf(raw_inputs=raw_inputs, cdf=cdf, logout=logout)
    # ans is batch_size x 94 x 94 x 100

    from numpy.random import default_rng
    rng = default_rng()

    if data_format == "channels_first":
        if permute:
            selection = np.zeros((ans.shape[0], 100, ans.shape[1], ans.shape[2]),
                                 dtype=np.int64)
            for ii in range(ans.shape[0]):
                for jj in range(ans.shape[1]):
                    for kk in range(ans.shape[2]):
                        selection[ii, :, jj, kk] = rng.permutation(100)
        else:
            selection = rng.integers(low=0, high=ans.shape[3],
                                     size=(ans.shape[0], ensemble_size, ans.shape[1], ans.shape[2]),
                                     endpoint=False)
        # naive version of code is:
        # for ii in range(ans.shape[0]):
        #     for jj in range(ensemble_size):
        #         for kk in range(ans.shape[1]):
        #             for ll in range(ans.shape[2]):
        #                 out[ii, jj, kk, ll] = ans[ii, kk, ll, selection[ii, jj, kk, ll]]
        iarr = np.arange(ans.shape[0])
        jarr = np.arange(1)
        karr = np.arange(ans.shape[1])
        larr = np.arange(ans.shape[2])
        ix, _, kx, lx = np.ix_(iarr, jarr, karr, larr)
        small_output = ans[ix, kx, lx, selection]
        upscaled_ans = np.repeat(np.repeat(small_output, 10, axis=-1), 10, axis=-2)

    elif data_format == "channels_last":
        if permute:
            selection = np.zeros(ans.shape, dtype=np.int64)
            for ii in range(ans.shape[0]):
                for jj in range(ans.shape[1]):
                    for kk in range(ans.shape[2]):
                        selection[ii, jj, kk, :] = rng.permutation(100)
        else:
            selection = rng.integers(low=0, high=ans.shape[3],
                                     size=ans.shape[:-1] + (ensemble_size,),
                                     endpoint=False)
        # naive version of code is:
        # for ii in range(ans.shape[0]):
        #     for jj in range(ans.shape[1]):
        #         for kk in range(ans.shape[2]):
        #             for ll in range(ensemble_size):
        #                 out[ii, jj, kk, ll] = ans[ii, jj, kk, selection[ii, jj, kk, ll]]
        iarr = np.arange(ans.shape[0])
        jarr = np.arange(ans.shape[1])
        karr = np.arange(ans.shape[2])
        larr = np.arange(1)
        ix, jx, kx, _ = np.ix_(iarr, jarr, karr, larr)
        small_output = ans[ix, jx, kx, selection]
        upscaled_ans = np.repeat(np.repeat(small_output, 10, axis=-2), 10, axis=-3)

    return upscaled_ans


def crps(data_generator, ecpointcdf, log=False):
    import properscoring as ps
    totscore = 0
    for i in range(data_generator.__len__()):
        print(i)
        dta = data_generator.__getitem__(i)
        pred = predictupscalecdf(dta[0]['lo_res_inputs'], cdf=ecpointcdf, logout=log)
        if log:
            scr = ps.crps_ensemble(np.log10(1+dta[1]['output']), pred)
        else:
            scr = ps.crps_ensemble(dta[1]['output'], pred)
        totscore += scr.mean()
    totscore /= data_generator.__len__()
    return totscore
