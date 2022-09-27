""" File for handling data loading and saving. """
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import read_config


data_paths = read_config.get_data_paths()
NIMROD_PATH = data_paths["GENERAL"]["NIMROD_PATH"]
IFS_PATH = data_paths["GENERAL"]["IFS_PATH"]
CONSTANTS_PATH = data_paths["GENERAL"]["CONSTANTS_PATH"]
CONSTANTS_PATH_ORO = data_paths["GENERAL"]["CONSTANTS_PATH_ORO"]

all_ifs_fields = ['tp', 'cp', 'sp', 'tisr', 'cape', 'tclw', 'tcwv', 'u700', 'v700']
ifs_hours = np.array([i for i in range(5)] + [i for i in range(6, 17)] + [i for i in range(18, 24)])


def denormalise(x):
    """
    Undo log-transform of rainfall.  Also cap at 500 (feel free to adjust according to application!)
    """
    return np.minimum(10**x - 1, 500.0)


def get_dates(year):
    """
    Return dates where we have radar data
    """
    from glob import glob
    files = glob(f"{NIMROD_PATH}/{year}/*.nc")
    dates = []
    for f in files:
        dates.append(f[:-3].split('_')[-1])
    return sorted(dates)


def load_nimrod(date, hour, log_precip=False, aggregate=1):
    year = date[:4]
    data = xr.open_dataset(f"{NIMROD_PATH}/{year}/metoffice-c-band-rain-radar_uk_{date}.nc")
    assert hour+aggregate < 25
    y = np.array(data['unknown'][hour:hour+aggregate, :, :]).sum(axis=0)
    data.close()
    # The remapping of NIMROD left a few negative numbers, so remove those
    y[y < 0.0] = 0.0
    if log_precip:
        return np.log10(1+y)
    else:
        return y


def logprec(y, log_precip=True):
    if log_precip:
        return np.log10(1+y)
    else:
        return y


def load_hires_constants(batch_size=1):
    df = xr.load_dataset(CONSTANTS_PATH_ORO)
    # LSM is already 0:1
    lsm = np.array(df['LSM'])[:, ::-1, :]  # flip latitudes

    # Orography.  Clip below, to remove spectral artifacts, and normalise by max
    z = df['z'].data
    z = z[:, ::-1, :]  # flip latitudes
    z[z < 5] = 5
    z = z/z.max()

    df.close()
    # print(z.shape, lsm.shape)
    return np.repeat(np.stack([z, lsm], -1), batch_size, axis=0)


def load_ifs_nimrod_batch(batch_dates, ifs_fields=all_ifs_fields, log_precip=False,
                          constants=False, hour=0, norm=False):
    batch_x = []
    batch_y = []

    if type(hour) == str:
        if hour == 'random':
            hours = ifs_hours[np.random.randint(22, size=[len(batch_dates)])]
        else:
            assert False, f"Not configured for {hour}"
    elif np.issubdtype(type(hour), np.integer):
        hours = len(batch_dates)*[hour]
    else:
        hours = hour

    for i, date in enumerate(batch_dates):
        h = hours[i]
        batch_x.append(load_ifsstack(ifs_fields, date, h, log_precip=log_precip, norm=norm))
        batch_y.append(load_nimrod(date, h, log_precip=log_precip))

    if (not constants):
        return np.array(batch_x), np.array(batch_y)
    else:
        return [np.array(batch_x), load_hires_constants(len(batch_dates))], np.array(batch_y)


def load_ifs(ifield, date, hour, log_precip=False, norm=False):
    # Get the time required (compensating for IFS saving precip at the end of the timestep)
    time = datetime(year=int(date[:4]), month=int(date[4:6]), day=int(date[6:8]), hour=hour) + timedelta(hours=1)

    # Get the correct forecast starttime
    if time.hour < 6:
        tmpdate = time - timedelta(days=1)
        loaddate = datetime(year=tmpdate.year, month=tmpdate.month, day=tmpdate.day, hour=18)
        loadtime = '12'
    elif 6 <= time.hour < 18:
        tmpdate = time
        loaddate = datetime(year=tmpdate.year, month=tmpdate.month, day=tmpdate.day, hour=6)
        loadtime = '00'
    elif 18 <= time.hour < 24:
        tmpdate = time
        loaddate = datetime(year=tmpdate.year, month=tmpdate.month, day=tmpdate.day, hour=18)
        loadtime = '12'
    else:
        assert False, "Not acceptable time"
    dt = time - loaddate
    time_index = int(dt.total_seconds()//3600)
    assert time_index >= 1, "Cannot use first hour of retrival"
    loaddata_str = loaddate.strftime("%Y%m%d") + '_' + loadtime

    field = ifield
    if field in ['u700', 'v700']:
        fleheader = 'winds'
        field = field[:1]
    elif field in ['cdir', 'tcrw']:
        fleheader = 'missing'
    else:
        fleheader = 'sfc'

    ds = xr.open_dataset(f"{IFS_PATH}/{fleheader}_{loaddata_str}.nc")
    data = ds[field]
    field = ifield
    if field in ['tp', 'cp', 'cdir', 'tisr']:
        data = data[time_index, :, :] - data[time_index-1, :, :]
    else:
        data = data[time_index, :, :]

    y = np.array(data[::-1, :])  # flip latitudes
    data.close()
    ds.close()
    if field in ['tp', 'cp', 'pr', 'prl', 'prc']:
        # print('pass')
        y[y < 0] = 0.
        y = 1000*y
    if log_precip and field in ['tp', 'cp', 'pr', 'prc', 'prl']:
        # precip is measured in metres, so multiply up
        return np.log10(1+y)  # *1000)
    elif norm:
        return (y-ifs_norm[field][0])/ifs_norm[field][1]
    else:
        return y


def load_ifsstack(fields, date, hour, log_precip=False, norm=False):
    field_arrays = []
    for f in fields:
        field_arrays.append(load_ifs(f, date, hour, log_precip=log_precip, norm=norm))
    return np.stack(field_arrays, -1)


def getifsstats(field, year=2018):
    import datetime

    # create date objects
    begin_year = datetime.date(year, 1, 1)
    end_year = datetime.date(year, 12, 31)
    one_day = datetime.timedelta(days=1)
    next_day = begin_year

    mi = 0
    mx = 0
    mn = 0
    sd = 0
    nsamples = 0
    for day in range(0, 366):  # includes potential leap year
        if next_day > end_year:
            break
        for hour in ifs_hours:
            try:
                dta = load_ifs(field, next_day.strftime("%Y%m%d"), hour)
                mi = min(mi, dta.min())
                mx = max(mx, dta.max())
                mn += dta.mean()
                sd += dta.std()**2
                nsamples += 1
            except:  # noqa
                print(f"Problem loading {next_day.strftime('%Y%m%d')}, {hour}")
        next_day += one_day
    mn /= nsamples
    sd = (sd / nsamples)**0.5
    return mi, mx, mn, sd


def gen_ifs_norm(year=2018):

    """
    One-off function, used to generate normalisation constants, which are used to normalise the various input fields for training/inference.
    
    Depending on the field, we may subtract the mean and divide by the std. dev., or just divide by the max observed value.
    """

    import pickle
    stats_dic = {}
    for f in all_ifs_fields:
        stats = getifsstats(f, year)
        if f == 'sp':
            stats_dic[f] = [stats[2], stats[3]]
        elif f == "u700" or f == "v700":
            stats_dic[f] = [0, max(-stats[0], stats[1])]
        else:
            stats_dic[f] = [0, stats[1]]
    with open(f'{CONSTANTS_PATH}/IFSNorm{year}.pkl', 'wb') as f:
        pickle.dump(stats_dic, f, 0)
    return


def load_ifs_norm(year=2018):
    import pickle
    with open(f'{CONSTANTS_PATH}/IFSNorm{year}.pkl', 'rb') as f:
        return pickle.load(f)

try:
    ifs_norm = load_ifs_norm(2018)
except:  # noqa
    ifs_norm = None
