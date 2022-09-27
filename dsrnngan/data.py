""" File for handling data loading and saving. """
import os
from datetime import datetime, timedelta
from meta import ensure_list

import numpy as np
import pandas as pd
import xarray as xr
# from dask.array.chunk import coarsen

import read_config


data_paths = read_config.get_data_paths()
ERA_PATH = data_paths["GENERAL"]["ERA_PATH"]
NIMROD_PATH = data_paths["GENERAL"]["NIMROD_PATH"]
IFS_PATH = data_paths["GENERAL"]["IFS_PATH"]
CONSTANTS_PATH = data_paths["GENERAL"]["CONSTANTS_PATH"]
CONSTANTS_PATH_OLD_ORO = data_paths["GENERAL"]["CONSTANTS_PATH_OLD_ORO"]
CONSTANTS_PATH_NEW_ORO = data_paths["GENERAL"]["CONSTANTS_PATH_NEW_ORO"]

all_era_fields = ['pr', 'prc', 'prl', 'tcrw', 'lwp', 'prw', 'cape', 'tosr', 'psl', 'u700', 'v700']
all_ifs_fields = ['tp', 'cp', 'sp', 'tisr', 'cape', 'tclw', 'tcwv', 'u700', 'v700']
ifs_hours = np.array([i for i in range(5)] + [i for i in range(6, 17)] + [i for i in range(18, 24)])
# Global constants
UK_LAT_LIMIT = (50, 60)
UK_LON_LIMIT = (-2, 12)


def denormalise(x):
    return np.minimum(10**x - 1, 500.0)


def select_precip_region(data, lat_limit, lon_limit):

    # Get the region's minimum and maximum latitudes and longitudes
    lat_min = lat_limit[0]
    lat_max = lat_limit[1]
    lon_min = lon_limit[0]
    lon_max = lon_limit[1]

    # Crop data to region
    data_region = data.isel({'latitude': np.logical_and(data.latitude > lat_min, data.latitude < lat_max),
                             'longitude': np.logical_and(data.longitude > lon_min, data.longitude < lon_max)})

    return data_region


def calculate_image_dimensions(lat_limit, lon_limit):

    # Calculate size of region, multiply by ten to convert degrees to pixels
    x_dim = 10*(lon_limit[1]-lon_limit[0])
    y_dim = 10*(lat_limit[1]-lat_limit[0])

    img_dim = np.array([x_dim, y_dim])

    return img_dim


def calculate_input_dimensions(img_dim, upscaling_factor):

    # Calculate input image dimensions after upscaling
    img_dim_input = img_dim//upscaling_factor

    return img_dim_input


def select_masks_region(masks, lat_limit, lon_limit):
    return [select_precip_region(mask, lat_limit, lon_limit) for mask in masks]


def add_masks(data, masks, mask_paths):
    for mask in masks:
        mask = xr.open_dataset(mask_paths)
        data = concatenate_mask_to_data(data, mask)
    return data


def concatenate_mask_to_data(data, mask):
    data = np.concatenate(data, mask, axis=-1)
    return data


def preprocess_masks(masks, lat_limit, lon_limit):
    masks_region = select_masks_region(masks, lat_limit, lon_limit)
    masks_region = [np.expand_dims(np.array(mask_region), axis=-1) for mask_region in masks_region]
    return masks_region


def load_masks_data(mask_paths):
    masks = []
    for mask_path in mask_paths:
        masks.append(xr.open_dataset(mask_path))
    return masks


def load_and_preprocess_masks(mask_paths, lat_limit=UK_LAT_LIMIT, lon_limit=UK_LON_LIMIT):
    masks = load_masks_data(mask_paths)
    return preprocess_masks(masks, lat_limit, lon_limit)


def load_precip_data(data_path, upscaling_factor, lat_limit=UK_LAT_LIMIT, lon_limit=UK_LON_LIMIT, masks=[]):

    data = xr.open_dataset(data_path)

    # Get a reference to the precip data
    precip = data['precipitationcal']

    # Select region according to latitude and longitude limits
    data_region = select_precip_region(precip, lat_limit, lon_limit)

    # Add 'channel' axis to the data
    data_region = np.expand_dims(np.array(data_region), axis=-1)

    # Add the masks to the data channels
    data_region = add_masks(data_region, masks)

    # Upscale data by the upscaling factor to provide input to network at first time instance
    # TODO: change time indexing when working with recurrent network
    # TODO: allow for more complex/noisy upscaling
    x = np.array(data_region[0, ::upscaling_factor, ::upscaling_factor])
    y = np.array(data_region[0])

    return (x, y)


def load_precip_data_batch(batch_data_paths, upscaling_factor, lat_limit=UK_LAT_LIMIT, lon_limit=UK_LON_LIMIT, masks=[]):
    batch_x = []
    batch_y = []

    for data_path in batch_data_paths:
        x, y = load_precip_data(data_path, upscaling_factor, lat_limit, lon_limit, masks)
        batch_x.append(x)
        batch_y.append(y)

    batch_x = np.array(batch_x, dtype="float32")
    batch_y = np.array(batch_y, dtype="float32")

    return batch_x, batch_y


def gather_files_in_dir(file_dir):
    """ Collect the paths of all the files in a given directory """
    file_ids = os.listdir(file_dir)
    file_paths = [os.path.join(file_dir, file_id) for file_id in file_ids]
    file_paths = [file_path for file_path in file_paths if ".nc" in file_path]

    return file_paths


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


def load_nimrod(date, hour, log_precip=False, aggregate=1, crop=None):
    year = date[:4]
    data = xr.open_dataset(f"{NIMROD_PATH}/{year}/metoffice-c-band-rain-radar_uk_{date}.nc")
    assert hour+aggregate < 25
    y = np.array(data['unknown'][hour:hour+aggregate, :, :]).sum(axis=0)
    data.close()
    # The remapping of NIMROD left a few negative numbers
    # So remove those
    y[y < 0] = 0
    if crop is not None and crop != 0:
        if type(crop) == tuple:
            y = y[crop[0]:crop[1], crop[0]:crop[1]]
        elif type(crop) == int:
            y = y[crop:-crop, crop:-crop]
    if log_precip:
        return np.log10(1+y)
    else:
        return y


def logprec(y, log_precip=True):
    if log_precip:
        return np.log10(1+y)
    else:
        return y


def load_nimrod_coarse(date, hour, log_precip=False, aggregate=1):
    nim_hr = load_nimrod(date, hour, log_precip=False, aggregate=aggregate)
    y = coarsen_nimrod(nim_hr)
    if log_precip:
        return np.log10(1+y)
    else:
        return y


def coarsen_nimrod(nim_hr):
    nim_hr2 = np.pad(nim_hr, ((12,), (12,)), mode="reflect")
    da = xr.DataArray(nim_hr2, dims=("x", "y"), coords={"x": np.arange(nim_hr2.shape[0]), "y": np.arange(nim_hr2.shape[1])})
    da2 = da.coarsen(x=25, y=25).mean()
    return da2.data


def load_era(field, date, hour, log_precip=False, norm=False, crop=2):
    year = date[:4]
    data = xr.open_dataarray(f"{ERA_PATH}/{year}/{field}{date}.nc")
    if crop is None or crop == 0:
        y = np.array(data[hour, :, :])
    else:
        y = np.array(data[hour, crop:-crop, crop:-crop])
    data.close()
    if log_precip and field in ['pr', 'prc', 'prl']:
        # ERA precip is measure in meters, so multiple up
        return np.log10(1+y*1000)
    elif norm:
        return (y-era_norm[field][0])/era_norm[field][1]
    else:
        return y


def load_erastack(fields, date, hour, log_precip=False, norm=False, crop=2,
                  aggregate=1):
    field_arrays = []
    for f in fields:
        for agg in range(aggregate):
            field_arrays.append(load_era(f, date, hour+agg, log_precip=log_precip, norm=norm, crop=crop))
    return np.stack(field_arrays, -1)


def load_era_nimrod(nimrodfile=None, date=None, era_fields=['pr', 'prc'], hour=0,
                    log_precip=False, norm=False, era_crop=2):
    if nimrodfile is not None:
        date = nimrodfile[:-3].split('_')[-1]
    return load_erastack(era_fields, date, hour, log_precip=log_precip, norm=norm, crop=era_crop),\
        load_nimrod(date, hour, log_precip=log_precip)


def load_hires_constants(batch_size=1, crop=False,
                         new_oro=True):
    df = xr.load_dataset(CONSTANTS_PATH_OLD_ORO)
    # LSM is already 0:1
    lsm = np.array(df['LSM'])[:, ::-1, :]
    if new_oro:
        df = xr.load_dataset(CONSTANTS_PATH_NEW_ORO)
        # Should rewrite this file to have increasing latitudes
        z = df['z'].data
        z = z[:, ::-1, :]
        z[z < 5] = 5
        # Normalise orography by max
        z = z/z.max()
    else:
        # Should rewrite this file to have increasing latitudes
        z = df['Z'].data
        z = z[:, ::-1, :]
        # z = np.array(df['Z'])[:, ::-1, :]
        # Normalise orography by max
        z = z/z.max()

    df.close()
    print(z.shape, lsm.shape)
    if crop:
        lsm = lsm[..., 5:-6, 5:-6]
        z = z[..., 5:-6, 5:-6]
    return np.repeat(np.stack([z, lsm], -1), batch_size, axis=0)


def load_era_nimrod_batch(batch_dates, era_fields, log_precip=False,
                          constants=False, hour=0, norm=False, era_crop=2):
    batch_x = []
    batch_y = []
    if hour == 'random':
        hours = np.random.randint(24, size=[len(batch_dates)])
    elif type(hour) == int:
        hours = len(batch_dates)*[hour]

    for i, date in enumerate(batch_dates):
        h = hours[i]
        batch_x.append(load_erastack(era_fields, date, h, log_precip=log_precip, norm=norm))
        batch_y.append(load_nimrod(date, h, log_precip=log_precip))

    if not constants:
        return np.array(batch_x), np.array(batch_y)
    else:
        return [np.array(batch_x), load_hires_constants(len(batch_dates))], np.array(batch_y)


def load_ifs_nimrod_batch(batch_dates, ifs_fields=all_ifs_fields, log_precip=False,
                          constants=False, hour=0, norm=False,
                          crop=False,
                          nim_crop=0,
                          ifs_crop=0):
    batch_x = []
    batch_y = []
    if crop:
        ifs_crop = (1, -1)
        nim_crop = (5, -6)

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
        batch_x.append(load_ifsstack(ifs_fields, date, h, log_precip=log_precip, norm=norm, crop=ifs_crop))
        batch_y.append(load_nimrod(date, h, log_precip=log_precip, crop=nim_crop))

    if (not constants):
        return np.array(batch_x), np.array(batch_y)
    else:
        return [np.array(batch_x), load_hires_constants(len(batch_dates))], np.array(batch_y)


def load_nimrod_nimrod_batch(batch_dates, log_precip=False,
                             constants=False, hour=0, aggregate=1):
    batch_x = []
    batch_y = []
    if hour == 'random':
        hours = np.random.randint(25 - aggregate, size=[len(batch_dates)])
    elif type(hour) == int:
        hours = len(batch_dates)*[hour]

    for i, date in enumerate(batch_dates):
        h = hours[i]
        dta = load_nimrod(date, h, log_precip=False, aggregate=aggregate)
        crs = coarsen_nimrod(dta)
        batch_x.append(logprec(crs, log_precip=log_precip))
        batch_y.append(logprec(dta, log_precip=log_precip))

    if not constants:
        return np.array(batch_x), np.array(batch_y)
    else:
        return [np.array(batch_x), load_hires_constants(len(batch_dates))], np.array(batch_y)


def load_ifs(ifield, date, hour, log_precip=False, norm=False, crop=0):
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

    if crop is None or crop == 0:
        y = np.array(data[::-1, :])
    else:
        y = np.array(data[::-1, :])
        if type(crop) == tuple:
            y = y[crop[0]:crop[1], crop[0]:crop[1]]
        elif type(crop) == int:
            y = y[crop:-crop, crop:-crop]
        else:
            assert False, "Not accepted cropping type"
    data.close()
    ds.close()
    if field in ['tp', 'cp', 'pr', 'prl', 'prc']:
        # print('pass')
        y[y < 0] = 0.
        y = 1000*y
    if log_precip and field in ['tp', 'cp', 'pr', 'prc', 'prl']:
        # ERA precip is measure in meters, so multiple up
        return np.log10(1+y)  # *1000)
    elif norm:
        return (y-ifs_norm[field][0])/ifs_norm[field][1]
    else:
        return y


def load_ifsstack(fields, date, hour, log_precip=False, norm=False, crop=0.):
    field_arrays = []
    for f in fields:
        field_arrays.append(load_ifs(f, date, hour, log_precip=log_precip, norm=norm, crop=crop))
    return np.stack(field_arrays, -1)


def getstats(field, year=2016):
    # Get various stats of the ERA field
    ds = xr.load_dataarray(f'{ERA_PATH}/original/{field}{year}.nc')
    mi = ds.min().data
    mx = ds.max().data
    mn = ds.mean().data
    sd = ds.std().data
    return mi, mx, mn, sd


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


def gen_norm(year=2016):
    # Save stats for specific year
    import pickle
    stats_dic = {}
    for f in all_era_fields:
        stats = getstats(f, year)
        if f == 'psl':
            stats_dic[f] = [stats[2], stats[3]]
        elif f == "u700" or f == "v700":
            stats_dic[f] = [0, max(-stats[0], stats[1])]
        else:
            stats_dic[f] = [0, stats[1]]
    with open(f'{CONSTANTS_PATH}/ERANorm{year}.pkl', 'wb') as f:
        pickle.dump(stats_dic, f, 0)
    return


def gen_ifs_norm(year=2018):
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
    with open(f'{CONSTANTS_PATH}/IFSNorm{year}F.pkl', 'wb') as f:
        pickle.dump(stats_dic, f, 0)
    return


def load_norm(year=2016):
    import pickle
    with open(f'{CONSTANTS_PATH}/ERANorm{year}.pkl', 'rb') as f:
        return pickle.load(f)


def load_ifs_norm(year=2016, tag=''):
    import pickle
    with open(f'{CONSTANTS_PATH}/IFSNorm{year}{tag}.pkl', 'rb') as f:
        return pickle.load(f)


try:
    era_norm = load_norm()
except:  # noqa
    era_norm = None
try:
    ifs_norm = load_ifs_norm(2018, tag='F')
except:  # noqa
    ifs_norm = None
