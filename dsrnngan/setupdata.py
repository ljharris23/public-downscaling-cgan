import gc

import tfrecords_generator
from tfrecords_generator import DataGenerator

from data import all_fcst_fields


def setup_batch_gen(train_years,
                    val_years,
                    batch_size=64,
                    val_size=None,
                    downsample=False,
                    weights=None,
                    val_fixed=True):

    tfrecords_generator.return_dic = False
    print(f"downsample flag is {downsample}")
    train = None if train_years is None \
        else DataGenerator(train_years,
                           batch_size=batch_size,
                           downsample=downsample, weights=weights)

    # note -- using create_fixed_dataset with a batch size not divisible by 16 will cause problems [is this true?]
    # create_fixed_dataset will not take a list
    if val_size is not None:
        # assume that val_size is small enough that we can just use one batch
        val = tfrecords_generator.create_fixed_dataset(val_years, batch_size=val_size, downsample=downsample)
        val = val.take(1)
        if val_fixed:
            val = val.cache()
    else:
        val = tfrecords_generator.create_fixed_dataset(val_years, batch_size=batch_size, downsample=downsample)
    return train, val


def setup_full_image_dataset(years,
                             batch_size=1,
                             downsample=False):

    from data_generator import DataGenerator as DataGeneratorFull
    from data import get_dates

    dates = get_dates(years)
    data_full = DataGeneratorFull(dates=dates,
                                  fcst_fields=all_fcst_fields,
                                  batch_size=batch_size,
                                  log_precip=True,
                                  shuffle=True,
                                  constants=True,
                                  hour='random',
                                  fcst_norm=True,
                                  downsample=downsample)
    return data_full


def setup_data(train_years=None,
               val_years=None,
               val_size=None,
               downsample=False,
               weights=None,
               batch_size=None,
               load_full_image=False):

    if load_full_image is True:
        batch_gen_train = None if train_years is None \
            else setup_full_image_dataset(train_years,
                                          batch_size=batch_size,
                                          downsample=downsample)
        batch_gen_valid = None if val_years is None \
            else setup_full_image_dataset(val_years,
                                          batch_size=batch_size,
                                          downsample=downsample)

    else:
        batch_gen_train, batch_gen_valid = setup_batch_gen(
            train_years=train_years,
            val_years=val_years,
            batch_size=batch_size,
            val_size=val_size,
            downsample=downsample,
            weights=weights)

    gc.collect()
    return batch_gen_train, batch_gen_valid