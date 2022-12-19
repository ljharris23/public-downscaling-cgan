import gc

import tfrecords_generator
from tfrecords_generator import DataGenerator

from data import all_fcst_fields


# Incredibly slim wrapper around tfrecords_generator.DataGenerator.  Can probably remove...
def setup_batch_gen(train_years,
                    batch_size=16,
                    autocoarsen=False,
                    weights=None):

    tfrecords_generator.return_dic = False
    print(f"autocoarsen flag is {autocoarsen}")
    batch_gen_train = DataGenerator(train_years,
                                    batch_size=batch_size,
                                    autocoarsen=autocoarsen,
                                    weights=weights)
    return batch_gen_train


def setup_full_image_dataset(years,
                             batch_size=1,
                             autocoarsen=False):

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
                                  autocoarsen=autocoarsen)
    return data_full


def setup_data(train_years=None,
               val_years=None,
               autocoarsen=False,
               weights=None,
               batch_size=None):

    batch_gen_train = None if train_years is None \
        else setup_batch_gen(train_years=train_years,
                             batch_size=batch_size,
                             autocoarsen=autocoarsen,
                             weights=weights)

    data_gen_valid = None if val_years is None \
        else setup_full_image_dataset(val_years,
                                      autocoarsen=autocoarsen)

    gc.collect()
    return batch_gen_train, data_gen_valid
