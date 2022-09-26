import glob

import numpy as np
import tensorflow as tf

import read_config

return_dic = True

data_paths = read_config.get_data_paths()
tfrecords_path = data_paths["TFRecords"]["tfrecords_path"]


def DataGenerator(year, batch_size, repeat=True):
    return create_mixed_dataset(year, batch_size, repeat=repeat)


def create_random_dataset(year, batch_size,
                          era_shape=(10, 10, 9),
                          con_shape=(250, 250, 2),
                          out_shape=(250, 250, 1),
                          repeat=True,
                          folder=tfrecords_path,
                          shuffle_size=1024):
    dataset = create_dataset(year, '*',
                             era_shape=era_shape,
                             con_shape=con_shape,
                             out_shape=out_shape,
                             folder=folder,
                             repeat=repeat,
                             shuffle_size=shuffle_size)
    return dataset.batch(batch_size).prefetch(2)


def create_mixed_dataset(year, batch_size,
                         era_shape=(10, 10, 9),
                         con_shape=(250, 250, 2),
                         out_shape=(250, 250, 1),
                         repeat=True,
                         folder=tfrecords_path,
                         shuffle_size=1024):

    classes = 8
    datasets = [create_dataset(year, i,
                               era_shape=era_shape,
                               con_shape=con_shape,
                               out_shape=out_shape,
                               folder=folder,
                               shuffle_size=shuffle_size,
                               repeat=repeat)
                for i in range(classes)]
    sampled_ds = tf.data.experimental.sample_from_datasets(
        datasets,
        weights=[1./classes]*classes
    ).batch(batch_size).prefetch(2)
    return sampled_ds

# Note, if we wanted fewer classes, we can use glob syntax to grab multiple classes as once
# e.g. create_dataset(2015,"[67]")
# will take classes 6 & 7 together


def _parse_batch(record_batch,
                 insize=(10, 10, 9),
                 consize=(250, 250, 2),
                 outsize=(250, 250, 1)):
    # Create a description of the features
    feature_description = {
        'lo_res_inputs': tf.io.FixedLenFeature(insize, tf.float32),
        'hi_res_inputs': tf.io.FixedLenFeature(consize, tf.float32),
        'output': tf.io.FixedLenFeature(outsize, tf.float32),
    }

    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_example(record_batch, feature_description)
    if return_dic:
        return {'lo_res_inputs': example['lo_res_inputs'], 'hi_res_inputs': example['hi_res_inputs']},\
            {'output': example['output']}
    else:
        return example['lo_res_inputs'], example['hi_res_inputs'], example['output']


def create_dataset(year, clss,
                   era_shape=(10, 10, 9),
                   con_shape=(250, 250, 2),
                   out_shape=(250, 250, 1),
                   folder=tfrecords_path,
                   shuffle_size=1024,
                   repeat=True):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if type(year) == str or type(year) == int:
        fl = glob.glob(f"{folder}/{year}_*.{clss}.tfrecords")
    elif type(year) == list:
        fl = []
        for y in year:
            fl += glob.glob(f"{folder}/{y}_*.{clss}.tfrecords")
    else:
        assert False, f"TFRecords not configure for type {type(year)}"

    files_ds = tf.data.Dataset.list_files(fl)
    ds = tf.data.TFRecordDataset(files_ds,
                                 num_parallel_reads=AUTOTUNE)
    ds = ds.shuffle(shuffle_size)
    ds = ds.map(lambda x: _parse_batch(x, insize=era_shape, consize=con_shape,
                                       outsize=out_shape))
    if repeat:
        return ds.repeat()
    else:
        return ds


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def write_data(year,
               era_fields=['pr', 'prc', 'psl', 'tosr', 'cape', 'lwp', 'prw', 'u700', 'v700'],
               hours=range(24),
               era_chunk_width=10,
               num_class=8,
               log_precip=True,
               era_norm=True):
    from data import get_dates
    import data_generator
    dates = get_dates(year)

    nim_size = 951
    era_size = 39

    upscaling_factor = 25
    half_width = int(np.ceil(upscaling_factor/2))

    nimrod_chunk_width = upscaling_factor * era_chunk_width
    nimrod_starts = np.arange(half_width, nim_size-half_width, nimrod_chunk_width)
    nimrod_starts[-1] = nim_size-half_width-nimrod_chunk_width
    nimrod_ends = nimrod_starts+nimrod_chunk_width
    era_starts = np.arange(1, era_size-1, era_chunk_width)
    era_starts[-1] = era_size - 1 - era_chunk_width
    era_ends = era_starts + era_chunk_width
    print(nimrod_starts, nimrod_ends)
    print(era_starts, era_ends)

    for hour in hours:
        dgc = data_generator.DataGenerator(dates=dates,
                                           era_fields=era_fields,
                                           batch_size=1, log_precip=log_precip, constants=True,
                                           hour=hour, era_norm=era_norm)
        fle_hdles = []
        for fh in range(num_class):
            flename = f"{tfrecords_path}/{year}_{hour}.{fh}.tfrecords"
            fle_hdles.append(tf.io.TFRecordWriter(flename))
        for batch in range(len(dates)):
            sample = dgc.__getitem__(batch)
            for k in range(sample[1]['output'].shape[0]):
                for i, idx in enumerate(nimrod_starts):
                    idx1 = nimrod_ends[i]
                    for j, jdx in enumerate(nimrod_starts):
                        jdx1 = nimrod_ends[j]
                        nimrod = sample[1]['output'][k, idx:idx1, jdx:jdx1].flatten()
                        const = sample[0]['hi_res_inputs'][k, idx:idx1, jdx:jdx1, :].flatten()
                        era = sample[0]['lo_res_inputs'][k, era_starts[i]:era_ends[i], era_starts[j]:era_ends[j], :].flatten()
                        feature = {
                            'lo_res_inputs': _float_feature(era),
                            'hi_res_inputs': _float_feature(const),
                            'output': _float_feature(nimrod)
                        }
                        features = tf.train.Features(feature=feature)
                        example = tf.train.Example(features=features)
                        example_to_string = example.SerializeToString()
                        clss = min(int(np.floor(((nimrod > 0.1).mean()*num_class))), num_class-1)
                        fle_hdles[clss].write(example_to_string)
        for fh in fle_hdles:
            fh.close()
