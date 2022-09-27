import glob

import numpy as np
import tensorflow as tf

import read_config
from data import all_ifs_fields, ifs_hours


return_dic = True

data_paths = read_config.get_data_paths()
records_folder = data_paths["TFRecords"]["tfrecords_path"]


def DataGenerator(year, batch_size, repeat=True, downsample=False, weights=None):
    return create_mixed_dataset(year, batch_size, repeat=repeat, downsample=downsample, weights=weights)

def create_mixed_dataset(year,
                         batch_size,
                         img_shape=(20, 20, 9),
                         con_shape=(200, 200, 2),
                         out_shape=(200, 200, 1),
                         repeat=True,
                         downsample=False,
                         folder=records_folder,
                         shuffle_size=1024,
                         weights=None):

    classes = 4
    if weights is None:
        weights = [1./classes]*classes
    datasets = [create_dataset(year,
                               i,
                               img_shape=img_shape,
                               con_shape=con_shape,
                               out_shape=out_shape,
                               folder=folder,
                               shuffle_size=shuffle_size,
                               repeat=repeat)
                for i in range(classes)]
    sampled_ds = tf.data.Dataset.sample_from_datasets(datasets,
                                                      weights=weights).batch(batch_size)

    if downsample and return_dic:
        sampled_ds = sampled_ds.map(_dataset_downsampler)
    elif downsample and not return_dic:
        sampled_ds = sampled_ds.map(_dataset_downsampler_list)
    sampled_ds = sampled_ds.prefetch(2)
    return sampled_ds

# Note, if we wanted fewer classes, we can use glob syntax to grab multiple classes as once
# e.g. create_dataset(2015,"[67]")
# will take classes 6 & 7 together


def _dataset_downsampler(inputs, outputs):
    image = outputs['output']
    kernel_tf = tf.constant(0.01, shape=(10, 10, 1, 1), dtype=tf.float32)
    image = tf.nn.conv2d(image, filters=kernel_tf, strides=[1, 10, 10, 1], padding='VALID',
                         name='conv_debug', data_format='NHWC')
    inputs['lo_res_inputs'] = image
    return inputs, outputs


def _dataset_downsampler_list(inputs, constants, outputs):
    image = outputs
    kernel_tf = tf.constant(0.01, shape=(10, 10, 1, 1), dtype=tf.float32)
    image = tf.nn.conv2d(image, filters=kernel_tf, strides=[1, 10, 10, 1], padding='VALID', name='conv_debug', data_format='NHWC')
    inputs = image
    return inputs, constants, outputs


def _parse_batch(record_batch,
                 insize=(20, 20, 9),
                 consize=(200, 200, 2),
                 outsize=(200, 200, 1)):
    # Create a description of the features
    feature_description = {
        'generator_input': tf.io.FixedLenFeature(insize, tf.float32),
        'constants': tf.io.FixedLenFeature(consize, tf.float32),
        'generator_output': tf.io.FixedLenFeature(outsize, tf.float32),
    }

    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_example(record_batch, feature_description)
    if return_dic:
        return ({'lo_res_inputs': example['generator_input'],
                 'hi_res_inputs': example['constants']},
                {'output': example['generator_output']})
    else:
        return example['generator_input'], example['constants'], example['generator_output']


def create_dataset(year,
                   clss,
                   img_shape=(20, 20, 9),
                   con_shape=(200, 200, 2),
                   out_shape=(200, 200, 1),
                   folder=records_folder,
                   shuffle_size=1024,
                   repeat=True):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if isinstance(year, (str, int)):
        fl = glob.glob(f"{folder}/{year}_*.{clss}.tfrecords")
    elif isinstance(year, list):
        fl = []
        for y in year:
            fl += glob.glob(f"{folder}/{y}_*.{clss}.tfrecords")
    else:
        assert False, f"TFRecords not configure for type {type(year)}"
    files_ds = tf.data.Dataset.list_files(fl)
    ds = tf.data.TFRecordDataset(files_ds,
                                 num_parallel_reads=AUTOTUNE)
    ds = ds.shuffle(shuffle_size)
    ds = ds.map(lambda x: _parse_batch(x,
                                       insize=img_shape,
                                       consize=con_shape,
                                       outsize=out_shape))
    if repeat:
        return ds.repeat()
    else:
        return ds


def create_fixed_dataset(year=None,
                         mode='validation',
                         batch_size=16,
                         downsample=False,
                         img_shape=(20, 20, 9),
                         con_shape=(200, 200, 2),
                         out_shape=(200, 200, 1),
                         name=None,
                         folder=records_folder):
    assert year is not None or name is not None, "Must specify year or file name"
    if folder[-1] != '/':
        folder = folder + '/'
    if name is None:
        name = f"{folder}{mode}{year}.tfrecords"
    else:
        if name[0] != '/':
            name = folder + name
    fl = glob.glob(name)
    files_ds = tf.data.Dataset.list_files(fl)
    ds = tf.data.TFRecordDataset(files_ds,
                                 num_parallel_reads=1)
    ds = ds.map(lambda x: _parse_batch(x,
                                       insize=img_shape,
                                       consize=con_shape,
                                       outsize=out_shape))
    ds = ds.batch(batch_size)
    if downsample and return_dic:
        ds = ds.map(_dataset_downsampler)
    elif downsample and not return_dic:
        ds = ds.map(_dataset_downsampler_list)
    return ds


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def write_data(year,
               ifs_fields=all_ifs_fields,
               hours=ifs_hours,
               img_chunk_width=20,
               num_class=4,
               log_precip=True,
               ifs_norm=True):
    from data import get_dates
    from data_generator_ifs import DataGenerator

    dates = get_dates(year)

    # nim_size = 940
    img_size = 94

    upscaling_factor = 10

    nsamples = (img_size//img_chunk_width + 1)**2
    print("Samples per image:", nsamples)
    import random

    for hour in hours:
        dgc = DataGenerator(dates=dates,
                            ifs_fields=ifs_fields,
                            batch_size=1,
                            log_precip=log_precip,
                            shuffle=False,
                            constants=True,
                            hour=hour,
                            ifs_norm=ifs_norm)
        fle_hdles = []
        for fh in range(num_class):
            flename = f"{records_folder}{year}_{hour}.{fh}.tfrecords"
            fle_hdles.append(tf.io.TFRecordWriter(flename))
        for batch in range(len(dates)):
            print(hour, batch)
            sample = dgc.__getitem__(batch)
            for k in range(sample[1]['output'].shape[0]):
                for ii in range(nsamples):
                    # e.g. for image width 94 and img_chunk_width 20, can have 0:20 up to 74:94
                    idx = random.randint(0, img_size-img_chunk_width)
                    idy = random.randint(0, img_size-img_chunk_width)

                    nimrod = sample[1]['output'][k,
                                                 idx*upscaling_factor:(idx+img_chunk_width)*upscaling_factor,
                                                 idy*upscaling_factor:(idy+img_chunk_width)*upscaling_factor].flatten()
                    const = sample[0]['hi_res_inputs'][k,
                                                       idx*upscaling_factor:(idx+img_chunk_width)*upscaling_factor,
                                                       idy*upscaling_factor:(idy+img_chunk_width)*upscaling_factor,
                                                       :].flatten()
                    input_img = sample[0]['lo_res_inputs'][k,
                                                     idx:idx+img_chunk_width,
                                                     idy:idy+img_chunk_width,
                                                     :].flatten()
                    feature = {
                        'generator_input': _float_feature(input_img),
                        'constants': _float_feature(const),
                        'generator_output': _float_feature(nimrod)
                    }
                    features = tf.train.Features(feature=feature)
                    example = tf.train.Example(features=features)
                    example_to_string = example.SerializeToString()
                    clss = min(int(np.floor(((nimrod > 0.1).mean()*num_class))), num_class-1)  # all class binning is here!
                    fle_hdles[clss].write(example_to_string)
        for fh in fle_hdles:
            fh.close()


def save_dataset(tfrecords_dataset, flename, max_batches=None):

    assert return_dic, "Only works with return_dic=True"
    flename = f"{records_folder}/{flename}"
    fle_hdle = tf.io.TFRecordWriter(flename)
    for ii, sample in enumerate(tfrecords_dataset):
        print(ii)
        if max_batches is not None:
            if ii == max_batches:
                break
        for k in range(sample[1]['output'].shape[0]):
            feature = {
                'generator_input': _float_feature(sample[0]['lo_res_inputs'][k, ...].numpy().flatten()),
                'constants': _float_feature(sample[0]['hi_res_inputs'][k, ...].numpy().flatten()),
                'generator_output': _float_feature(sample[1]['output'][k, ...].numpy().flatten())
            }
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            example_to_string = example.SerializeToString()
            fle_hdle.write(example_to_string)
    fle_hdle.close()
    return
