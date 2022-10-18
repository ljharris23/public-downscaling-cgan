""" Data generator class for full-image evaluation of precipitation downscaling network """
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from data import load_fcst_radar_batch, load_hires_constants, fcst_hours
import read_config

return_dic = True


class DataGenerator(Sequence):
    def __init__(self, dates, fcst_fields, batch_size, log_precip=True,
                 shuffle=True, constants=None, hour='random', fcst_norm=True,
                 downsample=False, seed=9999):
        self.dates = dates

        if isinstance(hour, str):
            if hour == 'random':
                self.hours = np.repeat(fcst_hours, len(self.dates))
                self.dates = np.tile(self.dates, len(fcst_hours))
            else:
                assert False, f"Unsupported hour {hour}"

        elif isinstance(hour, (int, np.integer)):
            self.hours = np.repeat(hour, len(self.dates))
            self.dates = np.tile(self.dates, 1)  # lol

        elif isinstance(hour, (list, np.ndarray)):
            self.hours = np.repeat(hour, len(self.dates))
            self.dates = np.tile(self.dates, len(hour))

        else:
            assert False, f"Unsupported hour {hour}"

        self.shuffle = shuffle
        if self.shuffle:
            np.random.seed(seed)
            self.shuffle_data()

        self.batch_size = batch_size
        self.fcst_fields = fcst_fields
        self.log_precip = log_precip
        self.shuffle = shuffle
        self.hour = hour
        self.fcst_norm = fcst_norm
        self.downsample = downsample
        if self.downsample:
            # read downscaling factor from file
            df_dict = read_config.read_downscaling_factor()  # read downscaling params
            self.ds_factor = df_dict["downscaling_factor"]
        if constants is None:
            self.constants = constants
        elif constants is True:
            self.constants = load_hires_constants(self.batch_size)
        else:
            self.constants = np.repeat(constants, self.batch_size, axis=0)

    def __len__(self):
        # Number of batches in dataset
        return len(self.dates) // self.batch_size

    def _dataset_downsampler(self, radar):
        kernel_tf = tf.constant(1.0/(self.ds_factor*self.ds_factor), shape=(self.ds_factor, self.ds_factor, 1, 1), dtype=tf.float32)
        image = tf.nn.conv2d(radar, filters=kernel_tf, strides=[1, self.ds_factor, self.ds_factor, 1], padding='VALID',
                             name='conv_debug', data_format='NHWC')
        return image

    def __getitem__(self, idx):
        # Get batch at index idx
        dates_batch = self.dates[idx*self.batch_size:(idx+1)*self.batch_size]
        hours_batch = self.hours[idx*self.batch_size:(idx+1)*self.batch_size]

        # Load and return this batch of images
        data_x_batch, data_y_batch = load_fcst_radar_batch(
            dates_batch,
            fcst_fields=self.fcst_fields,
            log_precip=self.log_precip,
            hour=hours_batch,
            norm=self.fcst_norm)
        if self.downsample:
            # replace forecast data by coarsened radar data!
            data_x_batch = self._dataset_downsampler(data_y_batch[..., np.newaxis])

        if self.constants is None:
            if return_dic:
                return {"lo_res_inputs": data_x_batch}, {"output": data_y_batch}
            else:
                return data_x_batch, data_y_batch
        else:
            if return_dic:
                return {"lo_res_inputs": data_x_batch,
                        "hi_res_inputs": self.constants},\
                        {"output": data_y_batch}
            else:
                return data_x_batch, self.constants, data_y_batch

    def shuffle_data(self):
        assert len(self.hours) == len(self.dates)
        p = np.random.permutation(len(self.hours))
        self.hours = self.hours[p]
        self.dates = self.dates[p]
        return

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_data()


if __name__ == "__main__":
    pass
