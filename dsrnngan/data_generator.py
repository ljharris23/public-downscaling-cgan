""" Data generator class for batched training of precipitation downscaling network """
import numpy as np
from tensorflow.keras.utils import Sequence

from data import load_era_nimrod_batch, load_hires_constants

return_dic = True


class DataGenerator(Sequence):
    def __init__(self, dates, era_fields, batch_size, log_precip=True,
                 shuffle=True, constants=None, hour='random', era_norm=True):
        self.dates = dates
        self.batch_size = batch_size
        self.era_fields = era_fields
        self.log_precip = log_precip
        self.shuffle = shuffle
        self.hour = hour
        self.era_norm = era_norm
        if constants is None:
            self.constants = constants
        elif constants is True:
            self.constants = load_hires_constants(self.batch_size)
        else:
            self.constants = np.repeat(constants, self.batch_size, axis=0)

    def __len__(self):
        # Number of batches in dataset
        return len(self.dates) // self.batch_size

    def __getitem__(self, idx):
        # Get batch at index idx
        dates_batch = self.dates[idx*self.batch_size:(idx+1)*self.batch_size]

        # Load and return this batch of images
        data_x_batch, data_y_batch = load_era_nimrod_batch(
            dates_batch,
            era_fields=self.era_fields,
            log_precip=self.log_precip,
            hour=self.hour,
            era_norm=self.era_norm)

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
        np.random.shuffle(self.dates)

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_data()


if __name__ == "__main__":
    pass
