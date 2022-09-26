""" Data generator class for batched training of precipitation downscaling network """
import numpy as np
from tensorflow.keras.utils import Sequence

from data import get_long_dates, get_long_data, load_hires_constants

return_dic = True


class DataGenerator(Sequence):
    def __init__(self, year, lead_time, ifs_fields, batch_size, log_precip=True,
                 crop=False,
                 shuffle=True, constants=None, hour='random', ifs_norm=True,
                 downsample=False, seed=9999, start_times=['00', '12']):
        self.lead_time = lead_time
        assert lead_time > 0 and lead_time < 73, "Lead time not in [1,72]"

        self.year = year
        self.dates, self.start_times = get_long_dates(year,
                                                      self.lead_time,
                                                      start_times)
        self.shuffle = shuffle
        if self.shuffle:
            np.random.seed(seed)
            self.shuffle_data()

        self.batch_size = batch_size
        self.ifs_fields = ifs_fields
        self.log_precip = log_precip
        self.shuffle = shuffle
        self.hour = hour
        self.ifs_norm = ifs_norm
        self.crop = crop
        self.downsample = downsample
        assert not self.downsample, "Not configured for pure super-res problem"
        if constants is None:
            self.constants = constants
        elif constants is True:
            self.constants = load_hires_constants(self.batch_size, crop=self.crop)
        else:
            self.constants = np.repeat(constants, self.batch_size, axis=0)
        self.master_idx = 0

    def __len__(self):
        # Number of batches in dataset
        return len(self.dates) // self.batch_size

    def __getitem__(self, idx):
        # Get batch at index idx
        data_x_batch = []
        data_y_batch = []
        while self.master_idx < len(self.dates):
            a, b = get_long_data(fields=self.ifs_fields,
                                 start_date=self.dates[self.master_idx],
                                 start_hour=self.start_times[self.master_idx],
                                 lead_time=self.lead_time,
                                 log_precip=self.log_precip,
                                 crop=self.crop,
                                 norm=self.ifs_norm)
            self.master_idx += 1
            if a is not None:
                data_x_batch.append(a)
                data_y_batch.append(b)
            if len(data_x_batch) == self.batch_size:
                break

        data_x_batch = np.stack(data_x_batch, axis=0)
        data_y_batch = np.stack(data_y_batch, axis=0)

        if self.constants is None:
            if return_dic:
                return {"lo_res_inputs": data_x_batch}, {"output": data_y_batch}
            else:
                return data_x_batch, data_y_batch
        else:
            if return_dic:
                return {"lo_res_inputs": data_x_batch,
                        "hi_res_inputs": self.constants[:data_x_batch.shape[0], ...]},\
                        {"output": data_y_batch}
            else:
                return data_x_batch, self.constants[:data_x_batch.shape[0], ...], data_y_batch

    def shuffle_data(self):
        assert len(self.start_times) == len(self.dates)
        p = np.random.permutation(len(self.start_times))
        self.start_times = self.start_times[p]
        self.dates = self.dates[p]
        return

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_data()


if __name__ == "__main__":
    pass
