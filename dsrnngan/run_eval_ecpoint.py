import argparse
import gc
import os

import matplotlib; matplotlib.use("Agg")  # noqa
import numpy as np
from tensorflow.python.keras.utils import generic_utils

import benchmarks
import crps
import ecpoint
import evaluation
import read_config
from data import get_dates
from data_generator_ifs import DataGenerator as DataGeneratorFull

read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode

parser = argparse.ArgumentParser()
parser.add_argument('--log_folder', type=str,
                    help="directory to store results")
parser.add_argument('--eval_year', type=int,
                    help="year to predict on", default=2019)
parser.add_argument('--ensemble_members', type=int,
                    help="number of ensemble members", default=100)
parser.add_argument('--num_batches', type=int,
                    help="number of images to predict on", default=256)
parser.add_argument('--ecpoint_model', type=str,
                    help="type of ecpoint model to calculate ranks for", default='full-corr')
args = parser.parse_args()

eval_year = args.eval_year
ensemble_members = args.ensemble_members
num_batches = args.num_batches
log_folder = args.log_folder
ecpoint_model = args.ecpoint_model

# input parameters
out_fn = os.path.join(log_folder, f"eval_noise_full_image_ecpoint_{eval_year}.txt")
load_full_image = True
add_noise = True
noise_factor = 1e-3
show_progress = True
normalize_ranks = True
batch_size = 1  # memory issues

evaluation.log_line(out_fn, "Method CRPS mean std")

# setup data
dates = get_dates(eval_year, ecmwf_file_order=True)
data_ecpoint = DataGeneratorFull(dates=dates,
                                 ifs_fields=ecpoint.ifs_fields,
                                 batch_size=batch_size,
                                 log_precip=False,
                                 crop=True,
                                 shuffle=True,
                                 constants=True,
                                 hour='random',
                                 ifs_norm=False)

batch_gen_iter = iter(data_ecpoint)
crps_scores = []
ranks = []
lowress = []
hiress = []
tpidx = ecpoint.ifs_fields.index('tp')

if show_progress:
    # Initialize progbar and batch counter
    progbar = generic_utils.Progbar(num_batches)

for k in range(num_batches):
    # load truth images
    if load_full_image:
        inputs, outputs = next(batch_gen_iter)
        cond = inputs['lo_res_inputs']
        const = inputs['hi_res_inputs']
        sample_truth = outputs['output']
        sample_truth = np.expand_dims(np.array(sample_truth), axis=-1)  # must be 4D tensor for pooling NHWC
    else:
        raise RuntimeError("Small image evaluation not implemented")

    # generate predictions
    samples_ecpoint = []
    if ecpoint_model == 'no-corr':  # pred_ensemble will be batch_size x H x W x ens
        sample_ecpoint = benchmarks.ecpointmodel(inputs['lo_res_inputs'],
                                                 data_format="channels_last")
    elif ecpoint_model == 'part-corr':
        sample_ecpoint = benchmarks.ecpointboxensmodel(inputs['lo_res_inputs'],
                                                       data_format="channels_last")
    elif ecpoint_model == 'full-corr':  # this has ens=100 every time
        sample_ecpoint = benchmarks.ecpointPDFmodel(inputs['lo_res_inputs'],
                                                    data_format="channels_last")
    elif ecpoint_model == 'mean':  # this has ens=100 every time
        sample_ecpoint = np.mean(benchmarks.ecpointPDFmodel(inputs['lo_res_inputs'],
                                                            data_format="channels_last"), axis=-1)
    else:
        raise Exception("Please correctly specify ecpoint model!")

    if add_noise:
        noise_dim_1, noise_dim_2 = sample_truth[0, ..., 0].shape
        noise = np.random.rand(batch_size, noise_dim_1, noise_dim_2, 1)*noise_factor
        sample_truth += noise
        ecpoint_dim_1, ecpoint_dim_2 = sample_ecpoint[0, ..., 0].shape
        noise_ecpoint = np.random.rand(batch_size, ecpoint_dim_1, ecpoint_dim_2, ensemble_members)*noise_factor
        sample_ecpoint += noise_ecpoint

    samples_ecpoint.append(sample_ecpoint.astype("float32"))

    # turn list into array
    samples_ecpoint = np.stack(samples_ecpoint, axis=-1)  # shape of samples_ecpoint is [n, h, w, c] e.g. [1, 940, 940, 100]
    samples_ecpoint = np.squeeze(samples_ecpoint, axis=-1)

    # calculate ranks
    # currently ranks only calculated without pooling
    # probably fine but may want to threshold in the future, e.g. <1mm, >5mm
    sample_truth_ranks = sample_truth.ravel()  # unwrap into one long array, then unwrap samples_gen in same format
    samples_ecpoint_ranks = samples_ecpoint.reshape((-1, ensemble_members))  # unknown batch size/img dims, known number of samples
    rank = np.count_nonzero(sample_truth_ranks[:, None] >= samples_ecpoint_ranks, axis=-1)  # mask array where truth > samples gen, count
    ranks.append(rank)
    cond_exp = np.repeat(np.repeat(cond[..., tpidx].astype(np.float32), 10, axis=-1), 10, axis=-2)
    lowress.append(cond_exp.ravel())
    hiress.append(sample_truth.astype(np.float32).ravel())
    del samples_ecpoint_ranks, sample_truth_ranks
    gc.collect()

    # calculate CRPS score
    # crps_ensemble expects truth dims [N, H, W], pred dims [N, H, W, C]
    crps_score = crps.crps_ensemble(np.squeeze(sample_truth, axis=-1), samples_ecpoint).mean()
    crps_scores.append(crps_score)

    if show_progress:
        losses = [("CRPS", np.mean(crps_scores))]
        progbar.add(1, values=losses)

ranks = np.concatenate(ranks)
lowress = np.concatenate(lowress)
hiress = np.concatenate(hiress)
gc.collect()
if normalize_ranks:
    ranks = (ranks / ensemble_members).astype(np.float32)
    gc.collect()

# calculate mean and standard deviation
mean = ranks.mean()
std = ranks.std()

crps_score = np.asarray(crps_scores).mean()
evaluation.log_line(out_fn, "{} {:.6f} {:.6f} {:.6f} ".format(ecpoint_model, crps_score, mean, std))

fname = f'ranksnew-ecpoint-{eval_year}.npz'
np.savez_compressed(os.path.join(log_folder, fname), ranks=ranks, lowres=lowress, hires=hiress)
