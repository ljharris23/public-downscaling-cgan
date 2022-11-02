import argparse
import gc
import os

import numpy as np
from properscoring import crps_ensemble

import benchmarks
import read_config
from data import all_fcst_fields, get_dates
from data_generator import DataGenerator as DataGeneratorFull
from evaluation import calculate_ralsd_rmse, log_line
from pooling import pool

read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode
ds_fac = read_config.read_downscaling_factor()["downscaling_factor"]

parser = argparse.ArgumentParser()
parser.add_argument('--log_folder', type=str,
                    help="directory to store results")
parser.add_argument('--predict_year', type=int,
                    help="year to predict on", default=2019)
parser.add_argument('--num_images', type=int,
                    help="number of images to predict on", default=256)
parser.set_defaults(include_nn_interp=True)
parser.set_defaults(include_zeros=True)
parser.add_argument('--include_nn_interp', dest='include_nn_interp', action='store_true',
                    help="Include nearest-neighbour upsampling as benchmark")
parser.add_argument('--include_zeros', dest='include_zeros', action='store_true',
                    help="Include all-zero prediction as benchmark")
args = parser.parse_args()

predict_year = args.predict_year
num_images = args.num_images
log_folder = args.log_folder
batch_size = 1  # memory issues
log_fname = os.path.join(log_folder, f"benchmarks_{predict_year}_{num_images}.txt")

# setup data
dates = get_dates(predict_year)
data_benchmarks = DataGeneratorFull(dates=dates,
                                    fcst_fields=all_fcst_fields,
                                    batch_size=batch_size,
                                    log_precip=False,  # no need to denormalise data
                                    shuffle=True,
                                    constants=True,
                                    hour='random',
                                    fcst_norm=False)

benchmark_methods = []
if args.include_nn_interp:
    benchmark_methods.append('nn_interp')
if args.include_zeros:
    benchmark_methods.append('zeros')

CRPS_pooling_methods = ['no_pooling', 'max_4', 'max_16', 'avg_4', 'avg_16']

log_line(log_fname, "Model CRPS CRPS_max_4 CRPS_max_16 CRPS_avg_4 CRPS_avg_16 RMSE EMRMSE RALSD MAE")

sample_crps = {}
crps_scores = {}
mse_scores = {}
emmse_scores = {}
mae_scores = {}
ralsd_scores = {}

tpidx = all_fcst_fields.index('tp')

for benchmark in benchmark_methods:
    crps_scores[benchmark] = {}
    mse_scores[benchmark] = []
    emmse_scores[benchmark] = []
    mae_scores[benchmark] = []
    ralsd_scores[benchmark] = []
    print(f"calculating for benchmark method {benchmark}")
    data_benchmarks_iter = iter(data_benchmarks)

    for ii in range(num_images):
        print(f"calculating for sample number {ii+1} of {num_images}")
        inp, outp = next(data_benchmarks_iter)
        # pooling requires 4 dimensions NHWC
        truth = np.expand_dims(outp['output'], axis=-1)

        # These two simple comparison models both produce single predictions.
        # A more complicated method could produce an ensemble (with different
        # ensemble members in the last dimension)
        if benchmark == 'nn_interp':
            sample_benchmark = benchmarks.nn_interp_model(inp['lo_res_inputs'][..., tpidx], ds_fac)
        elif benchmark == 'zeros':
            sample_benchmark = benchmarks.zeros_model(inp['lo_res_inputs'][..., tpidx], ds_fac)
        else:
            raise RuntimeError("Benchmark not recognised")

        sample_benchmark = np.expand_dims(sample_benchmark, axis=-1)
        for method in CRPS_pooling_methods:
            if method == 'no_pooling':
                truth_pooled = truth
                sample_benchmark_pooled = sample_benchmark
            else:
                truth_pooled = pool(truth, method)
                sample_benchmark_pooled = pool(sample_benchmark, method)

            # crps_ensemble expects truth dims [N, H, W], pred dims [N, H, W, C]
            crps_score = crps_ensemble(np.squeeze(truth_pooled, axis=-1), sample_benchmark_pooled)
            del truth_pooled, sample_benchmark_pooled

            if method not in crps_scores[benchmark]:
                crps_scores[benchmark][method] = []
            crps_scores[benchmark][method].append(crps_score)
            gc.collect()

        mse_score = ((truth - sample_benchmark)**2).mean(axis=(1, 2))
        mae_score = np.abs(truth - sample_benchmark).mean(axis=(1, 2))
        if benchmark == 'zeros':
            ralsd_score = np.nan
        else:
            ralsd_score = calculate_ralsd_rmse(np.squeeze(truth, axis=-1), [np.squeeze(sample_benchmark, axis=-1)])
        mse_scores[benchmark].append(mse_score)
        emmse_scores[benchmark].append(mse_score)  # no ensemble
        mae_scores[benchmark].append(mae_score)
        ralsd_scores[benchmark].append(ralsd_score)
        gc.collect()

    CRPS_pixel = np.asarray(crps_scores[benchmark]['no_pooling']).mean()
    CRPS_max_4 = np.asarray(crps_scores[benchmark]['max_4']).mean()
    CRPS_max_16 = np.asarray(crps_scores[benchmark]['max_16']).mean()
    CRPS_avg_4 = np.asarray(crps_scores[benchmark]['avg_4']).mean()
    CRPS_avg_16 = np.asarray(crps_scores[benchmark]['avg_16']).mean()

    rmse = np.sqrt(np.array(mse_scores[benchmark]).mean())
    emrmse = np.sqrt(np.array(emmse_scores[benchmark]).mean())
    mae = np.array(mae_scores[benchmark]).mean()
    ralsd = np.nanmean(np.array(ralsd_scores[benchmark]))

    log_line(log_fname, f"{benchmark} {CRPS_pixel:.6f} {CRPS_max_4:.6f} {CRPS_max_16:.6f} {CRPS_avg_4:.6f} {CRPS_avg_16:.6f} {rmse:.6f} {emrmse:.6f} {ralsd:.6f} {mae:.6f}")
