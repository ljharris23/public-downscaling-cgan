import argparse
import gc
import os
import numpy as np
import benchmarks
import read_config
from data import all_fcst_fields, get_dates
from data_generator import DataGenerator as DataGeneratorFull
from evaluation import rapsd_batch, log_line
from pooling import pool

read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode

parser = argparse.ArgumentParser()
parser.add_argument('--log_folder', type=str,
                    help="directory to store results")
parser.add_argument('--predict_year', type=int,
                    help="year to predict on", default=2019)
parser.add_argument('--ensemble_members', type=int,
                    help="number of ensemble members", default=100)
parser.add_argument('--num_batches', type=int,
                    help="number of images to predict on", default=256)
parser.set_defaults(max_pooling=False)
parser.set_defaults(avg_pooling=False)
parser.set_defaults(include_constant=False)
parser.set_defaults(include_zeros=False)
parser.add_argument('--include_constant', dest='include_constant', action='store_true',
                    help="Include constant upscaling as benchmark")
parser.add_argument('--include_zeros', dest='include_zeros', action='store_true',
                    help="Include zero prediction benchmark")
parser.add_argument('--max_pooling', dest='max_pooling', action='store_true',
                    help="Include max pooling for CRPS")
parser.add_argument('--avg_pooling', dest='avg_pooling', action='store_true',
                    help="Include average pooling for CRPS")
args = parser.parse_args()

predict_year = args.predict_year
ensemble_members = args.ensemble_members
num_batches = args.num_batches
log_folder = args.log_folder
batch_size = 1  # memory issues
log_fname = os.path.join(log_folder, "benchmarks_{}_{}_{}.txt".format(predict_year, num_batches, ensemble_members))

# setup data
dates = get_dates(predict_year)
data_benchmarks = DataGeneratorFull(dates=dates,
                                    fcst_fields=all_fcst_fields,
                                    batch_size=batch_size,
                                    log_precip=False,
                                    shuffle=True,
                                    constants=True,
                                    hour='random',
                                    fcst_norm=False)

benchmark_methods = []
if args.include_constant:
    benchmark_methods.append('constant')
if args.include_zeros:
    benchmark_methods.append('zeros')

pooling_methods = ['no_pooling']
if args.max_pooling:
    pooling_methods.append('max_4')
    pooling_methods.append('max_16')
    # pooling_methods.append('max_10_no_overlap')
if args.avg_pooling:
    pooling_methods.append('avg_4')
    pooling_methods.append('avg_16')
    # pooling_methods.append('avg_10_no_overlap')

# log_line(log_fname, "Number of samples {}".format(num_batches))
# log_line(log_fname, "Evaluation year {}".format(predict_year))
log_line(log_fname, "Model CRPS CRPS_avg_4 CRPS_max_4 CRPS_avg_16 CRPS_max_16 RMSE EMRMSE MAE RAPSD")

sample_crps = {}
crps_scores = {}
mse_scores = {}
emmse_scores = {}
mae_scores = {}
rapsd_scores = {}

tpidx = all_fcst_fields.index('tp')

for benchmark in benchmark_methods:
    crps_scores[benchmark] = {}
    mse_scores[benchmark] = []
    emmse_scores[benchmark] = []
    mae_scores[benchmark] = []
    rapsd_scores[benchmark] = []
    print(f"calculating for benchmark method {benchmark}")
    data_benchmarks_iter = iter(data_benchmarks)
    for i in range(num_batches):
        print(f"calculating for sample number {i+1} of {num_batches}")
        (inp, outp) = next(data_benchmarks_iter)
        # pooling requires 4 dimensions NHWC
        sample_truth = np.expand_dims(outp['output'], axis=-1)
        if benchmark == 'constant':
            sample_benchmark = benchmarks.constantupscalemodel(inp['lo_res_inputs'][..., tpidx])
        elif benchmark == 'zeros':
            sample_benchmark = benchmarks.zerosmodel(inp['lo_res_inputs'][..., tpidx])
        else:
            assert False

        sample_benchmark = np.expand_dims(sample_benchmark, axis=-1)
        for method in pooling_methods:
            if method == 'no_pooling':
                sample_truth_pooled = sample_truth
                sample_benchmark_pooled = sample_benchmark
            else:
                sample_truth_pooled = pool(sample_truth, method)
                sample_benchmark_pooled = pool(sample_benchmark, method)
            sample_truth_pooled = np.squeeze(sample_truth_pooled)
            sample_benchmark_pooled = np.squeeze(sample_benchmark_pooled)

            crps_score = benchmarks.mean_crps(sample_truth_pooled, sample_benchmark_pooled)
            del sample_truth_pooled, sample_benchmark_pooled

            if method not in crps_scores[benchmark]:
                crps_scores[benchmark][method] = []
            crps_scores[benchmark][method].append(crps_score)
            gc.collect()

        mse_score = ((sample_truth - sample_benchmark)**2).mean(axis=(1, 2))
        mae_score = np.abs(sample_truth - sample_benchmark).mean(axis=(1, 2))
        if benchmark == 'zeros':
            rapsd_score = np.nan
        else:
            rapsd_score = rapsd_batch(sample_truth, sample_benchmark)
        mse_scores[benchmark].append(mse_score)
        emmse_scores[benchmark].append(mse_score)  # no ensemble
        mae_scores[benchmark].append(mae_score)
        rapsd_scores[benchmark].append(rapsd_score)
        gc.collect()

    if not args.max_pooling:
        crps_scores[benchmark]['max_4'] = np.nan
        crps_scores[benchmark]['max_16'] = np.nan
        # crps_scores[benchmark]['max_10_no_overlap'] = np.nan
    if not args.avg_pooling:
        crps_scores[benchmark]['avg_4'] = np.nan
        crps_scores[benchmark]['avg_16'] = np.nan
        # crps_scores[benchmark]['avg_10_no_overlap'] = np.nan
    log_line(log_fname, "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}"
             .format(benchmark,
                     np.array(crps_scores[benchmark]['no_pooling']).mean(),
                     np.array(crps_scores[benchmark]['avg_4']).mean(),
                     np.array(crps_scores[benchmark]['max_4']).mean(),
                     np.array(crps_scores[benchmark]['avg_16']).mean(),
                     np.array(crps_scores[benchmark]['max_16']).mean(),
                     # np.array(crps_scores[benchmark]['max_10_no_overlap']).mean(),
                     # np.array(crps_scores[benchmark]['avg_10_no_overlap']).mean(),
                     np.sqrt(np.array(mse_scores[benchmark]).mean()),
                     np.sqrt(np.array(emmse_scores[benchmark]).mean()),
                     np.array(mae_scores[benchmark]).mean(),
                     np.nanmean(np.array(rapsd_scores[benchmark]))
                     ))