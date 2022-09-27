import argparse
import gc
import os

import numpy as np

import benchmarks
import crps
import ecpoint
import read_config
from data import get_dates
from data_generator_ifs import DataGenerator as DataGeneratorFull
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
parser.set_defaults(include_Lanczos=False)
parser.set_defaults(include_RainFARM=False)
parser.set_defaults(include_ecPoint=False)
parser.set_defaults(include_ecPoint_mean=False)
parser.set_defaults(include_constant=False)
parser.set_defaults(include_zeros=False)
parser.add_argument('--include_Lanczos', dest='include_Lanczos', action='store_true',
                    help="Include Lanczos benchmark")
parser.add_argument('--include_RainFARM', dest='include_RainFARM', action='store_true',
                    help="Include RainFARM benchmark")
parser.add_argument('--include_ecPoint', dest='include_ecPoint', action='store_true',
                    help="Include ecPoint benchmark")
parser.add_argument('--include_ecPoint_mean', dest='include_ecPoint_mean', action='store_true',
                    help="Include ecPoint mean benchmark")
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
                                    ifs_fields=ecpoint.ifs_fields,
                                    batch_size=batch_size,
                                    log_precip=False,
                                    crop=True,
                                    shuffle=True,
                                    constants=True,
                                    hour='random',
                                    ifs_norm=False)

benchmark_methods = []
if args.include_Lanczos:
    benchmark_methods.append('lanczos')
if args.include_RainFARM:
    benchmark_methods.append('rainfarm')
if args.include_ecPoint:
    benchmark_methods.append('ecpoint_no-corr')
    benchmark_methods.append('ecpoint_part-corr')
    benchmark_methods.append('ecpoint_full-corr')
if args.include_ecPoint_mean:
    benchmark_methods.append('ecpoint_mean')
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
        if benchmark == 'lanczos':
            sample_benchmark = benchmarks.lanczosmodel(inp['lo_res_inputs'][..., 1])
        elif benchmark == 'rainfarm':  # this has ens=100 every time
            sample_benchmark = benchmarks.rainfarmensemble(inp['lo_res_inputs'][..., 1])
        elif benchmark == 'ecpoint_no-corr':  # pred_ensemble will be batch_size x H x W x 100
            sample_benchmark = benchmarks.ecpointmodel(inp['lo_res_inputs'],
                                                       data_format="channels_last")
        elif benchmark == 'ecpoint_part-corr':
            sample_benchmark = benchmarks.ecpointboxensmodel(inp['lo_res_inputs'],
                                                             data_format="channels_last")
        elif benchmark == 'ecpoint_full-corr':  # this has ens=100 every time
            sample_benchmark = benchmarks.ecpointPDFmodel(inp['lo_res_inputs'],
                                                          data_format="channels_last")
        elif benchmark == 'ecpoint_mean':  # this has ens=100 every time
            sample_benchmark = np.mean(benchmarks.ecpointPDFmodel(inp['lo_res_inputs'],
                                                                  data_format="channels_last"), axis=-1)
        elif benchmark == 'constant':
            sample_benchmark = benchmarks.constantupscalemodel(inp['lo_res_inputs'][..., 1])
        elif benchmark == 'zeros':
            sample_benchmark = benchmarks.zerosmodel(inp['lo_res_inputs'][..., 1])
        else:
            assert False

        if benchmark in ['rainfarm', 'ecpoint_no-corr', 'ecpoint_part-corr', 'ecpoint_full-corr']:
            # these benchmarks produce an ensemble of samples
            for method in pooling_methods:
                if method == 'no_pooling':
                    sample_truth_pooled = sample_truth
                    sample_benchmark_pooled = sample_benchmark
                else:
                    sample_truth_pooled = pool(sample_truth, method)
                    sample_benchmark_pooled = pool(sample_benchmark, method)
                sample_truth_pooled = np.squeeze(sample_truth_pooled, axis=-1)
                crps_score = crps.crps_ensemble(sample_truth_pooled, sample_benchmark_pooled).mean()
                del sample_truth_pooled, sample_benchmark_pooled
                if method not in crps_scores[benchmark]:
                    crps_scores[benchmark][method] = []
                crps_scores[benchmark][method].append(crps_score)
            mse_tmp = []
            mae_tmp = []
            rapsd_tmp = []
            # sample_truth dims should match sample_benchmark[...,j] dims
            sample_truth = np.squeeze(sample_truth, axis=-1)
            for j in range(sample_benchmark.shape[-1]):
                mse = ((sample_truth - sample_benchmark[..., j])**2).mean(axis=(1, 2))
                mse_tmp.append(mse)
                mae = (np.abs(sample_truth - sample_benchmark[..., j])).mean(axis=(1, 2))
                mae_tmp.append(mae)
                rapsd = rapsd_batch(sample_truth, sample_benchmark[..., j])
                rapsd_tmp.append(rapsd)
            mse_score = np.asarray(mse_tmp).mean()
            mae_score = np.asarray(mae_tmp).mean()
            rapsd_score = np.asarray(rapsd_tmp).mean()
            del mse_tmp, mae_tmp, rapsd_tmp

            ensmean = sample_benchmark.mean(axis=-1)
            emmse_score = ((sample_truth - ensmean)**2).mean(axis=(1, 2))

            mse_scores[benchmark].append(mse_score)
            emmse_scores[benchmark].append(emmse_score)
            mae_scores[benchmark].append(mae_score)
            rapsd_scores[benchmark].append(rapsd_score)
            gc.collect()

        else:
            # these benchmarks produce a single sample
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
