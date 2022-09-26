import argparse
import gc
import os
import pickle
import yaml

import numpy as np
from tensorflow.python.keras.utils import generic_utils

import benchmarks
import crps
import data
import ecpoint
import read_config
from data_generator_ifsall import DataGenerator
from evaluation import _init_VAEGAN
from noise import NoiseGenerator
from setupmodel import setup_model

read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to configuration file")
parser.add_argument('--num_batches', type=int,
                    help="max number of batches", default='64')
parser.add_argument('--rank_samples', type=int,
                    help="ensemble size", default='10')
parser.add_argument('--start_time', type=int,
                    help="lead time to start at", default='1')
parser.add_argument('--stop_time', type=int,
                    help="lead time to stop at", default='72')
parser.add_argument('--stride', type=int,
                    help="time stride", default='1')
parser.add_argument('--name', type=str,
                    help="name suffix for file save", default='1')
parser.add_argument('--model_number', type=str,
                    help="model number for GAN", default='0147200')
args = parser.parse_args()

num_batches = args.num_batches
rank_samples = args.rank_samples
stride = args.stride

# read in the configurations
if args.config is not None:
    config_path = args.config
else:
    raise Exception("Please specify configuration!")

with open(config_path, 'r') as f:
    try:
        setup_params = yaml.safe_load(f)
        print(setup_params)
    except yaml.YAMLError as exc:
        print(exc)

mode = setup_params["GENERAL"]["mode"]
arch = setup_params["MODEL"]["architecture"]
padding = setup_params["MODEL"]["padding"]
log_folder = setup_params["SETUP"]["log_folder"]
problem_type = setup_params["GENERAL"]["problem_type"]
filters_gen = setup_params["GENERATOR"]["filters_gen"]
noise_channels = setup_params["GENERATOR"]["noise_channels"]
latent_variables = setup_params["GENERATOR"]["latent_variables"]
filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]
batch_size = 1

if mode not in ['GAN', 'VAEGAN', 'det']:
    raise ValueError("Mode type is restricted to 'GAN' 'VAEGAN' 'det'")
if problem_type != 'normal':
    raise RuntimeError("Problem type is restricted to 'normal', lead time datagenerator not configured for pure super-res problem")

downsample = False
input_channels = 9
eval_year = 2020  # only 2020 data is available
all_ifs_fields = ['tp', 'cp', 'sp', 'tisr', 'cape', 'tclw', 'tcwv', 'u700', 'v700']
denormalise_data = True
# start_times = ['00', '12']
start_times = ['00']

if mode == "det":
    rank_samples = 1  # can't generate an ensemble deterministically

# where to find model weights
weights_fn = os.path.join(log_folder, 'models', 'gen_weights-{}.h5'.format(args.model_number))
# where to save results
lead_time_fname_model_00 = os.path.join(log_folder, "model-lead-time-{}-00.pickle".format(args.name))
lead_time_fname_ecpoint_00 = os.path.join(log_folder, "ecpoint-lead-time-{}-00.pickle".format(args.name))
lead_time_fname_ifs_00 = os.path.join(log_folder, "ifs-lead-time-{}-00.pickle".format(args.name))
lead_time_fname_model_12 = os.path.join(log_folder, "model-lead-time-{}-12.pickle".format(args.name))
lead_time_fname_ecpoint_12 = os.path.join(log_folder, "ecpoint-lead-time-{}-12.pickle".format(args.name))
lead_time_fname_ifs_12 = os.path.join(log_folder, "ifs-lead-time-{}-12.pickle".format(args.name))

# initialise model
model = setup_model(mode=mode,
                    arch=arch,
                    input_channels=input_channels,
                    filters_gen=filters_gen,
                    filters_disc=filters_disc,
                    noise_channels=noise_channels,
                    latent_variables=latent_variables,
                    padding=padding)

gen = model.gen
if mode == "VAEGAN":
    # bit of a hack, second argument _should_ be a DataGenerator that the VAE-GAN can do a prediction with
    # but actually _init_VAEGAN skips that at the moment and just sets gen.built = True
    _init_VAEGAN(gen, None, True, batch_size, latent_variables)
gen.load_weights(weights_fn)

crps_scores_model = {}
crps_scores_ecpoint = {}
mae_scores_ifs = {}

for forecast_hour in start_times:
    for hour in range(args.start_time, args.stop_time+1, stride):
        print(f"calculating for hour {hour} of {args.stop_time}")
        print(f"forecast start hour is {forecast_hour}")
        # load data generators for this hour
        data_gen = DataGenerator(year=eval_year,
                                 lead_time=hour,
                                 ifs_fields=all_ifs_fields,
                                 batch_size=batch_size,
                                 log_precip=True,
                                 crop=True,
                                 shuffle=True,
                                 constants=True,
                                 hour='random',
                                 ifs_norm=True,
                                 downsample=downsample,
                                 start_times=forecast_hour)

        data_gen_iter = iter(data_gen)

        data_benchmarks = DataGenerator(year=eval_year,
                                        lead_time=hour,
                                        ifs_fields=ecpoint.ifs_fields,
                                        batch_size=batch_size,
                                        log_precip=False,
                                        crop=True,
                                        shuffle=True,
                                        constants=True,
                                        hour='random',
                                        ifs_norm=False,
                                        start_times=forecast_hour)

        data_benchmarks_iter = iter(data_benchmarks)

        # do at most num_batches examples, but no more than we have available!
        nbatches = min(num_batches, len(data_gen.dates)//batch_size)
        # Initialize progbar and batch counter
        progbar = generic_utils.Progbar(nbatches)

        for k in range(nbatches):
            # retrieve model data
            inputs, outputs = next(data_gen_iter)
            cond = inputs['lo_res_inputs']
            const = inputs['hi_res_inputs']
            sample_truth = outputs['output']  # NWH
            sample_ifs = inputs['lo_res_inputs'][..., 0]  # first field is total precip [NWH]
            if denormalise_data:
                sample_truth = data.denormalise(sample_truth)
                sample_ifs = data.denormalise(sample_ifs)

            # retrieve ecpoint data
            inp_benchmarks, outp_benchmarks = next(data_benchmarks_iter)
            ecpoint_sample = benchmarks.ecpointPDFmodel(inp_benchmarks['lo_res_inputs'])
            ecpoint_truth = outp_benchmarks['output']

            # generate predictions, depending on model type
            samples_gen = []
            if mode == "GAN":
                noise_shape = np.array(cond)[0, ..., 0].shape + (noise_channels,)
                noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
                for i in range(rank_samples):
                    nn = noise_gen()
                    sample_gen = gen.predict([cond, const, nn])
                    samples_gen.append(sample_gen.astype("float32"))
            elif mode == "det":
                sample_gen = gen.predict([cond, const])
                samples_gen.append(sample_gen.astype("float32"))
            elif mode == 'VAEGAN':
                # call encoder once
                (mean, logvar) = gen.encoder([cond, const])
                noise_shape = np.array(cond)[0, ..., 0].shape + (latent_variables,)
                noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
                for i in range(rank_samples):
                    nn = noise_gen()
                    # generate ensemble of preds with decoder
                    sample_gen = gen.decoder.predict([mean, logvar, nn, const])
                    samples_gen.append(sample_gen.astype("float32"))
            for ii in range(len(samples_gen)):
                sample_gen = np.squeeze(samples_gen[ii], axis=-1)  # squeeze out trival dim
                # sample_gen shape should be [n, h, w] e.g. [1, 940, 940]
                if denormalise_data:
                    sample_gen = data.denormalise(sample_gen)
                samples_gen[ii] = sample_gen
            # turn list into array
            samples_gen = np.stack(samples_gen, axis=-1)  # shape of samples_gen is [n, h, w, c] e.g. [1, 940, 940, 10]

            # calculate MAE for IFS pred
            sample_ifs = np.repeat(sample_ifs, 10, axis=1)
            sample_ifs = np.repeat(sample_ifs, 10, axis=2)
            mae_score_ifs = np.abs(sample_truth - sample_ifs).mean(axis=(1, 2))
            del sample_ifs
            gc.collect()

            # calculate CRPS score. Pooling not implemented here.
            # crps_ensemble expects truth dims [N, H, W], pred dims [N, H, W, C]
            crps_score_model = crps.crps_ensemble(sample_truth, samples_gen).mean()
            crps_score_ecpoint = crps.crps_ensemble(ecpoint_truth, ecpoint_sample).mean()
            del sample_truth, samples_gen, ecpoint_truth, ecpoint_sample
            gc.collect()

            # save results
            if hour not in crps_scores_model.keys():
                crps_scores_model[hour] = []
                crps_scores_ecpoint[hour] = []
                mae_scores_ifs[hour] = []
            crps_scores_model[hour].append(crps_score_model)
            crps_scores_ecpoint[hour].append(crps_score_ecpoint)
            mae_scores_ifs[hour].append(mae_score_ifs)

            crps_mean = np.mean(crps_scores_model[hour])
            losses = [("CRPS", crps_mean)]
            progbar.add(1, values=losses)

    if forecast_hour == '00':
        lead_time_fname_model = lead_time_fname_model_00
        lead_time_fname_ecpoint = lead_time_fname_ecpoint_00
        lead_time_fname_ifs = lead_time_fname_ifs_00
    elif forecast_hour == '12':
        lead_time_fname_model = lead_time_fname_model_12
        lead_time_fname_ecpoint = lead_time_fname_ecpoint_12
        lead_time_fname_ifs = lead_time_fname_ifs_12

    with open(lead_time_fname_model, 'wb') as handle:
        pickle.dump(crps_scores_model, handle)

    with open(lead_time_fname_ecpoint, 'wb') as handle:
        pickle.dump(crps_scores_ecpoint, handle)

    with open(lead_time_fname_ifs, 'wb') as handle:
        pickle.dump(mae_scores_ifs, handle)
