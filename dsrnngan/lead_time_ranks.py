import argparse
import gc
import os
import yaml

import numpy as np
from tensorflow.python.keras.utils import generic_utils

import benchmarks
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
                    help="max number of batches", default='999')
parser.add_argument('--rank_samples', type=int,
                    help="ensemble size", default='100')
parser.add_argument('--start_time', type=int,
                    help="lead time to start at", default='24')
parser.add_argument('--stop_time', type=int,
                    help="lead time to stop at", default='72')
parser.add_argument('--stride', type=int,
                    help="time stride", default='24')
parser.add_argument('--name', type=str,
                    help="name suffix for file save", default='1')
parser.add_argument('--model_number', type=str,
                    help="model number for GAN", default='0460800')
parser.set_defaults(include_ecPoint=False)
parser.add_argument('--include_ecPoint', dest='include_ecPoint', action='store_true',
                    help="Include ecPoint benchmark")

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
add_noise = True
noise_factor = 1e-3

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

# index of total precip in the field list
tpidx = all_ifs_fields.index('tp')
if args.include_ecPoint:
    ectpidx = ecpoint.ifs_fields.index('tp')

if mode == "det":
    rank_samples = 1  # can't generate an ensemble deterministically

# where to find model weights
weights_fn = os.path.join(log_folder, 'models', 'gen_weights-{}.h5'.format(args.model_number))

# results filename prefix (need to append lead-time used)
model_00_prefix = "model-ranksLT-00Z-{}".format(args.name)
model_12_prefix = "model-ranksLT-12Z-{}".format(args.name)

if args.include_ecPoint:
    ecpoint_00_prefix = "ecpoint-ranksLT-00Z-{}".format(args.name)
    ecpoint_12_prefix = "ecpoint-ranksLT-12Z-{}".format(args.name)

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

for forecast_hour in start_times:
    for hour in range(args.start_time, args.stop_time+1, stride):
        print(f"calculating for hour {hour} of {args.stop_time}")
        print(f"forecast start hour is {forecast_hour}")

        # reset these, since outputted per-hour
        ranks = []
        lowress = []
        if args.include_ecPoint:
            ecpranks = []
            ecplowress = []
        gc.collect()

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

        if args.include_ecPoint:
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
            if denormalise_data:
                sample_truth = data.denormalise(sample_truth).astype("float32")

            # generate predictions, depending on model type
            samples_gen = []
            if mode == "GAN":
                noise_shape = np.array(cond)[0, ..., 0].shape + (noise_channels,)
                noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
                for ii in range(rank_samples):
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
                for ii in range(rank_samples):
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
            samples_gen = np.stack(samples_gen, axis=-1)  # shape of samples_gen is [n, h, w, c] e.g. [1, 940, 940, 100]

            if add_noise:
                sample_truth += noise_factor*np.random.rand(*sample_truth.shape).astype("float32")
                samples_gen += noise_factor*np.random.rand(*samples_gen.shape).astype("float32")

            # do ranks stuff, copied from evaluation.py
            sample_truth_ranks = sample_truth.ravel()  # unwrap into one long array, then unwrap samples_gen in same format
            samples_gen_ranks = samples_gen.reshape((-1, rank_samples))  # 'unknown' batch size/img dims, known number of samples
            rank = np.count_nonzero(sample_truth_ranks[:, None] >= samples_gen_ranks, axis=-1)  # mask array where truth > samples gen, count
            ranks.append(rank)
            cond_exp = np.repeat(np.repeat(data.denormalise(cond[..., tpidx]).astype(np.float32), 10, axis=-1), 10, axis=-2)
            lowress.append(cond_exp.ravel())
            del samples_gen, samples_gen_ranks  # big arrays, 940x940x100
            gc.collect()

            if args.include_ecPoint:
                # do the same for ecPoint
                inp_benchmarks, outp_benchmarks = next(data_benchmarks_iter)
                ecpoint_sample = benchmarks.ecpointPDFmodel(inp_benchmarks['lo_res_inputs'],
                                                            data_format="channels_last")
                ecpoint_truth = outp_benchmarks['output'].astype("float32")

                if add_noise:
                    ecpoint_truth += noise_factor*np.random.rand(*ecpoint_truth.shape).astype("float32")
                    ecpoint_sample += noise_factor*np.random.rand(*ecpoint_sample.shape).astype("float32")

                ecpoint_truth_flat = ecpoint_truth.ravel()
                ecpoint_sample_flat = ecpoint_sample.reshape((-1, 100))
                ecpoint_rank = np.count_nonzero(ecpoint_truth_flat[:, None] >= ecpoint_sample_flat, axis=-1)
                ecpranks.append(ecpoint_rank)
                ecpoint_inp_exp = np.repeat(np.repeat(inp_benchmarks['lo_res_inputs'][..., ectpidx].astype(np.float32), 10, axis=-1), 10, axis=-2)
                ecplowress.append(ecpoint_inp_exp.ravel())
                del ecpoint_sample, ecpoint_sample_flat  # big arrays, 940x940x100
                gc.collect()

            progbar.add(1)

        # Save data...
        ranks = np.concatenate(ranks)
        lowress = np.concatenate(lowress)
        gc.collect()
        ranks = (ranks / rank_samples).astype(np.float32)
        gc.collect()

        if forecast_hour == '00':
            fn = model_00_prefix + "-{}".format(hour) + ".npz"
        elif forecast_hour == '12':
            fn = model_12_prefix + "-{}".format(hour) + ".npz"
        fname = os.path.join(log_folder, fn)
        np.savez_compressed(fname, ranks=ranks, lowres=lowress)
        del ranks, lowress
        gc.collect()

        if args.include_ecPoint:
            ecpranks = np.concatenate(ecpranks)
            ecplowress = np.concatenate(ecplowress)
            gc.collect()
            ecpranks = (ecpranks/100).astype(np.float32)
            gc.collect()

            if forecast_hour == '00':
                fn = ecpoint_00_prefix + "-{}".format(hour) + ".npz"
            elif forecast_hour == '12':
                fn = ecpoint_12_prefix + "-{}".format(hour) + ".npz"
            fname = os.path.join(log_folder, fn)
            np.savez_compressed(fname, ranks=ecpranks, lowres=ecplowress)
            del ecpranks, ecplowress
            gc.collect()
