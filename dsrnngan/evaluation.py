import gc
import os
import warnings

import numpy as np
from properscoring import crps_ensemble
from tensorflow.python.keras.utils import generic_utils

import data
import read_config
import setupdata
import setupmodel
from noise import NoiseGenerator
from pooling import pool
from rapsd import rapsd

warnings.filterwarnings("ignore", category=RuntimeWarning)

path = os.path.dirname(os.path.abspath(__file__))
ds_fac = read_config.read_downscaling_factor()["downscaling_factor"]


def setup_inputs(*,
                 mode,
                 arch,
                 downscaling_steps,
                 val_years,
                 autocoarsen,
                 input_channels,
                 filters_gen,
                 filters_disc,
                 noise_channels,
                 latent_variables,
                 padding):

    # initialise model
    model = setupmodel.setup_model(mode=mode,
                                   arch=arch,
                                   downscaling_steps=downscaling_steps,
                                   input_channels=input_channels,
                                   filters_gen=filters_gen,
                                   filters_disc=filters_disc,
                                   noise_channels=noise_channels,
                                   latent_variables=latent_variables,
                                   padding=padding)

    gen = model.gen

    # always uses full-sized images
    print('Loading full sized image dataset')
    _, data_gen_valid = setupdata.setup_data(
        val_years=val_years,
        autocoarsen=autocoarsen)
    return gen, data_gen_valid


def _init_VAEGAN(gen, data_gen, batch_size, latent_variables):
    if False:
        # this runs the model on one batch, which is what the internet says
        # but this doesn't actually seem to be necessary?!
        data_gen_iter = iter(data_gen)
        inputs, _ = next(data_gen_iter)
        cond = inputs['lo_res_inputs']
        const = inputs['hi_res_inputs']

        noise_shape = np.array(cond)[0, ..., 0].shape + (latent_variables,)
        noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
        mean, logvar = gen.encoder([cond, const])
        gen.decoder.predict([mean, logvar, noise_gen(), const])
    # even after running the model on one batch, this needs to be set(?!)
    gen.built = True
    return


def eval_one_chkpt(*,
                   mode,
                   gen,
                   data_gen,
                   noise_channels,
                   latent_variables,
                   num_images,
                   add_noise,
                   ensemble_size,
                   noise_factor,
                   denormalise_data=True,
                   normalize_ranks=True,
                   show_progress=True):

    ranks = []
    lowress = []
    hiress = []
    crps_scores = {}
    mae_all = []
    mse_all = []
    emmse_all = []
    ralsd_all = []

    data_gen_iter = iter(data_gen)
    tpidx = data.all_fcst_fields.index('tp')
    batch_size = 1  # do one full-size image at a time

    if mode == "det":
        ensemble_size = 1  # can't generate an ensemble deterministically

    if show_progress:
        # Initialize progbar
        progbar = generic_utils.Progbar(num_images,
                                        stateful_metrics=("CRPS", "EM-MSE"))

    CRPS_pooling_methods = ['no_pooling', 'max_4', 'max_16', 'avg_4', 'avg_16']
    rng = np.random.default_rng()

    for kk in range(num_images):
        # load truth images
        inputs, outputs = next(data_gen_iter)
        cond = inputs['lo_res_inputs']
        const = inputs['hi_res_inputs']
        truth = outputs['output']
        truth = np.expand_dims(np.array(truth), axis=-1)  # must be 4D tensor for pooling NHWC
        if denormalise_data:
            truth = data.denormalise(truth)

        # generate predictions, depending on model type
        samples_gen = []
        if mode == "GAN":
            noise_shape = np.array(cond)[0, ..., 0].shape + (noise_channels,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
            for ii in range(ensemble_size):
                nn = noise_gen()
                sample_gen = gen.predict([cond, const, nn])
                samples_gen.append(sample_gen.astype("float32"))
        elif mode == "det":
            sample_gen = gen.predict([cond, const])
            samples_gen.append(sample_gen.astype("float32"))
        elif mode == 'VAEGAN':
            # call encoder once
            mean, logvar = gen.encoder([cond, const])
            noise_shape = np.array(cond)[0, ..., 0].shape + (latent_variables,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
            for ii in range(ensemble_size):
                nn = noise_gen()
                # generate ensemble of preds with decoder
                sample_gen = gen.decoder.predict([mean, logvar, nn, const])
                samples_gen.append(sample_gen.astype("float32"))

        # samples generated, now process them (e.g., undo log transform) and calculate MAE etc
        for ii in range(ensemble_size):
            sample_gen = samples_gen[ii]
            # sample_gen shape should be [n, h, w, c] e.g. [1, 940, 940, 1]
            if denormalise_data:
                sample_gen = data.denormalise(sample_gen)

            # Calculate MAE, MSE for this sample
            mae = ((np.abs(truth - sample_gen)).mean(axis=(1, 2)))
            mse = ((truth - sample_gen)**2).mean(axis=(1, 2))

            mae_all.append(mae.flatten())
            mse_all.append(mse.flatten())

            if ii == 0:
                # reset on first ensemble member
                ensmean = np.zeros_like(sample_gen)
            ensmean += sample_gen

            sample_gen = np.squeeze(sample_gen, axis=-1)  # squeeze out trival dim
            samples_gen[ii] = sample_gen

        # Calculate Ensemble Mean MSE
        ensmean /= ensemble_size
        emmse = ((truth - ensmean)**2).mean(axis=(1, 2))
        emmse_all.append(emmse.flatten())

        # Do all RALSD at once, to avoid re-calculating power spectrum of truth image
        ralsd = calculate_ralsd_rmse(np.squeeze(truth, axis=-1), samples_gen)
        ralsd_all.append(ralsd.flatten())

        # turn list of predictions into array, for CRPS/rank calculations
        samples_gen = np.stack(samples_gen, axis=-1)  # shape of samples_gen is [n, h, w, c] e.g. [1, 940, 940, 10]

        # calculate CRPS scores for different pooling methods
        for method in CRPS_pooling_methods:
            if method == 'no_pooling':
                truth_pooled = truth
                samples_gen_pooled = samples_gen
            else:
                truth_pooled = pool(truth, method)
                samples_gen_pooled = pool(samples_gen, method)
            # crps_ensemble expects truth dims [N, H, W], pred dims [N, H, W, C]
            crps_score = crps_ensemble(np.squeeze(truth_pooled, axis=-1), samples_gen_pooled).mean()
            del truth_pooled, samples_gen_pooled
            gc.collect()

            if method not in crps_scores:
                crps_scores[method] = []
            crps_scores[method].append(crps_score)

        # calculate ranks; only calculated without pooling

        # Add noise to truth and generated samples to make 0-handling fairer
        # NOTE truth and sample_gen are 'polluted' after this, hence we do this last
        if add_noise:
            truth += rng.random(size=truth.shape, dtype=np.float32)*noise_factor
            samples_gen += rng.random(size=samples_gen.shape, dtype=np.float32)*noise_factor

        truth_flat = truth.ravel()  # unwrap into one long array, then unwrap samples_gen in same format
        samples_gen_ranks = samples_gen.reshape((-1, ensemble_size))  # unknown batch size/img dims, known number of samples
        rank = np.count_nonzero(truth_flat[:, None] >= samples_gen_ranks, axis=-1)  # mask array where truth > samples gen, count
        ranks.append(rank)
        # keep track of input and truth rainfall values, to facilitate further ranks processing
        cond_exp = np.repeat(np.repeat(data.denormalise(cond[..., tpidx]).astype(np.float32), ds_fac, axis=-1), ds_fac, axis=-2)
        lowress.append(cond_exp.ravel())
        hiress.append(truth.astype(np.float32).ravel())
        del samples_gen_ranks, truth_flat
        gc.collect()

        if show_progress:
            emmse_so_far = np.sqrt(np.mean(np.concatenate(emmse_all)))
            crps_mean = np.mean(crps_scores['no_pooling'])
            losses = [("EM-MSE", emmse_so_far), ("CRPS", crps_mean)]
            progbar.add(1, values=losses)

    mae_all = np.concatenate(mae_all)
    mse_all = np.concatenate(mse_all)
    emmse_all = np.concatenate(emmse_all)
    ralsd_all = np.concatenate(ralsd_all)

    other = {}
    other['mae'] = mae_all
    other['mse'] = mse_all
    other['emmse'] = emmse_all
    other['ralsd'] = ralsd_all

    ranks = np.concatenate(ranks)
    lowress = np.concatenate(lowress)
    hiress = np.concatenate(hiress)
    gc.collect()
    if normalize_ranks:
        ranks = (ranks / ensemble_size).astype(np.float32)
        gc.collect()
    arrays = (ranks, lowress, hiress)

    return arrays, crps_scores, other


def rank_OP(norm_ranks, num_ranks=100):
    op = np.count_nonzero(
        (norm_ranks == 0) | (norm_ranks == 1)
    )
    op = float(op)/len(norm_ranks)
    return op


def log_line(log_fname, line):
    with open(log_fname, 'a') as f:
        print(line, file=f)


def evaluate_multiple_checkpoints(*,
                                  mode,
                                  arch,
                                  val_years,
                                  log_fname,
                                  weights_dir,
                                  autocoarsen,
                                  add_noise,
                                  noise_factor,
                                  model_numbers,
                                  ranks_to_save,
                                  num_images,
                                  filters_gen,
                                  filters_disc,
                                  input_channels,
                                  latent_variables,
                                  noise_channels,
                                  padding,
                                  ensemble_size):

    df_dict = read_config.read_downscaling_factor()

    gen, data_gen_valid = setup_inputs(mode=mode,
                                       arch=arch,
                                       downscaling_steps=df_dict["steps"],
                                       val_years=val_years,
                                       autocoarsen=autocoarsen,
                                       input_channels=input_channels,
                                       filters_gen=filters_gen,
                                       filters_disc=filters_disc,
                                       noise_channels=noise_channels,
                                       latent_variables=latent_variables,
                                       padding=padding)

    log_line(log_fname, f"Samples per image: {ensemble_size}")
    log_line(log_fname, f"Initial dates/times: {data_gen_valid.dates[0:4]}, {data_gen_valid.hours[0:4]}")
    log_line(log_fname, "N CRPS CRPS_max_4 CRPS_max_16 CRPS_avg_4 CRPS_avg_16 RMSE EMRMSE RALSD MAE OP")

    for model_number in model_numbers:
        gen_weights_file = os.path.join(weights_dir, f"gen_weights-{model_number:07d}.h5")

        if not os.path.isfile(gen_weights_file):
            print(gen_weights_file, "not found, skipping")
            continue

        print(gen_weights_file)
        if mode == "VAEGAN":
            _init_VAEGAN(gen, data_gen_valid, 1, latent_variables)
        gen.load_weights(gen_weights_file)
        arrays, crps, other = eval_one_chkpt(mode=mode,
                                             gen=gen,
                                             data_gen=data_gen_valid,
                                             noise_channels=noise_channels,
                                             latent_variables=latent_variables,
                                             num_images=num_images,
                                             add_noise=add_noise,
                                             ensemble_size=ensemble_size,
                                             noise_factor=noise_factor)
        ranks, lowress, hiress = arrays
        OP = rank_OP(ranks)
        CRPS_pixel = np.asarray(crps['no_pooling']).mean()
        CRPS_max_4 = np.asarray(crps['max_4']).mean()
        CRPS_max_16 = np.asarray(crps['max_16']).mean()
        CRPS_avg_4 = np.asarray(crps['avg_4']).mean()
        CRPS_avg_16 = np.asarray(crps['avg_16']).mean()

        mae = other['mae'].mean()
        rmse = np.sqrt(other['mse'].mean())
        emrmse = np.sqrt(other['emmse'].mean())
        ralsd = np.nanmean(other['ralsd'])

        log_line(log_fname, f"{model_number} {CRPS_pixel:.6f} {CRPS_max_4:.6f} {CRPS_max_16:.6f} {CRPS_avg_4:.6f} {CRPS_avg_16:.6f} {rmse:.6f} {emrmse:.6f} {ralsd:.6f} {mae:.6f} {OP:.6f}")

        # save one directory up from model weights, in same dir as logfile
        ranks_folder = os.path.dirname(log_fname)

        if model_number in ranks_to_save:
            fname = f"ranksnew-{val_years}-{model_number}.npz"
            np.savez_compressed(os.path.join(ranks_folder, fname), ranks=ranks, lowres=lowress, hires=hiress)


def calculate_ralsd_rmse(truth, samples):
    # check 'batch size' is 1; can rewrite to handle batch size > 1 if necessary
    assert truth.shape[0] == 1
    assert samples[0].shape[0] == 1

    # truth has shape 1 x W x H
    # samples is a list, each of shape 1 x W x H

    # avoid producing infinite or misleading values by not doing RALSD calc
    # for images that are mostly zeroes
    if truth.mean() < 0.002:
        return np.array([np.nan])
    # calculate RAPSD of truth once, not repeatedly!
    fft_freq_truth = rapsd(np.squeeze(truth, axis=0), fft_method=np.fft)
    dBtruth = 10 * np.log10(fft_freq_truth)

    ralsd_all = []
    for pred in samples:
        if pred.mean() < 0.002:
            ralsd_all.append(np.nan)
        else:
            fft_freq_pred = rapsd(np.squeeze(pred, axis=0), fft_method=np.fft)
            dBpred = 10 * np.log10(fft_freq_pred)
            rmse = np.sqrt(np.nanmean((dBtruth-dBpred)**2))
            ralsd_all.append(rmse)
    return np.array(ralsd_all)
