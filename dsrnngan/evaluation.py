import gc
import os
import warnings

import numpy as np
from tensorflow.python.keras.utils import generic_utils

import crps
import data
import msssim
import setupdata
import setupmodel
from noise import NoiseGenerator
from pooling import pool
from rapsd import rapsd

warnings.filterwarnings("ignore", category=RuntimeWarning)

path = os.path.dirname(os.path.abspath(__file__))


def setup_inputs(*,
                 mode,
                 arch,
                 val_years,
                 downsample,
                 weights,
                 input_channels,
                 batch_size,
                 num_batches,
                 filters_gen,
                 filters_disc,
                 noise_channels,
                 latent_variables,
                 padding,
                 load_full_image):

    if load_full_image:
        # small batch size to prevent memory issues
        batch_size = 1
    else:
        batch_size = batch_size
        num_batches = num_batches

    # initialise model
    model = setupmodel.setup_model(mode=mode,
                                   arch=arch,
                                   input_channels=input_channels,
                                   filters_gen=filters_gen,
                                   filters_disc=filters_disc,
                                   noise_channels=noise_channels,
                                   latent_variables=latent_variables,
                                   padding=padding)

    gen = model.gen

    if load_full_image:
        print('Loading full sized image dataset')
        # load full size image
        _, batch_gen_valid = setupdata.setup_data(
            load_full_image=True,
            val_years=val_years,
            batch_size=batch_size,
            downsample=downsample)
    else:
        print('Evaluating with smaller image dataset')
        _, batch_gen_valid = setupdata.setup_data(
            load_full_image=False,
            val_years=val_years,
            val_size=batch_size*num_batches,
            downsample=downsample,
            weights=weights,
            batch_size=batch_size)
    return (gen, batch_gen_valid)


def _init_VAEGAN(gen, batch_gen, load_full_image, batch_size, latent_variables):
    if False:
        # this runs the model on one batch, which is what the internet says
        # but this doesn't actually seem to be necessary?!
        batch_gen_iter = iter(batch_gen)
        if load_full_image:
            inputs, outputs = next(batch_gen_iter)
            cond = inputs['lo_res_inputs']
            const = inputs['hi_res_inputs']
        else:
            cond, const, _ = next(batch_gen_iter)

        noise_shape = np.array(cond)[0, ..., 0].shape + (latent_variables,)
        noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
        mean, logvar = gen.encoder([cond, const])
        gen.decoder.predict([mean, logvar, noise_gen(), const])
    # even after running the model on one batch, this needs to be set(?!)
    gen.built = True
    return


def randomize_nans(x, rnd_mean, rnd_range):
    nan_mask = np.isnan(x)
    nan_shape = x[nan_mask].shape
    x[nan_mask] = rnd_mean + \
        (np.random.rand(*nan_shape)-0.5)*rnd_range


def ensemble_ranks(*,
                   mode,
                   gen,
                   batch_gen,
                   noise_channels,
                   latent_variables,
                   batch_size,
                   num_batches,
                   noise_offset=0.0,
                   noise_mul=1.0,
                   denormalise_data=True,
                   add_noise=True,
                   rank_samples=100,
                   noise_factor=None,
                   normalize_ranks=True,
                   load_full_image=False,
                   max_pooling=False,
                   avg_pooling=False,
                   show_progress=True):

    ranks = []
    lowress = []
    hiress = []
    crps_scores = {}
    batch_gen_iter = iter(batch_gen)
    tpidx = data.all_fcst_fields.index('tp')

    if mode == "det":
        rank_samples = 1  # can't generate an ensemble deterministically

    if show_progress:
        # Initialize progbar and batch counter
        progbar = generic_utils.Progbar(
            num_batches)

    pooling_methods = ['no_pooling']
    if max_pooling:
        pooling_methods.append('max_4')
        pooling_methods.append('max_16')
        pooling_methods.append('max_10_no_overlap')
    if avg_pooling:
        pooling_methods.append('avg_4')
        pooling_methods.append('avg_16')
        pooling_methods.append('avg_10_no_overlap')

    for k in range(num_batches):
        # load truth images
        if load_full_image:
            inputs, outputs = next(batch_gen_iter)
            cond = inputs['lo_res_inputs']
            const = inputs['hi_res_inputs']
            sample_truth = outputs['output']
            sample_truth = np.expand_dims(np.array(sample_truth), axis=-1)  # must be 4D tensor for pooling NHWC
        else:
            cond, const, sample = next(batch_gen_iter)
            sample_truth = sample.numpy()
        if denormalise_data:
            sample_truth = data.denormalise(sample_truth)
        if add_noise:
            noise_dim_1, noise_dim_2 = sample_truth[0, ..., 0].shape
            noise = np.random.rand(batch_size, noise_dim_1, noise_dim_2, 1)*noise_factor
            sample_truth += noise

        # generate predictions, depending on model type
        samples_gen = []
        if mode == "GAN":
            noise_shape = np.array(cond)[0, ..., 0].shape + (noise_channels,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
            for i in range(rank_samples):
                nn = noise_gen()
                nn *= noise_mul
                nn -= noise_offset
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
                nn *= noise_mul
                nn -= noise_offset
                # generate ensemble of preds with decoder
                sample_gen = gen.decoder.predict([mean, logvar, nn, const])
                samples_gen.append(sample_gen.astype("float32"))
        for ii in range(len(samples_gen)):
            sample_gen = np.squeeze(samples_gen[ii], axis=-1)  # squeeze out trival dim
            # sample_gen shape should be [n, h, w] e.g. [1, 940, 940]
            if denormalise_data:
                sample_gen = data.denormalise(sample_gen)
            if add_noise:
                (noise_dim_1, noise_dim_2) = sample_gen[0, ...].shape
                noise = np.random.rand(batch_size, noise_dim_1, noise_dim_2)*noise_factor
                sample_gen += noise
            samples_gen[ii] = sample_gen
        # turn list into array
        samples_gen = np.stack(samples_gen, axis=-1)  # shape of samples_gen is [n, h, w, c] e.g. [1, 940, 940, 10]

        # calculate ranks
        # currently ranks only calculated without pooling
        # probably fine but may want to threshold in the future, e.g. <1mm, >5mm
        sample_truth_ranks = sample_truth.ravel()  # unwrap into one long array, then unwrap samples_gen in same format
        samples_gen_ranks = samples_gen.reshape((-1, rank_samples))  # unknown batch size/img dims, known number of samples
        rank = np.count_nonzero(sample_truth_ranks[:, None] >= samples_gen_ranks, axis=-1)  # mask array where truth > samples gen, count
        ranks.append(rank)
        cond_exp = np.repeat(np.repeat(data.denormalise(cond[..., tpidx]).astype(np.float32), 10, axis=-1), 10, axis=-2)
        lowress.append(cond_exp.ravel())
        hiress.append(sample_truth.astype(np.float32).ravel())
        del samples_gen_ranks, sample_truth_ranks
        gc.collect()

        # calculate CRPS scores for different pooling methods
        for method in pooling_methods:
            if method == 'no_pooling':
                sample_truth_pooled = sample_truth
                samples_gen_pooled = samples_gen
            else:
                sample_truth_pooled = pool(sample_truth, method)
                samples_gen_pooled = pool(samples_gen, method)
            # crps_ensemble expects truth dims [N, H, W], pred dims [N, H, W, C]
            crps_score = crps.crps_ensemble(np.squeeze(sample_truth_pooled, axis=-1), samples_gen_pooled).mean()
            del sample_truth_pooled, samples_gen_pooled
            gc.collect()

            if method not in crps_scores:
                crps_scores[method] = []
            crps_scores[method].append(crps_score)

        if show_progress:
            crps_mean = np.mean(crps_scores['no_pooling'])
            losses = [("CRPS", crps_mean)]
            progbar.add(1, values=losses)

    ranks = np.concatenate(ranks)
    lowress = np.concatenate(lowress)
    hiress = np.concatenate(hiress)
    gc.collect()
    if normalize_ranks:
        ranks = (ranks / rank_samples).astype(np.float32)
        gc.collect()
    arrays = (ranks, lowress, hiress)

    return (arrays, crps_scores)


def rank_OP(norm_ranks, num_ranks=100):
    op = np.count_nonzero(
        (norm_ranks == 0) | (norm_ranks == 1)
    )
    op = float(op)/len(norm_ranks)
    return op


def log_line(log_fname, line):
    with open(log_fname, 'a') as f:
        print(line, file=f)


def rank_metrics_by_time(*,
                         mode,
                         arch,
                         val_years,
                         log_fname,
                         weights_dir,
                         downsample=False,
                         weights=None,
                         add_noise=True,
                         noise_factor=None,
                         load_full_image=False,
                         model_numbers=None,
                         ranks_to_save=None,
                         batch_size=None,
                         num_batches=None,
                         filters_gen=None,
                         filters_disc=None,
                         input_channels=None,
                         latent_variables=None,
                         noise_channels=None,
                         padding=None,
                         rank_samples=None,
                         max_pooling=False,
                         avg_pooling=False):

    (gen, batch_gen_valid) = setup_inputs(mode=mode,
                                          arch=arch,
                                          val_years=val_years,
                                          downsample=downsample,
                                          weights=weights,
                                          input_channels=input_channels,
                                          batch_size=batch_size,
                                          num_batches=num_batches,
                                          filters_gen=filters_gen,
                                          filters_disc=filters_disc,
                                          noise_channels=noise_channels,
                                          latent_variables=latent_variables,
                                          padding=padding,
                                          load_full_image=load_full_image)

    log_line(log_fname, "N OP CRPS CRPS_max_4 CRPS_max_16 CRPS_max_10_no_overlap CRPS_avg_4 CRPS_avg_16 CRPS_avg_10_no_overlap mean std")

    for model_number in model_numbers:
        gen_weights_file = os.path.join(weights_dir, "gen_weights-{:07d}.h5".format(model_number))

        if not os.path.isfile(gen_weights_file):
            print(gen_weights_file, "not found, skipping")
            continue

        print(gen_weights_file)
        if mode == "VAEGAN":
            _init_VAEGAN(gen, batch_gen_valid, load_full_image, batch_size, latent_variables)
        gen.load_weights(gen_weights_file)
        arrays, crps_scores = ensemble_ranks(mode=mode,
                                             gen=gen,
                                             batch_gen=batch_gen_valid,
                                             noise_channels=noise_channels,
                                             latent_variables=latent_variables,
                                             batch_size=batch_size,
                                             num_batches=num_batches,
                                             add_noise=add_noise,
                                             rank_samples=rank_samples,
                                             noise_factor=noise_factor,
                                             load_full_image=load_full_image,
                                             max_pooling=max_pooling,
                                             avg_pooling=avg_pooling)
        ranks, lowress, hiress = arrays
        OP = rank_OP(ranks)
        CRPS_no_pool = np.asarray(crps_scores['no_pooling']).mean()
        if max_pooling:
            CRPS_max_4 = np.asarray(crps_scores['max_4']).mean()
            CRPS_max_16 = np.asarray(crps_scores['max_16']).mean()
            CRPS_max_10_no_overlap = np.asarray(crps_scores['max_10_no_overlap']).mean()
        else:
            CRPS_max_4 = np.nan
            CRPS_max_16 = np.nan
            CRPS_max_10_no_overlap = np.nan
        if avg_pooling:
            CRPS_avg_4 = np.asarray(crps_scores['avg_4']).mean()
            CRPS_avg_16 = np.asarray(crps_scores['avg_16']).mean()
            CRPS_avg_10_no_overlap = np.asarray(crps_scores['avg_10_no_overlap']).mean()
        else:
            CRPS_avg_4 = np.nan
            CRPS_avg_16 = np.nan
            CRPS_avg_10_no_overlap = np.nan
        mean = ranks.mean()
        std = ranks.std()

        log_line(log_fname, "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(
            model_number, OP, CRPS_no_pool, CRPS_max_4, CRPS_max_16, CRPS_max_10_no_overlap,
            CRPS_avg_4, CRPS_avg_16, CRPS_avg_10_no_overlap, mean, std))

        # save one directory up from model weights, in same dir as logfile
        ranks_folder = os.path.dirname(log_fname)

        if model_number in ranks_to_save:
            fname = f"ranksnew-{val_years}-{model_number}.npz"
            np.savez_compressed(os.path.join(ranks_folder, fname), ranks=ranks, lowres=lowress, hires=hiress)


def calculate_rapsd_rmse(truth, pred):
    # avoid producing inf values by removing RAPSD calc for images
    # that are mostly zeroes (mean pixel value < 0.01)
    if (truth.mean()) < 0.002 or (pred.mean()) < 0.002:
        return np.nan
    fft_freq_truth = rapsd(truth, fft_method=np.fft)
    fft_freq_pred = rapsd(pred, fft_method=np.fft)
    truth = 10 * np.log10(fft_freq_truth)
    pred = 10 * np.log10(fft_freq_pred)
    rmse = np.sqrt(np.nanmean((truth-pred)**2))
    return rmse


def rapsd_batch(batch1, batch2):
    # radially averaged power spectral density
    # squeeze out final dimension (channels)
    if len(batch1.shape) == 4:
        batch1 = np.squeeze(batch1, axis=-1)
    if len(batch2.shape) == 4:
        batch2 = np.squeeze(batch2, axis=-1)
    rapsd_batch = []
    for i in range(batch1.shape[0]):
        rapsd_score = calculate_rapsd_rmse(
                        batch1[i, ...], batch2[i, ...])
        if rapsd_score:
            rapsd_batch.append(rapsd_score)
    return np.array(rapsd_batch)


def image_quality(*,
                  mode,
                  gen,
                  batch_gen,
                  noise_channels,
                  latent_variables,
                  batch_size,
                  rank_samples,
                  num_batches=100,
                  load_full_image=False,
                  denormalise_data=True,
                  show_progress=True):

    batch_gen_iter = iter(batch_gen)

    mae_all = []
    mse_all = []
    emmse_all = []
    ssim_all = []
    rapsd_all = []

    if show_progress:
        # Initialize progbar and batch counter
        progbar = generic_utils.Progbar(num_batches,
                                        stateful_metrics=["RMSE"])

    for k in range(num_batches):
        if load_full_image:
            (inputs, outputs) = next(batch_gen_iter)
            cond = inputs['lo_res_inputs']
            const = inputs['hi_res_inputs']
            truth = outputs['output']
            truth = np.expand_dims(np.array(truth), axis=-1)
        else:
            (cond, const, truth) = next(batch_gen_iter)
            truth = truth.numpy()

        if denormalise_data:
            truth = data.denormalise(truth)

        if mode == "GAN":
            noise_shape = np.array(cond)[0, ..., 0].shape + (noise_channels,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
        elif mode == "VAEGAN":
            noise_shape = np.array(cond)[0, ..., 0].shape + (latent_variables,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
            # call encoder once
            mean, logvar = gen.encoder([cond, const])

        for ii in range(rank_samples):
            if mode == "GAN":
                img_gen = gen.predict([cond, const, noise_gen()])
            elif mode == "det":
                img_gen = gen.predict([cond, const])
            elif mode == 'VAEGAN':
                img_gen = gen.decoder.predict([mean, logvar, noise_gen(), const])
            else:
                try:
                    img_gen = gen.predict([cond, const])
                except:  # noqa
                    assert False, 'image quality metrics not implemented for mode type'

            if denormalise_data:
                img_gen = data.denormalise(img_gen)

            mae = ((np.abs(truth - img_gen)).mean(axis=(1, 2)))
            mse = ((truth - img_gen)**2).mean(axis=(1, 2))
            ssim = msssim.MultiScaleSSIM(truth, img_gen, 1.0)
            rapsd = rapsd_batch(truth, img_gen)
            mae_all.append(mae.flatten())
            mse_all.append(mse.flatten())
            ssim_all.append(ssim.flatten())
            rapsd_all.append(rapsd.flatten())

            if ii == 0:
                # reset on first ensemble member
                ensmean = np.zeros_like(img_gen)
            ensmean += img_gen

        ensmean /= rank_samples
        emmse = ((truth - ensmean)**2).mean(axis=(1, 2))
        emmse_all.append(emmse.flatten())
        if show_progress:
            rmse_so_far = np.sqrt(np.mean(np.concatenate(mse_all)))
            losses = [("RMSE", rmse_so_far)]
            progbar.add(1, values=losses)

    mae_all = np.concatenate(mae_all)
    mse_all = np.concatenate(mse_all)
    emmse_all = np.concatenate(emmse_all)
    ssim_all = np.concatenate(ssim_all)
    rapsd_all = np.concatenate(rapsd_all)

    imgqualret = {}
    imgqualret['mae'] = mae_all
    imgqualret['mse'] = mse_all
    imgqualret['emmse'] = emmse_all
    imgqualret['ssim'] = ssim_all
    imgqualret['rapsd'] = rapsd_all

    return imgqualret


def quality_metrics_by_time(*,
                            mode,
                            arch,
                            val_years,
                            log_fname,
                            weights_dir,
                            downsample=False,
                            weights=None,
                            load_full_image=False,
                            model_numbers=None,
                            batch_size=None,
                            num_batches=None,
                            filters_gen=None,
                            filters_disc=None,
                            input_channels=None,
                            latent_variables=None,
                            noise_channels=None,
                            rank_samples=100,
                            padding=None):

    (gen, batch_gen_valid) = setup_inputs(mode=mode,
                                          arch=arch,
                                          val_years=val_years,
                                          downsample=downsample,
                                          weights=weights,
                                          input_channels=input_channels,
                                          batch_size=batch_size,
                                          num_batches=num_batches,
                                          filters_gen=filters_gen,
                                          filters_disc=filters_disc,
                                          noise_channels=noise_channels,
                                          latent_variables=latent_variables,
                                          padding=padding,
                                          load_full_image=load_full_image)

    log_line(log_fname, "Samples per image: {}".format(rank_samples))
    log_line(log_fname, "Initial dates/times: {}, {}".format(batch_gen_valid.dates[0:4], batch_gen_valid.hours[0:4]))
    log_line(log_fname, "N RMSE EMRMSE MSSSIM RAPSD MAE")

    for model_number in model_numbers:
        gen_weights_file = os.path.join(weights_dir, "gen_weights-{:07d}.h5".format(model_number))

        if not os.path.isfile(gen_weights_file):
            print(gen_weights_file, "not found, skipping")
            continue

        print(gen_weights_file)
        if mode == "VAEGAN":
            _init_VAEGAN(gen, batch_gen_valid, load_full_image, batch_size, latent_variables)
        gen.load_weights(gen_weights_file)
        imgqualret = image_quality(mode=mode,
                                   gen=gen,
                                   batch_gen=batch_gen_valid,
                                   noise_channels=noise_channels,
                                   latent_variables=latent_variables,
                                   batch_size=batch_size,
                                   rank_samples=rank_samples,
                                   num_batches=num_batches,
                                   load_full_image=load_full_image)
        mae = imgqualret['mae']
        mse = imgqualret['mse']
        emmse = imgqualret['emmse']
        ssim = imgqualret['ssim']
        rapsd = imgqualret['rapsd']

        log_line(log_fname, "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(
            model_number,
            np.sqrt(mse.mean()),
            np.sqrt(emmse.mean()),
            ssim.mean(),
            np.nanmean(rapsd),
            mae.mean()))
