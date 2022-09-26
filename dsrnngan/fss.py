# BSD 3-Clause License

# Copyright (c) 2019, PySteps developers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import gc
import os
import pickle

import numpy as np
from scipy.ndimage.filters import uniform_filter

import benchmarks
import data
import ecpoint
import setupmodel
from data import get_dates, all_ifs_fields
from data_generator_ifs import DataGenerator as DataGeneratorFull
from evaluation import _init_VAEGAN
from noise import NoiseGenerator
from tfrecords_generator_ifs import create_fixed_dataset


def plot_fss_curves(*,
                    mode,
                    arch,
                    log_folder,
                    weights_dir,
                    model_numbers,
                    problem_type,
                    filters_gen,
                    filters_disc,
                    noise_channels,
                    latent_variables,
                    padding,
                    predict_year,
                    predict_full_image,
                    ensemble_members=100,
                    plot_ecpoint=True):

    if problem_type == "normal":
        downsample = False
        input_channels = 9
    elif problem_type == "superresolution":
        downsample = True
        input_channels = 1
    else:
        raise Exception("no such problem type, try again!")

    if predict_full_image:
        batch_size = 1  # this will stop your computer having a strop
        num_batches = 256
    else:
        batch_size = 16
        num_batches = 50

    if mode == 'det':
        ensemble_members = 1  # in this case, only used for printing

    precip_values = np.array([0.1, 0.5, 2.0, 5.0])
    spatial_scales = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # initialise model
    model = setupmodel.setup_model(mode=mode,
                                   arch=arch,
                                   input_channels=input_channels,
                                   filters_gen=filters_gen,
                                   filters_disc=filters_disc,
                                   noise_channels=noise_channels,
                                   latent_variables=latent_variables,
                                   padding=padding)

    # load appropriate dataset
    if predict_full_image:
        dates = get_dates(predict_year, ecmwf_file_order=True)
        data_predict = DataGeneratorFull(dates=dates,
                                         ifs_fields=all_ifs_fields,
                                         batch_size=batch_size,
                                         log_precip=True,
                                         crop=True,
                                         shuffle=True,
                                         constants=True,
                                         hour='random',
                                         ifs_norm=True,
                                         downsample=downsample)

    if not predict_full_image:
        data_predict = create_fixed_dataset(predict_year,
                                            batch_size=batch_size,
                                            downsample=downsample)

    if plot_ecpoint:
        if not predict_full_image:
            raise RuntimeError('Data generator for benchmarks not implemented for small images')
        # requires a different data generator with different fields and no ifs_norm
        data_benchmarks = DataGeneratorFull(dates=dates,
                                            ifs_fields=ecpoint.ifs_fields,
                                            batch_size=batch_size,
                                            log_precip=False,
                                            crop=True,
                                            shuffle=True,
                                            constants=True,
                                            hour="random",
                                            ifs_norm=False)

    # tidier to iterate over GAN checkpoints and ecPoint using joint code
    model_numbers_ec = model_numbers.copy()
    if plot_ecpoint:
        model_numbers_ec.extend(["ecPoint-nocorr", "ecPoint-partcorr", "ecPoint-fullcorr", "ecPoint-mean", "constupsc"])

    method1 = {}  # method 1 - "ensemble FSS"
    method2 = {}  # method 2 - "no-ensemble FSS"
    for model_number in model_numbers_ec:
        method1[model_number] = {}
        method2[model_number] = {}
        for pv in precip_values:
            method1[model_number][pv] = {}
            method2[model_number][pv] = {}
            for spasc in spatial_scales:
                method1[model_number][pv][spasc] = {}
                method1[model_number][pv][spasc]["fssobj"] = fss_init(pv, spasc)
                method2[model_number][pv][spasc] = {}
                method2[model_number][pv][spasc]["fssobj"] = fss_init(pv, spasc)

    for model_number in model_numbers_ec:
        print(f"calculating for model number {model_number}")
        if model_number in model_numbers:
            # only load weights for GAN, not ecPoint
            gen_weights_file = os.path.join(weights_dir, "gen_weights-{:07d}.h5".format(model_number))
            if not os.path.isfile(gen_weights_file):
                print(gen_weights_file, "not found, skipping")
                continue
            print(gen_weights_file)
            if mode == "VAEGAN":
                _init_VAEGAN(model.gen, data_predict, predict_full_image, batch_size, latent_variables)
            model.gen.load_weights(gen_weights_file)

        if model_number in model_numbers:
            data_pred_iter = iter(data_predict)  # "restarts" GAN data iterator
        else:
            data_benchmarks_iter = iter(data_benchmarks)  # ecPoint data iterator

        for ii in range(num_batches):
            print(ii, num_batches)

            if model_number in model_numbers:
                # GAN, not ecPoint
                inputs, outputs = next(data_pred_iter)
                # need to denormalise
                if predict_full_image:
                    im_real = data.denormalise(outputs['output']).astype(np.single)  # shape: batch_size x H x W
                else:
                    im_real = (data.denormalise(outputs['output'])[..., 0]).astype(np.single)
            else:
                # ecPoint, no need to denormalise
                inputs, outputs = next(data_benchmarks_iter)
                im_real = outputs['output'].astype(np.single)  # shape: batch_size x H x W

            if model_number in model_numbers:
                # get GAN predictions
                pred_ensemble = []
                if mode == 'det':
                    pred_ensemble.append(data.denormalise(model.gen.predict([inputs['lo_res_inputs'],
                                                                             inputs['hi_res_inputs']]))[..., 0])
                else:
                    if mode == 'GAN':
                        noise_shape = inputs['lo_res_inputs'][0, ..., 0].shape + (noise_channels,)
                    elif mode == 'VAEGAN':
                        noise_shape = inputs['lo_res_inputs'][0, ..., 0].shape + (latent_variables,)
                    noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
                    if mode == 'VAEGAN':
                        # call encoder once
                        mean, logvar = model.gen.encoder([inputs['lo_res_inputs'], inputs['hi_res_inputs']])
                    for jj in range(ensemble_members):
                        inputs['noise_input'] = noise_gen()
                        if mode == 'GAN':
                            pred_ensemble.append(data.denormalise(model.gen.predict([inputs['lo_res_inputs'],
                                                                                     inputs['hi_res_inputs'],
                                                                                     inputs['noise_input']]))[..., 0])
                        elif mode == 'VAEGAN':
                            dec_inputs = [mean, logvar, inputs['noise_input'], inputs['hi_res_inputs']]
                            pred_ensemble.append(data.denormalise(model.gen.decoder.predict(dec_inputs))[..., 0])

                # turn accumulated list into numpy array
                pred_ensemble = np.stack(pred_ensemble, axis=1)  # shape: batch_size x ensemble_mem x H x W

                # list is large, so force garbage collect
                gc.collect()
            else:
                # pred_ensemble will be batch_size x ens x H x W
                if model_number == "ecPoint-nocorr":
                    pred_ensemble = benchmarks.ecpointmodel(inputs['lo_res_inputs'],
                                                            data_format="channels_first")
                elif model_number == "ecPoint-partcorr":
                    pred_ensemble = benchmarks.ecpointboxensmodel(inputs['lo_res_inputs'],
                                                                  data_format="channels_first")
                elif model_number == "ecPoint-fullcorr":
                    # this has ens=100 every time
                    pred_ensemble = benchmarks.ecpointPDFmodel(inputs['lo_res_inputs'],
                                                               data_format="channels_first")
                elif model_number == "ecPoint-mean":
                    # this has ens=100 every time
                    pred_ensemble = benchmarks.ecpointPDFmodel(inputs['lo_res_inputs'],
                                                               data_format="channels_first")
                    pred_ensemble = np.mean(pred_ensemble, axis=1)
                    pred_ensemble = np.expand_dims(pred_ensemble, 1)  # batch_size x 1 x H x W
                elif model_number == "constupsc":
                    pred_ensemble = np.expand_dims(inputs['lo_res_inputs'][:, :, :, 1], 1)
                    pred_ensemble = np.repeat(np.repeat(pred_ensemble, 10, axis=-1), 10, axis=-2)
                else:
                    raise RuntimeError('Unknown model_number {}' % model_number)

            for kk in range(batch_size):
                for pv in precip_values:
                    for spasc in spatial_scales:
                        # method 1: "ensemble skill"
                        fss_ens_accum(method1[model_number][pv][spasc]["fssobj"],
                                      pred_ensemble[kk, :, :, :],
                                      im_real[kk, :, :])
                        # method 2: "ensemble member skill"
                        fss_accumall(method2[model_number][pv][spasc]["fssobj"],
                                     pred_ensemble[kk, :, :, :],
                                     im_real[kk, :, :])
                gc.collect()

            # pred_ensemble is pretty large
            del im_real
            del pred_ensemble
            gc.collect()

        # once image iteration is done, might as well compute score for this method!
        for pv in precip_values:
            for spasc in spatial_scales:
                method1[model_number][pv][spasc]["score"] = fss_compute(method1[model_number][pv][spasc]["fssobj"])
                method2[model_number][pv][spasc]["score"] = fss_compute(method2[model_number][pv][spasc]["fssobj"])

        if model_number in model_numbers:
            fname1 = "FSS-GAN-" + str(model_number) + "-1.pickle"
            fname2 = "FSS-GAN-" + str(model_number) + "-2.pickle"
        else:
            fname1 = "FSS-" + model_number + "-1.pickle"
            fname2 = "FSS-" + model_number + "-2.pickle"
        fnamefull1 = os.path.join(log_folder, fname1)
        fnamefull2 = os.path.join(log_folder, fname2)

        with open(fnamefull1, 'wb') as f:
            pickle.dump(method1[model_number], f)
        with open(fnamefull2, 'wb') as f:
            pickle.dump(method2[model_number], f)

#     # do plots, one for each precip_value
#     for pv in precip_values:
#         plt.figure(figsize=(7, 5))
#         lw = 1
#         colors = ['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'navy']
#         for modnum, color in zip(model_numbers_ec, colors):
#             m1_scores = [method1[modnum][pv][spasc]["score"] for spasc in spatial_scales]
#             m2_scores = [method2[modnum][pv][spasc]["score"] for spasc in spatial_scales]
#             plt.plot(spatial_scales, m1_scores, '-', color=color, lw=lw,
#                      label=f"{modnum} method 1")
#             plt.plot(spatial_scales, m2_scores, ':', color=color, lw=lw,
#                      label=f"{modnum} method 2")

#         # plt.plot([0, 1], [0, 1], 'k--', lw=lw)  # no-skill
#         # plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.0])
#         plt.xlabel('Spatial scale (km)')
#         plt.ylabel('Fractions skill score (FSS)')
#         plt.title(f'FSS curve for precip threshold {pv}')
#         plt.legend(loc="best")
#         plt.savefig("{}/FSS-{}.pdf".format(log_folder, pv), bbox_inches='tight')
#         plt.close()


# Probably don't use this function directly -- ATTM
# def fss(X_f, X_o, thr, scale):
#     """
#     Compute the fractions skill score (FSS) for a deterministic forecast field
#     and the corresponding observation field.
#     Parameters
#     ----------
#     X_f: array_like
#         Array of shape (m, n) containing the forecast field.
#     X_o: array_like
#         Array of shape (m, n) containing the observation field.
#     thr: float
#         The intensity threshold.
#     scale: int
#         The spatial scale in pixels. In practice, the scale represents the size
#         of the moving window that it is used to compute the fraction of pixels
#         above the threshold.
#     Returns
#     -------
#     out: float
#         The fractions skill score between 0 and 1.
#     References
#     ----------
#     :cite:`RL2008`, :cite:`EWWM2013`
#     """

#     fss = fss_init(thr, scale)
#     fss_accum(fss, X_f, X_o)
#     return fss_compute(fss)


def fss_init(thr, scale):
    """Initialize a fractions skill score (FSS) verification object.
    Parameters
    ----------
    thr: float
        The intensity threshold.
    scale: float
        The spatial scale in pixels. In practice, the scale represents the size
        of the moving window that it is used to compute the fraction of pixels
        above the threshold.
    Returns
    -------
    fss: dict
        The initialized FSS verification object.
    """
    fss = dict(thr=thr, scale=scale, sum_fct_sq=0.0, sum_fct_obs=0.0, sum_obs_sq=0.0)

    return fss


def fss_accumall(fss, X_f, X_o):
    """Accumulate ensemble forecast-observation pairs to an FSS object.
    Does each ensemble member separately.
    Parameters
    -----------
    fss: dict
        The FSS object initialized with
        :py:func:`pysteps.verification.spatialscores.fss_init`.
    X_f: array_like
        Array of shape (c, m, n) containing an ensemble of c forecast fields.
    X_o: array_like
        Array of shape (m, n) containing the observation field.
    """
    if len(X_f.shape) != 3 or len(X_o.shape) != 2 or X_f.shape[-2:] != X_o.shape:
        message = "X_f and X_o must be three- and two-dimensional arrays"
        message += " having the same image dimensions"
        raise ValueError(message)

#     X_f = X_f.copy()
#     X_f[~np.isfinite(X_f)] = fss["thr"] - 1
#     X_o = X_o.copy()
#     X_o[~np.isfinite(X_o)] = fss["thr"] - 1

    # Convert to binary fields with the given intensity threshold
    I_f = (X_f >= fss["thr"]).astype(np.single)
    I_o = (X_o >= fss["thr"]).astype(np.single)

    # Compute fractions of pixels above the threshold within a square
    # neighboring area by applying a 2D moving average to the binary fields
    if fss["scale"] > 1:
        S_o = uniform_filter(I_o, size=fss["scale"], mode="constant", cval=0.0)
    else:
        S_o = I_o

    for ii in range(X_f.shape[0]):
        if fss["scale"] > 1:
            S_f = uniform_filter(I_f[ii, :, :], size=fss["scale"], mode="constant", cval=0.0)
        else:
            S_f = I_f[ii, :, :]

        fss["sum_obs_sq"] += np.nansum(S_o ** 2)
        fss["sum_fct_obs"] += np.nansum(S_f * S_o)
        fss["sum_fct_sq"] += np.nansum(S_f ** 2)


def fss_ens_accum(fss, X_f, X_o):
    """Accumulate ensemble forecast-observation pairs to an FSS object.
    Does ensemble mean of thresholded arrays.
    Parameters
    -----------
    fss: dict
        The FSS object initialized with
        :py:func:`pysteps.verification.spatialscores.fss_init`.
    X_f: array_like
        Array of shape (c, m, n) containing an ensemble of c forecast fields.
    X_o: array_like
        Array of shape (m, n) containing the observation field.
    """
    if len(X_f.shape) != 3 or len(X_o.shape) != 2 or X_f.shape[-2:] != X_o.shape:
        message = "X_f and X_o must be three- and two-dimensional arrays"
        message += " having the same image dimensions"
        raise ValueError(message)

#     X_f = X_f.copy()
#     X_f[~np.isfinite(X_f)] = fss["thr"] - 1
#     X_o = X_o.copy()
#     X_o[~np.isfinite(X_o)] = fss["thr"] - 1

    # Convert to binary fields with the given intensity threshold
    I_f = np.mean((X_f >= fss["thr"]), axis=0, dtype=np.single)
    I_o = (X_o >= fss["thr"]).astype(np.single)

    # Compute fractions of pixels above the threshold within a square
    # neighboring area by applying a 2D moving average to the binary fields
    if fss["scale"] > 1:
        S_f = uniform_filter(I_f, size=fss["scale"], mode="constant", cval=0.0)
        S_o = uniform_filter(I_o, size=fss["scale"], mode="constant", cval=0.0)
    else:
        S_f = I_f
        S_o = I_o

    fss["sum_obs_sq"] += np.nansum(S_o ** 2)
    fss["sum_fct_obs"] += np.nansum(S_f * S_o)
    fss["sum_fct_sq"] += np.nansum(S_f ** 2)


def fss_merge(fss_1, fss_2):
    """Merge two FSS objects.
    Parameters
    ----------
    fss_1: dict
      A FSS object initialized with
      :py:func:`pysteps.verification.spatialscores.fss_init`.
      and populated with
      :py:func:`pysteps.verification.spatialscores.fss_accum`.
    fss_2: dict
      Another FSS object initialized with
      :py:func:`pysteps.verification.spatialscores.fss_init`.
      and populated with
      :py:func:`pysteps.verification.spatialscores.fss_accum`.
    Returns
    -------
    out: dict
      The merged FSS object.
    """

    # checks
    if fss_1["thr"] != fss_2["thr"]:
        raise ValueError(
            "cannot merge: the thresholds are not same %s!=%s"
            % (fss_1["thr"], fss_2["thr"])
        )
    if fss_1["scale"] != fss_2["scale"]:
        raise ValueError(
            "cannot merge: the scales are not same %s!=%s"
            % (fss_1["scale"], fss_2["scale"])
        )

    # merge the FSS objects
    fss = fss_1.copy()
    fss["sum_obs_sq"] += fss_2["sum_obs_sq"]
    fss["sum_fct_obs"] += fss_2["sum_fct_obs"]
    fss["sum_fct_sq"] += fss_2["sum_fct_sq"]

    return fss


def fss_compute(fss):
    """Compute the FSS.
    Parameters
    ----------
    fss: dict
       An FSS object initialized with
       :py:func:`pysteps.verification.spatialscores.fss_init`
       and accumulated with
       :py:func:`pysteps.verification.spatialscores.fss_accum`.
    Returns
    -------
    out: float
        The computed FSS value.
    """
    numer = fss["sum_fct_sq"] - 2.0 * fss["sum_fct_obs"] + fss["sum_obs_sq"]
    denom = fss["sum_fct_sq"] + fss["sum_obs_sq"]

    return 1.0 - numer / denom
