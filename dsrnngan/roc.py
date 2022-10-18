import gc
import os
import pickle

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

import data
import setupmodel
from benchmarks import nn_interp_model
from data import all_fcst_fields, get_dates
from data_generator import DataGenerator as DataGeneratorFull
from evaluation import _init_VAEGAN
from noise import NoiseGenerator
from pooling import pool
from read_config import read_downscaling_factor
from tfrecords_generator import create_fixed_dataset


def calculate_roc(*,
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
                  ensemble_members,
                  calc_upsample):

    ds_fac = read_downscaling_factor()["downscaling_factor"]
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

    pooling_methods = ['no_pooling', 'max_4', 'max_16', 'avg_4', 'avg_16']

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
        dates = get_dates(predict_year)
        data_predict = DataGeneratorFull(dates=dates,
                                         fcst_fields=all_fcst_fields,
                                         batch_size=batch_size,
                                         log_precip=True,
                                         shuffle=True,
                                         constants=True,
                                         hour='random',
                                         fcst_norm=True,
                                         downsample=downsample)

    if not predict_full_image:
        data_predict = create_fixed_dataset(predict_year,
                                            batch_size=batch_size,
                                            downsample=downsample)

    if calc_upsample:
        if not predict_full_image:
            raise RuntimeError('Data generator for benchmarks not implemented for small images')
        # requires a different data generator with different fields and no fcst_norm
        data_benchmarks = DataGeneratorFull(dates=dates,
                                            fcst_fields=all_fcst_fields,
                                            batch_size=batch_size,
                                            log_precip=False,
                                            shuffle=True,
                                            constants=True,
                                            hour="random",
                                            fcst_norm=False)
        tpidx = all_fcst_fields.index('tp')

    auc_scores_roc = {}  # will only contain GAN AUCs; used for "progress vs time" plot
    auc_scores_pr = {}
    for method in pooling_methods:
        auc_scores_roc[method] = []
        auc_scores_pr[method] = []

    # tidier to iterate over GAN checkpoints and NN-interp using joint code
    model_numbers_ec = model_numbers.copy()
    if calc_upsample:
        model_numbers_ec.extend(["nn_interp"])

    for model_number in model_numbers_ec:
        print(f"calculating for model number {model_number}")
        if model_number in model_numbers:
            # only load weights for GAN, not upscale
            gen_weights_file = os.path.join(weights_dir, "gen_weights-{:07d}.h5".format(model_number))
            if not os.path.isfile(gen_weights_file):
                print(gen_weights_file, "not found, skipping")
                continue
            print(gen_weights_file)
            if mode == "VAEGAN":
                _init_VAEGAN(model.gen, data_predict, predict_full_image, batch_size, latent_variables)
            model.gen.load_weights(gen_weights_file)

        y_true = {}
        y_score = {}
        for method in pooling_methods:
            y_true[method] = {}
            y_score[method] = {}
            for value in precip_values:
                y_true[method][value] = []
                y_score[method][value] = []

        if model_number in model_numbers:
            data_pred_iter = iter(data_predict)  # "restarts" GAN data iterator
        else:
            data_benchmarks_iter = iter(data_benchmarks)  # benchmarks data iterator

        for ii in range(num_batches):
            print(ii, num_batches)

            if model_number in model_numbers:
                # GAN, not upscale
                inputs, outputs = next(data_pred_iter)
                # need to denormalise
                if predict_full_image:
                    im_real = data.denormalise(outputs['output']).astype(np.single)  # shape: batch_size x H x W
                else:
                    im_real = (data.denormalise(outputs['output'])[..., 0]).astype(np.single)
            else:
                # upscale, no need to denormalise
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
                    for j in range(ensemble_members):
                        inputs['noise_input'] = noise_gen()
                        if mode == 'GAN':
                            pred_ensemble.append(data.denormalise(model.gen.predict([inputs['lo_res_inputs'],
                                                                                     inputs['hi_res_inputs'],
                                                                                     inputs['noise_input']]))[..., 0])
                        elif mode == 'VAEGAN':
                            dec_inputs = [mean, logvar, inputs['noise_input'], inputs['hi_res_inputs']]
                            pred_ensemble.append(data.denormalise(model.gen.decoder.predict(dec_inputs))[..., 0])

                # turn accumulated list into numpy array
                pred_ensemble = np.stack(pred_ensemble, axis=1)  # shape: batch_size x ens x H x W

                # list is large, so force garbage collect
                gc.collect()
            else:
                # pred_ensemble will be batch_size x ens x H x W
                if model_number == "nn_interp":
                    pred_ensemble = np.expand_dims(inputs['lo_res_inputs'][:, :, :, tpidx], 1)
                    pred_ensemble = nn_interp_model(pred_ensemble, ds_fac)

                else:
                    raise RuntimeError('Unknown model_number {}' % model_number)

            # need to calculate averages each batch; can't store n_images x n_ensemble x H x W!
            for method in pooling_methods:
                if method == 'no_pooling':
                    im_real_pooled = im_real.copy()
                    pred_ensemble_pooled = pred_ensemble.copy()
                else:
                    # im_real only has 3 dims but the pooling needs 4,
                    # so add a fake extra dim and squeeze back down
                    im_real_pooled = np.expand_dims(im_real, axis=1)
                    im_real_pooled = pool(im_real_pooled, method, data_format='channels_first')
                    im_real_pooled = np.squeeze(im_real_pooled, axis=1)
                    pred_ensemble_pooled = pool(pred_ensemble, method, data_format='channels_first')

                for value in precip_values:
                    # binary instance of truth > threshold
                    # append an array of shape batch_size x H x W
                    y_true[method][value].append((im_real_pooled > value))
                    # check what proportion of pred > threshold
                    # collapse over ensemble dim, so append an array also of shape batch_size x H x W
                    y_score[method][value].append(np.mean(pred_ensemble_pooled > value, axis=1, dtype=np.single))
                del im_real_pooled
                del pred_ensemble_pooled
                gc.collect()

            # pred_ensemble is pretty large
            del im_real
            del pred_ensemble
            gc.collect()

        for method in pooling_methods:
            for value in precip_values:
                # turn list of batch_size x H x W into a single array
                y_true[method][value] = np.concatenate(y_true[method][value], axis=0)  # n_images x W x H
                gc.collect()  # clean up the list representation of y_true[value]

            for value in precip_values:
                # ditto
                y_score[method][value] = np.concatenate(y_score[method][value], axis=0)  # n_images x W x H
                gc.collect()

        fpr = {}; tpr = {}; rec = {}; pre = {}; baserates = {}; roc_auc = {}; pr_auc = {}  # noqa
        for method in pooling_methods:
            fpr[method] = []  # list of ROC fprs
            tpr[method] = []  # list of ROC tprs
            rec[method] = []  # list of precision-recall recalls
            pre[method] = []  # list of precision-recall precisions
            baserates[method] = []  # precision-recall 'no-skill' levels
            roc_auc[method] = []  # list of ROC AUCs
            pr_auc[method] = []  # list of p-r AUCs

            print("Computing ROC and prec-recall for", method)
            for value in precip_values:
                # Compute ROC curve and ROC area for each precip value
                fpr_pv, tpr_pv, _ = roc_curve(np.ravel(y_true[method][value]), np.ravel(y_score[method][value]), drop_intermediate=False)
                gc.collect()
                pre_pv, rec_pv, _ = precision_recall_curve(np.ravel(y_true[method][value]), np.ravel(y_score[method][value]))
                gc.collect()
                # note: fpr_pv, tpr_pv, etc., are at most the size of the number of unique values of y_score.
                # for us, this is just "fraction of ensemble members > threshold" which is relatively small,
                # but if y_score took arbirary values, this could be really large (particularly with drop_intermediate=False)
                roc_auc_pv = auc(fpr_pv, tpr_pv)
                pr_auc_pv = auc(rec_pv, pre_pv)
                fpr[method].append(fpr_pv)
                tpr[method].append(tpr_pv)
                pre[method].append(pre_pv)
                rec[method].append(rec_pv)
                baserates[method].append(y_true[method][value].mean())
                roc_auc[method].append(roc_auc_pv)
                pr_auc[method].append(pr_auc_pv)
                del y_true[method][value]
                del y_score[method][value]
                gc.collect()

            if model_number in model_numbers:
                # i.e., don't do this for upscale
                auc_scores_roc[method].append(np.array(roc_auc[method]))
                auc_scores_pr[method].append(np.array(pr_auc[method]))

        if model_number in model_numbers:
            fname1 = "ROC-GAN-" + str(model_number) + "-fpr.pickle"
            fname2 = "ROC-GAN-" + str(model_number) + "-tpr.pickle"
            fname3 = "ROC-GAN-" + str(model_number) + "-auc.pickle"
            fname4 = "PRC-GAN-" + str(model_number) + "-rec.pickle"
            fname5 = "PRC-GAN-" + str(model_number) + "-pre.pickle"
            fname6 = "PRC-GAN-" + str(model_number) + "-auc.pickle"
            fname7 = "PRC-GAN-" + str(model_number) + "-base.pickle"
        else:
            fname1 = "ROC-" + model_number + "-fpr.pickle"
            fname2 = "ROC-" + model_number + "-tpr.pickle"
            fname3 = "ROC-" + model_number + "-auc.pickle"
            fname4 = "PRC-" + model_number + "-rec.pickle"
            fname5 = "PRC-" + model_number + "-pre.pickle"
            fname6 = "PRC-" + model_number + "-auc.pickle"
            fname7 = "PRC-" + model_number + "-base.pickle"
        fnamefull1 = os.path.join(log_folder, fname1)
        fnamefull2 = os.path.join(log_folder, fname2)
        fnamefull3 = os.path.join(log_folder, fname3)
        fnamefull4 = os.path.join(log_folder, fname4)
        fnamefull5 = os.path.join(log_folder, fname5)
        fnamefull6 = os.path.join(log_folder, fname6)
        fnamefull7 = os.path.join(log_folder, fname7)

        with open(fnamefull1, 'wb') as f:
            pickle.dump(fpr, f)
        with open(fnamefull2, 'wb') as f:
            pickle.dump(tpr, f)
        with open(fnamefull3, 'wb') as f:
            pickle.dump(roc_auc, f)
        with open(fnamefull4, 'wb') as f:
            pickle.dump(rec, f)
        with open(fnamefull5, 'wb') as f:
            pickle.dump(pre, f)
        with open(fnamefull6, 'wb') as f:
            pickle.dump(pr_auc, f)
        with open(fnamefull7, 'wb') as f:
            pickle.dump(baserates, f)
