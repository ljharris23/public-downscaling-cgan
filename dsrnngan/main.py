import argparse
import json
import os
import yaml
from pathlib import Path

import matplotlib; matplotlib.use("Agg")  # noqa: E702
import numpy as np
import pandas as pd

import evaluation
import plots
import read_config
import setupdata
import setupmodel
import train


if __name__ == "__main__":
    read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode
    df_dict = read_config.read_downscaling_factor()  # read downscaling params

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration file")
    parser.set_defaults(do_training=True)
    parser.add_argument('--no_train', dest='do_training', action='store_false',
                        help="Do NOT carry out training, only perform eval")
    parser.add_argument('--restart', dest='restart', action='store_true',
                        help="Restart training from latest checkpoint")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--eval_full', dest='evalnum', action='store_const', const="full")
    group.add_argument('--eval_short', dest='evalnum', action='store_const', const="short")
    group.add_argument('--eval_blitz', dest='evalnum', action='store_const', const="blitz")
    parser.set_defaults(evalnum=None)
    parser.set_defaults(rank=False)
    parser.set_defaults(qual=False)
    parser.set_defaults(plot_ranks=False)
    parser.add_argument('--rank', dest='rank', action='store_true',
                        help="Include CRPS/rank evaluation on full-size images")
    parser.add_argument('--qual', dest='qual', action='store_true',
                        help="Include image quality metrics on full-size images")
    parser.add_argument('--plot_ranks', dest='plot_ranks', action='store_true',
                        help="Plot rank histograms")
    args = parser.parse_args()

    if args.evalnum is None and (args.rank or args.qual):
        raise RuntimeError("You asked for evaluation to occur, but did not pass in '--eval_full', '--eval_short', or '--eval_blitz' to specify length of evaluation")

    # Read in the configurations
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
    lr_gen = setup_params["GENERATOR"]["learning_rate_gen"]
    noise_channels = setup_params["GENERATOR"]["noise_channels"]
    latent_variables = setup_params["GENERATOR"]["latent_variables"]
    filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]
    lr_disc = setup_params["DISCRIMINATOR"]["learning_rate_disc"]
    train_years = setup_params["TRAIN"]["train_years"]
    training_weights = setup_params["TRAIN"]["training_weights"]
    num_samples = setup_params["TRAIN"]["num_samples"]
    steps_per_checkpoint = setup_params["TRAIN"]["steps_per_checkpoint"]
    batch_size = setup_params["TRAIN"]["batch_size"]
    kl_weight = setup_params["TRAIN"]["kl_weight"]
    ensemble_size = setup_params["TRAIN"]["ensemble_size"]
    CLtype = setup_params["TRAIN"]["CL_type"]
    content_loss_weight = setup_params["TRAIN"]["content_loss_weight"]
    val_years = setup_params["VAL"]["val_years"]
    val_size = setup_params["VAL"]["val_size"]
    num_batches = setup_params["EVAL"]["num_batches"]
    add_noise = setup_params["EVAL"]["add_postprocessing_noise"]
    noise_factor = setup_params["EVAL"]["postprocessing_noise_factor"]
    max_pooling = setup_params["EVAL"]["max_pooling"]
    avg_pooling = setup_params["EVAL"]["avg_pooling"]

    # otherwise these are of type string, e.g. '1e-5'
    lr_gen = float(lr_gen)
    lr_disc = float(lr_disc)
    kl_weight = float(kl_weight)
    noise_factor = float(noise_factor)
    content_loss_weight = float(content_loss_weight)

    if mode not in ['GAN', 'VAEGAN', 'det']:
        raise ValueError("Mode type is restricted to 'GAN' 'VAEGAN' 'det'")
    if problem_type not in ['normal', 'superresolution']:
        raise ValueError("Problem type is restricted to 'normal' 'superresolution'")
    if ensemble_size is not None:
        if CLtype not in ["CRPS", "CRPS_phys", "ensmeanMSE", "ensmeanMSE_phys"]:
            raise ValueError("Content loss type is restricted to 'CRPS', 'CRPS_phys', 'ensmeanMSE', 'ensmeanMSE_phys'")

    num_checkpoints = int(num_samples/(steps_per_checkpoint * batch_size))
    checkpoint = 1

    # create log folder and model save/load subfolder if they don't exist
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    model_weights_root = os.path.join(log_folder, "models")
    Path(model_weights_root).mkdir(parents=True, exist_ok=True)

    # save setup parameters
    save_config = os.path.join(log_folder, 'setup_params.yaml')
    with open(save_config, 'w') as outfile:
        yaml.dump(setup_params, outfile, default_flow_style=False)

    if problem_type == "normal":
        downsample = False
        input_channels = 9
    elif problem_type == "superresolution":
        downsample = True
        input_channels = 1
    else:
        raise ValueError("no such problem type, try again!")

    if args.do_training:
        # initialize GAN
        model = setupmodel.setup_model(
            mode=mode,
            arch=arch,
            downscaling_steps=df_dict["steps"],
            input_channels=input_channels,
            latent_variables=latent_variables,
            filters_gen=filters_gen,
            filters_disc=filters_disc,
            noise_channels=noise_channels,
            padding=padding,
            lr_disc=lr_disc,
            lr_gen=lr_gen,
            kl_weight=kl_weight,
            ensemble_size=ensemble_size,
            CLtype=CLtype,
            content_loss_weight=content_loss_weight)

        batch_gen_train, batch_gen_valid = setupdata.setup_data(
            train_years=train_years,
            val_years=val_years,
            val_size=val_size,
            downsample=downsample,
            weights=training_weights,
            batch_size=batch_size,
            load_full_image=False)

        if args.restart:  # load weights and run status

            model.load(model.filenames_from_root(model_weights_root))
            with open(log_folder + "-run_status.json", 'r') as f:
                run_status = json.load(f)
            training_samples = run_status["training_samples"]
            checkpoint = int(training_samples / (steps_per_checkpoint * batch_size)) + 1

            log_file = "{}/log.txt".format(log_folder)
            log = pd.read_csv(log_file)
            log_list = [log]

        else:  # initialize run status
            training_samples = 0

            log_file = os.path.join(log_folder, "log.txt")
            log_list = []

        plot_fname = os.path.join(log_folder, "progress.pdf")

        while (training_samples < num_samples):  # main training loop

            print("Checkpoint {}/{}".format(checkpoint, num_checkpoints))

            # train for some number of batches
            loss_log = train.train_model(model=model,
                                         mode=mode,
                                         batch_gen_train=batch_gen_train,
                                         batch_gen_valid=batch_gen_valid,
                                         noise_channels=noise_channels,
                                         latent_variables=latent_variables,
                                         checkpoint=checkpoint,
                                         steps_per_checkpoint=steps_per_checkpoint,
                                         plot_samples=val_size,
                                         plot_fn=plot_fname)

            training_samples += steps_per_checkpoint * batch_size
            checkpoint += 1

            # save results
            model.save(model_weights_root)
            run_status = {
                "training_samples": training_samples,
            }
            with open(os.path.join(log_folder, "run_status.json"), 'w') as f:
                json.dump(run_status, f)

            data = {"training_samples": [training_samples]}
            for foo in loss_log:
                data[foo] = loss_log[foo]

            log_list.append(pd.DataFrame(data=data))
            log = pd.concat(log_list)
            log.to_csv(log_file, index=False, float_format="%.6f")

            # Save model weights each checkpoint
            gen_weights_file = os.path.join(model_weights_root, "gen_weights-{:07d}.h5".format(training_samples))
            model.gen.save_weights(gen_weights_file)

    else:
        print("Training skipped...")

    rank_fname = os.path.join(log_folder, "rank.txt")
    qual_fname = os.path.join(log_folder, "qual.txt")

    # model iterations to save full rank data to disk for during evaluations;
    # necessary for plot rank histograms. these are large files, so small
    # selection used to avoid storing gigabytes of data
    interval = steps_per_checkpoint * batch_size
    finalchkpt = num_samples // interval
    # last 4 checkpoints, or all checkpoints if < 4
    ranks_to_save = [(finalchkpt - ii)*interval for ii in range(3, -1, -1)] if finalchkpt >= 4 else [ii*interval for ii in range(1, finalchkpt+1)]

    if args.evalnum == "blitz":
        model_numbers = ranks_to_save.copy()  # should not be modifying list in-place, but just in case!
    elif args.evalnum == "short":
        # last 1/3rd of checkpoints
        Neval = max(finalchkpt // 3, 1)
        model_numbers = [(finalchkpt - ii)*interval for ii in range((Neval-1), -1, -1)]
    elif args.evalnum == "full":
        model_numbers = np.arange(0, num_samples + 1, interval)[1:].tolist()

    # evaluate model performance
    if args.qual:
        evaluation.quality_metrics_by_time(mode=mode,
                                           arch=arch,
                                           val_years=val_years,
                                           log_fname=qual_fname,
                                           weights_dir=model_weights_root,
                                           downsample=downsample,
                                           weights=training_weights,
                                           model_numbers=model_numbers,
                                           batch_size=1,  # do inference 1 at a time, in case of memory issues
                                           num_batches=num_batches,
                                           filters_gen=filters_gen,
                                           filters_disc=filters_disc,
                                           input_channels=input_channels,
                                           latent_variables=latent_variables,
                                           noise_channels=noise_channels,
                                           rank_samples=2,
                                           padding=padding)

    if args.rank:
        evaluation.rank_metrics_by_time(mode=mode,
                                        arch=arch,
                                        val_years=val_years,
                                        log_fname=rank_fname,
                                        weights_dir=model_weights_root,
                                        downsample=downsample,
                                        weights=training_weights,
                                        add_noise=add_noise,
                                        noise_factor=noise_factor,
                                        model_numbers=model_numbers,
                                        ranks_to_save=ranks_to_save,
                                        batch_size=1,  # ditto
                                        num_batches=num_batches,
                                        filters_gen=filters_gen,
                                        filters_disc=filters_disc,
                                        input_channels=input_channels,
                                        latent_variables=latent_variables,
                                        noise_channels=noise_channels,
                                        padding=padding,
                                        rank_samples=10,
                                        max_pooling=max_pooling,
                                        avg_pooling=avg_pooling)

    if args.plot_ranks:
        plots.plot_histograms(log_folder, val_years, ranks=ranks_to_save, N_ranks=11)
