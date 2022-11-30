import os
import yaml

import matplotlib; matplotlib.use("Agg")  # noqa

import evaluation
import read_config

read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode

# input parameters
log_folder = '/network/group/aopp/predict/TIP018_HARRIS_TENSORFL/andrew-output/rev-mainGAN'; model_numbers = [460800]  # noqa: E702
val_years = 2020

model_weights_root = os.path.join(log_folder, "models")
config_path = os.path.join(log_folder, 'setup_params.yaml')
with open(config_path, 'r') as f:
    try:
        setup_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

mode = setup_params["GENERAL"]["mode"]
arch = setup_params["MODEL"]["architecture"]
padding = setup_params["MODEL"]["padding"]
problem_type = setup_params["GENERAL"]["problem_type"]
filters_gen = setup_params["GENERATOR"]["filters_gen"]
noise_channels = setup_params["GENERATOR"]["noise_channels"]
latent_variables = setup_params["GENERATOR"]["latent_variables"]
filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]
add_noise = setup_params["EVAL"]["add_postprocessing_noise"]
noise_factor = setup_params["EVAL"]["postprocessing_noise_factor"]
noise_factor = float(noise_factor)
max_pooling = setup_params["EVAL"]["max_pooling"]
avg_pooling = setup_params["EVAL"]["avg_pooling"]
num_images = 256

if problem_type == 'normal':
    input_channels = 9
    autocoarsen = False
elif problem_type == 'autocoarsen':
    input_channels = 1
    autocoarsen = True

if mode in ("GAN", "VAEGAN"):
    ensemble_size = 100
elif mode == "det":
    ensemble_size = 1

out_fn = "{}/eval_{}.txt".format(log_folder, str(val_years))

evaluation.evaluate_multiple_checkpoints(mode=mode,
                                         arch=arch,
                                         val_years=val_years,
                                         log_fname=out_fn,
                                         weights_dir=model_weights_root,
                                         autocoarsen=autocoarsen,
                                         add_noise=add_noise,
                                         noise_factor=noise_factor,
                                         model_numbers=model_numbers,
                                         ranks_to_save=model_numbers,
                                         num_images=num_images,
                                         filters_gen=filters_gen,
                                         filters_disc=filters_disc,
                                         input_channels=input_channels,
                                         latent_variables=latent_variables,
                                         noise_channels=noise_channels,
                                         padding=padding,
                                         ensemble_size=ensemble_size)
