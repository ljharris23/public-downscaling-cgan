import os
import yaml

import fss
import read_config

read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode

# input parameters
log_folder = '/network/group/aopp/predict/TIP018_HARRIS_TENSORFL/andrew-output/rev-mainGAN'; model_numbers = [460800]
plot_upscale = False
predict_year = 2020
predict_full_image = True
ensemble_members = 100

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

fss.plot_fss_curves(mode=mode,
                    arch=arch,
                    log_folder=log_folder,
                    model_numbers=model_numbers,
                    weights_dir=model_weights_root,
                    problem_type=problem_type,
                    filters_gen=filters_gen,
                    filters_disc=filters_disc,
                    noise_channels=noise_channels,
                    latent_variables=latent_variables,
                    padding=padding,
                    predict_year=predict_year,
                    predict_full_image=predict_full_image,
                    ensemble_members=ensemble_members,
                    plot_upscale=plot_upscale)