# Downscaling cGAN requirements:

Use conda (even better, mamba); install everything from conda-forge
- Python 3
- TensorFlow 2
    - we use v2.7; other versions should require no/minimal changes
    - GPU strongly recommended (use a GPU-enabled TensorFlow build, compatable with the installed CUDA version)
- matplotlib
- seaborn
- cartopy
- jupyter
- xarray
- pandas
- netcdf4
- h5netcdf
- scikit-learn
- cfgrib
- dask
- tqdm
- properscoring
- climlab
- iris

May also require cudatoolkit

# High-level data overview

You should have two main datasets:
1. Forecast data (low-resolution)
2. Truth data, for example radar (high-resolution)

All images in each dataset should be the same size, and there should be a constant resolution scaling factor between them.

In the paper, we also used a third, static, dataset:

3. "Constant" data - orography and land-sea mask, at the same resolution as the truth data.

Ideally these datasets are as 'clean' as possible.  We recommend you generate these with offline scripts that perform all regridding, cropping, interpolation, etc., so that the files can be loaded in and read directly with as little further processing as possible.  We saved these as netCDF files, which we read in using xarray.

We assume that it is possible to perform inference using full-size images, but that the images are too large to use whole during training.

For this reason, part of our dataflow involves generating training data separately (small portions of the full image), and storing these in .tfrecords files.  We think this has two advantages:
1. The training data is in an 'optimised' format; only the data needed for training is loaded in, rather than needing to open several files and extract a single timestep from each.
2. The training data can be split into different bins, and data can be drawn from these bins in a specific ratio.  We use this to artificially increase the prevalence of rainy data.

# Setting up the downscaling cGAN with your own data







# Training and evaluating a network

1. Training the model

First, set up your model parameters in the configuration (.yaml) file.
An example is provided in the main directory. We recommend copying this 
file to somewhere on your local machine before training.

Run the following to start the training:

python main.py --config <path/to/config/file>

There are a number of options you can use at this point. These will 
evaluate your model after it has finished training:

--rank_small or --rank_full flags will run (small/large image) CRPS/rank based evalution
--qual_small or --qual_full flags will run (small/large image) image quality based evalution
--plot_ranks will plot rank histograms
	     N.B. you must run rank based eval first
	   
If you choose to run --rank_small/full and/or --qual_small/full, you must
also specify if you want to do this for all model iterations or 
just a selection. Do this using 

- --eval_full	  (all model checkpoints)
- --eval_short	  (the final 1/3rd of model checkpoints)
- --eval_blitz	  (the final 4 model checkpoints)

Two things to note:
- These three options work well with the 100 checkpoints that we 
have been working with. If this changes, you may want to update
them accordingly.
- Calculating everything, for all model iterations, will take a long 
time. Possibly weeks. You have been warned. 

As an example, to train a model and evaluate it fully but for only the
last few model iterations, on full images, you could run:

python main.py --config <path/to/config/file> --eval_blitz --rank_full
--qual_full --plot_ranks

2. If you've already trained your model, and you just want to run some 
evaluation, use the --no_train flag, for example:

python main.py --config <path/to/config/file> --no_train --eval_full

3. To generate plots of the output from a trained model, use predict.py
This requires a path to the directory where the weights of the model are 
stored, and will read in the setup parameters directly. Use the following
arguments:

Necessary arguments:
- --log_folder	    <path/to/model/directory>
- --model_number    model iteration you want to use to predict

Optional arguments:
- --predict_year	    year of data to predict on (2019/2020)
- --num_samples             number of different input images to predict on
- --pred_ensemble_size	    number of predictions to draw from ensemble
		    N.B. if you run mode == 'det' we have hardcoded this
		    to 1 for obvious reasons.

There are also the following optional arguments:
- --predict_full_image          to predict on the full image dataset
- --include_Lanczos     	includes Lanczos interpolation prediction
- --include_ecPoint     	includes ecPoint-based prediction
- --include_RainFARM    	includes RainFARM prediction
- --include_deterministic       includes deterministic prediction for comparison.
			N.B. Probably don't use this for comparing against
			a deterministic model. 

For example:

python predict.py --log_folder <path/to/model> --model_number 0006400 
--num_samples 7 --predict_full_image --include_ecPoint

4. If, for whatever reason, you don't want to go through main.py but you
want to do some evaluation on specific models only, you can use the scripts

- run_eval.py
- run_qual.py
- run_roc.py

You'll have to open them and hardcode file paths and model numbers at the
top. I also made a plot_comparisons.py at one point but I've deleted it so
now you have to use predict.py. Sorry.

5. If you want to change model architectures, add them to the models.py
file. You can then use the config.yaml file to call a different model
architecture. We recommend not deleting the old one just yet.

6. run_benchmarks.py will generate benchmark scores for CRPS, RMSE, MAE
and RAPSD for the specified benchmark models (input arguments). We have
set this up to use the same evaluation setup as the NN models but note
the benchmark evaluation uses 100 ensemble members (NN default is 10 for
time & memory reasons). So, for a proper comparison you *must* evaluate
your model and specify 100 ensemble members.

For example:
python run_benchmarks.py --log_folder /path/to/model --include_Lanczos 
--include_ecPoint --include_RainFARM --include_ecPoint_mean 
--include_constant --include_zeros

7. This is research code so please let us know if something is wrong and
also note that it definitely isn't perfect :)
