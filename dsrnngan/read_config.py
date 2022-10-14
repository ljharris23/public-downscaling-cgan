import os
import sys
import yaml

import tensorflow as tf


def read_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_config.yaml')
    try:
        with open(config_path, 'r') as f:
            try:
                localconfig = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)
                sys.exit(1)
            return localconfig
    except FileNotFoundError as e:
        print(e)
        print("You must set local_config.yaml in the main folder. Copy local_config-example.yaml and adjust appropriately.")
        sys.exit(1)


def get_data_paths():
    data_config_paths = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_paths.yaml')

    try:
        with open(data_config_paths, 'r') as f:
            try:
                all_data_paths = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)
                sys.exit(1)
    except FileNotFoundError as e:
        print(e)
        print("data_paths.yaml not found. Should exist in main folder")
        sys.exit(1)

    lc = read_config()
    data_paths = all_data_paths[lc['data_paths']]
    return data_paths


def set_gpu_mode():
    lc = read_config()
    if lc['use_gpu']:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # remove environment variable (if it doesn't exist, nothing happens)
        # set memory allocation to incremental if desired
        if lc['gpu_mem_incr']:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)
        else:
            # if gpu_mem_incr False, do nothing
            pass
        if lc['disable_tf32']:
            tf.config.experimental.enable_tensor_float_32_execution(False)
    else:
        # if use_gpu False
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def read_downscaling_factor():
    df_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downscaling_factor.yaml')
    try:
        with open(df_path, 'r') as f:
            try:
                df = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)
                sys.exit(1)
            return df
    except FileNotFoundError as e:
        print(e)
        print("downscaling_factor.yaml not found in the main folder.")
        sys.exit(1)
