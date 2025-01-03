import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))

# List of directories to add to the Python path
directories_to_add = [
    '../models',
    '../data_utilities'
]

# Insert directories at the beginning of the path (for higher priority)
for directory in directories_to_add:
    module_dir = os.path.join(script_dir, directory) 
    sys.path.insert(0, module_dir) 
import datetime
import json
import argparse
from joblib import dump
import tensorflow as tf
from downstream_models import CnnLstmModel, Encoder_MLP, MN_MLP
from prepare_and_split_mimic_iii import prepare_mimic_iii_dataset


# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# XLA optimization for faster performance(up to 10-15 minutes total time saved)
tf.config.optimizer.set_jit(True)


def main(args):

    # Load configuration
    with open(args.config_file_path) as config_file:
        config = json.load(config_file)

    if config["dataset_name"] == 'mimic_iii':
        
        train_data, val_data, test_data = prepare_mimic_iii_dataset(config, downstream_task=True, classifier_name=config["classifier_name"]) 

        if config["classifier_name"] == 'CNN-LSTM':
            # Cnn-Lstm Model
            results = [CnnLstmModel(config, train_data, val_data, test_data, rep).evaluation() for rep in range(config["repeat"])]
            dump([results], os.path.join(config["results_directory"], config["dataset_name"], f'{config["classifier_name"]}_results.pkl'))
            print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Completed {config["classifier_name"]} training on {config["dataset_name"]} dataset ...')

        elif config["classifier_name"] == 'MLP':
            # Encoder_MLP Model
            results = [Encoder_MLP(config, train_data, val_data, test_data, rep).evaluation() for rep in range(config["repeat"])]
            dump([results], os.path.join(config["results_directory"], config["dataset_name"], f'{config["encoder_name"]}+{config["classifier_name"]}_results.pkl'))
            print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Completed {config["classifier_name"]} training with {config["encoder_name"]} on {config["dataset_name"]} dataset ...')

        elif config["classifier_name"] == 'MN_MLP':
            # MobileNet Model
            results = [MN_MLP(config, train_data, val_data, test_data, rep).evaluation() for rep in range(config["repeat"])]
            dump([results], os.path.join(config["results_directory"], config["dataset_name"], f'{config["encoder_name"]}+{config["classifier_name"]}_results.pkl'))
            print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Completed {config["classifier_name"]} training with {config["encoder_name"]} on {config["dataset_name"]} dataset ...')

        else:
            raise ValueError("Inexistent classifier name ...")
        
    else:
        raise ValueError("Inexistent dataset name ...")

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='SSL BP estimation - Downstream Stage')
        
    # configuration files
    parser.add_argument('--config_file_path', nargs='?', type=str, help='path to the configuration file (json)', default='downstream_config_mimic_iii.json')
    
    args = parser.parse_args()

    # Get a list of available GPUs
    gpus = tf.config.list_physical_devices('GPU')

    # Select the desired GPU (e.g., the first GPU)
    gpu_to_use = gpus[0] 

    # Make the selected GPU visible to TensorFlow
    tf.config.set_visible_devices(gpu_to_use, 'GPU')

    # Verify that only the selected GPU is visible
    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Using GPU {tf.config.list_logical_devices("GPU")}') 
    
    main(args)

