import os
import sys
import sys
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
import json
import argparse
from joblib import dump
import tensorflow as tf
from downstream_models import CnnLstmModel, Encoder_MLP
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

        elif config["classifier_name"] == 'MLP':
            # Encoder_MLP Model
            results = [Encoder_MLP(config, train_data, val_data, test_data, rep).evaluation() for rep in range(config["repeat"])]
            dump([results], os.path.join(config["results_directory"], config["dataset_name"], f'{config["encoder_name"]}+{config["classifier_name"]}_results.pkl'))

        else:
            raise ValueError("Inexistent classifier name ...")
        
        
    else:
        raise ValueError("Inexistent dataset name ...")

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='SSL BP estimation - Downstream Stage')
        
    # configuration files
    parser.add_argument('--config_file_path', nargs='?', type=str, help='path to the configuration file (json)', default='Downstream_config_mimic_iii.json')
    
    args = parser.parse_args()
    
    main(args)

