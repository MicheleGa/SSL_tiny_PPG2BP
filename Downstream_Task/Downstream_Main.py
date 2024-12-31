import os
import sys
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(script_dir, '../Pretext_Task') 
sys.path.insert(0, module_dir)
import json
import argparse
from joblib import dump

from Downstream_Models import CnnLstmModel, Encoder_MLP

from Pretext_DataProcess_mimic_iii import prepare_mimic_iii_dataset


# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

