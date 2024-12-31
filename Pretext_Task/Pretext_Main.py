import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(script_dir, '../models') 
sys.path.insert(0, module_dir)
import gc
import json
import datetime
import argparse
import joblib
import numpy as np
from numpy.random import seed
import tensorflow as tf
from tensorflow.random import set_seed
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from Pretext_DataProcess_mimic_iii import prepare_mimic_iii_dataset

from barlow_twins import WarmUpCosine, BarlowModel, BarlowLoss


# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# XLA optimization for faster performance(up to 10-15 minutes total time saved)
tf.config.optimizer.set_jit(True)

def main(args):

    # Load configuration
    with open(args.config_file_path) as config_file:
        config = json.load(config_file)

    # Load data using parameters from config and start the preprocessing step
    if config["dataset_name"] == "mimic_iii":

        folders = config["folders"] 
        training_persons = []
        
        for folder in folders:
            patients_folder = os.path.join(config["data_directory"], config["dataset_name"], f'p0{str(folder)}')
            for subject in os.listdir(patients_folder):
                training_persons.append(os.path.join(patients_folder, subject))
        
        train_data, _, _ = prepare_mimic_iii_dataset(config, training_persons)   
        
    else:
        raise ValueError("Inexistent dataset name ...")


    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: {config["dataset_name"]} dataset ready for training ...')

    # Training Model
    repeat = config["repeat"]
    results = []
    for r in range(repeat):

        K.clear_session()

        if config["dataset_name"] == "mimic_iii":

            #train_model(config, train_data, val_data, test_data, r)
            seed(int(config["seed"]))
            set_seed(int(config["seed"]))

            model_params = config["encoder_params"]
            epochs = model_params["epochs"]
            batch_size = model_params["batch_size"]
            optimizer_params = model_params["optimizer"]

            num_samples_per_split = np.load(os.path.join(config["data_directory"], config["dataset_name"], 'num_samples_per_split.npy'))
            n_train_samples = num_samples_per_split[0]
            steps_per_epoch = n_train_samples // batch_size

            warmup_epochs = int(epochs * 0.1)
            warmup_steps = int(warmup_epochs * steps_per_epoch)

            lr_decayed_fn = WarmUpCosine(
                learning_rate_base=1e-3,
                total_steps=epochs * steps_per_epoch,
                warmup_learning_rate=0.0,
                warmup_steps=warmup_steps
            )
            
            optimizer = optimizers.SGD(learning_rate=lr_decayed_fn, **optimizer_params)

            tb = TensorBoard(log_dir=os.path.join(config["saved_model_directory"], config["dataset_name"], f'tensorboard_{config["encoder_name"]}_{config["dataset_name"]}_{repeat}'))

            bm_model = BarlowModel(config)
            
            loss = BarlowLoss(batch_size)

            bm_model.compile(optimizer=optimizer, loss=loss)

            history = bm_model.fit(train_data,
                                   epochs=epochs,
                                   steps_per_epoch=steps_per_epoch,
                                   verbose=1,
                                   callbacks=[tb])
            
            bm_model.mobile_net.save_weights(os.path.join(config["saved_model_directory"], config["dataset_name"], f'{config["encoder_name"]}_{config["dataset_name"]}_{repeat}.h5'))

            print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: End of run {r + 1}/{repeat}')
            results.append(history.history["loss"])
        else:
            raise ValueError("Inexistent dataset name ...")
        
        del bm_model
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

    results_filename = f'{config["encoder_name"]}_results.pkl'
    joblib.dump([results], os.path.join(config["results_directory"], config["dataset_name"], results_filename))

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='SSL BP estimation - Pretext Stage')
        
    # Configuration files
    parser.add_argument('--config_file_path', nargs='?', type=str, help='path to the configuration file (json)', default='Pretext_config_mimic_iii.json')
    
    args = parser.parse_args()
    
    main(args)