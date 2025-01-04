import os
import datetime
import h5py
import gc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from signal_processing import load_and_preprocess_mimic_iii_data


def process_data(i, output_file):
    """
    Processes a single data point.

    Args:
        i: Index of the data point.
        ppg_h5: h5py dataset object for PPG data.
        bp_h5: h5py dataset object for BP data.

    Returns:
        Tuple of processed PPG and BP data.
    """
    with h5py.File(output_file, 'r') as f:
        ppg_h5 = f.get('/ppg')[i, :]
        bp_h5 = f.get('/label')[i, :]
    ppg = np.array(ppg_h5)
    bp = np.array(bp_h5, dtype=np.float32)
    return ppg, bp

def parallel_processing(idxs, output_file, cores):
    """
    Processes data in parallel using joblib.

    Args:
        idxs: Array of indices to process.
        output_file: Path to the input h5 file.

    Returns:
        Tuples of processed PPG and BP data.
    """

    results = Parallel(n_jobs=cores)(delayed(process_data)(idx, output_file) for idx in idxs)

    return zip(*results) 


def save_split(config, output_file, idxs, save_path_dir):

    PPG, BP = parallel_processing(idxs, output_file, config["n_cores"])

    np.save(os.path.join(save_path_dir, 'ppg.npy'), np.vstack(PPG))
    np.save(os.path.join(save_path_dir, 'bp.npy'), np.vstack(BP))
    

def prepare_mimic_iii_dataset(config):

    folders = config["folders"] 
    training_persons = []
    
    for folder in folders:
        patients_folder = os.path.join(config["data_directory"], config["dataset_name"], f'p0{str(folder)}')
        for subject in os.listdir(patients_folder):
            training_persons.append(os.path.join(patients_folder, subject))

    params = config["prepare_datasets_params"]

    output_file = os.path.join(config["data_directory"], config["dataset_name"], f'{config["dataset_name"]}.h5')
    if not os.path.isfile(output_file):
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: {config["dataset_name"]} preprocessing ...')
        with h5py.File(output_file, "a") as f:
            f.create_dataset('ppg', (0, params["win_len"] * params["signal_freq"]), maxshape=(None, params["win_len"] * params["signal_freq"]), chunks=(100, params["win_len"] * params["signal_freq"]))
            f.create_dataset('label', (0,2), maxshape=(None, 2), dtype=int, chunks=(100, 2))
            f.create_dataset('subject_idx', (0,1), maxshape=(None, 1), dtype=int, chunks=(100, 1))

        for subject_idx in range(len(training_persons)):
            load_and_preprocess_mimic_iii_data(config, subject_idx, training_persons[subject_idx], output_file) 
        
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: {config["dataset_name"]} {config["dataset_name"]} preprocessing completed ...')
        gc.collect()

        # Train/Val/Test Split
        numpy_record_path_train = os.path.join(config["data_directory"], config["dataset_name"], 'train')
        if not os.path.isdir(numpy_record_path_train):
            os.mkdir(numpy_record_path_train)
        numpy_record_path_val = os.path.join(config["data_directory"], config["dataset_name"], 'val')
        if not os.path.isdir(numpy_record_path_val):
            os.mkdir(numpy_record_path_val)
        numpy_record_path_test = os.path.join(config["data_directory"], config["dataset_name"], 'test')
        if not os.path.isdir(numpy_record_path_test):
            os.mkdir(numpy_record_path_test)

        with h5py.File(output_file, 'r') as f:
            BP = np.array(f.get('/label'))
            BP = np.round(BP)
            BP = np.transpose(BP)
            subject_idx = np.squeeze(np.array(f.get('/subject_idx')))

        N_samp_total = BP.shape[1]
        subject_idx = subject_idx[:N_samp_total]

        valid_idx = np.arange(subject_idx.shape[-1])

        # divide the subjects into training, validation and test subjects
        subject_labels = np.unique(subject_idx)
        
        subjects_train_labels, subjects_test_labels = train_test_split(subject_labels, test_size=float(config["test_percentage"]), random_state=int(config["seed"]))
        subjects_train_labels, subjects_val_labels = train_test_split(subjects_train_labels, test_size=float(config["val_percentage"]), random_state=int(config["seed"]))

        with open(os.path.join(config["data_directory"], config["dataset_name"], 'dataset_subjects_split.txt'), 'w') as file:
            file.write('Training subjects:\n')
            for subject in subjects_train_labels:
                file.write(f' {subject},')
            file.write('\nValidation subjects:\n')
            for subject in subjects_val_labels:
                file.write(f' {subject},')
            file.write("\nTesting subjects:\n")
            for subject in subjects_test_labels:
                file.write(f' {subject},')

        np.save(os.path.join(config["data_directory"], config["dataset_name"], 'num_subjects_per_split.npy'), np.array([len(subjects_train_labels), len(subjects_val_labels), len(subjects_test_labels)]))

        # Calculate samples belong to training, validation and test subjects
        idx_train = valid_idx[np.isin(subject_idx, subjects_train_labels)]
        idx_val = valid_idx[np.isin(subject_idx, subjects_val_labels)]
        idx_test = valid_idx[np.isin(subject_idx, subjects_test_labels)]

        np.save(os.path.join(config["data_directory"], config["dataset_name"], 'num_samples_per_split.npy'), np.array([len(idx_train), len(idx_val), len(idx_test)]))

        # save ground truth BP values of training, validation and test set in csv-files for future reference
        BP_train = BP[:,idx_train]
        d = {"SBP": np.transpose(BP_train[0, :]), "DBP": np.transpose(BP_train[1, :])}
        train_set = pd.DataFrame(d)
        train_set.to_csv(os.path.join(numpy_record_path_train, f'{config["dataset_name"]}_trainset.csv'))
        BP_val = BP[:,idx_val]
        d = {"SBP": np.transpose(BP_val[0, :]), "DBP": np.transpose(BP_val[1, :])}
        train_set = pd.DataFrame(d)
        train_set.to_csv(os.path.join(numpy_record_path_val, f'{config["dataset_name"]}_valset.csv'))
        BP_test = BP[:,idx_test]
        d = {"SBP": np.transpose(BP_test[0, :]), "DBP": np.transpose(BP_test[1, :])}
        train_set = pd.DataFrame(d)
        train_set.to_csv(os.path.join(numpy_record_path_test, f'{config["dataset_name"]}_testset.csv'))

        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Saving train/val/test splits to numpy arrays ...')

        save_split(config, output_file, idx_train, numpy_record_path_train)
        save_split(config, output_file, idx_val, numpy_record_path_val)
        save_split(config, output_file, idx_test, numpy_record_path_test)

        gc.collect()

