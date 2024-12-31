""" Convert PPG/BP samples pairs to a binary format that is used for training neural networks using Tensorflow

This script reads a dataset consisting of PPG and BP samples from a .h5 file and converts them into a binary format that
can be used for as input data for a neural network during training. The dataset can be divided into training, validation
and test set by (i) dividing the dataset on a subject basis ensuring that data from one subject are not scattered across
training, validation and test set or (ii) dividing the dataset randomly.

File: prepare_MIMIC_dataset.py
Author: Dr.-Ing. Fabian Schrumpf
E-Mail: Fabian.Schrumpf@htwk-leipzig.de
Date created: 8/6/2021
Date last modified: 8/6/2021
"""

import datetime
import os
import h5py
import tensorflow as tf
# ks.enable_eager_execution()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def ppg_hdf2tfrecord(h5_file, tfrecord_path, samp_idx, downstream_task=False):
    # Function that converts PPG/BP sample pairs into the binary .tfrecord file format. This function creates a .tfrecord
    # file containing a defined number os samples
    #
    # Parameters:
    # h5_file: file containing ppg and BP data
    # tfrecordpath: full path for storing the .tfrecord files
    # samp_idx: sample indizes of the data in the .h5 file to be stored in the .tfrecord file
    # weights_SBP: sample weights for the systolic BP (optional)
    # weights_DBP: sample weights for the diastolic BP (optional)

    N_samples = len(samp_idx)
    # open the .h5 file and get the samples with the indizes specified by samp_idx
    with h5py.File(h5_file, 'r') as f:
        # load ppg and BP data as well as the subject numbers the samples belong to
        ppg_h5 = f.get('/ppg')
        BP = f.get('/label')
        subject_idx = f.get('/subject_idx')

        writer = tf.io.TFRecordWriter(tfrecord_path)

        # iterate over each sample index and convert the corresponding data to a binary format
        for i in np.nditer(samp_idx):

            ppg = np.array(ppg_h5[i,:])
            target = np.array(BP[i,:], dtype=np.float32)
            sub_idx = np.array(subject_idx[i])

            # create a dictionary containing the serialized data
            data = \
                {'ppg': _float_feature(ppg.tolist()),
                'label': _float_feature(target.tolist()),
                'subject_idx': _float_feature(sub_idx.tolist()),
                'Nsamples': _float_feature([N_samples])}

            # write data to the .tfrecord target file
            feature = tf.train.Features(feature=data)
            example = tf.train.Example(features=feature)
            serialized = example.SerializeToString()

            writer.write(serialized)
        writer.close()


def ppg_hdf2tfrecord_sharded(config, h5_file, samp_idx, tfrecordpath, Nsamp_per_shard, modus='train', downstream_task=False):
    # Save PPG/BP pairs as .tfrecord files. Save defined number os samples per file (Sharding)
    # Weights can be defined for each sample
    #
    # Parameters:
    # h5_file: File that contains the whole dataset (in .h5 format), created by
    # samp_idx: sample indizes from the dataset in the h5. file that are used to create this tfrecords dataset
    # tfrecordpath: full path for storing the .tfrecord files
    # N_samp_per_shard: number of samples per shard/.tfrecord file
    # modus: define if the data is stored in the "train", "val" or "test" subfolder of "tfrecordpath"
    # weights_SBP: sample weights for the systolic BP (optional)
    # weights_DBP: sample weights for the diastolic BP (optional)

    base_filename = os.path.join(tfrecordpath, config["dataset_name"])

    N_samples = len(samp_idx)

    # calculate the number of Files/shards that are needed to stroe the whole dataset
    N_shards = np.ceil(N_samples / Nsamp_per_shard).astype(int)

    # iterate over every shard
    for i in range(N_shards):
        idx_start = i * Nsamp_per_shard
        idx_stop = (i + 1) * Nsamp_per_shard
        if idx_stop > N_samples:
            idx_stop = N_samples

        idx_curr = samp_idx[idx_start:idx_stop]
        output_filename = '{0}_{1}_{2:05d}_of_{3:05d}.tfrecord'.format(base_filename,
                                                                       modus,
                                                                       i + 1,
                                                                       N_shards)
        
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Processing {modus} shard {str(i + 1)} of {str(N_shards)}')
        
        ppg_hdf2tfrecord(h5_file, output_filename, idx_curr, downstream_task)


def h5_to_tfrecords(config, downstream_task=False):

    tfrecord_path_train = os.path.join(config["data_directory"], config["dataset_name"], 'train')
    if not os.path.isdir(tfrecord_path_train):
        os.mkdir(tfrecord_path_train)
    tfrecord_path_val = os.path.join(config["data_directory"], config["dataset_name"], 'val')
    if not os.path.isdir(tfrecord_path_val):
        os.mkdir(tfrecord_path_val)
    tfrecord_path_test = os.path.join(config["data_directory"], config["dataset_name"], 'test')
    if not os.path.isdir(tfrecord_path_test):
        os.mkdir(tfrecord_path_test)

    Nsamp_per_shard = 1000

    dataset_h5py_path = os.path.join(config["data_directory"], config["dataset_name"], f'{config["dataset_name"]}.h5')

    with h5py.File(dataset_h5py_path, 'r') as f:
        BP = np.array(f.get('/label'))
        BP = np.round(BP)
        BP = np.transpose(BP)
        subject_idx = np.squeeze(np.array(f.get('/subject_idx')))

    N_samp_total = BP.shape[1]
    subject_idx = subject_idx[:N_samp_total]

    # Divide the dataset by subjevct into training, validation and test set
    # -------------------------------------------------------------------------------
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
    train_set.to_csv(os.path.join(tfrecord_path_train, f'{config["dataset_name"]}_trainset.csv'))
    BP_val = BP[:,idx_val]
    d = {"SBP": np.transpose(BP_val[0, :]), "DBP": np.transpose(BP_val[1, :])}
    train_set = pd.DataFrame(d)
    train_set.to_csv(os.path.join(tfrecord_path_val, f'{config["dataset_name"]}_valset.csv'))
    BP_test = BP[:,idx_test]
    d = {"SBP": np.transpose(BP_test[0, :]), "DBP": np.transpose(BP_test[1, :])}
    train_set = pd.DataFrame(d)
    train_set.to_csv(os.path.join(tfrecord_path_test, f'{config["dataset_name"]}_testset.csv'))

    # create tfrecord dataset
    # ----------------------------
    np.random.shuffle(idx_train)
    ppg_hdf2tfrecord_sharded(config, dataset_h5py_path, idx_train, tfrecord_path_train, Nsamp_per_shard, modus='train', downstream_task=downstream_task)
    ppg_hdf2tfrecord_sharded(config, dataset_h5py_path, idx_val, tfrecord_path_val, Nsamp_per_shard, modus='val', downstream_task=downstream_task)
    ppg_hdf2tfrecord_sharded(config, dataset_h5py_path, idx_test, tfrecord_path_test, Nsamp_per_shard, modus='test', downstream_task=downstream_task)
    

    
