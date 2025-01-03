import os
import datetime
import h5py
import numpy as np
from functools import partial
import tensorflow as tf
from signal_processing import load_and_preprocess_mimic_iii_data
from h5_to_tfrecord import h5_to_tfrecords
from augmentations import RandomAugmentor
    
    
def read_tfrecord_and_augment(example, n_input=875, random_augmentor=None):

    tfrecord_format = (
        {
            'ppg': tf.io.FixedLenFeature([n_input], tf.float32),
            'label': tf.io.FixedLenFeature([2], tf.float32)
        }
    )
    ppg = tf.io.parse_single_example(example, tfrecord_format)['ppg']
    return random_augmentor(ppg), random_augmentor(ppg)


def read_tfrecord(example, n_input=875):

    tfrecord_format = (
        {
            'ppg': tf.io.FixedLenFeature([n_input], tf.float32),
            'label': tf.io.FixedLenFeature([2], tf.float32)
        }
    )
    parsed_features = tf.io.parse_single_example(example, tfrecord_format)

    return parsed_features['ppg'], (parsed_features['label'][0], parsed_features['label'][1])


def create_tf_dataset(config, split_name='dataset', downstream_task=False, classifier_name='', random_augmentor=None):

    if classifier_name != '':
        if classifier_name == 'CNN-LSTM':
            params = config["cnn_lstm_params"]
            n_input = params["n_steps"] 
            batch_size = params["batch_size"]
        elif classifier_name == 'MLP':
            params = config["encoder_mlp_params"]
            n_input = params["n_steps"] 
            batch_size = params["batch_size"]
        elif classifier_name == 'MN_MLP':
            params = config["mn_mlp_params"]
            n_input = params["n_steps"] 
            batch_size = params["batch_size"] 
        else:
            raise ValueError("Inexistent classifier name ...")
    else:
        params = config["encoder_params"]
        n_input = params["n_steps"] 
        batch_size = params["batch_size"]

    pattern = os.path.join(config["data_directory"], config["dataset_name"], split_name, config["dataset_name"] + "_" + split_name + "_?????_of_?????.tfrecord")
    dataset = tf.data.TFRecordDataset.list_files(pattern)

    if split_name == 'train':
        dataset = dataset.shuffle(1000, seed=int(config["seed"]), reshuffle_each_iteration=True)
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=800, block_length=400)
    else:
        dataset = dataset.interleave(tf.data.TFRecordDataset)
    
    if downstream_task:
        dataset = dataset.map(partial(read_tfrecord, n_input=n_input), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(partial(read_tfrecord_and_augment, n_input=n_input, random_augmentor=random_augmentor), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.repeat()

    return dataset


def prepare_mimic_iii_dataset(config, training_persons=None, downstream_task=False, classifier_name=''):

    """
    Prepare training and testing datasets.

    """

    params = config["prepare_datasets_params"]

    output_file = os.path.join(config["data_directory"], config["dataset_name"], f'{config["dataset_name"]}.h5')
    if not os.path.isfile(output_file):
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: {config["dataset_name"]} preprocessing ...')
        with h5py.File(output_file, "a") as f:
            f.create_dataset('ppg', (0, params["win_len"] * params["signal_freq"]), maxshape=(None, params["win_len"] * params["signal_freq"]), chunks=(100, params["win_len"] * params["signal_freq"]))
            f.create_dataset('label', (0,2), maxshape=(None,2), dtype=int, chunks=(100, 2))
            f.create_dataset('subject_idx', (0,1), maxshape=(None,1), dtype=int, chunks=(100, 1))

        for subject_idx in range(len(training_persons)):
            load_and_preprocess_mimic_iii_data(config, subject_idx, training_persons[subject_idx], output_file) 
        
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: {config["dataset_name"]} train/val/test split and convertion to TFRecords ...')
        h5_to_tfrecords(config, downstream_task)
    
    num_subjects_per_split = np.load(os.path.join(config["data_directory"], config["dataset_name"], 'num_subjects_per_split.npy'))
    n_train_subjects = num_subjects_per_split[0]
    n_val_subjects = num_subjects_per_split[1]
    n_test_subjects = num_subjects_per_split[2]

    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Total # of subjects for training/validation/testing {n_train_subjects}/{n_val_subjects}/{n_test_subjects}')

    num_samples_per_split = np.load(os.path.join(config["data_directory"], config["dataset_name"], 'num_samples_per_split.npy'))
    n_train_samples = num_samples_per_split[0]
    n_val_samples = num_samples_per_split[1]
    n_test_samples = num_samples_per_split[2]
    
    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Total # of samples for training/validation/testing {n_train_samples}/{n_val_samples}/{n_test_samples}')

    if downstream_task:
        train_set = create_tf_dataset(config, split_name='train', downstream_task=downstream_task, classifier_name=classifier_name)
        val_set = create_tf_dataset(config, split_name='val', downstream_task=downstream_task, classifier_name=classifier_name)
        test_set = create_tf_dataset(config, split_name='test', downstream_task=downstream_task, classifier_name=classifier_name)
    else:
        bt_augmentor = RandomAugmentor(config["seed"])
        train_set = create_tf_dataset(config, split_name='train', random_augmentor=bt_augmentor)
        val_set = None
        test_set = None         

    return train_set, val_set, test_set


