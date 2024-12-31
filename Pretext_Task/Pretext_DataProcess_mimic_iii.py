import os
import datetime
import h5py
from functools import partial
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt, find_peaks, savgol_filter, welch
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from h5_to_tfrecord import h5_to_tfrecords


class Augmentation(Layer):
    """Base augmentation class.
    https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py#L8

    Base augmentation class. Contains the random_execute method.

    Methods:
        random_execute: method that returns true or false based 
          on a probability. Used to determine whether an augmentation 
          will be run.
    """

    def __init__(self):
        super().__init__()

    @tf.function
    def random_execute(self, prob: float) -> bool:
        """random_execute function.

        Arguments:
            prob: a float value from 0-1 that determines the 
              probability.

        Returns:
            returns true or false based on the probability.
        """

        return tf.random.uniform([], minval=0, maxval=1) < prob


class Jitter(Augmentation):
    def __init__(self, sigma=0.03):
        super().__init__()
        self.sigma = sigma

    def call(self, inputs):
        if self.random_execute(prob=0.1):
            noise = tf.random.normal(shape=(1000,), mean=0.0, stddev=self.sigma)
            return inputs + noise
        else:
            return inputs

class Scaling(Augmentation):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def call(self, inputs):
        if self.random_execute(prob=0.1):
            factor = tf.random.normal(shape=(1000,), mean=1.0, stddev=self.sigma)
            return inputs * factor
        else:
            return inputs

class Rotation(Augmentation):
    def call(self, inputs):
        if self.random_execute(prob=0.1):
            flip = tf.random.uniform(shape=(1000,), minval=-1, maxval=1, dtype=tf.int32)
            flip = tf.cast(flip, dtype=tf.float32)
            rotate_axis = tf.random.shuffle(tf.range(1000))
            return flip * tf.gather(inputs, rotate_axis)
        else:
            return inputs
    

class RandomAugmentor(Model):
    """RandomAugmentor class.

    RandomAugmentor class. Chains all the augmentations into 
    one pipeline.

    Attributes:
        time_shift: Instance variable representing the TimeShift layer.
        scaling: Instance variable representing the Scaling layer.
        noise_injection: Instance variable representing the NoiseInjection layer.
        time_warp: Instance variable representing the TimeWarp layer.

    Methods:
        call: chains layers in pipeline together
    """

    def __init__(self, n_input):
        super().__init__()
        self.jitter = Jitter()
        self.scaling = Scaling()
        self.rotation = Rotation()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.jitter(x)
        x = self.scaling(x)
        x = self.rotation(x)
        return x
    
    
def read_tfrecord_and_augment(example, n_input=1000, random_augmentor=None):

    tfrecord_format = (
        {
            'ppg': tf.io.FixedLenFeature([n_input], tf.float32),
            'label': tf.io.FixedLenFeature([2], tf.float32)
        }
    )
    ppg = tf.io.parse_single_example(example, tfrecord_format)['ppg']
    return random_augmentor(ppg), random_augmentor(ppg)

def read_tfrecord(example, n_input=1000):

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


def interpolate_nan_pchip(data):
    r"""
    Interpolates NaN values in a 1D NumPy array using PCHIP.

    Args:
        data: The input 1D NumPy array.

    Returns:
        The interpolated array.
    """

    valid_indices = np.where(~np.isnan(data))[0]
    invalid_indices = np.where(np.isnan(data))[0]

    interpolator = PchipInterpolator(valid_indices, data[valid_indices])
    interpolated_values = interpolator(invalid_indices)

    data[invalid_indices] = interpolated_values
    return data


def create_windows(win_len, fs, N_samp, overlap):

    win_len = win_len * fs
    overlap = np.round(overlap * win_len)
    N_samp = N_samp - win_len + 1

    idx_start = np.round(np.arange(0,N_samp, win_len-overlap)).astype(int)
    idx_stop = np.round(idx_start + win_len - 1)

    return idx_start, idx_stop


def plot_signal(signal : np.array, fs : int, flat_locs_sig : np.array = None, peaks : np.array = None, valleys: np.array = None, title : str = '') -> None:
    r"""
    Handy function to plot signals and their anomalies when prvodided.

    Parameters:
    ------------

    signal: np.array,
        the signal to analyze
    flat_locs_sig: np.array,
        the locations of the flat lines
    peaks: np.array,
        the locations of the peaks
    valleys: np.array,
        the locations of the valleys
    fs: int,
        the sampling rate of the signal
    title: str,
        the title of the plot

    Return
    ------------
    None
    """
    import matplotlib.pyplot as plt
    t = np.arange(0, (len(signal) / fs), 1.0 / fs)
    plt.title(f'{title}')
    plt.xlabel('s')
    plt.plot(t, signal, color='black', label='signal')

    if peaks is not None:
        x_vals = np.arange(len(signal))
        up_env = np.interp(x_vals, peaks, signal[peaks])
        plt.plot(t, up_env, color='red', label='up envelope', marker='o', linestyle='dashed', linewidth=1, markersize=1)
        plt.scatter(t[peaks], signal[peaks], color='red', label='peaks')

    if valleys is not None:
        x_vals = np.arange(len(signal))
        down_env = np.interp(x_vals, valleys, signal[valleys])
        plt.plot(t, down_env, color='blue', label='down envelope', marker='o', linestyle='dashed', linewidth=1, markersize=1)
        plt.scatter(t[valleys], signal[valleys], color='blue', label='valleys')

    if flat_locs_sig is not None:
        plt.scatter(t[flat_locs_sig], signal[flat_locs_sig], color='green', label='flat lines')

    plt.legend()
    plt.show()
    plt.clf()
    

def load_and_preprocess_mimic_iii_data(config, idx, id_data, output_file):

    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Processing subject {id_data.split("/")[-1]}')

    params = config["prepare_datasets_params"]
    valid_bp_ranges = config["valid_bp_ranges"]

    win_len = params["win_len"]
    overlap = params["overlap"]
    fs = params["signal_freq"]
    filter_order = params["filter_order"]
    
    SBP_min, SBP_max = valid_bp_ranges["low_sbp"], valid_bp_ranges["up_sbp"] 
    DBP_min, DBP_max = valid_bp_ranges["low_dbp"], valid_bp_ranges["up_dbp"]

    n_samp_total = 0

    PPG_RECORD = np.empty((0, win_len * fs))
    OUTPUT = np.empty((0, 2))

    for record in os.listdir(id_data):
        
        record_path = os.path.join(id_data, record) 
        
        ppg = np.load(os.path.join(record_path, 'ppg.npy'))
        abp = np.load(os.path.join(record_path, 'abp.npy'))

        abp_peaks = find_peaks(abp)[0]
        abp_valleys = find_peaks(-abp)[0]
        
        # Create start and stop indizes for time windows
        n_samples = abp.shape[0]
        win_start, win_stop = create_windows(win_len, fs, n_samples, overlap)
        n_win = len(win_start)
        n_samp_total += n_win

        ppg_record = np.zeros((n_win, win_len*fs))
        output = np.zeros((n_win, 2))

        for i in range(0, n_win):
        
            idx_start = win_start[i]
            idx_stop = win_stop[i]

            ppg_win = ppg[idx_start:idx_stop+1]

            # Sanity check if enough peak values are present and if the number of SBP peaks matches the number of DBP peaks
            ABP_sys_idx_win = abp_peaks[np.logical_and(abp_peaks >= idx_start, abp_peaks < idx_stop)].astype(int)
            ABP_dia_idx_win = abp_valleys[np.logical_and(abp_valleys >= idx_start, abp_valleys < idx_stop)].astype(int)

            if ABP_sys_idx_win.shape[-1] < (win_len/60)*40 or ABP_sys_idx_win.shape[-1] > (win_len/60)*120:
                output[i, :] = np.nan
                continue

            if ABP_dia_idx_win.shape[-1] < (win_len/60)*40 or ABP_dia_idx_win.shape[-1] > (win_len/60)*120:
                output[i, :] = np.nan
                continue

            if len(ABP_sys_idx_win) != len(ABP_dia_idx_win):
                if ABP_sys_idx_win[0] > ABP_dia_idx_win[0]:
                    ABP_dia_idx_win = np.delete(ABP_dia_idx_win,0)
                if ABP_sys_idx_win[-1] > ABP_dia_idx_win[-1]:
                    ABP_sys_idx_win = np.delete(ABP_sys_idx_win,-1)

            ABP_sys_win = abp[ABP_sys_idx_win]
            ABP_dia_win = abp[ABP_dia_idx_win]

            # check if any of the SBP or DBP values exceed reasonable vlaues
            if np.any(np.logical_or(ABP_sys_win < SBP_min, ABP_sys_win > SBP_max)):
                output[i, :] = np.nan
                continue

            if np.any(np.logical_or(ABP_dia_win < DBP_min, ABP_dia_win > DBP_max)):
                output[i, :] = np.nan
                continue

            # check for NaN in the detected SBP and DBP peaks
            if np.any(np.isnan(ABP_sys_win)) or np.any(np.isnan(ABP_dia_win)):
                output[i, :] = np.nan
                continue

            # First replace NaNs that may affect the computation
            ppg_win = interpolate_nan_pchip(ppg_win)

            # S-G filter to smooth the signal
            ppg_win = savgol_filter(ppg_win, window_length=25, polyorder=7)
            
            # S-G filter to remove baseline wander
            filtered_ppg_win = savgol_filter(ppg_win, window_length=211, polyorder=5)
            ppg_win = ppg_win - filtered_ppg_win
            
            # Harmonic filtering from CardioID
            fh, Pxx_den = welch(ppg_win, fs=fs, window='hann', axis=0)
            
            # Find the peak frequency (+1 to avoid having a zero as first frequency)
            first_harmonic = fh[np.argmax(Pxx_den) + 1]

            # Adaptive Butterworth filtering
            lowcut = 2 * first_harmonic
            highcut = 5.5 * first_harmonic
            
            sos_ppg = butter(filter_order,
                            [lowcut, highcut],
                            btype='bp',
                            analog=False,
                            output='sos',
                            fs=fs)
            ppg_win = sosfiltfilt(sos_ppg, ppg_win)

            # Standardize PPG
            ppg_win = ppg_win - np.mean(ppg_win)
            ppg_win = ppg_win / np.std(ppg_win)

            # calculate the BP ground truth as the median of all SBP and DBP values in the present window
            BP_sys = np.median(ABP_sys_win).astype(int)
            BP_dia = np.median(ABP_dia_win).astype(int)
            
            ppg_record[i, :] = ppg_win
            output[i, :] = [BP_sys, BP_dia]
     
        idx_nans = np.isnan(output[:,0])
        OUTPUT = np.vstack((OUTPUT, output[np.invert(idx_nans),:]))
        PPG_RECORD = np.vstack((PPG_RECORD, ppg_record[np.invert(idx_nans),:]))

    if OUTPUT.shape[0] > 1:
        
        # add data to .h5 file
        with h5py.File(output_file, "a") as f:

            BP_dataset = f['label']
            DatasetCurrLength = BP_dataset.shape[0]
            DatasetNewLength = DatasetCurrLength + OUTPUT.shape[0]
            BP_dataset.resize(DatasetNewLength, axis=0)
            BP_dataset[-OUTPUT.shape[0]:,:] = OUTPUT

            ppg_dataset = f['ppg']

            ppg_dataset.resize(DatasetNewLength, axis=0)
            ppg_dataset[-PPG_RECORD.shape[0]:,:] = PPG_RECORD

            subject_dataset = f['subject_idx']
            subject_dataset.resize(DatasetNewLength, axis=0)
            subject_dataset[-OUTPUT.shape[0]:,:] = idx * np.ones((OUTPUT.shape[0], 1))

            print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Processed {OUTPUT.shape[0]} samples ({DatasetNewLength} samples total)')
    else:

        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Skipped subject {id_data.split("/")[-1]}')


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
        bt_augmentor = RandomAugmentor(params["n_input"])
        train_set = create_tf_dataset(config, split_name='train', random_augmentor=bt_augmentor)
        val_set = None
        test_set = None         

    return train_set, val_set, test_set


