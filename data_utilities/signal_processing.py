import os
import datetime
import h5py
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt, find_peaks, savgol_filter, welch


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
