import torch
import numpy as np
from scipy.signal import butter, filtfilt, welch
from torch.utils.data import Dataset, DataLoader


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', output='ba')
    return b, a


def filter_signal(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def filter_ecg(signal, sampling_rate):
    """
    Filter the ECG signal.
    Input:
        Signal: The original ECG signal, a numerical array.
        Sampling_ Rate: The sampling rate of the signal.
    Output:
        Filtered: The filtered ECG signal.
    """
    return filter_signal(signal, 3, 45, sampling_rate, order=5)


def filter_ppg(signal, sampling_rate):
    """
    Filter the PPG signal.
    Input:
        Signal: The original PPG signal is a numerical array.
        Sampling_ Rate: The sampling rate of the signal.
    Output:
        Filtered: The filtered PPG signal.
    """
    return filter_signal(signal, 1, 8, sampling_rate, order=4)


def transform_to_frequency(signal, fs):
    """
    Convert time series data into spectral data.
    Input:
        Signal: Time series data, a numerical array.
        Fs: The sampling rate of the signal.
    Output:
        f: Frequency value array.
        Pxx: Spectral density.
    """
    f, Pxx = welch(signal, fs, nperseg=1024)
    return f, Pxx


def get_dataloader(ppg_data, ecg_data, sampling_rate, batch_size=32, shuffle=True):
    dataset = CustomDataset(ppg_data, ecg_data, sampling_rate)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class CustomDataset(Dataset):
    def __init__(self, ppg_data, ecg_data, sampling_rate):
        self.ppg_data = ppg_data
        self.ecg_data = ecg_data
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.ppg_data)

    def __getitem__(self, idx):
        ppg = self.ppg_data[idx]
        ecg = self.ecg_data[idx]

        # Preprocessing
        ppg_filtered = filter_ppg(ppg, self.sampling_rate)
        ecg_filtered = filter_ecg(ecg, self.sampling_rate)

        # Convert to Spectral data
        _, ppg_freq = transform_to_frequency(ppg_filtered, self.sampling_rate)
        _, ecg_freq = transform_to_frequency(ecg_filtered, self.sampling_rate)

        return {
            'ppg': torch.tensor(ppg_filtered, dtype=torch.float),
            'ecg': torch.tensor(ecg_filtered, dtype=torch.float),
            'ppg_freq': torch.tensor(ppg_freq, dtype=torch.float),
            'ecg_freq': torch.tensor(ecg_freq, dtype=torch.float)
        }