import numpy as np
from scipy.signal import butter, filtfilt, welch


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
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