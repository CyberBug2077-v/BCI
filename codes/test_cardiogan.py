import os
import socket
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import sklearn.preprocessing as skp
import matplotlib.pyplot as plt
import tflib

import module 
import preprocessing

tf.keras.backend.set_floatx('float64')
tf.autograph.set_verbosity(0)

@tf.function
def sample_P2E(P, model):
    fake_ecg = model(P, training=False)
    return fake_ecg




########### params ###########
ecg_sampling_freq = 128
ppg_sampling_freq = 128
window_size = 4
ecg_segment_size = ecg_sampling_freq*window_size
ppg_segment_size = ppg_sampling_freq*window_size
model_dir = 'Huang\weights'

""" model """
Gen_PPG2ECG = module.generator_attention()
""" resotre """
tflib.Checkpoint(dict(Gen_PPG2ECG=Gen_PPG2ECG), model_dir).restore()
print("model loaded successfully")


# Load data
file_path = r'part_1_PPG_beats_P2P_Aug_2022.csv'
x_ppg = np.loadtxt(file_path, delimiter=',')

# Resample to 128Hz
# Note: cv2.resize returns a two-dimensional array, which we need to convert to a one-dimensional array.
x_ppg_resized = cv2.resize(x_ppg, (ppg_segment_size, 1), interpolation=cv2.INTER_LINEAR).flatten()

# Filter data
x_ppg_filtered = preprocessing.filter_ppg(x_ppg_resized, ppg_sampling_freq)

# Adjust data shape to match model input requirements
x_ppg_normalized = np.reshape(x_ppg_filtered, (-1, 512))

# Data normalization
x_ppg_normalized = skp.minmax_scale(x_ppg_normalized, feature_range=(-1, 1), axis=1)

# Use model to predict
x_ecg = sample_P2E(x_ppg_normalized, Gen_PPG2ECG)

# Plotting ECG Waveforms
plt.figure(figsize=(10, 4))  # Graph size
plt.plot(x_ecg[0])       # Draw waveform
plt.title('ECG Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()  
# load the data: x_ppg = np.loadtxt()
# make sure loaded data is a numpy array: x_ppg = np.array(x_ppg)
# resample to 128 Hz using: cv2.resize(x_ppg, (1,ppg_segment_size), interpolation = cv2.INTER_LINEAR)
# filter the data using: preprocessing.filter_ppg(x_ppg, 128)
# make an array to N x 512 [this is the input shape of x_ppg], where Nx512=len(x_ppg)
# normalize the data b/w -1 to 1: x_ppg = skp.minmax_scale(x_ppg, (-1, 1), axis=1)
#######
#x_ecg = sample_P2E(x_ppg, Gen_PPG2ECG)
#######
