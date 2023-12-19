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


# 加载数据
file_path = r'C:\Users\曹淮德\Desktop\study\Huang\ppg2ecg-cardiogan\codes\part_1_PPG_beats_P2P_Aug_2022.csv'
x_ppg = np.loadtxt(file_path, delimiter=',')

# 重采样到 128 Hz
# 注意：cv2.resize 返回的是二维数组，我们需要将其转换为一维数组
x_ppg_resized = cv2.resize(x_ppg, (ppg_segment_size, 1), interpolation=cv2.INTER_LINEAR).flatten()

# 过滤数据
x_ppg_filtered = preprocessing.filter_ppg(x_ppg_resized, ppg_sampling_freq)

# 调整数据形状以符合模型输入要求
x_ppg_normalized = np.reshape(x_ppg_filtered, (-1, 512))

# 数据归一化
x_ppg_normalized = skp.minmax_scale(x_ppg_normalized, feature_range=(-1, 1), axis=1)

# 使用模型进行预测
x_ecg = sample_P2E(x_ppg_normalized, Gen_PPG2ECG)

# 绘制 ECG 波形图
plt.figure(figsize=(10, 4))  # 设置图形的大小
plt.plot(x_ecg[0])       # 绘制波形图
plt.title('ECG Waveform')    # 设置标题
plt.xlabel('Time')           # 设置 X 轴标签
plt.ylabel('Amplitude')      # 设置 Y 轴标签
plt.grid(True)               # 显示网格
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
