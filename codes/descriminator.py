from layers import Conv1DBlock
import torch
import torch.nn as nn


class DescriminatorT(nn.Module):
    """
    Time domain discriminator, used to distinguish between real and generated ECG signals.
    Input:
        Input_ Channels (int): The number of channels for the input signal.
    Output:
        The discrimination result represents the probability that the input signal is a true signal.
    """
    def __init__(self, input_channels, signal_length):
        super(DescriminatorT, self).__init__()
        self.model = nn.Sequential(
            Conv1DBlock(input_channels, 64, kernel_size=3, stride=2, padding='same', activation='leaky_relu'),
            Conv1DBlock(64, 128, kernel_size=3, stride=2, padding='same', activation='leaky_relu'),
            nn.Flatten(),
            nn.Linear(128 * (signal_length // 4), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class DescriminatorF(nn.Module):
    """
    Frequency domain discriminator, used to distinguish the spectrum of real and generated ECG signals.
    Input:
        Input_ Channels (int): The number of channels in the input spectrogram.
        Spectrogram_ Length (int): The length of the input spectrogram.
    Output:
        The discrimination result indicates that the input spectrogram is based on the probability of the real signal.
    """
    def __init__(self, input_channels, spectrogram_length):
        super(DescriminatorF, self).__init__()
        self.model = nn.Sequential(
            Conv1DBlock(input_channels, 64, kernel_size=3, stride=2, padding='same', activation='leaky_relu'),
            Conv1DBlock(64, 128, kernel_size=3, stride=2, padding='same', activation='leaky_relu'),
            nn.Flatten(),
            nn.Linear(128 * (spectrogram_length // 4), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
