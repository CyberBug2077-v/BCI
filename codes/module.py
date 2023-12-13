import torch
import torch.nn as nn
from layers import Conv1DBlock, DeConv1DBlock, AttentionBlock1D


class Generator(nn.Module):
    """
    A generator model based on attention mechanism.
    Input:
        Input_ Channels (int): The number of channels for the input signal.
        Output_ Channels (int): The number of channels for the output signal.
        Filter_ Sizes (list of int): The number of filters for each layer.
        Kernel_ Sizes (list of int): The size of the convolution kernel for each layer.
    Output:
        The generated ECG signal.
    """
    def __init__(self, input_channels, output_channels, filter_sizes, kernel_sizes):
        super(Generator, self).__init__()
        self.n_layers = len(filter_sizes)
        self.filter_sizes = filter_sizes
        self.kernel_sizes = kernel_sizes
        self.input_channels = input_channels
        self.output_channels = output_channels

    def _downsample_layers(self):
        layers = nn.ModuleList()
        for i in range(self.n_layers):
            layers.append(
                Conv1DBlock(self.input_channels if i == 0 else self.filter_sizes[i-1],
                            self.filter_sizes[i],
                            self.kernel_sizes[i],
                            stride=2,
                            padding='same',
                            activation='leaky_relu')
            )
        return layers

    def _upsample_layers(self):
        layers = nn.ModuleList()
        for i in range(self.n_layers-1, -1, -1):
            layers.append(
                DeConv1DBlock(self.filter_sizes[i],
                              self.filter_sizes[i-1] if i > 0 else self.output_channels,
                              self.kernel_sizes[i],
                              stride=2,
                              padding='same',
                              activation='relu')
            )
        return layers

    def forward(self, x):
        downsample_layers = self._downsample_layers()
        upsample_layers = self._upsample_layers()
        attention_blocks = [AttentionBlock1D(size) for size in self.filter_sizes]

        # Downsample
        connections = []
        for layer in downsample_layers:
            x = layer(x)
            connections.append(x)

        # Upsample and attention mechanism
        for i, (layer, attention_block) in enumerate(zip(upsample_layers, attention_blocks)):
            x = layer(x)
            if i < self.n_layers - 1:
                attention = attention_block(x, connections[self.n_layers - 2 - i])
                x = x + attention

        return x
