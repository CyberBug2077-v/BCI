import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    """
    One dimensional convolutional layer.
    Input:
        In_ Channels (int): Input number of channels.
        Out_ Channels (int): Output number of channels.
        Kernel_ Size (int): The size of the convolution kernel.
        Stride (int, optional): Convolution step size. The default is 1.
        Padding (str, optional): Fill type, 'valid' or 'same'. The default is' valid '.
        Activation (str, optional): Activation function type, such as' relu ',' leaky '_ Relu ',' sigmoid ',' tanh '. The default is None.
    Output:
        Tensors processed by convolution and activation functions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='valid', activation=None):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0 if padding == 'valid' else 1)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'leaky_relu':
            return F.leaky_relu(x, 0.2)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        return x


class DeConv1DBlock(nn.Module):
    """
    One dimensional deconvolution (transposed convolution) layer.
    Input:
        In_ Channels (int): Input number of channels.
        Out_ Channels (int): Output number of channels.
        Kernel_ Size (int): The size of the convolution kernel.
        Stride (int, optional): Convolution step size. The default is 1.
        Padding (str, optional): Fill type, 'valid' or 'same'. The default is' valid '.
        Activation (str, optional): Activation function type, such as' relu ',' leaky '_ Relu ',' sigmoid ',' tanh '. The default is None.
    Output:
        Tensors processed by deconvolution and activation functions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='valid', activation=None):
        super(DeConv1DBlock, self).__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=0 if padding == 'valid' else 1)
        self.activation = activation

    def forward(self, x):
        x = self.deconv(x)
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'leaky_relu':
            return F.leaky_relu(x, 0.2)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        return x


class DenseBlock(nn.Module):
    """
    Densely connected layer (fully connected layer) block.
    Input:
        In_ Features (int): Input number of features.
        Out_ Features (int): Output of features.
        Activation (str, optional): Activation function type, such as' relu ',' leaky '_ Relu ',' sigmoid ',' softmax ',' tanh '. The default is None.
    Output:
        Tensors processed through fully connected layers and activation functions.
    """
    def __init__(self, in_features, out_features, activation=None):
        super(DenseBlock, self).__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.dense(x)
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'leaky_relu':
            return F.leaky_relu(x, 0.2)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'softmax':
            return F.softmax(x, dim=-1)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        return x


class BatchNormBlock(nn.Module):
    """
    Batch normalization layer block.
    Input:
        Num_ Features (int): Input the number of features (i.e. the number of channels).
    Output:
        Tensors that have undergone batch normalization.
    """
    def __init__(self, num_features):
        super(BatchNormBlock, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.batch_norm(x)


class DropoutBlock(nn.Module):
    """
    Dropout layer block.
    Input:
        P (float, optional): Dropout ratio. The default is 0.5.
    Output:
        Tensor processed by Dropout.
    """
    def __init__(self, p=0.5):
        super(DropoutBlock, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)


class AttentionBlock1D(nn.Module):
    """
    One dimensional attention mechanism block based on U-Net.
    Input:
        Curr_ Layer (Tensor): The feature map of the current layer.
        Conn_ Layer (Tensor): A feature map of skip connections.
    Output:
        Feature maps weighted by attention.
    """
    def __init__(self, in_channels):
        super(AttentionBlock1D, self).__init__()
        self.theta = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.phi = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.psi = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, curr_layer, conn_layer):
        theta_x = self.theta(conn_layer)
        phi_g = self.phi(curr_layer)
        f = F.relu(theta_x + phi_g)
        psi_f = self.psi(f)
        rate = torch.sigmoid(psi_f)
        attr_x = conn_layer * rate
        return attr_x
