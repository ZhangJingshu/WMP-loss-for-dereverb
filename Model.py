import numpy as np
import librosa
import os
import sys
import math

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.parameter import Parameter


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class MyPReLU(nn.Module):
    def __init__(self, n_channels, n_features, init = 0.25):
        super(MyPReLU, self).__init__()
        self.num_parameters = n_channels * n_features
        self.n_channels = n_channels
        self.n_features = n_features
        self.weight = Parameter(torch.Tensor(self.num_parameters).fill_(init))

    def forward(self, input):
        weight = torch.reshape(self.weight, [1, self.n_channels, 1, self.n_features])
        return F.prelu(input, weight)

    def extra_repr(self):
        return 'n_channels = {}, n_features = {}'.format(self.n_channels, self.n_features)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_features, stride=1, padding=0, bias=True):
        super(Down, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(num_parameters = out_channels, init = 0.1)

        init_layer(self.conv)
        init_bn(self.bn)

    def forward(self, input):
        return self.prelu(self.bn(self.conv(input)))


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_features, stride=1, padding=0, bias=True):
        super(Up, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(num_parameters = out_channels, init = 0.1)

        init_layer(self.conv)
        init_bn(self.bn)

    def forward(self, x1, x2):

        x1 = self.prelu(self.bn(self.conv(x1)))

        # Following https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py#L54
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        return torch.cat([x2, x1], dim=1)


class Recurrent(nn.Module):
    def __init__(self, input_size, hidden_size, conv_channels, type, bidirectional, num_layers = 1, dropout = 0, batch_first = True):
        super(Recurrent, self).__init__()
        self.type = type
        self.conv_channels = conv_channels
        self.bidirectional = bidirectional
        if type != "None":
            if type == "LSTM":
                self.recurrent = torch.nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers,
                                        batch_first = batch_first, dropout = dropout, bidirectional = bidirectional)
            else:
                self.recurrent = torch.nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers,
                                        batch_first = batch_first, dropout = dropout, bidirectional = bidirectional)

            if bidirectional is True:
                self.conv = torch.nn.Conv2d(in_channels = conv_channels*3, out_channels = conv_channels, kernel_size = [1,1])
            else:
                self.conv = torch.nn.Conv2d(in_channels = conv_channels*2, out_channels = conv_channels, kernel_size = [1,1])

    def forward(self, x_in):
        if self.type == "None":
            return x_in
        batch_size, n_channels, length, n_features = x_in.size()
        x_in_re = x_in.transpose(1,2).reshape(batch_size, length, n_channels*n_features)
        x = self.recurrent(x_in_re)[0]

        if self.bidirectional is True:
            x = x.reshape(batch_size, length, self.conv_channels*2, -1).transpose(1,2)
            x = torch.cat((x, x_in), dim = 1)
            x = self.conv(x)
        else:
            x = x.reshape(batch_size, length, self.conv_channels, -1).transpose(1,2)
            x = torch.cat((x, x_in), dim = 1)
            x = self.conv(x)
        return x


class UNet_recurrent(nn.Module):
    def __init__(self, config):
        """Jansson, Andreas, Eric Humphrey, Nicola Montecchio, Rachel Bittner, Aparna Kumar, and Tillman Weyde. "Singing voice separation with deep U-Net convolutional networks." (2017).
        recurrent layer is added in Unet structure
        """
        super(UNet_recurrent, self).__init__()
        self.config = config
        self.n_conv_layers = len(config["model"]["channels_up"])
        window_size = config["data"]["frame_length"]
        hop_size = config["data"]["hop_length"]
        center = True
        pad_mode = 'reflect'
        window = 'hann'

        stride_mode = config["model"]["stride_mode"]
        channels_down = config["model"]["channels_down"]
        channels_up = config["model"]["channels_up"]
        kernel_size = config["model"]["kernel_size"]
        if config["training"]["target"] == "cIRM" or config["training"]["target"] == "spec":
            channels_up[-1][1] = 2
        else:
            channels_up[-1][1] = 1
        if stride_mode == 0:                             #0: stride = 2 along T domain; 1: stride = 1 & 2, changes in each layer; 2: stride = 1 (no stride along T axis)
            stride = config["model"]["stride"][0]
        elif stride_mode == 1:
            stride = config["model"]["stride"]
        elif stride_mode == 2:
            stride = config["model"]["stride"][1]

        padding = config["model"]["padding"]

        recurrent_type = config["model"]["recurrent_type"]
        bidirectional = config["model"]["bidirectional"]

        dropout = config["training"]["dropout"]

        self.stft = STFT(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        self.istft = ISTFT(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        if stride_mode == 1:
            self.down1 = Down(in_channels=channels_down[0][0], out_channels=channels_down[0][1], kernel_size = kernel_size, n_features = 129, stride = stride[1], padding = padding, bias = False)
            self.down2 = Down(in_channels=channels_down[1][0], out_channels=channels_down[1][1], kernel_size = kernel_size, n_features = 65, stride = stride[0], padding = padding, bias = False)
            self.down3 = Down(in_channels=channels_down[2][0], out_channels=channels_down[2][1], kernel_size = kernel_size, n_features = 33, stride = stride[1], padding = padding, bias = False)
            self.down4 = Down(in_channels=channels_down[3][0], out_channels=channels_down[3][1], kernel_size = kernel_size, n_features = 17, stride = stride[0], padding = padding, bias = False)
            self.down5 = Down(in_channels=channels_down[4][0], out_channels=channels_down[4][1], kernel_size = kernel_size, n_features = 9, stride = stride[1], padding = padding, bias = False)
            if self.n_conv_layers == 6:
                self.down6 = Down(in_channels=channels_down[5][0], out_channels=channels_down[5][1], kernel_size = kernel_size, n_features = 5, stride = stride[1], padding = padding, bias = False)
                self.recurrent = Recurrent(input_size = 512*5, hidden_size = 512*5, conv_channels = 512, type = recurrent_type, bidirectional = bidirectional, num_layers = 1, dropout = dropout)
            else:
                self.recurrent = Recurrent(input_size = 256*9, hidden_size = 256*9, conv_channels = 256, type = recurrent_type, bidirectional = bidirectional, num_layers = 1, dropout = dropout)
            self.up1 = Up(in_channels=channels_up[0][0], out_channels=channels_up[0][1], kernel_size = kernel_size, n_features = 9, stride = stride[1], padding = padding, bias = False)
            self.up2 = Up(in_channels=channels_up[1][0], out_channels=channels_up[1][1], kernel_size = kernel_size, n_features = 17, stride = stride[0], padding = padding, bias = False)
            self.up3 = Up(in_channels=channels_up[2][0], out_channels=channels_up[2][1], kernel_size = kernel_size, n_features = 33, stride = stride[1], padding = padding, bias = False)
            self.up4 = Up(in_channels=channels_up[3][0], out_channels=channels_up[3][1], kernel_size = kernel_size, n_features = 65, stride = stride[0], padding = padding, bias = False)
            if self.n_conv_layers == 6:
                self.up5 = Up(in_channels=channels_up[4][0], out_channels=channels_up[4][1], kernel_size = kernel_size, n_features =129, stride = stride[1], padding = padding, bias = False)
                self.final_conv = nn.ConvTranspose2d(in_channels=channels_up[5][0], out_channels=channels_up[5][1], kernel_size=kernel_size, stride=stride[1], padding=padding, bias=True)
            else:
                self.final_conv = nn.ConvTranspose2d(in_channels=channels_up[4][0], out_channels=channels_up[4][1], kernel_size=kernel_size, stride=stride[1], padding=padding, bias=True)
        else:
            self.down1 = Down(in_channels=channels_down[0][0], out_channels=channels_down[0][1], kernel_size = kernel_size, n_features = 129, stride = stride, padding = padding, bias = False)
            self.down2 = Down(in_channels=channels_down[1][0], out_channels=channels_down[1][1], kernel_size = kernel_size, n_features = 65, stride = stride, padding = padding, bias = False)
            self.down3 = Down(in_channels=channels_down[2][0], out_channels=channels_down[2][1], kernel_size = kernel_size, n_features = 33, stride = stride, padding = padding, bias = False)
            self.down4 = Down(in_channels=channels_down[3][0], out_channels=channels_down[3][1], kernel_size = kernel_size, n_features = 17, stride = stride, padding = padding, bias = False)
            self.down5 = Down(in_channels=channels_down[4][0], out_channels=channels_down[4][1], kernel_size = kernel_size, n_features = 9, stride = stride, padding = padding, bias = False)
            if self.n_conv_layers == 6:
                self.down6 = Down(in_channels=channels_down[5][0], out_channels=channels_down[5][1], kernel_size = kernel_size, n_features = 5, stride = stride, padding = padding, bias = False)
                self.recurrent = Recurrent(input_size = 512*5, hidden_size = 512*5, conv_channels = 512, type = recurrent_type, bidirectional = bidirectional, num_layers = 1, dropout = dropout)
            else:
                self.recurrent = Recurrent(input_size = 256*9, hidden_size = 256*9, conv_channels = 256, type = recurrent_type, bidirectional = bidirectional, num_layers = 1, dropout = dropout)
            self.up1 = Up(in_channels=channels_up[0][0], out_channels=channels_up[0][1], kernel_size = kernel_size, n_features = 9, stride = stride, padding = padding, bias = False)
            self.up2 = Up(in_channels=channels_up[1][0], out_channels=channels_up[1][1], kernel_size = kernel_size, n_features = 17, stride = stride, padding = padding, bias = False)
            self.up3 = Up(in_channels=channels_up[2][0], out_channels=channels_up[2][1], kernel_size = kernel_size, n_features = 33, stride = stride, padding = padding, bias = False)
            self.up4 = Up(in_channels=channels_up[3][0], out_channels=channels_up[3][1], kernel_size = kernel_size, n_features = 65, stride = stride, padding = padding, bias = False)
            if self.n_conv_layers == 6:
                self.up5 = Up(in_channels=channels_up[4][0], out_channels=channels_up[4][1], kernel_size = kernel_size, n_features =129, stride = stride, padding = padding, bias = False)
                self.final_conv = nn.ConvTranspose2d(in_channels=channels_up[5][0], out_channels=channels_up[5][1], kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
            else:
                self.final_conv = nn.ConvTranspose2d(in_channels=channels_up[4][0], out_channels=channels_up[4][1], kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.final_conv)

    def spectrogram(self, input):
        (real, imag) = self.stft(input)
        return (real ** 2 + imag ** 2) ** 0.5

    def wav_to_spectrogram(self, input):
        """Wav to spectrogram using STFT.

        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        return self.spectrogram(input[:, :, 0])

    def spectrogram_to_wav(self, input, spectrogram, length=None):
        """Spectrogram to wav using ISTFT.

        Args:
          input: (batch_size, segment_samples, channels_num)
          spectrogram: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, segment_samples, channels_num)
        """
        (real, imag) = self.stft(input[:, :, 0])
        (_, cos, sin) = magphase(real, imag)
        return self.istft(spectrogram * cos, spectrogram * sin, length)

    def forward(self, input):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        #sp = self.wav_to_spectrogram(input)    # (batch_size, channels_num, time_steps, freq_bins)

        x1 = self.down1(input)
        del input
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        if self.n_conv_layers == 6:
            x6 = self.down6(x5)
            x7 = self.recurrent(x6)
            x = self.up1(x7, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)
        else:
            x6 = self.recurrent(x5)
            x = self.up1(x6, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        #x = torch.sigmoid(self.final_conv(x))
        x = self.final_conv(x)

        # Spectrogram
        if self.config["training"]["target"] == "spec":
            sp_out = x
        elif self.config["training"]["target"] == "cIRM" or self.config["training"]["target"] == "PSM":
            sp_out = torch.tanh(x)
        elif self.config["training"]["target"] == "IRM":
            sp_out = torch.sigmoid(x)
        #sp_out = x

        # Wav
        #length = input.shape[1]
        #wav_out = self.spectrogram_to_wav(input, sp_out, length)

        return sp_out
