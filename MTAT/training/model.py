# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchaudio

from modules import Conv_1d, ResSE_1d, Conv_2d, Res_2d, Conv_V, Conv_H, HarmonicSTFT, Res_2d_mp
from attention_modules import BertConfig, BertEncoder, BertEmbeddings, BertPooler, PositionalEncoding

class Maxout(nn.Module):
    def __init__(self, in_features, out_features, pool_size=2, batch_norm=True):
        super(Maxout, self).__init__()
        self.out_features = out_features
        self.pool_size = pool_size
        self.linear = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(pool_size)])
        if batch_norm:
            self.bn = nn.ModuleList([nn.BatchNorm1d(out_features) for _ in range(pool_size)])
        else:
            self.bn = nn.ModuleList([nn.Identity() for _ in range(pool_size)])

    def forward(self, x):
        maxout_layers = torch.stack([self.bn[i](layer(x)) for i, layer in enumerate(self.linear)], dim=2) 
        maxout_output = torch.max(maxout_layers, dim=2).values
        return maxout_output

class MaxPlus(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init="full"):
        super(MaxPlus, self).__init__()
        self.bias = bias
        if init == "full":
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
        else:
            w = torch.randn(out_features, in_features)
            mask = torch.rand(out_features, in_features) > init/(out_features * in_features)
            w = w - 1e9 * mask
            self.weight = nn.Parameter(w)
        self.b = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = x + self.weight
        if self.bias:
            x = torch.cat([x, self.b.view(1,-1,1).repeat(x.size(0),1,1)], dim=2)
        x, _ = torch.max(x, dim=2)
        return x

class ShortChunkCNN(nn.Module):
    '''
    Short-chunk CNN architecture.
    So-called vgg-ish model with a small receptive field.
    Deeper layers, smaller pooling (2x2).
    '''
    def __init__(self,
                n_channels=128,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=128,
                n_class=50,
                method='relu'):
        super(ShortChunkCNN, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Conv_2d(1, n_channels, pooling=2)
        self.layer2 = Conv_2d(n_channels, n_channels, pooling=2)
        self.layer3 = Conv_2d(n_channels, n_channels*2, pooling=2)
        self.layer4 = Conv_2d(n_channels*2, n_channels*2, pooling=2)
        self.layer5 = Conv_2d(n_channels*2, n_channels*2, pooling=2)
        self.layer6 = Conv_2d(n_channels*2, n_channels*2, pooling=2)
        self.layer7 = Conv_2d(n_channels*2, n_channels*4, pooling=2)

        # Dense
        if method == "relu":
            self.dense1 = nn.Linear(n_channels*4, n_channels*4)
            self.bn = nn.BatchNorm1d(n_channels*4)
            self.dense2 = nn.Linear(n_channels*4, n_class)
            self.dropout = nn.Dropout(0.5)
            self.relu = nn.ReLU()
        elif method == "maxout": 
            self.dense1 = Maxout(n_channels*4, n_channels*4, pool_size=2, batch_norm=True)
            self.bn = nn.Identity()
            self.dense2 = nn.Linear(n_channels*4, n_class)
            self.dropout = nn.Dropout(0.5)
            self.relu = nn.Identity()
        elif method == "zhang":
            self.dense1 = nn.Linear(n_channels*4, n_channels*4, bias=False)
            self.bn = nn.Identity()
            self.dropout = nn.Dropout(0.5)
            self.dense2 = MaxPlus(n_channels*4, n_class)
            self.relu = nn.ReLU()
        elif method == "lmpl": 
            self.dense1 = nn.Linear(n_channels*4, n_channels*4, bias=False)
            self.relu = MaxPlus(n_channels*4, n_channels*4)
            self.bn = nn.Identity()
            self.dense2 = nn.Linear(n_channels*4, n_class)
            self.dropout = nn.Identity()
        elif method == "lmplbn": 
            self.dense1 = nn.Linear(n_channels*4, n_channels*4, bias=False)
            self.relu = MaxPlus(n_channels*4, n_channels*4)
            self.bn = nn.BatchNorm1d(n_channels*4)
            self.dense2 = nn.Linear(n_channels*4, n_class)
            self.dropout = nn.Identity()
        elif method == "lmpl2":
            self.dense1 = nn.Linear(n_channels*4, n_channels*4, bias=False)
            self.relu = MaxPlus(n_channels*4, n_channels*4, init=8*n_channels)
            self.bn = nn.Identity()
            self.dense2 = nn.Linear(n_channels*4, n_class)
            self.dropout = nn.Identity()
        elif method == "lmpl2bn":
            self.dense1 = nn.Linear(n_channels*4, n_channels*4, bias=False)
            self.relu = MaxPlus(n_channels*4, n_channels*4, init=8*n_channels)
            self.bn = nn.BatchNorm1d(n_channels*4)
            self.dense2 = nn.Linear(n_channels*4, n_class)
            self.dropout = nn.Identity()
        else:
            print("Invalid method!!!")
            exit(-1)

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # if self.noise:
        #     x = add_salt_and_pepper_noise(x, 0.8)
        #     x = self.layer0(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dens
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x
