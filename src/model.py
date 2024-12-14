import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from einops import rearrange

import my_utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('mps') if torch.mps.is_available() else device

class FFDNet(nn.Module):

    def __init__(self):
        super(FFDNet, self).__init__()
        self.num_conv_layers = 12
        self.downsampled_channels = 15
        self.num_feature_maps = 96
        self.output_features = 12
        
        self.kernel_size = 3
        self.padding = 1
        
        layers = []
        # Conv + Relu
        layers.append(nn.Conv2d(in_channels=self.downsampled_channels, out_channels=self.num_feature_maps, \
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Conv + BN + Relu
        for _ in range(self.num_conv_layers - 2):
            layers.append(nn.Conv2d(in_channels=self.num_feature_maps, out_channels=self.num_feature_maps, \
                                    kernel_size=self.kernel_size, padding=self.padding, bias=False))
            layers.append(nn.BatchNorm2d(self.num_feature_maps))
            layers.append(nn.ReLU(inplace=True))
        
        # Conv
        layers.append(nn.Conv2d(in_channels=self.num_feature_maps, out_channels=self.output_features, \
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))

        self.intermediate_dncnn = nn.Sequential(*layers)

    def forward(self, x, noise_sigma):
        noise_map = noise_sigma.view(x.shape[0], 1, 1, 1).repeat(1, x.shape[1], x.shape[2] // 2, x.shape[3] // 2)
        noise_map = noise_map.to(device)
        x_up = my_utils.downsample(x.data) # 4 * C * H/2 * W/2
        x_up = x_up.to(device)
        x_cat = torch.cat((noise_map.data, x_up), 1) # 4 * (C + 1) * H/2 * W/2
        x_cat = Variable(x_cat)
        x_cat = x_cat.to(device)
        # x_cat.to('mps')                           #Just set to mps for on mac
        h_dncnn = self.intermediate_dncnn(x_cat)
        y_pred = my_utils.upsample(h_dncnn)
        return y_pred

class resNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=.2):
        super().__init__()
        self.activation = nn.ReLU()
        # self.norm = nn.BatchNorm2d(num_features=in_channels)

        self.resBlock = nn.Sequential(
            # nn.BatchNorm2d(in_channels),
            # nn.ReLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self.resBlock(x)
class ResidualFFDNet(nn.Module):

    def __init__(self):
        super(ResidualFFDNet, self).__init__()

        self.num_conv_layers = 12
        self.downsampled_channels = 15
        self.num_feature_maps = 96
        self.output_features = 12
            
        self.kernel_size = 3
        self.padding = 1
        
        layers = []
        # Conv + Relu
        layers.append(nn.Conv2d(in_channels=self.downsampled_channels, out_channels=self.num_feature_maps, \
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Conv + BN + Relu
        for _ in range(self.num_conv_layers - 2):
            # layers.append(nn.Conv2d(in_channels=self.num_feature_maps, out_channels=self.num_feature_maps, \
            #                         kernel_size=self.kernel_size, padding=self.padding, bias=False))
            # layers.append(nn.BatchNorm2d(self.num_feature_maps))
            # layers.append(nn.ReLU(inplace=True))
            layers.append(resNetBlock(in_channels=self.num_feature_maps,out_channels=self.num_feature_maps))
        
        # Conv
        layers.append(nn.Conv2d(in_channels=self.num_feature_maps, out_channels=self.output_features, \
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))

        self.intermediate_dncnn = nn.Sequential(*layers)

    def forward(self, x, noise_sigma):
        noise_map = noise_sigma.view(x.shape[0], 1, 1, 1).repeat(1, x.shape[1], x.shape[2] // 2, x.shape[3] // 2)
        noise_map = noise_map.to(device)
        x_up = my_utils.downsample(x.data) # 4 * C * H/2 * W/2
        x_up = x_up.to(device)
        x_cat = torch.cat((noise_map.data, x_up), 1) # 4 * (C + 1) * H/2 * W/2
        x_cat = Variable(x_cat)
        x_cat = x_cat.to(device)
        # x_cat.to('mps')                           #Just set to mps for on mac
        h_dncnn = self.intermediate_dncnn(x_cat)
        y_pred = my_utils.upsample(h_dncnn)
        return y_pred

class self_attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_q = nn.Conv1d(in_channels=12,out_channels=36,kernel_size=1)
        self.w_k = nn.Conv1d(in_channels=12,out_channels=36,kernel_size=1)
        self.w_v = nn.Conv1d(in_channels=12,out_channels=36,kernel_size=1)
        self.w_o = nn.Conv1d(in_channels=36,out_channels=12,kernel_size=1)
        # self.scaling_factor = 1/torch.sqrt(d_model)

        #What model parameters are needed for attention block?
        self.softmax = nn.Softmax(-1)
    def forward(self,x):
        dim = torch.tensor(x.shape[1])
        scaling_factor = 1/torch.sqrt(dim)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        inter = torch.matmul(k.transpose(1,2),q)
        value_weights = self.softmax(inter*scaling_factor)
        return self.w_o(torch.matmul(v,value_weights))

class AttentionFFDNet(nn.Module):

    def __init__(self):
        super(AttentionFFDNet, self).__init__()
        self.num_conv_layers = 12
        self.downsampled_channels = 15
        self.num_feature_maps = 96
        self.output_features = 12
        self.attention = self_attention(112)
        # self.attention = nn.MultiheadAttention(12544,1)
            
        self.kernel_size = 3
        self.padding = 1
        
        layers = []
        # Conv + Relu
        layers.append(nn.Conv2d(in_channels=self.downsampled_channels, out_channels=self.num_feature_maps, \
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Conv + BN + Relu
        for _ in range(self.num_conv_layers - 2):
            layers.append(nn.Conv2d(in_channels=self.num_feature_maps, out_channels=self.num_feature_maps, \
                                    kernel_size=self.kernel_size, padding=self.padding, bias=False))
            layers.append(nn.BatchNorm2d(self.num_feature_maps))
            layers.append(nn.ReLU(inplace=True))
        
        # Conv
        layers.append(nn.Conv2d(in_channels=self.num_feature_maps, out_channels=self.output_features, \
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))

        self.intermediate_dncnn = nn.Sequential(*layers)

    def forward(self, x, noise_sigma):
        noise_map = noise_sigma.view(x.shape[0], 1, 1, 1).repeat(1, x.shape[1], x.shape[2] // 2, x.shape[3] // 2)
        noise_map = noise_map.to(device)
        x_up = my_utils.downsample(x.data) # 4 * C * H/2 * W/2
        x_residual = x_up
        x_residual = x_residual.to(device)
        x_up = x_up.to(device)
        # x_up = x_up.squeeze().flatten(1)
        # x_up = self.attention(x_up).reshape((-1,12,112,112))
        x_cat = torch.cat((noise_map.data, x_up), 1) # 4 * (C + 1) * H/2 * W/2
        x_cat = Variable(x_cat)
        x_cat = x_cat.to(device)
        # x_cat.to('mps')                           #Just set to mps for on mac
        h_dncnn = self.intermediate_dncnn(x_cat)
        # print(h_dncnn.shape)
        # print(f"Before attention min/max: {h_dncnn.min()},{h_dncnn.max()}")
        # print(torch.min(torch.min(h_dncnn,3).values,2))
        # print(torch.max(torch.max(h_dncnn,3).values,2))

        h_dncnn = h_dncnn.flatten(2)
        # h_dncnn = h_dncnn.reshape((-1,12,112,112))
        h_dncnn = self.attention(h_dncnn).reshape((-1,12,112,112))
        # h_dncnn = self.attention(h_dncnn,h_dncnn,h_dncnn)[0].reshape((-1,12,112,112))
        # print(f"After attention min/max: {h_dncnn.min()},{h_dncnn.max()}") 
        # print(torch.min(torch.min(h_dncnn,3).values,2))
        # print(torch.max(torch.max(h_dncnn,3).values,2))
        # h_dncnn =               #Need different attention layer for this?
        y_pred = my_utils.upsample(h_dncnn+x_residual)
        # print(y_pred.shape)
        # print(torch.min(torch.min(y_pred,3).values,2))
        # print(torch.max(torch.max(y_pred,3).values,2))
        return y_pred