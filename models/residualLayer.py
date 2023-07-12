import torch
import torch.nn as nn

from models.causalconv2d import CausalConv2d
from models.GCN import gcn


import os
class config:
    #目录信息
    root_path = os.getcwd()
    train_data_path = os.path.join(root_path, 'data/Data.npy')
    test_data_path = os.path.join(root_path, 'data/test_x.npy')
    train_adj_path = os.path.join(root_path, 'data/Data.csv')

    mode = 'train'
    BATCH_SIZE = 10
    TRAIN_NUM = 6000
    VALID_NUM = 1130

    #划分训练集相关
    per_step = 2


    residual_channels = 32
    skip_channels=256
    num_sensor=170
    #TCN相关
    tcn_in_dim = 1

    #GCN相关
    gcn_dropout = 0.3


class ResidualLayer(nn.Module):    
    def __init__(self, dilation,config):
        super(ResidualLayer, self).__init__()
        self.config = config
        self.conv_filter = CausalConv2d(config.residual_channels, config.residual_channels,
                                         kernel_size=2, dilation=dilation)
        self.conv_gate = CausalConv2d(config.residual_channels, config.residual_channels,
                                         kernel_size=2, dilation=dilation)  
        self.skipconv = nn.Conv2d(config.residual_channels, config.skip_channels,
                                         kernel_size=(1,1))
        self.gconv = gcn(config=config)   
        self.bn = nn.BatchNorm2d(config.residual_channels)   
        
   
    def forward(self, x, adjs):
        residual = x
        conv_filter = self.conv_filter(residual)
        conv_gate = self.conv_gate(residual)  
        x = torch.tanh(conv_filter) * torch.sigmoid(conv_gate) 
        # print("x",x.shape)
        skip = self.skipconv(x)
        # print(skip.shape)
        x = self.gconv(x,adjs)
        
        x = x + residual
        x = self.bn(x)
        #skip=[batch,skip_channel,num_sensor,seq_len(12)]  x=[batch,residual_channel,num_sensor,seq_len]
        return x, skip
    

# if __name__ == '__main__':
    
#     x = torch.FloatTensor(10,12,170,32)
#     # x = torch.squeeze(x)
#     x = x.permute(0,3,2,1)
#     adj = torch.FloatTensor(170,170)
#     print(x.shape)
#     # x_transpose = x.transpose(1,3)
#     # print(x_transpose.shape)
#     # print(x_transpose[:,0,:,:].shape)
#     model = ResidualLayer(dilation=1,config=config)
#     print(model)
#     x,skip = model(x,adj)
#     print(x.shape,skip.shape)