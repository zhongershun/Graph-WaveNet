import torch
import torch.nn as nn

# 作为模型数据输入的开始
class start_conv(nn.Module):
    def __init__(self,config):
        super(start_conv, self).__init__()
        self.config = config
        self.conv = nn.Conv2d(in_channels=config.tcn_in_dim, out_channels=config.residual_channels,kernel_size=1)

    def forward(self,x):
        x = self.conv(x)
        return x
    

if __name__ == '__main__':
    x = torch.FloatTensor(10,12,170,1)
    # x = torch.squeeze(x)
    x = x.permute(0,3,1,2)
    print(x.shape)
    # x_transpose = x.transpose(1,3)
    # print(x_transpose.shape)
    # print(x_transpose[:,0,:,:].shape)
    model = start_conv(config=None)
    y = model(x)
    print(y.shape)