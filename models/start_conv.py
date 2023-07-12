import torch
import torch.nn as nn

class start_conv(nn.Module):
    def __init__(self,config):
        super(start_conv, self).__init__()
        self.config = config
        self.conv = nn.Conv2d(in_channels=1, out_channels=32,kernel_size=1)

    def forward(self,x):
        x = self.conv(x)
        return x
    

if __name__ == '__main__':
    x = torch.FloatTensor(10,12,170,1)
    # x = torch.squeeze(x)
    # x = x.permute(0,2,1)
    print(x.shape)
    x_transpose = x.transpose(1,3)
    print(x_transpose.shape)
    print(x_transpose[:,0,:,:].shape)
    model = start_conv(config=None)
    y = model(x_transpose)
    print(y.shape)