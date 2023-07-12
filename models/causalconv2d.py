import torch
import torch.nn as nn

class CausalConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=2, dilation=1):
        super(CausalConv2d, self).__init__()
        self.dilation = dilation
        pad = int((dilation+1)/2)
        self.causalconv2 = nn.Conv2d(in_channel, out_channel, kernel_size=(1,kernel_size), padding=(0,pad),dilation=dilation)

    def forward(self, x):
        fx = self.causalconv2(x)
        if fx.size(3) != x.size(3):
            fx = fx[:,:,:,:-x.size(3)]
        return x
    
if __name__ == '__main__':
    x = torch.FloatTensor(10,12,170,32)
    # x = torch.squeeze(x)
    x = x.permute(0,3,2,1)
    print(x.shape)
    # x_transpose = x.transpose(1,3)
    # print(x_transpose.shape)
    # print(x_transpose[:,0,:,:].shape)
    model = CausalConv2d(in_channel=32,out_channel=32,kernel_size=2)
    print(model)
    y = model(x)
    print(y.shape)