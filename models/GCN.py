import torch 
import torch.nn as nn
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        # x = (batchsize, resudial_size, num_sensor, time_step(12))
        # A = (num_sensor, num_sensor)
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        # x = (batchsize, resudial_size, num_sensor, time_step(12))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,config):
        super(gcn,self).__init__()
        self.config = config
        self.nconv = nconv()
        c_in = (config.adj_lens+1)*config.residual_channels
        self.mlp = linear(c_in,config.residual_channels)
        self.dropout = config.gcn_dropout

    def forward(self,x,adjs):
        out = [x]
        # print("touch")
        # print(out.shape)
        assert len(adjs) == self.config.adj_lens

        for adj in adjs :
            x1 = self.nconv(x,adj)
            out.append(x1)

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

# if __name__ == '__main__':
    
#     x = torch.FloatTensor(10,12,170,32)
#     # x = torch.squeeze(x)
#     x = x.permute(0,3,2,1)
#     adj = torch.FloatTensor(170,170)
#     print(x.shape)
#     # x_transpose = x.transpose(1,3)
#     # print(x_transpose.shape)
#     # print(x_transpose[:,0,:,:].shape)
#     model = gcn(c_in=config.residual_channels,config=config)
#     print(model)
#     y = model(x,adj)
#     print(y.shape)