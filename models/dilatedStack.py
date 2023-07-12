import torch
import torch.nn as nn

from models.residualLayer import ResidualLayer

class DilatedStack(nn.Module):
    def __init__(self, config):
        super(DilatedStack, self).__init__()
        self.config = config
        residual_stack = [ResidualLayer(dilation=d, config=self.config)
                         for d in config.dilation]
        self.residual_stack = nn.ModuleList(residual_stack)
        
    def forward(self, x, adjs):
        skips = []
        for layer in self.residual_stack:
            x, skip = layer(x,adjs)
            skips.append(skip.unsqueeze(0))
            #skip=[1,batch,skip_channel,num_sensor,seq_len(12)]  
            # x=[batch,residual_channel,num_sensor,seq_len]
        return torch.cat(skips, dim=0), x  # [layers,batch,skip_channel,num_sensor,seq_len(12)]