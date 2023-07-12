import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dilatedStack import DilatedStack
from models.start_conv import start_conv

class Mymodel(nn.Module):

    def __init__(self,adjs,config):

        super(Mymodel, self).__init__()
        self.config = config
        self.adjs = adjs
        config.adj_lens = len(adjs)+1
        self.input_conv = start_conv(config=config)        
        self.dilated_stacks = nn.ModuleList(
            [DilatedStack(config=config)
             for block in range(config.blocks)]
        )

        self.end_conv_1 = nn.Conv2d(in_channels=config.skip_channels,
                                  out_channels=config.end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=config.end_channels,
                                    out_channels=config.out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
        self.nodevec1 = nn.Parameter(torch.randn(config.num_sensor, 10).to(config.device), requires_grad=True).to(config.device)
        self.nodevec2 = nn.Parameter(torch.randn(10, config.num_sensor).to(config.device), requires_grad=True).to(config.device)

    def forward(self, x):

        # x=[batch,seq_len(12),num_sensor,1]
        x = x.permute(0,3,2,1)# [batch,1,num_sensor,seq_len(12)]

        x = self.input_conv(x) # [batch,residual_size,num_sensor,seq_len(12)]             

        skip_connections = []

        adjs = None
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        adjs = self.adjs + [adp]

        for cycle in self.dilated_stacks:

            skips, x = cycle(x,adjs)             
            # print()
            skip_connections.append(skips)

        skip_connections = torch.cat(skip_connections, dim=0)        
        ## skip_connection=[total_layers,batch,skip_channel,num_sensor,seq_len(12)]

        # gather all output skip connections to generate output, discard last residual output

        out = skip_connections.sum(dim=0) # [batch,skip_channel,num_sensor,seq_len(12)]

        out = F.relu(out)

        out = self.end_conv_1(out) # [batch,out_size,num_sensor,seq_len]
        out = F.relu(out)

        out=self.end_conv_2(out)
        return out     