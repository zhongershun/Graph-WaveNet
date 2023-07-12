import torch
from models.myModel import Mymodel
from config import config

import torch
from models.myModel import Mymodel

config.device = torch.device('cpu')
x = torch.FloatTensor(10,12,170,1)
    # x = torch.squeeze(x)
# x = x.permute(0,3,2,1)
adj = torch.FloatTensor(170,170)
print(x.shape)
    # x_transpose = x.transpose(1,3)
    # print(x_transpose.shape)
    # print(x_transpose[:,0,:,:].shape)
model = Mymodel(adjs=[adj],config=config)
print(model)
y = model(x)
print(y.shape)