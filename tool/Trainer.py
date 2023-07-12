import torch.optim as optim
from tqdm import tqdm
import torch

import numpy as np

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse

class trainer:
    def __init__(self,config,model):
        self.config = config
        self.model = model
        self.device = config.device
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.wd)
        self.loss = masked_mae
        self.clip = 5
    

    def train(self,x,y):
        self.model.train()
        self.optimizer.zero_grad()
        
        # print(x.shape)
        # print(y.shape)
        # exit(0)
        pred_y = self.model(x)
        # (batch,1,num_sensor,12)
        # real = torch.unsqueeze(y,dim=1)
        loss = self.loss(pred_y, y, 0.0)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = masked_mape(pred_y,y,0.0).item()
        rmse = masked_rmse(pred_y,y,0.0).item()
        return loss.item(),mape,rmse
    
    def valid(self, x, y):
        self.model.eval()
        
        pred_y = self.model(x)
        # (batch,1,num_sensor,12)
        # real = torch.unsqueeze(y,dim=1)
        loss = self.loss(pred_y, y, 0.0)
        mape = masked_mape(pred_y,y,0.0).item()
        rmse = masked_rmse(pred_y,y,0.0).item()
        return loss.item(),mape,rmse
    
    def predict(self, x):
        self.model.eval()
        
        pred_y = self.model(x)
        return pred_y
        # (batch,1,num_sensor,12)
        # real = torch.unsqueeze(y,dim=1)

