import torch
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,config):
        self.config = config
        if config.mode == 'train':
            # print('touch')
            train_data = np.load(config.train_data_path)
            # train_data = train_data[:-4]
            self.data = train_data
            # print(self.data.shape)
            # print(int((self.data.shape[0]-12-12)/self.config.per_step)+1)
            train_data_clip_x = []
            train_data_clip_y = []
            for hour_time_step in range(int((self.data.shape[0]-12-12)/self.config.per_step)+1):
                train_data_clip_x.append(self.data[hour_time_step*2:hour_time_step*2+12])
                train_data_clip_y.append(self.data[hour_time_step*2+12:hour_time_step*2+24])
            train_data_clip_x = np.array(train_data_clip_x)
            train_data_clip_y = np.array(train_data_clip_y)
            self.data_x = train_data_clip_x
            self.data_y = train_data_clip_y
        elif config.mode == 'test':
            test_data = np.load(config.test_data_path)
            self.data = test_data
            self.data_x = test_data
            self.data_y = None
        else:
            self.data = None
            
            
    def __getitem__(self, i):
        if self.config.mode=='train':
            return (self.data_x[i], self.data_y[i])
        elif self.config.mode=='test':
            return self.data[i]
        
    def __len__(self):
        return len(self.data_x)
    
    def getx_data(self):
        return self.data_x