import torch
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,config):
        self.config = config
        if config.mode == 'train':
            # print('touch')
            train_data = np.load(config.train_data_path)
            train_data = train_data[:-4]
            self.data = train_data
        elif config.mode == 'test':
            test_data = np.load(config.test_data_path)
            self.data = test_data
        else:
            self.data = None
            
            
    def __getitem__(self, i):
        if self.config.mode=='train':
            train_data_clip = []
            for hour_time_step in range(int(self.data.shape[0]/12)):
                train_data_clip.append(self.data[hour_time_step*12:(hour_time_step+1)*12])
            train_data_clip = np.array(train_data_clip)
            # print(train_data_clip.shape)
            return train_data_clip[i]
        elif self.config.mode=='test':
            return self.data[i]
        
    def __len__(self):
        return len(self.data)