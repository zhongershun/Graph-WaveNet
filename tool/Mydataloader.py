from tool.MyDataset import MyDataset
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torch

def collate_fn(batch):
    print(len(batch[0]))
    is_train = True if len(batch[0])==2 else False
    if is_train:
        # print('touch')
        x = [b[0] for b in batch]
        y = [b[1] for b in batch]
        return torch.FloatTensor(x), torch.FloatTensor(y)
    else:
        # x = [b for b in batch]
        return batch

def Mydataloader(config):
    dataset = MyDataset(config=config)
    if config.mode == 'train':
        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.BATCH_SIZE,
            collate_fn=collate_fn,
            sampler=sampler.RandomSampler(range(config.TRAIN_NUM))
        )
        valid_dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.BATCH_SIZE,
            collate_fn=collate_fn,
            sampler=sampler.RandomSampler(range(config.TRAIN_NUM,config.TRAIN_NUM+config.VALID_NUM))
        )
        return train_dataloader, valid_dataloader
    elif config.mode == 'test':
        test_dataloader = DataLoader(
            dataset=dataset,
            # collate_fn=collate_fn,
            batch_size=config.BATCH_SIZE
        )
        return test_dataloader
    
