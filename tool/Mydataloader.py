from tool.MyDataset import MyDataset
from torch.utils.data import DataLoader
from torch.utils.data import sampler

def Mydataloader(config):
    dataset = MyDataset(config=config)
    if config.mode == 'train':
        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.BATCH_SIZE,
            sampler=sampler.RandomSampler(range(config.TRAIN_NUM))
        )
        valid_dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.BATCH_SIZE,
            sampler=sampler.RandomSampler(range(config.TRAIN_NUM,config.TRAIN_NUM+config.VALID_NUM))
        )
        return train_dataloader, valid_dataloader
    elif config.mode == 'test':
        test_dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.BATCH_SIZE
        )
        return test_dataloader