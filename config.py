import os
class config:
    #目录信息
    root_path = os.getcwd()
    train_data_path = os.path.join(root_path, 'data/Data.npy')
    test_data_path = os.path.join(root_path, 'data/test_x.npy')
    train_adj_path = os.path.join(root_path, 'data/Data.csv')

    mode = 'train'
    BATCH_SIZE = 10
    TRAIN_NUM = 1000
    VALID_NUM = 190