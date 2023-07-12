import os
class config:
    #目录信息
    root_path = os.getcwd()
    train_data_path = os.path.join(root_path, 'data/Data.npy')
    test_data_path = os.path.join(root_path, 'data/test_x.npy')
    train_adj_path = os.path.join(root_path, 'data/Data.csv')

    mode = 'train'
    BATCH_SIZE = 10
    TRAIN_NUM = 6000
    VALID_NUM = 1130

    #划分训练集相关
    per_step = 2


    residual_channels = 32
    #TCN相关
    tcn_in_dim = 1