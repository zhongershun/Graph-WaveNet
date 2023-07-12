import os
class config:
    #目录信息
    root_path = os.getcwd()
    train_data_path = os.path.join(root_path, 'data/Data.npy')
    test_data_path = os.path.join(root_path, 'data/test_x.npy')
    train_adj_path = os.path.join(root_path, 'data/Data.csv')

    device = None

    mode = 'train'
    BATCH_SIZE = 10
    TRAIN_NUM = 6000
    VALID_NUM = 1130

    #划分训练集相关
    per_step = 2
    adj_lens = 1


    residual_channels = 32
    skip_channels = 256
    end_channels = 512
    out_dim = 12
    num_sensor=170
    dilation = [1,2,4,8]
    blocks = 4

    #TCN相关
    tcn_in_dim = 1

    #GCN相关
    gcn_dropout = 0.3
