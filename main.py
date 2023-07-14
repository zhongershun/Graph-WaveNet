import torch
from models.myModel import Mymodel
from config import config

from tool.Mydataloader import Mydataloader
from tool.MyDataset import MyDataset

from tool.Trainer import trainer

from tqdm import tqdm
import numpy as np
import pandas as pd

import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',default='train',help='train or test the model')
    parser.add_argument('--epoch',default=10,help='epochs of training',type=int)
    parser.add_argument('--device',type=str,default='cuda',help='')
    
    parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
    parser.add_argument('--num_sensor',type=int,default=170,help='number of sensor')
    parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--wd',type=float,default=0.0001,help='weight decay rate')
    parser.add_argument('--save',type=str,default='./save_model',help='save path')

    args = parser.parse_args()

    config.mode = args.mode
    config.epoch = args.epoch
    config.in_dim = args.in_dim
    config.gcn_dropout = args.dropout
    config.lr = args.lr
    config.wd = args.wd
    config.model_output = args.save

    config.device = torch.device(args.device)
    
    print(args)
    return



def prework():
    if config.mode=='train':
        train_dataloader, valid_dataloader, scaler = Mydataloader(config=config)
        return train_dataloader, valid_dataloader, scaler
    elif config.mode=='test':
        test_dataloader, scaler = Mydataloader(config=config)
        return test_dataloader, scaler
    

def train(train_dataloader, valid_dataloader, scaler):
    edge = pd.read_csv(config.train_adj_path)
    adj = np.zeros((config.num_sensor,config.num_sensor))*10000
    FROM = edge['from']
    TO = edge['to']
    cost = edge['cost']
    for i in range(len(FROM)):
        # print(FROM[i])
        adj[FROM[i]][TO[i]] = cost[i]
    # print(adj)
    adj = torch.Tensor(adj).to(config.device)
    model = Mymodel(adjs=[adj],config=config)
    mytrainer = trainer(model=model,config=config)

    his_loss =[]
    
    # 画图用
    all_train_loss = []
    his_val_loss =[]
    all_train_mape = []
    all_train_rmse = []
    all_val_mape = []
    all_val_rmse = []
    
    
    val_time = []
    train_time = []
    for e in range(config.epoch):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        print('-' * 20 + ' ' + 'Epoch ' + str(e+1) + ' ' + '-' * 20)
        i = 0
        for x, y in tqdm(train_dataloader):
            i = i+1
            x = torch.Tensor(x).to(config.device)
            y = torch.Tensor(y).to(config.device)
            metrics = mytrainer.train(x,y,scaler)

            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if i % 30 == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(i, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        all_train_loss.extend(train_loss)
        all_train_mape.extend(train_mape)
        all_train_rmse.extend(train_rmse)


        s1 = time.time()
        for x, y in tqdm(valid_dataloader):
            testx = torch.Tensor(x).to(config.device)
            testy = torch.Tensor(y).to(config.device)
            metrics = mytrainer.valid(testx, testy, scaler)

            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        all_val_mape.extend(valid_mape)
        all_val_rmse.extend(valid_rmse)
        print(log.format(e,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        
        his_val_loss.extend(valid_loss)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(e, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(mytrainer.model.state_dict(), config.model_output+"/epoch_"+str(e)+".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    all_train_loss = np.array(all_train_loss)
    his_val_loss = np.array(his_val_loss)
    np.savetxt('./DataLog/alllosses.csv', all_train_loss, delimiter=',', fmt='%.3f')
    np.savetxt('./DataLog/vallosses.csv', his_val_loss, delimiter=',', fmt='%.3f')
    np.savetxt('./DataLog/all_train_mape.csv', all_train_mape, delimiter=',', fmt='%.3f')
    np.savetxt('./DataLog/all_train_rmse.csv', all_train_rmse, delimiter=',', fmt='%.3f')
    np.savetxt('./DataLog/all_val_mape.csv', all_val_mape, delimiter=',', fmt='%.3f')
    np.savetxt('./DataLog/all_val_rmse.csv', all_val_rmse, delimiter=',', fmt='%.3f')

def test(test_dataloader,scaler):
    edge = pd.read_csv(config.train_adj_path)
    adj = np.zeros((config.num_sensor,config.num_sensor))*10000
    FROM = edge['from']
    TO = edge['to']
    cost = edge['cost']
    for i in range(len(FROM)):
        # print(FROM[i])
        adj[FROM[i]][TO[i]] = cost[i]
    # print(adj)
    model = Mymodel(adjs=[adj],config=config)
    model.to(config.device)
    model.load_state_dict(torch.load(''))
    model.eval()
    mytrainer = trainer(model=model,config=config)

    pred_ys = []
    for x in tqdm(test_dataloader):
        testx = torch.Tensor(x).to(config.device)
        # testy = torch.Tensor(y).to(config.device)
        pred_y = mytrainer.predict(testx,scaler)
        pred_ys.append(pred_y)

if __name__ == '__main__':
    parse_args()
    if config.mode == 'train':
        train_dataloader,valid_dataloader,scaler = prework()
        train(train_dataloader,valid_dataloader,scaler)
    elif config.mode == 'test':
        test_dataloader,scaler = prework()
        test(test_dataloader, scaler)
