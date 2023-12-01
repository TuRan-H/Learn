from ast import Module
from networkx import reconstruct_path
from sympy import beta, linear_eq_to_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import csv
import sys, os


myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


def set_device():
    """
    获取 cuda 设备
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


class COVID19Dataset(Dataset):
    def __init__(self, path, mode='train', target_only='False') -> None:
        """
        自定义的dataset

        Paramenter
        ---
        path: 数据集.csv文件的位置

        mode: 当前模式 test; train; test

        target_only: 对特定的目标进行处理
        """
        super(COVID19Dataset, self).__init__()
        self.path = path
        self.target_only = target_only 
        self.mode = mode

        # * data是csv文件, 一共有 2701 * 95
        data = pd.read_csv(path)    # 使用pandas读入csv文件, 这种方式会自动删除第一行(也就是csv文件的标题行) data: 2700 * 95
        data = np.array(data)[:, 1:]   # 将data裁剪掉第一列 data: 2700 * 94

        # TODO  选取特征列
        feats = list(range(93)) # 特征列是从0 ~ 92列, 其中93列是标签列


        if mode == 'test':
            data = data[:, feats]
            temp = data
            self.data = torch.FloatTensor(data)

        else:
            target =  data[:, -1] # 切片下最后一列, 最后一列作为target(label)使用
            data = data[:, feats] # 选取特征列作为输入

            # 单独创建索引, 用于供给train和dev分割数据
            if mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]  
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            self.data = torch.tensor(data[indices], dtype=torch.float32)
            self.target = torch.tensor(target[indices], dtype=torch.float32)

        # **************************************************normalization
        self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) / self.data[:, 40:].std(dim=0, keepdim=True)
        # **************************************************normalization

        self.dim = self.data.shape[1]
        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(self.mode, len(self.data), self.dim))
        

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
           return self.data[index] , self.target[index]
        elif self.mode == 'test':
            return self.data[index]

    def __len__(self):
        return len(self.data)

def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    """
    对于dataloader的初始化

    params
    ---
    path: 数据集的地址

    mode: train | dev | test

    n_jobs: 工作进程数

    target_only: 是否自定义特征值
    """
    dataset = COVID19Dataset(path, mode, target_only)
    if mode == 'train':
        dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=False, num_workers=n_jobs, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False, num_workers=n_jobs, pin_memory=True)
    
    return dataloader


class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        """
        Param
        ---
        input_dim: 输入数据的特征值的个数, 也可以理解为输入数据有多少列
        """
        super(NeuralNet, self).__init__()
        # TODO 修改网络的隐藏层, 以达到更好地效果
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x:torch.tensor):
        """
        前向传播
        """
        prediction = self.net(x).squeeze(1)    # ? 为什么使用了squneeze函数之后, 整个loss就变得特别小, 否则loss就始终维持在50?

        return prediction
    
    def cal_loss(self, prediction:torch.tensor, target:torch.tensor):
        """
        计算输出的loss
        """
        # TODO 实现L1/ L2 范数正规化
        loss = self.criterion(prediction, target)

        return loss


def dev(dev_dataloader:COVID19Dataset, model:NeuralNet, device):
    """
    计算 dev 的平均 loss

    params
    ---
    dev_dataloader: 验证集dev的dataloader
    model: 神经网络类

    return
    ---
    返回训练过的模型在dev集上的平均loss
    """
    model.eval()
    total_loss = 0
    for data, target in dev_dataloader:
        data , target = data.to(device), target.to(device)
        with torch.no_grad():
            prediction = model(data)
            loss = model.cal_loss(prediction, target)
        total_loss += loss.detach().cpu().item() * len(data)
    total_loss = total_loss / len(dev_dataloader.dataset)
    return total_loss


def train(train_dataloader:COVID19Dataset, dev_dataloader:COVID19Dataset, model:NeuralNet, config:dict, device):
    """
    对模型进行训练, 并将模型保存

    params
    ---
    train_dataloader: 训练数据集
    dev_dataloader: 验证数据集
    model: 模型
    device: 设备
    """
    n_epochs = config['n_epochs']
    early_stop = config['early_stop']
    save_path = config['save_path']
    optimizer:torch.optim.SGD = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    loss_record = {'train':[], 'dev':[]}
    model.to(device)

    epoch = 0
    min_mse = 1000
    while epoch < n_epochs:
        model.train()
        for data, target in train_dataloader:
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            prediction = model(data)
            loss = model.cal_loss(prediction, target)
            loss.backward()
            optimizer.step()
            train_mse = loss.detach().cpu().item()
            loss_record['train'].append(train_mse)
        dev_mse = dev(dev_dataloader, model, device)

        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss_dev = {:.4f}, loss_train = {:.4f})'.format(epoch + 1, min_mse, train_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        loss_record['dev'].append(dev_mse)
        epoch += 1

        if early_stop_cnt > early_stop:
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


def test(test_dataloader:COVID19Dataset, model:NeuralNet, device):
    """
    对模型行进测试, 并返回模型在 public test dataset 上面的测试结果
    """
    model.eval()
    model.to(device)
    predictions = []
    print()
    for data in test_dataloader:
        data = data.to(device)
        with torch.no_grad():
            prediction = model(data)
            predictions.append(prediction.detach().cpu())
    predictions = torch.cat(predictions, dim=0).numpy()
    print("the result of model has been predicted")
    
    return predictions

def save_result(predictions, config):
    """
    保存模型在 public test dataset 上面的测试结果到 output_file 中

    params
    ---
    predictions: 模型测测试结果

    config: 配置文件
    """
    output_file = config['output_path']
    with open(output_file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for index, prediction in enumerate(predictions):
            writer.writerow([index, prediction])


def load_model(load_path, input_dim) -> NeuralNet:
    """
    从pth文件中加载模型

    params
    ---
    load_path: pth文件地址

    input_dim: 模型的特征值个数
    """
    model = NeuralNet(input_dim)
    ckpt = torch.load(load_path, map_location='cpu')
    model.load_state_dict(ckpt)
    return model


# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 3000,                
    'batch_size': 64,
    'optimizer': 'SGD',              
    'optim_hparas': {                
        'lr': 0.001,
        'momentum': 0.10
    },
    'early_stop': 200,
    'save_path': 'models/model.pth',
    'output_path': 'output/model_result/result.csv'
}

device = set_device()
os.makedirs('models', exist_ok=True)
target_only = False                   # TODO: Using 40 states & 2 tested_positive features
train_path = 'datasets/covid.train.csv'
test_path = 'datasets/covid.test.csv'

train_set = prep_dataloader(train_path, 'train', config['batch_size'], 0, target_only)
dev_set = prep_dataloader(train_path, 'dev', config['batch_size'], target_only)
test_set = prep_dataloader(test_path, 'test', config['batch_size'], 0, target_only)

model = NeuralNet(train_set.dataset.dim).to(device)
min_mse, loss_record = train(train_set, dev_set, model, config, device)

predictions = test(test_set, model, device)
save_result(predictions, config)