"""
8.6 循环神经网络的简介实现
"""

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

from utils.data.time_machine import load_data_time_machine


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        """
        初始化RNN

        Args:
            rnn_layer (nn.Module): RNN layer, e.g., nn.RNN, nn.LSTM, nn.GRU.
        """
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        self.linear = nn.Linear(self.num_hiddens, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        return torch.zeros((self.rnn.num_layers, batch_size, self.num_hiddens), device=device)


if __name__ == "__main__":
    # 确定超参
    batch_size, num_steps = 32, 35
    num_hiddens = 256
    num_epochs, lr = 500, 1
    device = d2l.try_gpu()

    # 导入数据集
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)

    # 实例化模型
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)

    # 开始训练
    d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)